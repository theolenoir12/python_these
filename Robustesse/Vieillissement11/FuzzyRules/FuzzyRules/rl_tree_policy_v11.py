"""Policy 'regles apprises' par arbre de regression distille de la PD (V11-p=2).

Un unique ``DecisionTreeRegressor`` regresse la commande signee ``u_h2`` a
partir des features I0 ``[P_net_w, SoC_norm, E_h2_norm]`` (ou IS avec les trois
colonnes d'usure). Une zone morte annule les micro-commandes H2 pour eviter le
cyclage marche/arret (cout de demarrage). La policy expose la meme interface que
``flc_policy_v11`` : un callable au signature de boucle qui delegue la
faisabilite a ``get_lol``.

Choix (arretes le 2026-07-21) :

- regression directe de ``u_h2`` + zone morte ``|u_h2| < deadband_w -> 0`` ;
- selection de la profondeur par J3 en boucle fermee (Etape B), la fidelite MSE
  n'etant qu'un diagnostic ;
- chaque feuille de l'arbre est une regle boite ; leur nombre est un resultat.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from Common.get_lol import get_lol
from .rl_dataset_v11 import (
    I0_FEATURES,
    IF_FEATURES,
    IS_FEATURES,
    SOH_EOL,
    build_dataset,
)
from .rl_teacher_cache import SOC_NORM, H2_NORM

POLICY_ID_STEM = "rl-tree-distill-pd-v11-p2"

_FEATURE_KEY = {"I0": "X_i0", "IS": "X_is", "IF": "X_if"}
_FEATURE_NAMES = {"I0": I0_FEATURES, "IS": IS_FEATURES, "IF": IF_FEATURES}


def fit_tree(dataset, information_set="I0", max_depth=4, min_samples_leaf=200,
             sample_weight=None, random_state=0):
    """Ajuste l'arbre sur le bloc d'apprentissage (split == 'train')."""
    X = dataset[_FEATURE_KEY[information_set]]
    y = dataset["y"]
    train = dataset["split"] == "train"

    weight = None
    if sample_weight == "inv_freq":
        weight = dataset["inv_freq_weight"][train]

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X[train], y[train], sample_weight=weight)
    tree._genial_meta = {
        "information_set": information_set,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "sample_weight": sample_weight,
        "n_leaves": int(tree.get_n_leaves()),
        "features": list(_FEATURE_NAMES[information_set]),
    }
    return tree


def fidelity_diagnostics(tree, dataset):
    """MAE/RMSE de la commande par partition, accord marche/arret FC/ELY."""
    info = tree._genial_meta["information_set"]
    X = dataset[_FEATURE_KEY[info]]
    y = dataset["y"]
    split = dataset["split"]
    out = {}
    for name in ("train", "val", "test"):
        mask = split == name
        pred = tree.predict(X[mask])
        err = pred - y[mask]
        fc_true, fc_pred = y[mask] > 1e-9, pred > 1e-9
        ely_true, ely_pred = y[mask] < -1e-9, pred < -1e-9
        out[name] = {
            "n": int(mask.sum()),
            "mae_w": float(np.mean(np.abs(err))),
            "rmse_w": float(np.sqrt(np.mean(err ** 2))),
            "fc_onoff_agree": float(np.mean(fc_true == fc_pred)),
            "ely_onoff_agree": float(np.mean(ely_true == ely_pred)),
        }
    return out


def _feature_vector(P_net, SoC_t, E_h2_t, E_h2_init, information_set,
                    wear=(0.0, 0.0, 0.0)):
    soc_n = min(max((SoC_t - SOC_NORM[0]) / (SOC_NORM[1] - SOC_NORM[0]), 0.0), 1.0)
    h2_n = min(max((E_h2_t - H2_NORM[0]) / (E_h2_init - H2_NORM[0]), 0.0), 1.0)
    if information_set == "I0":
        return np.array([[float(P_net), soc_n, h2_n]])
    return np.array([[float(P_net), soc_n, h2_n, *wear]])


def make_tree_policy_v11(tree, deadband_w=0.0, candidate_id=None):
    """Construit la policy de boucle a partir d'un arbre ajuste."""
    info = tree._genial_meta["information_set"]
    meta = dict(tree._genial_meta)
    spec = {
        "policy_id_stem": POLICY_ID_STEM,
        "tree_params": {k: meta[k] for k in
                        ("information_set", "max_depth", "min_samples_leaf",
                         "sample_weight", "n_leaves")},
        "deadband_w": deadband_w,
        "rules_text_sha256": hashlib.sha256(
            export_text(tree, feature_names=meta["features"]).encode()
        ).hexdigest(),
    }
    spec_sha = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    policy_id = f"{POLICY_ID_STEM}-{info.lower()}-{spec_sha[:12]}"

    def reset():
        return None  # policy sans memoire

    def rule(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
             SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
             RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t

        wear = (0.0, 0.0, 0.0)
        if info == "IS":
            # Usure relative >= 0, sans borne haute (identique a _wear du
            # dataset : np.clip((1-SoH)/(1-SoH_EoL), 0.0, None)).
            wear = (
                max((1.0 - SoH_bat_t) / (1.0 - SOH_EOL["bat"]), 0.0),
                max((1.0 - SoH_fc_t) / (1.0 - SOH_EOL["fc"]), 0.0),
                max((1.0 - SoH_ely_t) / (1.0 - SOH_EOL["ely"]), 0.0),
            )
        p_net = float(P_tot_ref_t)
        x = _feature_vector(p_net, SoC_t, E_h2_t, E_h2_init, info, wear)
        u_h2 = float(tree.predict(x)[0])

        # Zone morte : pas de micro-commande H2.
        if abs(u_h2) < deadband_w:
            u_h2 = 0.0
        # Gardes de defaillance identiques a la FLC experte.
        if "FC" in defaillances:
            u_h2 = min(u_h2, 0.0)
        if "ELY" in defaillances:
            u_h2 = max(u_h2, 0.0)

        p_fc = max(u_h2, 0.0)
        p_ely = min(u_h2, 0.0)
        p_bat = p_net - p_fc - p_ely
        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), p_net, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    rule.reset = reset
    rule.policy_id = policy_id
    rule.information_set = info
    rule.tree = tree
    rule.deadband_w = deadband_w
    rule.candidate_id = candidate_id
    rule.rl_metadata = {
        "policy_id": policy_id,
        "spec_sha256": spec_sha,
        "information_set": info,
        "features": meta["features"],
        "n_leaves": meta["n_leaves"],
        "deadband_w": deadband_w,
        "target": "u_h2 = P_dc_fc + P_dc_ely (signed)",
    }
    return rule


def null_test_unused_permutation(tree, dataset, rng=None):
    """Permuter une feature d'importance nulle ne doit rien changer.

    Retourne (feature_permutee, max_delta) ; delta ~ 0 attendu. Si toutes les
    features sont utilisees, retourne (None, 0.0).
    """
    rng = rng or np.random.default_rng(0)
    info = tree._genial_meta["information_set"]
    X = dataset[_FEATURE_KEY[info]].copy()
    unused = np.where(tree.feature_importances_ <= 0.0)[0]
    if len(unused) == 0:
        return None, 0.0
    j = int(unused[0])
    base = tree.predict(X)
    X[:, j] = rng.permutation(X[:, j])
    delta = float(np.max(np.abs(tree.predict(X) - base)))
    return tree._genial_meta["features"][j], delta


if __name__ == "__main__":
    # Auto-diagnostic rapide (sans boucle fermee) : balayage de profondeur,
    # fidelite par partition et test nul.
    ds = build_dataset()
    print(f"{'depth':>5} {'leaves':>6} {'MAE_val':>9} {'RMSE_val':>9} "
          f"{'FC_agr_val':>10} {'ELY_agr_val':>11}")
    for depth in (2, 3, 4, 5, 6):
        tree = fit_tree(ds, "I0", max_depth=depth)
        diag = fidelity_diagnostics(tree, ds)
        v = diag["val"]
        print(f"{depth:5d} {tree.get_n_leaves():6d} {v['mae_w']:9.1f} "
              f"{v['rmse_w']:9.1f} {v['fc_onoff_agree']:10.4f} "
              f"{v['ely_onoff_agree']:11.4f}")
    tree = fit_tree(ds, "I0", max_depth=4)
    feat, delta = null_test_unused_permutation(tree, ds)
    print(f"\nnull test (permute unused): feature={feat} max_delta={delta:.3e}")
    pol = make_tree_policy_v11(tree, deadband_w=0.0)
    print("policy_id:", pol.policy_id, "| n_leaves:", pol.rl_metadata["n_leaves"])
