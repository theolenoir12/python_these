"""Policy ANFIS Takagi-Sugeno ordre 1, deux branches, V11-p=2.

Conformément au plan (§2.3), ANFIS est séparé en une branche DÉFICIT (P_net > 0,
consigne PEMFC) et une branche SURPLUS (P_net < 0, consigne PEMWE), chacune un
système Sugeno d'ordre 1 (``anfis_ts.AnfisTS1``). La cible reste la commande
signée ``u_h2`` du disciple ; la batterie ferme le bilan. La policy expose la
MÊME interface de boucle que l'arbre et la FLC (zone morte, gardes de défaillance,
délégation de faisabilité à ``get_lol``) — seul le régresseur change, ce qui rend
la comparaison de FAMILLE propre.

Ablations d'information : I0 = [P_net, SoC, H2] ; IS = I0 + usure bat/fc/ely
(colonnes nulles au neuf), comme pour l'arbre et la FLC.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from Common.get_lol import get_lol

from .anfis_ts import AnfisTS1
from .rl_dataset_v11 import I0_FEATURES, IS_FEATURES, SOH_EOL
from .rl_tree_policy_v11 import _feature_vector

POLICY_ID_STEM = "anfis-ts1-distill-pd-v11-p2"
_FEATURE_NAMES = {"I0": I0_FEATURES, "IS": IS_FEATURES}


class AnfisTwoBranch:
    """Deux ANFIS TS1 routés par le signe de P_net (colonne 0 des features)."""

    def __init__(self, deficit, surplus, information_set="I0"):
        self.deficit = deficit
        self.surplus = surplus
        self.information_set = information_set

    @classmethod
    def fit(cls, X, y, information_set="I0", n_mf=3, ridge=1e-6,
            sigma_scale=1.0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        pnet = X[:, 0]
        deficit_mask = pnet > 0.0
        surplus_mask = pnet < 0.0
        if deficit_mask.sum() < n_mf ** X.shape[1] or \
                surplus_mask.sum() < n_mf ** X.shape[1]:
            raise ValueError("pas assez d'échantillons par branche pour la grille")
        deficit = AnfisTS1.init_uniform(X[deficit_mask], n_mf, sigma_scale)
        deficit.fit_consequents(X[deficit_mask], y[deficit_mask], ridge)
        surplus = AnfisTS1.init_uniform(X[surplus_mask], n_mf, sigma_scale)
        surplus.fit_consequents(X[surplus_mask], y[surplus_mask], ridge)
        return cls(deficit, surplus, information_set)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        pnet = X[:, 0]
        out = np.zeros(X.shape[0])
        dm, sm = pnet > 0.0, pnet < 0.0
        if dm.any():
            out[dm] = self.deficit.predict(X[dm])
        if sm.any():
            out[sm] = self.surplus.predict(X[sm])
        return out

    def spec(self):
        return {
            "kind": "anfis-ts1-two-branch",
            "information_set": self.information_set,
            "deficit": self.deficit.spec(),
            "surplus": self.surplus.spec(),
            "rule_count": {"deficit": self.deficit.n_rules,
                           "surplus": self.surplus.n_rules},
        }


def make_anfis_policy_v11(model, deadband_w=0.0, candidate_id=None):
    """Construit la policy de boucle à partir d'un ``AnfisTwoBranch`` ajusté."""
    info = model.information_set
    spec = model.spec()
    policy_spec = {
        "policy_id_stem": POLICY_ID_STEM,
        "information_set": info,
        "deadband_w": deadband_w,
        "model_spec": spec,
    }
    spec_sha = hashlib.sha256(
        json.dumps(policy_spec, sort_keys=True).encode()).hexdigest()
    policy_id = f"{POLICY_ID_STEM}-{info.lower()}-{spec_sha[:12]}"

    def reset():
        return None  # policy sans mémoire

    def rule(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
             SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
             RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t

        wear = (0.0, 0.0, 0.0)
        if info == "IS":
            wear = (
                max((1.0 - SoH_bat_t) / (1.0 - SOH_EOL["bat"]), 0.0),
                max((1.0 - SoH_fc_t) / (1.0 - SOH_EOL["fc"]), 0.0),
                max((1.0 - SoH_ely_t) / (1.0 - SOH_EOL["ely"]), 0.0),
            )
        p_net = float(P_tot_ref_t)
        x = _feature_vector(p_net, SoC_t, E_h2_t, E_h2_init, info, wear)
        u_h2 = float(model.predict(x)[0])

        if abs(u_h2) < deadband_w:
            u_h2 = 0.0
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
    rule.model = model
    rule.deadband_w = deadband_w
    rule.candidate_id = candidate_id
    rule.anfis_metadata = {
        "policy_id": policy_id,
        "spec_sha256": spec_sha,
        "information_set": info,
        "features": list(_FEATURE_NAMES[info]),
        "rule_count": spec["rule_count"],
        "deadband_w": deadband_w,
        "target": "u_h2 = P_dc_fc + P_dc_ely (signed)",
    }
    return rule


if __name__ == "__main__":
    from .rl_dataset_v11 import build_dataset
    ds = build_dataset()
    train = ds["split"] == "train"
    model = AnfisTwoBranch.fit(ds["X_i0"][train], ds["y"][train], "I0", n_mf=3)
    pol = make_anfis_policy_v11(model)
    print("spec:", json.dumps(model.spec(), indent=2))
    print("policy_id:", pol.policy_id)
