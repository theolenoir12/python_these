"""Policy 'regles apprises' IF : arbre distille + feature de prevision H18.

Ensemble d'information IF = I0 + energie nette prevue sur H18. C'est l'ajout
mesure au diagnostic du jour (l'arbre I0 sous-performe par manque de foresight,
EENS). La feature est le meme signal ``E_net_H18`` que la FLC-IF promue, calcule
par la meme logique (oracle / gaussian_iid / gaussian_ar1 / persistance) pour
une comparaison strictement appariee entre familles.

A l'entrainement, la feature est l'energie nette **vraie** du maitre (oracle).
A l'inference, elle est calculee depuis ``P_tot_ref_future`` fourni par la
boucle, eventuellement bruite selon le scenario.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from sklearn.tree import export_text

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol

from .flc_forecast_policy_v11 import (
    DEFAULT_BIAS_KWH,
    DEFAULT_SIGMA_KWH,
    FORECAST_SCENARIOS,
)
from .rl_dataset_v11 import HORIZON_STEPS_IF, IF_FEATURES
from .rl_teacher_cache import SOC_NORM, H2_NORM

POLICY_ID_STEM = "rl-tree-distill-pd-v11-p2-if"
_DT_H = float(I.LOAD["Ts"]) / 3600.0


def make_forecast_tree_policy_v11(
    tree, deadband_w=0.0, forecast_scenario="oracle",
    horizon_steps=HORIZON_STEPS_IF, bias_kwh=DEFAULT_BIAS_KWH,
    sigma_inject_kwh=DEFAULT_SIGMA_KWH, noise_rho=0.0, noise_seed=0,
    candidate_id=None,
):
    """Policy de boucle IF a partir d'un arbre ajuste sur X_if.

    L'arbre doit avoir ete entraine avec ``information_set='IF'`` : sa derniere
    feature est ``E_net_h18_kwh``.
    """
    if tree._genial_meta["information_set"] != "IF":
        raise ValueError("l'arbre doit etre entraine en IF (X_if)")
    if forecast_scenario not in FORECAST_SCENARIOS:
        raise ValueError(f"scenario inconnu: {forecast_scenario}")
    if not 0.0 <= noise_rho < 1.0:
        raise ValueError("noise_rho doit appartenir a [0, 1[")
    meta = dict(tree._genial_meta)

    spec = {
        "policy_id_stem": POLICY_ID_STEM,
        "tree_params": {k: meta[k] for k in
                        ("information_set", "max_depth", "min_samples_leaf",
                         "sample_weight", "n_leaves")},
        "deadband_w": deadband_w,
        "forecast": {
            "scenario": forecast_scenario, "horizon_steps": horizon_steps,
            "bias_kwh": bias_kwh, "sigma_inject_kwh": sigma_inject_kwh,
            "noise_rho": noise_rho, "noise_seed": noise_seed,
        },
        "rules_text_sha256": hashlib.sha256(
            export_text(tree, feature_names=meta["features"]).encode()
        ).hexdigest(),
    }
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    policy_id = f"{POLICY_ID_STEM}-{spec_sha[:12]}"

    state = {"rng": np.random.default_rng(noise_seed), "eps": 0.0,
             "forecast_calls": 0}

    def reset():
        state.update({"rng": np.random.default_rng(noise_seed), "eps": 0.0,
                      "forecast_calls": 0})

    def _forecast_energy_kwh(P_tot_ref_t, P_tot_ref_future):
        if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
            return 0.0
        if forecast_scenario == "persistence":
            energy = float(P_tot_ref_t) * horizon_steps * _DT_H / 1000.0
        else:
            future = np.asarray(P_tot_ref_future[:horizon_steps], dtype=float)
            energy = float(np.sum(future) * _DT_H / 1000.0)
        if forecast_scenario in ("gaussian_iid", "gaussian_ar1"):
            innovation = float(state["rng"].standard_normal())
            if forecast_scenario == "gaussian_ar1":
                state["eps"] = (noise_rho * state["eps"]
                                + math.sqrt(1.0 - noise_rho ** 2) * innovation)
            else:
                state["eps"] = innovation
            energy += bias_kwh + sigma_inject_kwh * state["eps"]
        state["forecast_calls"] += 1
        return energy

    def rule(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
             SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
             RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
             P_tot_ref_future=None):
        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t
        del SoH_fc_t, SoH_ely_t

        p_net = float(P_tot_ref_t)
        soc_n = min(max((SoC_t - SOC_NORM[0]) / (SOC_NORM[1] - SOC_NORM[0]),
                        0.0), 1.0)
        h2_n = min(max((E_h2_t - H2_NORM[0]) / (E_h2_init - H2_NORM[0]), 0.0),
                   1.0)
        e_net = _forecast_energy_kwh(p_net, P_tot_ref_future)
        x = np.array([[p_net, soc_n, h2_n, e_net]])
        u_h2 = float(tree.predict(x)[0])

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
    rule.forecast_horizon_steps = horizon_steps
    rule.policy_id = policy_id
    rule.information_set = "IF"
    rule.tree = tree
    rule.deadband_w = deadband_w
    rule.candidate_id = candidate_id
    rule.rl_metadata = {
        "policy_id": policy_id, "spec_sha256": spec_sha,
        "information_set": "IF", "features": meta["features"],
        "n_leaves": meta["n_leaves"], "deadband_w": deadband_w,
        "forecast": spec["forecast"],
    }
    return rule


if __name__ == "__main__":
    from .rl_dataset_v11 import build_dataset
    from .rl_tree_policy_v11 import fit_tree, fidelity_diagnostics

    ds = build_dataset()
    print(f"IF features: {list(IF_FEATURES)}")
    print(f"{'depth':>5} {'leaves':>6} {'MAE_val':>9} {'FC_agr':>8} "
          f"{'ELY_agr':>8} {'feat_import(P_net,SoC,H2,Enet)'}")
    for depth in (3, 4, 5, 6):
        tree = fit_tree(ds, "IF", max_depth=depth)
        v = fidelity_diagnostics(tree, ds)["val"]
        imp = ", ".join(f"{x:.2f}" for x in tree.feature_importances_)
        print(f"{depth:5d} {tree.get_n_leaves():6d} {v['mae_w']:9.1f} "
              f"{v['fc_onoff_agree']:8.4f} {v['ely_onoff_agree']:8.4f}  [{imp}]")
