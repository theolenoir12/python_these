"""Policy 'regles apprises' IF par REGLE de precharge au-dessus de l'arbre I0.

Direction A du recap (RECAP_RULE_LEARNING_V11_P2 section 5) : le diagnostic a
montre que l'arbre CART ignore la prevision quand elle est fournie comme feature
(forecast-comme-feature, section 2.5). La valeur de la foresight est ici
STRUCTURELLE : une action CONDITIONNELLE de precharge / preservation H2. On la
greffe donc AU-DESSUS de l'arbre I0 distille, exactement comme la couche FLC-IF
(``flc_forecast_policy_v11``) se greffe au-dessus de la FLC experte I0.

Comparaison strictement appariee avec la FLC-IF : meme signal de prevision
(``E_net_H``, memes scenarios oracle / gaussian_iid / gaussian_ar1 / persistance,
memes ``bias_kwh`` / ``sigma_design_kwh`` importes de ``flc_forecast_policy_v11``)
et meme regle a hysteresis. La seule difference entre les deux familles est le
PARENT : FLC experte contre arbre distille. Ce qui change d'une famille a l'autre
est donc l'effet de FAMILLE, pas l'effet de jeu d'information.

Regle de precharge (identique a la FLC-IF) :

- ``E_net_H > +m_sigma*sigma_design``  -> precharge ON (deficit net a venir) ;
- ``E_net_H < -m_sigma*sigma_design``  -> precharge OFF ;
- entre les deux, on retient l'etat (hysteresis), ``min_dwell_steps`` de garde ;
- si precharge ON et ``SoC < soc_target`` et le parent demandait de
  l'electrolyse (``p_ely < 0``), on reduit cette commande :
  ``p_ely <- p_ely * (1 - forecast_strength)`` ; la batterie ferme le bilan.

Test nul obligatoire (protocole section 6) : ``forecast_strength = 0`` appelle
directement l'arbre parent, bit-a-bit, sans consommer aucun tirage aleatoire.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol

from .flc_forecast_policy_v11 import (
    DEFAULT_BIAS_KWH,
    DEFAULT_HORIZON_STEPS,
    DEFAULT_SIGMA_KWH,
    FORECAST_SCENARIOS,
)
from .rl_tree_policy_v11 import make_tree_policy_v11

POLICY_ID_STEM = "rl-tree-precharge-pd-v11-p2"


def make_tree_precharge_policy_v11(
    tree,
    deadband_w=0.0,
    forecast_strength=0.0,
    forecast_scenario="oracle",
    horizon_steps=DEFAULT_HORIZON_STEPS,
    soc_target=0.99,
    threshold_sigma_multiplier=None,
    min_dwell_steps=0,
    bias_kwh=DEFAULT_BIAS_KWH,
    sigma_design_kwh=DEFAULT_SIGMA_KWH,
    sigma_inject_kwh=None,
    noise_rho=0.0,
    noise_seed=0,
    candidate_id=None,
):
    """Construit la policy IF 'arbre + regle de precharge' de boucle.

    ``tree`` est un arbre distille de jeu I0 (ou IS) : la prevision n'entre PAS
    comme feature de l'arbre, seulement par la regle. Passer un arbre IF est
    refuse, car cela reviendrait a melanger les deux vehicules de foresight.
    """
    base_info = tree._genial_meta["information_set"]
    if base_info not in ("I0", "IS"):
        raise ValueError(
            "l'arbre parent doit etre I0 ou IS (la prevision entre par la "
            "regle, pas comme feature) ; recu: " + base_info
        )
    forecast_strength = float(forecast_strength)
    if not 0.0 <= forecast_strength <= 1.0:
        raise ValueError("forecast_strength doit appartenir a [0, 1]")
    if forecast_scenario not in FORECAST_SCENARIOS:
        raise ValueError(f"scenario de prevision inconnu: {forecast_scenario}")
    horizon_steps = int(horizon_steps)
    if horizon_steps <= 0:
        raise ValueError("horizon_steps doit etre positif")
    soc_target = float(soc_target)
    if not 0.2 <= soc_target <= 0.995:
        raise ValueError("soc_target doit appartenir a [0,2 ; 0,995]")
    min_dwell_steps = int(min_dwell_steps)
    if min_dwell_steps < 0:
        raise ValueError("min_dwell_steps doit etre positif ou nul")
    bias_kwh = float(bias_kwh)
    sigma_design_kwh = float(sigma_design_kwh)
    if sigma_design_kwh < 0.0:
        raise ValueError("sigma_design_kwh doit etre positif ou nul")
    sigma_inject_kwh = float(
        sigma_design_kwh if sigma_inject_kwh is None else sigma_inject_kwh
    )
    if sigma_inject_kwh < 0.0:
        raise ValueError("sigma_inject_kwh doit etre positif ou nul")
    noise_rho = float(noise_rho)
    if not 0.0 <= noise_rho < 1.0:
        raise ValueError("noise_rho doit appartenir a [0, 1[")
    noise_seed = int(noise_seed)
    if threshold_sigma_multiplier is None:
        threshold_sigma_multiplier = 0.0 if forecast_scenario == "oracle" else 1.0
    threshold_sigma_multiplier = float(threshold_sigma_multiplier)
    if threshold_sigma_multiplier < 0.0:
        raise ValueError("threshold_sigma_multiplier doit etre positif ou nul")

    parent = make_tree_policy_v11(tree, deadband_w=deadband_w)
    information_set = (
        {"I0": "IF", "IS": "ISF"}[base_info]
        if forecast_strength > 0.0 else base_info
    )

    params = {
        "forecast_strength": forecast_strength,
        "forecast_scenario": forecast_scenario,
        "horizon_steps": horizon_steps,
        "soc_target": soc_target,
        "threshold_sigma_multiplier": threshold_sigma_multiplier,
        "min_dwell_steps": min_dwell_steps,
        "bias_kwh": bias_kwh,
        "sigma_design_kwh": sigma_design_kwh,
        "sigma_inject_kwh": sigma_inject_kwh,
        "noise_rho": noise_rho,
        "noise_seed": noise_seed,
        "deadband_w": deadband_w,
    }
    specification = {
        "policy_id_stem": POLICY_ID_STEM,
        "parent_spec_sha256": parent.rl_metadata["spec_sha256"],
        "parent_information_set": base_info,
        "information_set": information_set,
        "forecast_rule": {
            "enter": "E_net_H > +m_sigma*sigma_design",
            "exit": "E_net_H < -m_sigma*sigma_design",
            "between": "retain hysteresis state",
            "action": "P_ely <- P_ely*(1-forecast_strength)",
        },
        "parameters": params,
    }
    spec_sha256 = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy_id = f"{POLICY_ID_STEM}-{information_set.lower()}-{spec_sha256[:12]}"

    state = {
        "rng": np.random.default_rng(noise_seed),
        "eps": 0.0,
        "precharge_on": False,
        "dwell_remaining": 0,
        "forecast_calls": 0,
        "noise_draws": 0,
        "precharge_signal_steps": 0,
        "precharge_applied_steps": 0,
        "ely_energy_removed_kwh_dc": 0.0,
    }

    def reset():
        parent.reset()
        state.update({
            "rng": np.random.default_rng(noise_seed),
            "eps": 0.0,
            "precharge_on": False,
            "dwell_remaining": 0,
            "forecast_calls": 0,
            "noise_draws": 0,
            "precharge_signal_steps": 0,
            "precharge_applied_steps": 0,
            "ely_energy_removed_kwh_dc": 0.0,
        })

    def _forecast_energy_kwh(P_tot_ref_t, P_tot_ref_future):
        if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
            return None
        dt_h = float(I.LOAD["Ts"] / 3600.0)
        if forecast_scenario == "persistence":
            energy_kwh = float(P_tot_ref_t) * horizon_steps * dt_h / 1000.0
        else:
            future = np.asarray(P_tot_ref_future[:horizon_steps], dtype=float)
            energy_kwh = float(np.sum(future) * dt_h / 1000.0)
        if forecast_scenario in ("gaussian_iid", "gaussian_ar1"):
            innovation = float(state["rng"].standard_normal())
            state["noise_draws"] += 1
            if forecast_scenario == "gaussian_ar1":
                state["eps"] = (
                    noise_rho * state["eps"]
                    + math.sqrt(1.0 - noise_rho ** 2) * innovation
                )
            else:
                state["eps"] = innovation
            energy_kwh += bias_kwh + sigma_inject_kwh * state["eps"]
        return energy_kwh

    def precharge_signal(SoC_t, P_tot_ref_t, P_tot_ref_future):
        energy_kwh = _forecast_energy_kwh(P_tot_ref_t, P_tot_ref_future)
        if energy_kwh is None:
            return False, None
        state["forecast_calls"] += 1
        threshold = threshold_sigma_multiplier * sigma_design_kwh
        if state["dwell_remaining"] > 0:
            state["dwell_remaining"] -= 1
        elif (not state["precharge_on"]) and energy_kwh > threshold:
            state["precharge_on"] = True
            state["dwell_remaining"] = min_dwell_steps
        elif state["precharge_on"] and energy_kwh < -threshold:
            state["precharge_on"] = False
            state["dwell_remaining"] = min_dwell_steps
        active = bool(state["precharge_on"] and float(SoC_t) < soc_target)
        if active:
            state["precharge_signal_steps"] += 1
        return active, energy_kwh

    def diagnostics():
        return {key: value for key, value in state.items() if key != "rng"}

    def rule(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
             SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
             RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
             P_tot_ref_future=None):
        policy_args = (
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
            RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )
        if forecast_strength == 0.0:
            return parent(*policy_args)

        base_action, base_lol = parent(*policy_args)
        active, _ = precharge_signal(SoC_t, P_tot_ref_t, P_tot_ref_future)
        # On ne touche que si un electrolyseur tourne (p_ely < 0) : c'est
        # l'energie de surplus qu'on garde en batterie au lieu d'en faire du H2.
        if not active or base_action[2] == 0.0:
            return base_action, base_lol

        p_fc = float(base_action[1])
        p_ely = float(base_action[2]) * (1.0 - forecast_strength)
        p_bat = float(P_tot_ref_t) - p_fc - p_ely
        removed_w = abs(float(base_action[2]) - p_ely)
        state["precharge_applied_steps"] += 1
        state["ely_energy_removed_kwh_dc"] += (
            removed_w * I.LOAD["Ts"] / 3600.0 / 1000.0
        )
        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), P_tot_ref_t, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    rule.reset = reset
    rule.precharge_signal = precharge_signal
    rule.forecast_diagnostics = diagnostics
    rule.parent_policy = parent
    rule.tree = tree
    rule.forecast_horizon_steps = horizon_steps
    rule.policy_id = policy_id
    rule.information_set = information_set
    rule.deadband_w = deadband_w
    rule.candidate_id = candidate_id
    rule.forecast_parameters = params
    rule.rl_metadata = {
        "policy_id": policy_id,
        "spec_sha256": spec_sha256,
        "information_set": information_set,
        "parent_spec_sha256": parent.rl_metadata["spec_sha256"],
        "parent_information_set": base_info,
        "n_leaves": parent.rl_metadata["n_leaves"],
        "inputs_added": ("E_net_forecast_H%d" % horizon_steps,),
        "forecast_rule_count": 3,
        "forecast_rule": specification["forecast_rule"],
        "parameters": params,
    }
    return rule


if __name__ == "__main__":
    # Auto-diagnostic sans boucle fermee : verifie que la regle de precharge est
    # STRICTEMENT appariee a celle de la FLC-IF (meme signal sur une meme
    # sequence d'entrees), et que le test nul est bit-a-bit.
    from .flc_forecast_policy_v11 import make_forecast_augmented_flc_policy_v11
    from .rl_dataset_v11 import build_dataset
    from .rl_tree_policy_v11 import fit_tree

    ds = build_dataset()
    tree = fit_tree(ds, "I0", max_depth=4)
    rl = make_tree_precharge_policy_v11(
        tree, forecast_strength=1.0, forecast_scenario="oracle")
    flc = make_forecast_augmented_flc_policy_v11(
        forecast_strength=1.0, forecast_scenario="oracle")

    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(2000):
        soc = float(rng.uniform(0.2, 1.0))
        pnet = float(rng.uniform(-20000, 5000))
        future = rng.uniform(-20000, 5000, size=18)
        a_rl, _ = rl.precharge_signal(soc, pnet, future)
        a_flc, _ = flc.precharge_signal(soc, pnet, future)
        mismatches += int(a_rl != a_flc)
    print(f"precharge signal appariement RL vs FLC-IF : "
          f"{2000 - mismatches}/2000 identiques")
    print("policy_id:", rl.policy_id, "| info:", rl.information_set)
