"""Couche de precharge IF/ISF au-dessus de la FLC experte V11-p=2."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol

from .flc_policy_v11 import make_tuned_expert_flc_policy_v11
from .flc_soh_policy_v11 import make_soh_augmented_flc_policy_v11


POLICY_ID = "flc-expert-v11-p2-if-isf-h18-v1-2026-07-21"
FORECAST_SCENARIOS = (
    "oracle", "gaussian_iid", "gaussian_ar1", "persistence",
)
DEFAULT_HORIZON_STEPS = 18
DEFAULT_BIAS_KWH = -2.3172222838936554
DEFAULT_SIGMA_KWH = 39.376786399148806
SELECTED_IF_FORECAST_STRENGTH = 1.0


def _validate_probability_parameters(noise_rho):
    noise_rho = float(noise_rho)
    if not 0.0 <= noise_rho < 1.0:
        raise ValueError("noise_rho doit appartenir a [0, 1[")
    return noise_rho


def make_forecast_augmented_flc_policy_v11(
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
    soh_strength_fc=0.0,
    soh_strength_ely=0.0,
):
    """Construit IF ou ISF par precharge anticipative de la batterie.

    La couche reduit uniquement la commande ELY deja demandee par son parent.
    Une force nulle appelle directement ce parent : l'ablation est donc une
    identite fonctionnelle et ne consomme aucun tirage aleatoire.
    """
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
    noise_rho = _validate_probability_parameters(noise_rho)
    noise_seed = int(noise_seed)
    if threshold_sigma_multiplier is None:
        threshold_sigma_multiplier = (
            0.0 if forecast_scenario == "oracle" else 1.0
        )
    threshold_sigma_multiplier = float(threshold_sigma_multiplier)
    if threshold_sigma_multiplier < 0.0:
        raise ValueError("threshold_sigma_multiplier doit etre positif ou nul")

    soh_strength_fc = float(soh_strength_fc)
    soh_strength_ely = float(soh_strength_ely)
    if soh_strength_fc == 0.0 and soh_strength_ely == 0.0:
        parent = make_tuned_expert_flc_policy_v11()
        information_set = "IF" if forecast_strength > 0.0 else "I0"
    else:
        parent = make_soh_augmented_flc_policy_v11(
            soh_strength_fc=soh_strength_fc,
            soh_strength_ely=soh_strength_ely,
        )
        information_set = "ISF" if forecast_strength > 0.0 else "IS"

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
        "soh_strength_fc": soh_strength_fc,
        "soh_strength_ely": soh_strength_ely,
    }
    specification = {
        "policy_id": POLICY_ID,
        "parent_spec_sha256": parent.flc_metadata["spec_sha256"],
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
        return {
            key: value for key, value in state.items()
            if key != "rng"
        }

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        P_tot_ref_future=None,
    ):
        policy_args = (
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
            RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )
        if forecast_strength == 0.0:
            return parent(*policy_args)

        base_action, base_lol = parent(*policy_args)
        active, _ = precharge_signal(
            SoC_t, P_tot_ref_t, P_tot_ref_future
        )
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
    rule.forecast_horizon_steps = horizon_steps
    rule.policy_id = POLICY_ID
    rule.information_set = information_set
    rule.forecast_parameters = params
    rule.flc_metadata = {
        "policy_id": POLICY_ID,
        "spec_sha256": spec_sha256,
        "information_set": information_set,
        "parent_spec_sha256": parent.flc_metadata["spec_sha256"],
        "parent_information_set": parent.information_set,
        "inputs_added": ("E_net_forecast_H18",),
        "forecast_rule_count": 3,
        "forecast_rule": specification["forecast_rule"],
        "parameters": params,
    }
    return rule


def make_selected_if_policy_v11(forecast_scenario="oracle", **simulation_kwargs):
    """Construit la variante IF retenue à l'issue du protocole 25 ans.

    En usage avec une prévision externe déjà calculée, le scénario ``oracle``
    signifie simplement qu'aucune erreur synthétique n'est ajoutée au tableau
    ``P_tot_ref_future`` fourni au contrôleur. Les autres scénarios servent aux
    expériences de robustesse reproductibles.
    """
    forbidden = {
        "forecast_strength", "soh_strength_fc", "soh_strength_ely",
    } & set(simulation_kwargs)
    if forbidden:
        raise ValueError(
            "la variante IF sélectionnée fige les paramètres: "
            + ", ".join(sorted(forbidden))
        )
    return make_forecast_augmented_flc_policy_v11(
        forecast_strength=SELECTED_IF_FORECAST_STRENGTH,
        forecast_scenario=forecast_scenario,
        soh_strength_fc=0.0,
        soh_strength_ely=0.0,
        **simulation_kwargs,
    )
