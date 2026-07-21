"""Couche floue SoH attribuable au-dessus du parent FLC I0 regle."""

from __future__ import annotations

import hashlib
import json
from itertools import product

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol

from .flc_policy_v11 import make_tuned_expert_flc_policy_v11
from .mamdani import (
    FuzzyRule,
    FuzzyVariable,
    MamdaniSystem,
    TrapezoidalMF,
    TriangularMF,
)


POLICY_ID = "flc-mamdani-expert-v11-p2-is-soh-v1-2026-07-21"
WEAR_TERMS = ("low", "medium", "high")
ADJUSTMENT_TERMS = ("decrease", "neutral", "increase")
WEAR_MF_SPEC = {
    "low": ("trapezoidal", 0.0, 0.0, 0.15, 0.40),
    "medium": ("triangular", 0.20, 0.50, 0.80),
    "high": ("trapezoidal", 0.60, 0.85, 1.0, 1.0),
}
ADJUSTMENT_MF_SPEC = {
    "decrease": ("trapezoidal", -1.0, -1.0, -0.65, -0.10),
    "neutral": ("triangular", -0.20, 0.0, 0.20),
    "increase": ("trapezoidal", 0.10, 0.65, 1.0, 1.0),
}

# Axes : usure normalisee du composant H2, puis usure normalisee batterie.
# Une ressource plus usee est soulagee au profit de la ressource moins usee.
HEALTH_RULE_TABLE = (
    ("neutral", "increase", "increase"),
    ("decrease", "neutral", "increase"),
    ("decrease", "decrease", "neutral"),
)


def normalized_wear(soh, soh_eol):
    """Fraction de vie SoH consommee, bornee a [0, 1]."""
    return float(np.clip(
        (1.0 - float(soh)) / (1.0 - float(soh_eol)), 0.0, 1.0
    ))


def _wear_variable(name):
    return FuzzyVariable(
        name=name,
        terms={
            "low": TrapezoidalMF(*WEAR_MF_SPEC["low"][1:]),
            "medium": TriangularMF(*WEAR_MF_SPEC["medium"][1:]),
            "high": TrapezoidalMF(*WEAR_MF_SPEC["high"][1:]),
        },
    )


def _adjustment_variable():
    return FuzzyVariable(
        name="command_adjustment",
        lower=-1.0,
        upper=1.0,
        terms={
            "decrease": TrapezoidalMF(*ADJUSTMENT_MF_SPEC["decrease"][1:]),
            "neutral": TriangularMF(*ADJUSTMENT_MF_SPEC["neutral"][1:]),
            "increase": TrapezoidalMF(*ADJUSTMENT_MF_SPEC["increase"][1:]),
        },
    )


def _health_controller(output_points):
    rules = []
    for component_i, battery_i in product(range(3), repeat=2):
        antecedents = (WEAR_TERMS[component_i], WEAR_TERMS[battery_i])
        consequent = HEALTH_RULE_TABLE[component_i][battery_i]
        rules.append(FuzzyRule(
            antecedents=antecedents,
            consequent=consequent,
            label=(
                f"component_wear={antecedents[0]},battery_wear={antecedents[1]}"
                f"->{consequent}"
            ),
        ))
    return MamdaniSystem(
        inputs=(
            _wear_variable("component_wear"),
            _wear_variable("battery_wear"),
        ),
        output=_adjustment_variable(),
        rules=tuple(rules),
        output_points=int(output_points),
        default_output=0.0,
    )


def _unit_interval(value, lower, upper):
    return min(max(
        (float(value) - float(lower)) / (float(upper) - float(lower)), 0.0
    ), 1.0)


def make_soh_augmented_flc_policy_v11(
    soh_strength_fc=0.0,
    soh_strength_ely=0.0,
    output_points=401,
):
    """Construit FLC-IS par modulation floue relative des usures.

    Les deux intensites nulles appellent directement le parent I0. Le test nul
    est donc une identite fonctionnelle, et pas seulement une egalite a une
    tolerance numerique.
    """
    soh_strength_fc = float(soh_strength_fc)
    soh_strength_ely = float(soh_strength_ely)
    if not 0.0 <= soh_strength_fc <= 1.0:
        raise ValueError("soh_strength_fc doit appartenir a [0, 1]")
    if not 0.0 <= soh_strength_ely <= 1.0:
        raise ValueError("soh_strength_ely doit appartenir a [0, 1]")

    parent = make_tuned_expert_flc_policy_v11()
    health = _health_controller(output_points)
    strengths = {"deficit": soh_strength_fc, "surplus": soh_strength_ely}
    component_eol = {
        "deficit": float(I.FC["SoH_EoL"]),
        "surplus": float(I.ELY["SoH_EoL"]),
    }
    params = {
        "soh_strength_fc": soh_strength_fc,
        "soh_strength_ely": soh_strength_ely,
        "output_points": int(output_points),
    }
    specification = {
        "policy_id": POLICY_ID,
        "parent_candidate_id": parent.candidate_id,
        "parent_spec_sha256": parent.flc_metadata["spec_sha256"],
        "wear_membership_functions": WEAR_MF_SPEC,
        "adjustment_membership_functions": ADJUSTMENT_MF_SPEC,
        "health_rule_table": HEALTH_RULE_TABLE,
        "parameters": params,
    }
    spec_sha256 = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def command_multiplier(
        branch, soh_battery, soh_component, return_trace=False
    ):
        if branch not in strengths:
            raise ValueError("branch doit valoir 'deficit' ou 'surplus'")
        strength = strengths[branch]
        if strength == 0.0:
            if return_trace:
                return 1.0, {
                    "strength": 0.0,
                    "raw_adjustment": 0.0,
                    "component_wear": None,
                    "battery_wear": None,
                    "null_branch": True,
                }
            return 1.0
        battery_wear = normalized_wear(soh_battery, I.BAT["SoH_EoL"])
        component_wear = normalized_wear(
            soh_component, component_eol[branch]
        )
        # Au BoL, la couche doit rester exactement neutre, meme si la
        # discretisation du centroide renvoie un residu de l'ordre de l'ulp.
        if battery_wear == 0.0 and component_wear == 0.0:
            if return_trace:
                return 1.0, {
                    "strength": strength,
                    "raw_adjustment": 0.0,
                    "component_wear": 0.0,
                    "battery_wear": 0.0,
                    "bol_neutral": True,
                }
            return 1.0
        inferred = health.infer(
            {
                "component_wear": component_wear,
                "battery_wear": battery_wear,
            },
            return_trace=return_trace,
        )
        adjustment = inferred[0] if return_trace else inferred
        multiplier = float(np.clip(1.0 + strength * adjustment, 0.0, 2.0))
        if not return_trace:
            return multiplier
        trace = dict(inferred[1])
        trace.update({
            "strength": strength,
            "raw_adjustment": float(adjustment),
            "component_wear": component_wear,
            "battery_wear": battery_wear,
            "multiplier": multiplier,
        })
        return multiplier, trace

    def reset():
        return parent.reset()

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        policy_args = (
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
            RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )
        if soh_strength_fc == 0.0 and soh_strength_ely == 0.0:
            return parent(*policy_args)

        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t
        soc = _unit_interval(SoC_t, 0.20, 0.995)
        h2 = _unit_interval(E_h2_t, 0.0, E_h2_init)
        p_net = float(P_tot_ref_t)
        nominal_fc_dc = float(I.FC["P_fc_max"] * I.CONV["eta"])
        nominal_ely_dc = float(I.ELY["P_ely_max"] / I.CONV["eta"])
        parent_params = parent.flc_parameters

        p_fc = 0.0
        p_ely = 0.0
        if p_net > 0.0 and "FC" not in defaillances:
            severity = min(p_net / parent_params["deficit_scale_w"], 1.0)
            fraction = parent.command_fraction("deficit", severity, soc, h2)
            fraction = float(np.clip(
                fraction * command_multiplier(
                    "deficit", SoH_bat_t, SoH_fc_t
                ), 0.0, 1.0
            ))
            p_fc = (
                fraction * parent_params["fc_ceiling_fraction"] * nominal_fc_dc
            )
        elif p_net < 0.0 and "ELY" not in defaillances:
            severity = min(-p_net / parent_params["surplus_scale_w"], 1.0)
            fraction = parent.command_fraction("surplus", severity, soc, h2)
            fraction = float(np.clip(
                fraction * command_multiplier(
                    "surplus", SoH_bat_t, SoH_ely_t
                ), 0.0, 1.0
            ))
            p_ely = -(
                fraction * parent_params["ely_ceiling_fraction"] * nominal_ely_dc
            )

        p_bat = p_net - p_fc - p_ely
        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), p_net, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    rule.reset = reset
    rule.command_multiplier = command_multiplier
    rule.health_controller = health
    rule.parent_policy = parent
    rule.policy_id = POLICY_ID
    rule.information_set = "IS"
    rule.soh_parameters = params
    rule.flc_metadata = {
        "policy_id": POLICY_ID,
        "spec_sha256": spec_sha256,
        "information_set": "IS",
        "parent_candidate_id": parent.candidate_id,
        "parent_spec_sha256": parent.flc_metadata["spec_sha256"],
        "inputs_added": ("SoH_bat", "SoH_fc", "SoH_ely"),
        "forecast_excluded": True,
        "health_rule_count": len(health.rules),
        "wear_membership_functions": WEAR_MF_SPEC,
        "adjustment_membership_functions": ADJUSTMENT_MF_SPEC,
        "health_rule_table": HEALTH_RULE_TABLE,
        "parameters": params,
    }
    return rule
