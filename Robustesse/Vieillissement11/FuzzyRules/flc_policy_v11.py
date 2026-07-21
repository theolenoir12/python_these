"""FLC experte Mamdani I0 pour le micro-reseau GENIAL V11-p=2.

La politique est online et sans prevision. Elle ne lit pour sa decision que la
puissance nette courante, le SoC et le remplissage H2. Les SoH et puissances
maximales vieillies ne servent qu'aux contraintes physiques communes appliquees
par ``get_lol``.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol

from .mamdani import (
    FuzzyRule,
    FuzzyVariable,
    MamdaniSystem,
    TrapezoidalMF,
    TriangularMF,
)


POLICY_ID = "flc-mamdani-expert-v11-p2-i0-v1-2026-07-21"
TUNED_I0_CANDIDATE_ID = "flc_8126e6f729c6"
TUNED_I0_PARAMETERS = {
    "fc_ceiling_fraction": 0.78375339,
    "ely_ceiling_fraction": 0.52282887,
    "deficit_scale_multiplier": 1.28769941,
    "surplus_scale_multiplier": 0.62373324,
    "output_deadband": 0.26735578,
}
INPUT_TERMS = ("low", "medium", "high")
OUTPUT_TERMS = ("off", "low", "medium", "high", "full")
INPUT_MF_SPEC = {
    "low": ("trapezoidal", 0.0, 0.0, 0.25, 0.50),
    "medium": ("triangular", 0.25, 0.50, 0.75),
    "high": ("trapezoidal", 0.50, 0.75, 1.0, 1.0),
}
OUTPUT_MF_SPEC = {
    "off": ("trapezoidal", 0.0, 0.0, 0.0, 0.25),
    "low": ("triangular", 0.0, 0.25, 0.50),
    "medium": ("triangular", 0.25, 0.50, 0.75),
    "high": ("triangular", 0.50, 0.75, 1.0),
    "full": ("trapezoidal", 0.75, 1.0, 1.0, 1.0),
}


# Axes : H2 (low, medium, high), SoC (low, medium, high), severite Pnet
# (low, medium, high). Ces tables sont le savoir expert explicite de la v1.
DEFICIT_RULE_TABLE = (
    (
        ("medium", "high", "full"),
        ("off", "medium", "high"),
        ("off", "low", "medium"),
    ),
    (
        ("high", "full", "full"),
        ("low", "medium", "high"),
        ("off", "low", "high"),
    ),
    (
        ("high", "full", "full"),
        ("medium", "high", "full"),
        ("low", "medium", "high"),
    ),
)

SURPLUS_RULE_TABLE = (
    (
        ("low", "medium", "high"),
        ("medium", "high", "full"),
        ("high", "full", "full"),
    ),
    (
        ("off", "low", "medium"),
        ("low", "medium", "high"),
        ("medium", "high", "full"),
    ),
    (
        ("off", "off", "low"),
        ("off", "low", "medium"),
        ("low", "medium", "high"),
    ),
)


def _input_variable(name):
    return FuzzyVariable(
        name=name,
        terms={
            "low": TrapezoidalMF(*INPUT_MF_SPEC["low"][1:]),
            "medium": TriangularMF(*INPUT_MF_SPEC["medium"][1:]),
            "high": TrapezoidalMF(*INPUT_MF_SPEC["high"][1:]),
        },
    )


def _output_variable():
    return FuzzyVariable(
        name="command_fraction",
        terms={
            "off": TrapezoidalMF(*OUTPUT_MF_SPEC["off"][1:]),
            "low": TriangularMF(*OUTPUT_MF_SPEC["low"][1:]),
            "medium": TriangularMF(*OUTPUT_MF_SPEC["medium"][1:]),
            "high": TriangularMF(*OUTPUT_MF_SPEC["high"][1:]),
            "full": TrapezoidalMF(*OUTPUT_MF_SPEC["full"][1:]),
        },
    )


def _rules_from_table(branch, table):
    rules = []
    for h2_i, soc_i, severity_i in product(range(3), repeat=3):
        antecedents = (
            INPUT_TERMS[severity_i], INPUT_TERMS[soc_i], INPUT_TERMS[h2_i]
        )
        consequent = table[h2_i][soc_i][severity_i]
        rules.append(FuzzyRule(
            antecedents=antecedents,
            consequent=consequent,
            label=(
                f"{branch}:severity={antecedents[0]},soc={antecedents[1]},"
                f"h2={antecedents[2]}->{consequent}"
            ),
        ))
    return tuple(rules)


def _make_controller(branch, table, output_points):
    return MamdaniSystem(
        inputs=(
            _input_variable("severity"),
            _input_variable("soc"),
            _input_variable("h2"),
        ),
        output=_output_variable(),
        rules=_rules_from_table(branch, table),
        output_points=output_points,
    )


def _unit_interval(value, lower, upper):
    return min(max((float(value) - float(lower)) / (float(upper) - float(lower)), 0.0), 1.0)


def make_expert_flc_policy_v11(
    fc_ceiling_fraction=1.0,
    ely_ceiling_fraction=1.0,
    deficit_scale_w=None,
    surplus_scale_w=None,
    output_deadband=0.10,
    output_points=401,
):
    """Construit la baseline FLC experte I0 a deux branches.

    ``fc_ceiling_fraction`` et ``ely_ceiling_fraction`` bornent les sorties
    floues par rapport aux puissances stack nominales. Les valeurs instantanees
    vieillies restent uniquement des contraintes de securite dans ``get_lol``.
    """
    fc_ceiling_fraction = float(fc_ceiling_fraction)
    ely_ceiling_fraction = float(ely_ceiling_fraction)
    output_deadband = float(output_deadband)
    if not 0.0 < fc_ceiling_fraction <= 1.0:
        raise ValueError("fc_ceiling_fraction doit appartenir a ]0, 1]")
    if not 0.0 < ely_ceiling_fraction <= 1.0:
        raise ValueError("ely_ceiling_fraction doit appartenir a ]0, 1]")
    if not 0.0 <= output_deadband < 1.0:
        raise ValueError("output_deadband doit appartenir a [0, 1[")

    nominal_fc_dc = float(I.FC["P_fc_max"] * I.CONV["eta"])
    nominal_ely_dc = float(I.ELY["P_ely_max"] / I.CONV["eta"])
    deficit_scale_w = float(
        nominal_fc_dc if deficit_scale_w is None else deficit_scale_w
    )
    surplus_scale_w = float(
        nominal_ely_dc if surplus_scale_w is None else surplus_scale_w
    )
    if deficit_scale_w <= 0.0 or surplus_scale_w <= 0.0:
        raise ValueError("les echelles de puissance doivent etre positives")

    controllers = {
        "deficit": _make_controller(
            "deficit", DEFICIT_RULE_TABLE, int(output_points)
        ),
        "surplus": _make_controller(
            "surplus", SURPLUS_RULE_TABLE, int(output_points)
        ),
    }

    params = {
        "fc_ceiling_fraction": fc_ceiling_fraction,
        "ely_ceiling_fraction": ely_ceiling_fraction,
        "deficit_scale_w": deficit_scale_w,
        "surplus_scale_w": surplus_scale_w,
        "output_deadband": output_deadband,
        "output_points": int(output_points),
    }
    specification = {
        "policy_id": POLICY_ID,
        "input_membership_functions": INPUT_MF_SPEC,
        "output_membership_functions": OUTPUT_MF_SPEC,
        "deficit_rule_table": DEFICIT_RULE_TABLE,
        "surplus_rule_table": SURPLUS_RULE_TABLE,
        "parameters": params,
    }
    spec_sha256 = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def command_fraction(branch, severity, soc, h2, return_trace=False):
        if branch not in controllers:
            raise ValueError("branch doit valoir 'deficit' ou 'surplus'")
        result = controllers[branch].infer(
            {"severity": severity, "soc": soc, "h2": h2},
            return_trace=return_trace,
        )
        fraction = result[0] if return_trace else result
        fraction = 0.0 if fraction <= output_deadband else float(fraction)
        if not return_trace:
            return fraction
        trace = dict(result[1])
        trace["after_deadband"] = fraction
        return fraction, trace

    def reset():
        # Politique sans memoire ; methode fournie pour l'interface commune.
        return None

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t
        del SoH_fc_t, SoH_ely_t

        soc = _unit_interval(SoC_t, 0.20, 0.995)
        h2 = _unit_interval(E_h2_t, 0.0, E_h2_init)
        p_net = float(P_tot_ref_t)

        p_fc = 0.0
        p_ely = 0.0
        if p_net > 0.0 and "FC" not in defaillances:
            severity = min(p_net / deficit_scale_w, 1.0)
            fraction = command_fraction("deficit", severity, soc, h2)
            p_fc = fraction * fc_ceiling_fraction * nominal_fc_dc
        elif p_net < 0.0 and "ELY" not in defaillances:
            severity = min(-p_net / surplus_scale_w, 1.0)
            fraction = command_fraction("surplus", severity, soc, h2)
            p_ely = -fraction * ely_ceiling_fraction * nominal_ely_dc

        p_bat = p_net - p_fc - p_ely
        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), p_net, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    rule.reset = reset
    rule.command_fraction = command_fraction
    rule.controllers = controllers
    rule.policy_id = POLICY_ID
    rule.information_set = "I0"
    rule.flc_parameters = params
    rule.flc_metadata = {
        "policy_id": POLICY_ID,
        "spec_sha256": spec_sha256,
        "information_set": "I0",
        "inference": {
            "type": "Mamdani",
            "and": "min",
            "implication": "min",
            "aggregation": "max",
            "defuzzification": "discrete_centroid",
        },
        "inputs": ("P_net_current", "SoC", "E_h2_fill"),
        "hidden_inputs_excluded": ("calendar", "forecast", "SoH"),
        "input_membership_functions": INPUT_MF_SPEC,
        "output_membership_functions": OUTPUT_MF_SPEC,
        "rule_count": {
            "deficit": len(controllers["deficit"].rules),
            "surplus": len(controllers["surplus"].rules),
        },
        "deficit_rule_table": DEFICIT_RULE_TABLE,
        "surplus_rule_table": SURPLUS_RULE_TABLE,
        "parameters": params,
    }
    return rule


def make_tuned_expert_flc_policy_v11():
    """Construit le compromis I0 promu apres screening 1 an et audit 25 ans."""
    nominal_fc_dc = float(I.FC["P_fc_max"] * I.CONV["eta"])
    nominal_ely_dc = float(I.ELY["P_ely_max"] / I.CONV["eta"])
    policy = make_expert_flc_policy_v11(
        fc_ceiling_fraction=TUNED_I0_PARAMETERS["fc_ceiling_fraction"],
        ely_ceiling_fraction=TUNED_I0_PARAMETERS["ely_ceiling_fraction"],
        deficit_scale_w=(
            TUNED_I0_PARAMETERS["deficit_scale_multiplier"] * nominal_fc_dc
        ),
        surplus_scale_w=(
            TUNED_I0_PARAMETERS["surplus_scale_multiplier"] * nominal_ely_dc
        ),
        output_deadband=TUNED_I0_PARAMETERS["output_deadband"],
    )
    policy.candidate_id = TUNED_I0_CANDIDATE_ID
    policy.tuning_parameters = dict(TUNED_I0_PARAMETERS)
    policy.tuning_run_fingerprint = "c21a6da6c16c"
    policy.validation_25y_fingerprint = "5d6c177f02a7"
    return policy
