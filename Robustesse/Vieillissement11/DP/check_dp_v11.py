"""Controles courts du portage PD vers Vieillissement11.

Ce script ne resout aucune annee de PD. Il verifie les couts elementaires du
backward contre les transitions canoniques V11, les ancres en densite de
courant et l'absence des anciennes constantes de degradation dans les trois
modules executables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dp_core as dp
from Common.degradation_v11 import (
    MODEL_ID,
    advance_ely_density,
    advance_fc_density,
    new_ely_state,
    new_fc_state,
    state_cost_eur,
)


def _assert_close(actual, expected, label, atol=1e-10):
    if not np.isclose(actual, expected, rtol=1e-10, atol=atol):
        raise AssertionError(
            f"{label}: actual={actual:.16g}, expected={expected:.16g}"
        )


def check_stage_costs():
    alpha_fc, alpha_ely = 0.04, 0.06
    controls = np.array([
        0.0,
        0.45 * dp.P_FC_MAX * dp.ETA,
        *dp.v11_control_anchors(alpha_ely, dp.P_ELY_MAX),
        -0.75 * dp.P_ELY_MAX / dp.ETA,
    ])
    pre = dp.precompute_controls(
        controls, alpha_fc=alpha_fc, alpha_ely=alpha_ely)
    _, current_fc_on, current_ely_on, cost_fc, cost_ely = pre

    p_fc = np.maximum(controls, 0.0) / dp.ETA
    p_ely = np.abs(np.minimum(controls, 0.0) * dp.ETA)
    j_fc = np.asarray(dp.fc_current_density(p_fc, alpha_fc))
    j_ely = np.asarray(dp.ely_current_density(p_ely, alpha_ely))

    for index in range(len(controls)):
        for previous_on in (0, 1):
            previous_j_fc = j_fc[index] if previous_on else 0.0
            fc_state = advance_fc_density(
                new_fc_state(), j_fc[index], previous_j_fc, dp.TS_H)
            _assert_close(
                cost_fc[index, previous_on], state_cost_eur('fc', fc_state),
                f"cout PEMFC u={index}, prev={previous_on}",
            )

            previous_j_ely = j_ely[index] if previous_on else 0.0
            ely_state = advance_ely_density(
                new_ely_state(), j_ely[index], previous_j_ely, dp.TS_H)
            _assert_close(
                cost_ely[index, previous_on], state_cost_eur('ely', ely_state),
                f"cout PEMWE u={index}, prev={previous_on}",
            )

        if current_fc_on[index] != int(j_fc[index] > 1e-9):
            raise AssertionError("etat marche/arret PEMFC incoherent")
        if current_ely_on[index] != int(j_ely[index] > 1e-9):
            raise AssertionError("etat marche/arret PEMWE incoherent")


def check_density_anchors():
    for alpha in (0.0, 0.08):
        anchors = dp.v11_control_anchors(alpha, dp.P_ELY_MAX)
        densities = [
            float(dp.ely_current_density(abs(control) * dp.ETA, alpha))
            for control in anchors
        ]
        for actual, expected in zip(densities, (1.0, 2.0)):
            _assert_close(actual, expected, f"ancre PEMWE alpha={alpha}", atol=2e-3)


def check_model_attribution():
    _assert_close(
        dp.ELY_V11['stress_exponent'], dp.NOMINAL_ELY_STRESS_EXPONENT,
        "exposant PEMWE nominal",
    )
    if dp.RB1_REFERENCE_PARAMS != (0.20, 0.40):
        raise AssertionError("mauvais reglage RB1 de reference")
    if dp.RB2_REFERENCE_PARAMS != (0.574, 0.465):
        raise AssertionError("mauvais reglage RB2 de reference")

    forbidden = (
        'FC_FHIGH', 'FC_FLOW', 'FC_ALPHA_HIGH', 'FC_ALPHA_LOW',
        'ELY_F30', 'ELY_F60', 'ELY_REC', 'sens_common',
    )
    root = Path(__file__).resolve().parent
    for name in ('dp_core.py', 'dp_aging.py', 'dp_pareto.py'):
        text = (root / name).read_text(encoding='utf-8')
        found = [token for token in forbidden if token in text]
        if found:
            raise AssertionError(f"{name}: symboles legacy encore presents: {found}")


def main():
    check_stage_costs()
    check_density_anchors()
    check_model_attribution()
    print(f"OK -- PD V11 p=2 -- {MODEL_ID}")


if __name__ == '__main__':
    main()
