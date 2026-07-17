import inspect
import os
import sys

import numpy as np
import pytest

V10 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, V10)
sys.path.insert(0, os.path.join(V10, "RB2"))

from Common.cost_fcn_total2 import ELY_REC, _ely_rates, get_cost_bat
from Common.Init_EMR_MG_v16_python import BAT
from rb2_policy import make_rb2_policy


def test_ely_irreversible_rate_anchors_and_continuity():
    assert _ely_rates(0.0)[0] == 0.0
    assert _ely_rates(1.0)[0] == pytest.approx(1.2)
    assert _ely_rates(2.0)[0] == pytest.approx(4.8)
    assert _ely_rates(3.0)[0] == pytest.approx(
        4.8 + ELY_REC["high_current_accel"]
    )
    assert _ely_rates(2.0 - 1e-8)[0] == pytest.approx(
        _ely_rates(2.0 + 1e-8)[0], abs=1e-6
    )


def test_high_current_irreversible_rate_is_not_capped():
    assert _ely_rates(3.0)[0] > _ely_rates(2.5)[0] > _ely_rates(2.0)[0]


def test_battery_current_scaling_has_unit_floor_below_one_c():
    """A excursion SoC identique, le facteur V10 vaut 1 sous 1C."""
    voltage = BAT["v_cell_nom"] * BAT["series_num"]
    current_1c = BAT["Q_bat"] * BAT["parallel_num"]
    soc = np.array([0.40, 0.50])
    cost_1c = get_cost_bat(np.array([voltage * current_1c]), soc, 1.0)
    cost_half_c = get_cost_bat(
        np.array([0.5 * voltage * current_1c]), soc, 1.0
    )
    assert cost_half_c == pytest.approx(cost_1c)


def test_battery_current_scaling_matches_v10_anchors():
    """Le choix V10 impose psi=1 jusqu'a 1C, puis psi(2C)=1.2956."""
    voltage = BAT["v_cell_nom"] * BAT["series_num"]
    current_1c = BAT["Q_bat"] * BAT["parallel_num"]
    soc = np.array([0.40, 0.50])
    base = get_cost_bat(np.array([voltage * current_1c]), soc, 1.0)
    assert get_cost_bat(
        np.array([0.01 * voltage * current_1c]), soc, 1.0
    ) == pytest.approx(base)
    assert get_cost_bat(
        np.array([2.0 * voltage * current_1c]), soc, 1.0
    ) == pytest.approx(1.2956 * base)

def test_rb2_policy_has_only_two_power_setpoints():
    assert list(inspect.signature(make_rb2_policy).parameters) == [
        "fc_setpoint",
        "ely_setpoint",
    ]
