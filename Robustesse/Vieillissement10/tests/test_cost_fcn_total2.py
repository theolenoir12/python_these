import inspect
import os
import sys

import pytest

V10 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, V10)
sys.path.insert(0, os.path.join(V10, "RB2"))

from Common.cost_fcn_total2 import ELY_REC, _ely_rates
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

def test_rb2_policy_has_only_two_power_setpoints():
    assert list(inspect.signature(make_rb2_policy).parameters) == [
        "fc_setpoint",
        "ely_setpoint",
    ]
