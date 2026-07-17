import pytest

from Common.rb2_policy import make_rb2_policy
from Common.rb2_soh_policy_v11 import make_rb2_soh_policy_v11


def _args(soh_bat=1.0, soh_fc=1.0, soh_ely=1.0):
    return (
        0.8, 7000.0, [], [], 0.0, 0.0, soh_bat, 200.0, 200.0,
        10_000.0, 10_000.0, float("inf"), float("inf"), soh_fc, soh_ely,
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"fc_self": 0.1, "ely_self": -0.1},
        {"fc_from_bat": 0.2, "ely_from_bat": -0.2},
        {"fc_from_ely": -0.1, "ely_from_fc": 0.1, "shape": 4.0},
    ],
)
def test_all_soh_one_is_exact_parent_rb2(coefficients):
    parent = make_rb2_policy(0.59, 0.49)
    augmented = make_rb2_soh_policy_v11(0.59, 0.49, **coefficients)
    assert augmented(*_args()) == parent(*_args())


def test_policy_remains_two_setpoint_dispatch_when_aged():
    policy = make_rb2_soh_policy_v11(
        0.59, 0.49, fc_from_bat=0.05, ely_self=-0.04
    )
    action, _ = policy(*_args(soh_bat=0.70, soh_ely=0.90))
    assert len(action) == 3
    assert policy.rb2_parameters["fc_from_bat"] == 0.05


def test_operando_source_is_explicit_and_supported():
    policy = make_rb2_soh_policy_v11(
        0.59, 0.49, fc_self=0.05, soh_source="operando"
    )
    context = {
        "fc": {"soh_operando": 0.9},
        "ely": {"soh_operando": 0.9},
    }
    action_operando, _ = policy(*_args(), aging_context=context)
    action_permanent, _ = make_rb2_soh_policy_v11(
        0.59, 0.49, fc_self=0.05, soh_source="permanent"
    )(*_args())
    assert action_operando != action_permanent
