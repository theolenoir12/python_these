import pytest

from Common.rb1_policy_v11 import make_rb1_policy_v11


def test_rb1_threshold_validation():
    with pytest.raises(ValueError):
        make_rb1_policy_v11(0.6, 0.4)


def test_rb1_parameters_are_explicit():
    rule = make_rb1_policy_v11(0.2, 0.35)
    assert rule.rb1_parameters == {"soc_low": 0.2, "soc_high": 0.35}
