import os
import sys

import numpy as np


V10 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if V10 not in sys.path:
    sys.path.insert(0, V10)

from Common import Init_EMR_MG_v16_python as I
from Common.rb2_policy import (
    _normalized_wear_factor,
    make_augmented_rb2_policy,
    make_rb2_policy,
)


def _call(policy, p_ref=2000.0, future=None, rul_fc=np.inf, rul_ely=np.inf):
    args = (
        0.5, p_ref, [], np.zeros(1), 0.0, 0.0, 1.0,
        100.0, 200.0, I.FC["P_fc_max"], I.ELY["P_ely_max"],
        rul_fc, rul_ely, 1.0, 1.0,
    )
    if future is None:
        return policy(*args)
    return policy(*args, np.asarray(future, dtype=float))


def test_augmented_null_case_is_exactly_base_rb2():
    base = make_rb2_policy(0.59, 0.49)
    augmented = make_augmented_rb2_policy(
        fc_setpoint=0.59, ely_setpoint=0.49,
        soh_gamma_fc=0.0, soh_gamma_ely=0.0,
        rul_gamma_fc=0.0, rul_gamma_ely=0.0,
        forecast_enabled=False,
    )
    for p_ref in (-20000.0, -500.0, 0.0, 500.0, 2000.0):
        action_base, lol_base = _call(base, p_ref=p_ref)
        action_aug, lol_aug = _call(augmented, p_ref=p_ref)
        np.testing.assert_allclose(action_aug, action_base, rtol=0.0, atol=0.0)
        assert lol_aug == lol_base


def test_unknown_rul_does_not_derate_setpoints():
    base = make_rb2_policy(0.59, 0.49)
    rul = make_augmented_rb2_policy(
        fc_setpoint=0.59, ely_setpoint=0.49,
        rul_ref_fc_days=3000.0, rul_ref_ely_days=8000.0,
        rul_gamma_fc=0.1, rul_gamma_ely=0.1,
    )
    action_base, lol_base = _call(base, p_ref=2000.0)
    action_rul, lol_rul = _call(rul, p_ref=2000.0)
    np.testing.assert_allclose(action_rul, action_base, rtol=0.0, atol=0.0)
    assert lol_rul == lol_base


def test_normalized_soh_null_case_is_exactly_base_rb2():
    base = make_rb2_policy(0.59, 0.49)
    normalized = make_augmented_rb2_policy(
        fc_setpoint=0.59, ely_setpoint=0.49,
        soh_mode="normalized_wear",
        soh_strength_fc=0.0, soh_strength_ely=0.0,
        soh_shape_fc=4.0, soh_shape_ely=4.0,
    )
    action_base, lol_base = _call(base, p_ref=2000.0)
    action_normalized, lol_normalized = _call(normalized, p_ref=2000.0)
    np.testing.assert_allclose(action_normalized, action_base, rtol=0.0, atol=0.0)
    assert lol_normalized == lol_base


def test_normalized_soh_strength_is_the_exact_eol_derating():
    assert _normalized_wear_factor(1.0, 0.9, 0.25, 1.0) == 1.0
    assert np.isclose(_normalized_wear_factor(0.95, 0.9, 0.25, 1.0), 0.875)
    assert np.isclose(_normalized_wear_factor(0.9, 0.9, 0.25, 1.0), 0.75)
    assert np.isclose(_normalized_wear_factor(0.0, 0.9, 0.25, 1.0), 0.75)


def test_prediction_only_inhibits_ely_setpoint():
    pred = make_augmented_rb2_policy(
        fc_setpoint=0.59, ely_setpoint=0.49,
        forecast_enabled=True, forecast_horizon_h=18,
        forecast_soc_target=0.99, forecast_noise_enabled=False,
        forecast_hysteresis_sigma=0.0, forecast_min_dwell_h=0.0,
    )
    action, _ = _call(pred, p_ref=-20000.0, future=np.full(18, 1000.0))
    p_bat, p_fc, p_ely = action
    assert p_fc == 0.0
    assert p_ely == 0.0
    assert p_bat < 0.0
