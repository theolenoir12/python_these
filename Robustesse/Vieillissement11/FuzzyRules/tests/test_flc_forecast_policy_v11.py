import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.flc_forecast_policy_v11 import (
    DEFAULT_SIGMA_KWH,
    make_forecast_augmented_flc_policy_v11,
    make_selected_if_policy_v11,
)
from FuzzyRules.flc_policy_v11 import make_tuned_expert_flc_policy_v11
from FuzzyRules.flc_soh_policy_v11 import make_soh_augmented_flc_policy_v11


def _args(
    soc=0.60, net=-12000.0, soh_bat=1.0, h2=100.0,
    soh_fc=1.0, soh_ely=1.0,
):
    return (
        soc, net, [], np.zeros(1), 0.0, 0.0, soh_bat, h2, 200.0,
        I.FC["P_fc_max"], I.ELY["P_ely_max"],
        float("inf"), float("inf"), soh_fc, soh_ely,
    )


class ForecastAugmentedFLCPolicyTests(unittest.TestCase):
    def test_zero_forecast_strength_is_bit_exact_i0_and_does_not_draw(self):
        parent = make_tuned_expert_flc_policy_v11()
        null = make_forecast_augmented_flc_policy_v11(
            forecast_strength=0.0,
            forecast_scenario="gaussian_iid",
            noise_seed=123,
        )
        future = np.full(18, 5000.0)
        for net in (-12000.0, 3000.0):
            args = _args(net=net)
            self.assertEqual(
                null(*args, P_tot_ref_future=future), parent(*args)
            )
        self.assertEqual(null.forecast_diagnostics()["noise_draws"], 0)

    def test_zero_forecast_strength_is_bit_exact_soh_parent(self):
        parent = make_soh_augmented_flc_policy_v11(0.0, 0.025)
        null = make_forecast_augmented_flc_policy_v11(
            forecast_strength=0.0,
            forecast_scenario="gaussian_iid",
            soh_strength_fc=0.0,
            soh_strength_ely=0.025,
        )
        args = _args(soh_bat=0.8, soh_ely=0.92)
        self.assertEqual(
            null(*args, P_tot_ref_future=np.ones(18)), parent(*args)
        )

    def test_oracle_deficit_forecast_reduces_ely_only(self):
        parent = make_tuned_expert_flc_policy_v11()
        policy = make_forecast_augmented_flc_policy_v11(
            forecast_strength=1.0,
            forecast_scenario="oracle",
        )
        args = _args(soc=0.5, net=-12000.0)
        base, _ = parent(*args)
        action, _ = policy(
            *args, P_tot_ref_future=np.full(18, 3000.0)
        )
        self.assertLess(base[2], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertEqual(action[1], base[1])
        self.assertAlmostEqual(sum(action), -12000.0, places=9)

    def test_missing_forecast_leaves_parent_unchanged(self):
        parent = make_tuned_expert_flc_policy_v11()
        policy = make_forecast_augmented_flc_policy_v11(
            forecast_strength=1.0,
        )
        args = _args()
        self.assertEqual(
            policy(*args, P_tot_ref_future=None), parent(*args)
        )

    def test_reset_reproduces_noise_and_hysteresis(self):
        policy = make_forecast_augmented_flc_policy_v11(
            forecast_strength=1.0,
            forecast_scenario="gaussian_iid",
            noise_seed=456,
        )
        future = np.zeros(18)
        first = policy.precharge_signal(0.5, 0.0, future)
        policy.precharge_signal(0.5, 0.0, future)
        policy.reset()
        replay = policy.precharge_signal(0.5, 0.0, future)
        self.assertEqual(first, replay)
        self.assertEqual(policy.forecast_diagnostics()["noise_draws"], 1)

    def test_hysteresis_threshold_uses_backtest_sigma(self):
        policy = make_forecast_augmented_flc_policy_v11(
            forecast_strength=1.0,
            forecast_scenario="oracle",
            threshold_sigma_multiplier=1.0,
        )
        below = np.full(18, (0.9 * DEFAULT_SIGMA_KWH * 1000.0 / 18.0))
        above = np.full(18, (1.1 * DEFAULT_SIGMA_KWH * 1000.0 / 18.0))
        self.assertFalse(policy.precharge_signal(0.5, 0.0, below)[0])
        self.assertTrue(policy.precharge_signal(0.5, 0.0, above)[0])

    def test_metadata_distinguishes_if_and_isf(self):
        if_policy = make_forecast_augmented_flc_policy_v11(1.0)
        isf_policy = make_forecast_augmented_flc_policy_v11(
            1.0, soh_strength_ely=0.025
        )
        self.assertEqual(if_policy.information_set, "IF")
        self.assertEqual(isf_policy.information_set, "ISF")
        self.assertEqual(if_policy.forecast_horizon_steps, 18)

    def test_selected_constructor_freezes_promoted_if(self):
        policy = make_selected_if_policy_v11()
        self.assertEqual(policy.information_set, "IF")
        self.assertEqual(policy.forecast_parameters["forecast_strength"], 1.0)
        self.assertEqual(policy.forecast_parameters["soh_strength_ely"], 0.0)
        with self.assertRaises(ValueError):
            make_selected_if_policy_v11(forecast_strength=0.5)


if __name__ == "__main__":
    unittest.main()
