import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.flc_policy_v11 import (
    TUNED_I0_CANDIDATE_ID,
    TUNED_I0_PARAMETERS,
    make_expert_flc_policy_v11,
    make_tuned_expert_flc_policy_v11,
)


def _args(
    soc=0.60, net=1000.0, soh_bat=1.0, h2=100.0,
    p_fc_max=None, p_ely_max=None, soh_fc=1.0, soh_ely=1.0,
    failures=None,
):
    return (
        soc, net, list(failures or []), np.zeros(1), 0.0, 0.0, soh_bat,
        h2, 200.0,
        I.FC["P_fc_max"] if p_fc_max is None else p_fc_max,
        I.ELY["P_ely_max"] if p_ely_max is None else p_ely_max,
        float("inf"), float("inf"), soh_fc, soh_ely,
    )


class ExpertFLCPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_expert_flc_policy_v11()

    def test_identity_and_rule_count_are_explicit(self):
        self.assertEqual(self.policy.information_set, "I0")
        self.assertIn("v11-p2-i0", self.policy.policy_id)
        self.assertEqual(len(self.policy.flc_metadata["spec_sha256"]), 64)
        self.assertEqual(self.policy.flc_metadata["rule_count"], {
            "deficit": 27, "surplus": 27,
        })

    def test_zero_net_power_returns_zero_action(self):
        action, lol = self.policy(*_args(net=0.0))
        self.assertEqual(action, (0.0, 0.0, 0.0))
        self.assertEqual(lol, 0.0)

    def test_deficit_uses_fc_and_battery_only_and_closes_balance(self):
        action, lol = self.policy(*_args(net=1000.0))
        self.assertGreaterEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), 1000.0, places=9)
        self.assertEqual(lol, 0.0)

    def test_surplus_uses_ely_and_battery_only_and_closes_balance(self):
        action, lol = self.policy(*_args(net=-5000.0))
        self.assertEqual(action[1], 0.0)
        self.assertLessEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), -5000.0, places=9)
        self.assertEqual(lol, 0.0)

    def test_physical_power_limits_are_applied_by_common_safety_layer(self):
        action_fc, _ = self.policy(*_args(
            soc=0.3, net=4000.0, p_fc_max=100.0
        ))
        action_ely, _ = self.policy(*_args(
            soc=0.9, net=-20000.0, p_ely_max=100.0
        ))
        self.assertLessEqual(action_fc[1] / I.CONV["eta"], 100.0)
        self.assertLessEqual(abs(action_ely[2]) * I.CONV["eta"], 100.0)

    def test_soh_fc_and_ely_are_not_hidden_decision_inputs(self):
        fresh, _ = self.policy(*_args(soh_fc=1.0, soh_ely=1.0))
        aged, _ = self.policy(*_args(soh_fc=0.9, soh_ely=0.9))
        self.assertEqual(fresh, aged)

    def test_deficit_semantics_follow_soc_and_h2_reserves(self):
        low_soc = self.policy.command_fraction("deficit", 0.6, 0.1, 0.8)
        high_soc = self.policy.command_fraction("deficit", 0.6, 0.9, 0.8)
        scarce_h2 = self.policy.command_fraction("deficit", 0.6, 0.6, 0.1)
        abundant_h2 = self.policy.command_fraction("deficit", 0.6, 0.6, 0.9)
        self.assertGreater(low_soc, high_soc)
        self.assertGreater(abundant_h2, scarce_h2)

    def test_surplus_semantics_follow_soc_and_tank_space(self):
        low_soc = self.policy.command_fraction("surplus", 0.6, 0.1, 0.2)
        high_soc = self.policy.command_fraction("surplus", 0.6, 0.9, 0.2)
        tank_space = self.policy.command_fraction("surplus", 0.6, 0.6, 0.1)
        tank_full = self.policy.command_fraction("surplus", 0.6, 0.6, 0.9)
        self.assertGreater(high_soc, low_soc)
        self.assertGreater(tank_space, tank_full)

    def test_failed_h2_component_is_replaced_by_battery_request(self):
        deficit, _ = self.policy(*_args(net=1000.0, failures=["FC"]))
        surplus, _ = self.policy(*_args(net=-1000.0, failures=["ELY"]))
        self.assertEqual(deficit, (1000.0, 0.0, 0.0))
        self.assertEqual(surplus, (-1000.0, 0.0, 0.0))

    def test_reset_is_exact_for_stateless_policy(self):
        before = self.policy(*_args(net=1000.0))
        self.policy.reset()
        after = self.policy(*_args(net=1000.0))
        self.assertEqual(before, after)

    def test_zero_power_scale_is_rejected_instead_of_silently_defaulted(self):
        with self.assertRaises(ValueError):
            make_expert_flc_policy_v11(deficit_scale_w=0.0)

    def test_promoted_tuned_policy_is_frozen_and_identified(self):
        tuned = make_tuned_expert_flc_policy_v11()
        self.assertEqual(tuned.candidate_id, TUNED_I0_CANDIDATE_ID)
        self.assertEqual(tuned.tuning_parameters, TUNED_I0_PARAMETERS)
        self.assertEqual(
            tuned.flc_metadata["spec_sha256"],
            "71c0531744f2ecf0b6cde6ee97a7ed0ba0d3d2468cebca06caa75643c2bd162d",
        )


if __name__ == "__main__":
    unittest.main()
