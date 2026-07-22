import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.flc_policy_v11 import make_tuned_expert_flc_policy_v11
from FuzzyRules.flc_soh_policy_v11 import (
    make_soh_augmented_flc_policy_v11,
    normalized_wear,
)


def _args(
    soc=0.60, net=1000.0, soh_bat=1.0, h2=100.0,
    soh_fc=1.0, soh_ely=1.0,
):
    return (
        soc, net, [], np.zeros(1), 0.0, 0.0, soh_bat, h2, 200.0,
        I.FC["P_fc_max"], I.ELY["P_ely_max"],
        float("inf"), float("inf"), soh_fc, soh_ely,
    )


class SoHAugmentedFLCPolicyTests(unittest.TestCase):
    def test_normalized_wear_uses_each_component_eol_interval(self):
        self.assertEqual(normalized_wear(1.0, 0.9), 0.0)
        self.assertAlmostEqual(normalized_wear(0.95, 0.9), 0.5)
        self.assertEqual(normalized_wear(0.9, 0.9), 1.0)
        self.assertEqual(normalized_wear(0.0, 0.9), 1.0)

    def test_zero_strength_is_bit_exact_parent_for_all_soh_values(self):
        parent = make_tuned_expert_flc_policy_v11()
        null = make_soh_augmented_flc_policy_v11(0.0, 0.0)
        for net in (-12000.0, -500.0, 0.0, 500.0, 3000.0):
            args = _args(
                net=net, soh_bat=0.73, soh_fc=0.91, soh_ely=0.92
            )
            self.assertEqual(null(*args), parent(*args))

    def test_nonzero_strength_is_exactly_neutral_at_bol(self):
        parent = make_tuned_expert_flc_policy_v11()
        augmented = make_soh_augmented_flc_policy_v11(0.5, 0.5)
        for net in (-12000.0, 3000.0):
            self.assertEqual(augmented(*_args(net=net)), parent(*_args(net=net)))

    def test_fc_is_relieved_when_more_worn_than_battery(self):
        policy = make_soh_augmented_flc_policy_v11(0.5, 0.0)
        relieved, _ = policy(*_args(
            net=3000.0, soh_bat=1.0, soh_fc=I.FC["SoH_EoL"]
        ))
        supported, _ = policy(*_args(
            net=3000.0, soh_bat=I.BAT["SoH_EoL"], soh_fc=1.0
        ))
        self.assertLess(relieved[1], supported[1])

    def test_ely_is_relieved_when_more_worn_than_battery(self):
        policy = make_soh_augmented_flc_policy_v11(0.0, 0.5)
        relieved, _ = policy(*_args(
            net=-12000.0, soh_bat=1.0, soh_ely=I.ELY["SoH_EoL"]
        ))
        supported, _ = policy(*_args(
            net=-12000.0, soh_bat=I.BAT["SoH_EoL"], soh_ely=1.0
        ))
        self.assertLess(abs(relieved[2]), abs(supported[2]))

    def test_inactive_component_soh_cannot_change_other_branch(self):
        policy = make_soh_augmented_flc_policy_v11(0.5, 0.5)
        deficit_fresh_ely = policy(*_args(net=3000.0, soh_ely=1.0))
        deficit_worn_ely = policy(*_args(net=3000.0, soh_ely=0.9))
        surplus_fresh_fc = policy(*_args(net=-12000.0, soh_fc=1.0))
        surplus_worn_fc = policy(*_args(net=-12000.0, soh_fc=0.9))
        self.assertEqual(deficit_fresh_ely, deficit_worn_ely)
        self.assertEqual(surplus_fresh_fc, surplus_worn_fc)

    def test_metadata_exposes_small_health_rule_base(self):
        policy = make_soh_augmented_flc_policy_v11(0.1, 0.2)
        self.assertEqual(policy.information_set, "IS")
        self.assertEqual(policy.flc_metadata["health_rule_count"], 9)
        self.assertEqual(policy.flc_metadata["parent_candidate_id"], "flc_8126e6f729c6")


if __name__ == "__main__":
    unittest.main()
