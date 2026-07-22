"""Tests de la policy IF 'arbre + regle de precharge' (direction A), V11-p=2.

Verifie les invariants obligatoires (PLAN_FUZZY_RULE_LEARNING_V11_P2 section 6) et
surtout l'APPARIEMENT STRICT du signal de precharge avec la FLC-IF : sur des
sequences d'entrees identiques, la regle greffee sur l'arbre et celle greffee sur
la FLC experte doivent produire exactement le meme signal (meme feature de
prevision, meme hysteresis). Ce qui distingue les deux familles est alors le seul
parent (arbre vs FLC), donc un effet de famille pur.
"""

import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.flc_forecast_policy_v11 import (
    make_forecast_augmented_flc_policy_v11,
)
from FuzzyRules.rl_tree_policy_v11 import fit_tree, make_tree_policy_v11
from FuzzyRules.rl_tree_precharge_policy_v11 import (
    POLICY_ID_STEM,
    make_tree_precharge_policy_v11,
)
from FuzzyRules.tests.test_rl_tree_policy_v11 import _I0_ROWS, _args, _toy_dataset


def _i0_tree():
    return fit_tree(_toy_dataset(_I0_ROWS, "I0"), "I0", max_depth=3,
                    min_samples_leaf=1)


def _if_tree():
    from FuzzyRules.tests.test_rl_tree_forecast_policy_v11 import _IF_ROWS
    return fit_tree(_toy_dataset(_IF_ROWS, "IF"), "IF", max_depth=3,
                    min_samples_leaf=1)


class ConstructionTests(unittest.TestCase):
    def test_rejects_if_tree_as_parent(self):
        with self.assertRaises(ValueError):
            make_tree_precharge_policy_v11(_if_tree())

    def test_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            make_tree_precharge_policy_v11(_i0_tree(),
                                          forecast_scenario="crystal_ball")

    def test_rejects_out_of_range_strength(self):
        with self.assertRaises(ValueError):
            make_tree_precharge_policy_v11(_i0_tree(), forecast_strength=1.5)

    def test_metadata_declares_if_from_i0_parent(self):
        policy = make_tree_precharge_policy_v11(_i0_tree(), forecast_strength=1.0)
        self.assertEqual(policy.information_set, "IF")
        self.assertEqual(policy.forecast_horizon_steps, 18)
        self.assertTrue(policy.policy_id.startswith(POLICY_ID_STEM))
        self.assertEqual(policy.rl_metadata["parent_information_set"], "I0")
        self.assertEqual(policy.rl_metadata["forecast_rule_count"], 3)


class NullTestBitExact(unittest.TestCase):
    """Force nulle = arbre parent bit-a-bit, sans aucun tirage aleatoire."""

    def test_zero_strength_is_bit_exact_and_draws_nothing(self):
        tree = _i0_tree()
        parent = make_tree_policy_v11(tree)
        null = make_tree_precharge_policy_v11(
            tree, forecast_strength=0.0, forecast_scenario="gaussian_iid",
            noise_seed=123,
        )
        future = np.full(18, 5000.0)
        for net in (-5000.0, 2000.0, 150.0):
            args = _args(net=net)
            self.assertEqual(null(*args, P_tot_ref_future=future), parent(*args))
        self.assertEqual(null.forecast_diagnostics()["noise_draws"], 0)


class PairingWithFLCIF(unittest.TestCase):
    """Le signal de precharge est identique a celui de la FLC-IF."""

    def _pair(self, scenario, seed=0):
        rl = make_tree_precharge_policy_v11(
            _i0_tree(), forecast_strength=1.0, forecast_scenario=scenario,
            noise_seed=seed,
        )
        flc = make_forecast_augmented_flc_policy_v11(
            forecast_strength=1.0, forecast_scenario=scenario, noise_seed=seed,
        )
        return rl, flc

    def test_oracle_signal_matches_over_a_sequence(self):
        rl, flc = self._pair("oracle")
        rng = np.random.default_rng(1)
        for _ in range(500):
            soc = float(rng.uniform(0.2, 1.0))
            pnet = float(rng.uniform(-20000.0, 5000.0))
            future = rng.uniform(-20000.0, 5000.0, size=18)
            a_rl, e_rl = rl.precharge_signal(soc, pnet, future)
            a_flc, e_flc = flc.precharge_signal(soc, pnet, future)
            self.assertEqual(a_rl, a_flc)
            self.assertAlmostEqual(e_rl, e_flc, places=9)

    def test_gaussian_iid_signal_matches_with_shared_seed(self):
        rl, flc = self._pair("gaussian_iid", seed=7)
        rng = np.random.default_rng(2)
        for _ in range(500):
            soc = float(rng.uniform(0.2, 1.0))
            pnet = float(rng.uniform(-20000.0, 5000.0))
            future = rng.uniform(-20000.0, 5000.0, size=18)
            self.assertEqual(rl.precharge_signal(soc, pnet, future)[0],
                             flc.precharge_signal(soc, pnet, future)[0])


class PrechargeActionTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_tree_precharge_policy_v11(
            _i0_tree(), forecast_strength=1.0, forecast_scenario="oracle")
        self.future_deficit = np.full(18, 3000.0)   # E_net_H > 0 -> precharge ON

    def test_precharge_moves_surplus_from_ely_to_battery(self):
        parent = make_tree_policy_v11(_i0_tree())
        base, _ = parent(*_args(net=-5000.0, soc=0.6))
        action, _ = self.policy(*_args(net=-5000.0, soc=0.6),
                                P_tot_ref_future=self.future_deficit)
        self.assertLess(base[2], 0.0)      # le parent electrolysait
        self.assertEqual(action[2], 0.0)   # precharge coupe l'ELY (force 1)
        self.assertEqual(action[1], 0.0)   # toujours pas de FC
        self.assertAlmostEqual(sum(action), -5000.0, places=9)  # bilan ferme

    def test_precharge_leaves_deficit_action_untouched(self):
        # En deficit le parent ne fait pas d'ELY -> rien a preserver.
        parent = make_tree_policy_v11(_i0_tree())
        base = parent(*_args(net=2000.0, soc=0.6))
        action = self.policy(*_args(net=2000.0, soc=0.6),
                             P_tot_ref_future=self.future_deficit)
        self.assertEqual(action, base)

    def test_soc_at_target_disables_precharge(self):
        # SoC >= soc_target : batterie deja pleine, pas de precharge meme si
        # un deficit est prevu.
        parent = make_tree_policy_v11(_i0_tree())
        base = parent(*_args(net=-5000.0, soc=0.995))
        action = self.policy(*_args(net=-5000.0, soc=0.995),
                             P_tot_ref_future=self.future_deficit)
        self.assertEqual(action, base)

    def test_no_future_leaves_parent_unchanged(self):
        parent = make_tree_policy_v11(_i0_tree())
        base = parent(*_args(net=-5000.0, soc=0.6))
        action = self.policy(*_args(net=-5000.0, soc=0.6),
                             P_tot_ref_future=None)
        self.assertEqual(action, base)

    def test_future_surplus_does_not_trigger_precharge(self):
        # E_net_H < 0 (surplus a venir) -> precharge OFF -> parent inchange.
        parent = make_tree_policy_v11(_i0_tree())
        base = parent(*_args(net=-5000.0, soc=0.6))
        action = self.policy(*_args(net=-5000.0, soc=0.6),
                             P_tot_ref_future=np.full(18, -3000.0))
        self.assertEqual(action, base)


class HysteresisTests(unittest.TestCase):
    def test_state_is_retained_between_thresholds(self):
        policy = make_tree_precharge_policy_v11(
            _i0_tree(), forecast_strength=1.0, forecast_scenario="gaussian_iid",
            sigma_design_kwh=10.0, threshold_sigma_multiplier=1.0,
            sigma_inject_kwh=0.0, bias_kwh=0.0, noise_seed=0,
        )
        # seuil = 1 * 10 = 10 kWh. E_net_H = sum(future)*1/1000.
        big_deficit = np.full(18, 2000.0)   # 36 kWh > +10 -> ON
        small = np.full(18, 100.0)          # 1.8 kWh, entre -10 et +10 -> retient
        big_surplus = np.full(18, -2000.0)  # -36 kWh < -10 -> OFF
        self.assertTrue(policy.precharge_signal(0.5, 0.0, big_deficit)[0])
        self.assertTrue(policy.precharge_signal(0.5, 0.0, small)[0])   # retenu ON
        self.assertFalse(policy.precharge_signal(0.5, 0.0, big_surplus)[0])
        self.assertFalse(policy.precharge_signal(0.5, 0.0, small)[0])  # retenu OFF

    def test_reset_replays_noise_and_hysteresis(self):
        policy = make_tree_precharge_policy_v11(
            _i0_tree(), forecast_strength=1.0, forecast_scenario="gaussian_iid",
            noise_seed=456,
        )
        future = np.zeros(18)
        first = policy.precharge_signal(0.5, 0.0, future)
        policy.precharge_signal(0.5, 0.0, future)
        policy.reset()
        replay = policy.precharge_signal(0.5, 0.0, future)
        self.assertEqual(first, replay)
        self.assertEqual(policy.forecast_diagnostics()["noise_draws"], 1)


class ISParentTests(unittest.TestCase):
    def test_is_parent_yields_isf(self):
        rows = [
            ((2000.0, 0.5, 0.5, 0.0, 0.0, 0.0), 1000.0),
            ((-5000.0, 0.5, 0.5, 0.0, 0.0, 0.0), -3000.0),
        ]
        tree = fit_tree(_toy_dataset(rows, "IS"), "IS", max_depth=3,
                        min_samples_leaf=1)
        policy = make_tree_precharge_policy_v11(tree, forecast_strength=1.0)
        self.assertEqual(policy.information_set, "ISF")
        self.assertEqual(policy.rl_metadata["parent_information_set"], "IS")


if __name__ == "__main__":
    unittest.main()
