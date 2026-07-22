"""Tests de la policy ANFIS deux branches (V11-p=2).

Le modèle ANFIS est ajusté sur des données synthétiques à cible CONSTANTE par
branche (déficit -> +1000 W, surplus -> -3000 W), ce qui rend ses prédictions
contrôlables sans le cache enseignant. On vérifie alors les mêmes invariants de
wrapper que l'arbre : bilan fermé, gardes de défaillance, zone morte, reset,
métadonnées (2 x 27 règles).
"""

import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.anfis_policy_v11 import (
    POLICY_ID_STEM,
    AnfisTwoBranch,
    make_anfis_policy_v11,
)


def _toy_model(deficit_u=1000.0, surplus_u=-3000.0, n=400, seed=0):
    rng = np.random.default_rng(seed)
    # déficit : P_net>0 ; surplus : P_net<0 ; SoC/H2 normalisés dans [0,1].
    Xd = np.column_stack([rng.uniform(200.0, 4000.0, n),
                          rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    Xs = np.column_stack([rng.uniform(-8000.0, -200.0, n),
                          rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    X = np.vstack([Xd, Xs])
    y = np.concatenate([np.full(n, deficit_u), np.full(n, surplus_u)])
    return AnfisTwoBranch.fit(X, y, "I0", n_mf=3)


def _args(net, soc=0.60, h2=100.0, soh_bat=1.0, soh_fc=1.0, soh_ely=1.0,
          failures=None, p_fc_max=None, p_ely_max=None):
    return (
        soc, net, list(failures or []), np.zeros(1), 0.0, 0.0, soh_bat,
        h2, 200.0,
        I.FC["P_fc_max"] if p_fc_max is None else p_fc_max,
        I.ELY["P_ely_max"] if p_ely_max is None else p_ely_max,
        float("inf"), float("inf"), soh_fc, soh_ely,
    )


class TwoBranchModelTests(unittest.TestCase):
    def test_routing_predicts_right_sign_per_branch(self):
        model = _toy_model()
        pred = model.predict(np.array([[2000.0, 0.5, 0.5], [-5000.0, 0.5, 0.5]]))
        self.assertGreater(pred[0], 0.0)   # déficit -> FC
        self.assertLess(pred[1], 0.0)      # surplus -> ELY

    def test_fit_rejects_too_few_samples_per_branch(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.uniform(-1, 1, 10), rng.uniform(0, 1, 10),
                             rng.uniform(0, 1, 10)])
        with self.assertRaises(ValueError):
            AnfisTwoBranch.fit(X, np.zeros(10), "I0", n_mf=3)


class AnfisPolicyInvariantTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_anfis_policy_v11(_toy_model())

    def test_metadata_and_rule_count(self):
        self.assertEqual(self.policy.information_set, "I0")
        self.assertTrue(self.policy.policy_id.startswith(f"{POLICY_ID_STEM}-i0-"))
        self.assertEqual(len(self.policy.anfis_metadata["spec_sha256"]), 64)
        self.assertEqual(self.policy.anfis_metadata["rule_count"],
                         {"deficit": 27, "surplus": 27})

    def test_deficit_uses_fc_and_battery_only_and_closes_balance(self):
        action, lol = self.policy(*_args(net=2000.0))
        self.assertGreater(action[1], 0.0)     # p_fc > 0
        self.assertEqual(action[2], 0.0)       # p_ely == 0
        self.assertAlmostEqual(sum(action), 2000.0, places=6)
        self.assertEqual(lol, 0.0)

    def test_surplus_uses_ely_and_battery_only_and_closes_balance(self):
        action, lol = self.policy(*_args(net=-5000.0))
        self.assertEqual(action[1], 0.0)       # p_fc == 0
        self.assertLess(action[2], 0.0)        # p_ely < 0
        self.assertAlmostEqual(sum(action), -5000.0, places=6)
        self.assertEqual(lol, 0.0)

    def test_large_deadband_routes_everything_to_battery(self):
        policy = make_anfis_policy_v11(_toy_model(), deadband_w=1e9)
        action, _ = policy(*_args(net=2000.0))
        self.assertEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertAlmostEqual(action[0], 2000.0, places=6)

    def test_deadband_below_prediction_keeps_command(self):
        policy = make_anfis_policy_v11(_toy_model(), deadband_w=500.0)
        action, _ = policy(*_args(net=2000.0))
        self.assertGreater(action[1], 0.0)     # ~1000 W > 500 -> conservé

    def test_fc_failure_clamps_to_nonpositive(self):
        action, _ = self.policy(*_args(net=2000.0, failures=["FC"]))
        self.assertEqual(action[1], 0.0)
        self.assertAlmostEqual(sum(action), 2000.0, places=6)

    def test_ely_failure_clamps_to_nonnegative(self):
        action, _ = self.policy(*_args(net=-5000.0, failures=["ELY"]))
        self.assertEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), -5000.0, places=6)

    def test_physical_power_limit_applied_by_get_lol(self):
        action, _ = self.policy(*_args(net=3000.0, soc=0.3, p_fc_max=100.0))
        self.assertLessEqual(action[1] / I.CONV["eta"], 100.0 + 1e-6)

    def test_reset_is_exact_for_stateless_policy(self):
        before = self.policy(*_args(net=2000.0))
        self.policy.reset()
        after = self.policy(*_args(net=2000.0))
        self.assertEqual(before, after)

    def test_soh_is_not_a_hidden_input_in_i0(self):
        fresh = self.policy(*_args(net=2000.0, soh_fc=1.0, soh_ely=1.0))
        aged = self.policy(*_args(net=2000.0, soh_fc=0.9, soh_ely=0.9))
        self.assertEqual(fresh, aged)


if __name__ == "__main__":
    unittest.main()
