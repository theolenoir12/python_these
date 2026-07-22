"""Tests unitaires de la policy 'regles apprises' IF (arbre + prevision H18).

Comme pour les tests I0/IS, aucun cache enseignant n'est requis : l'arbre jouet
est ajuste pour que sa prediction DEPENDE de la feature de prevision, ce qui rend
observables la mecanique de prevision (oracle vs persistance), l'injection de
bruit gaussien seede et la restauration par ``reset()``.
"""

import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.rl_dataset_v11 import HORIZON_STEPS_IF
from FuzzyRules.rl_tree_forecast_policy_v11 import (
    POLICY_ID_STEM,
    make_forecast_tree_policy_v11,
)
from FuzzyRules.rl_tree_policy_v11 import fit_tree
from FuzzyRules.tests.test_rl_tree_policy_v11 import _I0_ROWS, _args, _toy_dataset


# Regimes : la cible depend de P_net (colonne 0) ET, en deficit, du signe de
# l'energie nette prevue (colonne 3). Un futur negatif (deficit a venir) demande
# plus de FC ; un futur positif en demande moins.
_IF_ROWS = [
    ((2000.0, 0.5, 0.5, -50.0), 1500.0),   # deficit + futur deficitaire
    ((2000.0, 0.5, 0.5, 50.0), 500.0),     # deficit + futur excedentaire
    ((-5000.0, 0.5, 0.5, 0.0), -3000.0),   # surplus -> ELY
]


def _if_tree():
    ds = _toy_dataset(_IF_ROWS, "IF")
    return fit_tree(ds, "IF", max_depth=3, min_samples_leaf=1)


def _i0_tree():
    return fit_tree(_toy_dataset(_I0_ROWS, "I0"), "I0", max_depth=3,
                    min_samples_leaf=1)


class ForecastTreeConstructionTests(unittest.TestCase):
    def test_rejects_non_if_tree(self):
        with self.assertRaises(ValueError):
            make_forecast_tree_policy_v11(_i0_tree())

    def test_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            make_forecast_tree_policy_v11(_if_tree(),
                                         forecast_scenario="crystal_ball")

    def test_rejects_out_of_range_noise_rho(self):
        with self.assertRaises(ValueError):
            make_forecast_tree_policy_v11(_if_tree(), forecast_scenario="oracle",
                                         noise_rho=1.0)

    def test_metadata_declares_if_and_horizon(self):
        policy = make_forecast_tree_policy_v11(_if_tree())
        self.assertEqual(policy.information_set, "IF")
        self.assertEqual(policy.forecast_horizon_steps, HORIZON_STEPS_IF)
        self.assertTrue(policy.policy_id.startswith(POLICY_ID_STEM))
        self.assertEqual(policy.rl_metadata["forecast"]["scenario"], "oracle")


class ForecastMechanicsTests(unittest.TestCase):
    def test_none_future_gives_zero_energy_and_closes_balance(self):
        # Futur absent -> e_net=0 (>=0) -> feuille "500" ; bilan ferme.
        policy = make_forecast_tree_policy_v11(_if_tree())
        action, lol = policy(*_args(net=2000.0), P_tot_ref_future=None)
        self.assertGreaterEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), 2000.0, places=9)
        self.assertEqual(lol, 0.0)

    def test_oracle_reads_future_sign(self):
        policy = make_forecast_tree_policy_v11(_if_tree(),
                                              forecast_scenario="oracle")
        future_deficit = np.full(HORIZON_STEPS_IF, 3000.0)   # e_net > 0
        future_surplus = np.full(HORIZON_STEPS_IF, -3000.0)  # e_net < 0
        p_fc_pos, _ = policy(*_args(net=2000.0), P_tot_ref_future=future_deficit)
        policy.reset()
        p_fc_neg, _ = policy(*_args(net=2000.0), P_tot_ref_future=future_surplus)
        # Futur deficitaire (e_net<0) -> plus de FC que futur excedentaire.
        self.assertGreater(p_fc_neg[1], p_fc_pos[1])

    def test_persistence_uses_current_power_not_future_values(self):
        policy = make_forecast_tree_policy_v11(_if_tree(),
                                              forecast_scenario="persistence")
        # Futur non vide mais dont les valeurs sont ignorees par la persistance :
        # e_net = P_tot_ref_t * H * dt/1000 = 2000*18/1000 = 36 kWh > 0 -> "500".
        oracle = make_forecast_tree_policy_v11(_if_tree(),
                                              forecast_scenario="oracle")
        misleading = np.full(HORIZON_STEPS_IF, -99999.0)
        persist_fc, _ = policy(*_args(net=2000.0), P_tot_ref_future=misleading)
        oracle_fc, _ = oracle(*_args(net=2000.0), P_tot_ref_future=misleading)
        # L'oracle suit le futur (tres deficitaire) -> beaucoup de FC ; la
        # persistance ignore ces valeurs et suit la puissance courante.
        self.assertGreater(oracle_fc[1], persist_fc[1])

    def test_gaussian_noise_is_seeded_and_reset_reproducible(self):
        policy = make_forecast_tree_policy_v11(
            _if_tree(), forecast_scenario="gaussian_iid",
            sigma_inject_kwh=80.0, noise_seed=7,
        )
        future = np.full(HORIZON_STEPS_IF, 100.0)
        first = policy(*_args(net=2000.0), P_tot_ref_future=future)
        second = policy(*_args(net=2000.0), P_tot_ref_future=future)
        policy.reset()
        replay = policy(*_args(net=2000.0), P_tot_ref_future=future)
        self.assertEqual(first, replay)      # reset rejoue la meme sequence
        # Le bruit tire des valeurs differentes d'un pas au suivant.
        self.assertNotEqual(first, second)


class ForecastGuardTests(unittest.TestCase):
    def test_surplus_uses_ely_only_and_closes_balance(self):
        policy = make_forecast_tree_policy_v11(_if_tree())
        action, _ = policy(*_args(net=-5000.0),
                           P_tot_ref_future=np.zeros(HORIZON_STEPS_IF))
        self.assertEqual(action[1], 0.0)
        self.assertLessEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), -5000.0, places=9)

    def test_fc_failure_clamps_command_to_nonpositive(self):
        policy = make_forecast_tree_policy_v11(_if_tree())
        action, _ = policy(*_args(net=2000.0, failures=["FC"]),
                           P_tot_ref_future=np.full(HORIZON_STEPS_IF, -3000.0))
        self.assertEqual(action[1], 0.0)
        self.assertAlmostEqual(sum(action), 2000.0, places=9)

    def test_ely_failure_clamps_command_to_nonnegative(self):
        policy = make_forecast_tree_policy_v11(_if_tree())
        action, _ = policy(*_args(net=-5000.0, failures=["ELY"]),
                           P_tot_ref_future=np.zeros(HORIZON_STEPS_IF))
        self.assertEqual(action[2], 0.0)
        self.assertAlmostEqual(sum(action), -5000.0, places=9)

    def test_deadband_zeroes_micro_command(self):
        # Feuille "500" (deficit modere) coupee par une large zone morte.
        policy = make_forecast_tree_policy_v11(_if_tree(), deadband_w=800.0)
        action, _ = policy(*_args(net=2000.0),
                           P_tot_ref_future=np.full(HORIZON_STEPS_IF, 3000.0))
        self.assertEqual(action[1], 0.0)
        self.assertAlmostEqual(action[0], 2000.0, places=9)


if __name__ == "__main__":
    unittest.main()
