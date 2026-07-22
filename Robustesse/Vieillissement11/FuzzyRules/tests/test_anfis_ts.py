"""Tests du moteur ANFIS Takagi-Sugeno ordre 1 (NumPy pur), V11-p=2.

Aucune dépendance au cache ni au simulateur : le moteur est pur et déterministe.
On vérifie les appartenances, la partition de l'unité (couche de normalisation),
la récupération exacte d'une cible linéaire par moindres carrés, et la réduction
d'erreur sur une cible non linéaire.
"""

import unittest

import numpy as np

from FuzzyRules.anfis_ts import AnfisTS1, gaussian_mf


class GaussianMFTests(unittest.TestCase):
    def test_peak_at_center(self):
        self.assertAlmostEqual(gaussian_mf(2.0, 2.0, 1.0), 1.0)

    def test_symmetric_and_decreasing(self):
        self.assertAlmostEqual(gaussian_mf(1.0, 2.0, 0.5), gaussian_mf(3.0, 2.0, 0.5))
        self.assertGreater(gaussian_mf(2.4, 2.0, 0.5), gaussian_mf(2.9, 2.0, 0.5))


class StructureTests(unittest.TestCase):
    def test_grid_rule_count_is_nmf_power_ninputs(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        model = AnfisTS1.init_uniform(X, n_mf=3)
        self.assertEqual(model.n_rules, 27)
        self.assertEqual(model.spec()["n_consequent_params"], 27 * 4)

    def test_two_mf_two_inputs_gives_four_rules(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        self.assertEqual(AnfisTS1.init_uniform(X, n_mf=2).n_rules, 4)


class PartitionOfUnityTests(unittest.TestCase):
    def test_normalized_firing_sums_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        model = AnfisTS1.init_uniform(X, n_mf=3)
        wbar = model.normalized_firing(X)
        self.assertEqual(wbar.shape, (200, 27))
        np.testing.assert_allclose(wbar.sum(axis=1), 1.0, atol=1e-9)


class ConsequentFitTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.X = self.rng.normal(size=(1500, 3))

    def test_predict_before_fit_raises(self):
        model = AnfisTS1.init_uniform(self.X, n_mf=3)
        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_single_mf_recovers_linear_target_exactly(self):
        # n_mf=1 -> 1 règle -> la sortie EST une régression linéaire.
        y = 2.0 * self.X[:, 0] - 3.0 * self.X[:, 1] + 0.5 * self.X[:, 2] + 4.0
        model = AnfisTS1.init_uniform(self.X, n_mf=1).fit_consequents(
            self.X, y, ridge=1e-10)
        np.testing.assert_allclose(model.predict(self.X), y, atol=1e-3)

    def test_grid_model_represents_global_linear_target(self):
        y = self.X @ np.array([1.0, -2.0, 0.5]) + 3.0
        model = AnfisTS1.init_uniform(self.X, n_mf=3).fit_consequents(self.X, y)
        rmse = np.sqrt(np.mean((model.predict(self.X) - y) ** 2))
        self.assertLess(rmse, 1e-2)

    def test_fit_beats_constant_on_nonlinear_target(self):
        y = np.tanh(3.0 * self.X[:, 0]) + 0.5 * self.X[:, 1] ** 2
        model = AnfisTS1.init_uniform(self.X, n_mf=3).fit_consequents(self.X, y)
        rmse_model = np.sqrt(np.mean((model.predict(self.X) - y) ** 2))
        rmse_const = np.sqrt(np.mean((y - y.mean()) ** 2))
        self.assertLess(rmse_model, 0.5 * rmse_const)

    def test_constant_input_column_is_handled(self):
        X = self.X.copy()
        X[:, 2] = 5.0                       # écart-type nul -> scale forcé à 1
        y = X[:, 0] - X[:, 1]
        model = AnfisTS1.init_uniform(X, n_mf=2).fit_consequents(X, y)
        self.assertTrue(np.all(np.isfinite(model.predict(X))))


if __name__ == "__main__":
    unittest.main()
