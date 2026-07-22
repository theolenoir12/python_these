"""Tests du pipeline DAgger (direction B), V11-p=2.

Parties pures et testables sans cache PD ni simulateur : logger d'etats visites,
extraction de features (memes normalisations que le jeu d'apprentissage),
re-etiquetage plus-proche-voisin contre une grille de politique synthetique,
agregation. Le calcul de la vraie grille PD et la boucle fermee sont dans le
runner mesocentre et ne sont pas couverts ici.
"""

import unittest

import numpy as np

from FuzzyRules.rl_dataset_v11 import _normalize_features
from FuzzyRules.rl_dagger_v11 import (
    DiscipleStateLogger,
    build_i0_features,
    dagger_aggregate,
    onoff_from_action,
    relabel_with_policy_grid,
    _nearest_index,
)


class _StubDisciple:
    """Disciple jouet : renvoie une action fixee selon le signe de P_net."""

    def __init__(self):
        self.calls = 0
        self.was_reset = False

    def reset(self):
        self.was_reset = True

    def __call__(self, SoC_t, P_tot_ref_t, *rest, **kwargs):
        self.calls += 1
        if P_tot_ref_t >= 0:               # deficit -> FC
            action = (P_tot_ref_t - 1000.0, 1000.0, 0.0)
        else:                              # surplus -> ELY
            action = (P_tot_ref_t + 3000.0, 0.0, -3000.0)
        return action, 0.0


def _args(soc, net, e_h2=100.0, e_h2_init=200.0):
    return (soc, net, [], np.zeros(1), 0.0, 0.0, 1.0, e_h2, e_h2_init,
            1753.0, 18430.0, float("inf"), float("inf"), 1.0, 1.0)


class OnOffTests(unittest.TestCase):
    def test_fc_on_when_fc_power_positive(self):
        self.assertEqual(onoff_from_action((0.0, 500.0, 0.0)), (1, 0))

    def test_ely_on_when_ely_power_negative(self):
        self.assertEqual(onoff_from_action((0.0, 0.0, -2000.0)), (0, 1))

    def test_both_off_when_micro_powers(self):
        self.assertEqual(onoff_from_action((1000.0, 0.5, -0.5)), (0, 0))


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.stub = _StubDisciple()
        self.logger = DiscipleStateLogger(self.stub)

    def test_delegates_and_returns_disciple_action(self):
        out = self.logger(*_args(0.6, 2000.0))
        self.assertEqual(out, ((1000.0, 1000.0, 0.0), 0.0))
        self.assertEqual(self.stub.calls, 1)

    def test_logs_states_and_derived_onoff(self):
        self.logger(*_args(0.6, 2000.0))     # deficit -> fc_on
        self.logger(*_args(0.8, -5000.0))    # surplus -> ely_on
        v = self.logger.visited()
        np.testing.assert_array_equal(v["P_net"], [2000.0, -5000.0])
        np.testing.assert_array_equal(v["SoC"], [0.6, 0.8])
        np.testing.assert_array_equal(v["fc_on"], [1, 0])
        np.testing.assert_array_equal(v["ely_on"], [0, 1])
        np.testing.assert_array_equal(v["t"], [0, 1])

    def test_reset_clears_log_and_resets_disciple(self):
        self.logger(*_args(0.6, 2000.0))
        self.logger.reset()
        self.assertTrue(self.stub.was_reset)
        self.assertEqual(len(self.logger.visited()["t"]), 0)


class FeatureTests(unittest.TestCase):
    def test_features_match_dataset_normalization(self):
        visited = {
            "P_net": np.array([1500.0, -800.0]),
            "SoC": np.array([0.10, 0.90]),     # sous borne / interieur
            "E_h2": np.array([250.0, 50.0]),   # au-dessus reservoir / interieur
        }
        X = build_i0_features(visited)
        P, s, h = _normalize_features(visited["P_net"], visited["SoC"],
                                      visited["E_h2"])
        np.testing.assert_allclose(X[:, 0], P)
        np.testing.assert_allclose(X[:, 1], s)
        np.testing.assert_allclose(X[:, 2], h)
        self.assertEqual(X[0, 1], 0.0)   # SoC clampe
        self.assertEqual(X[0, 2], 1.0)   # H2 clampe


class NearestIndexTests(unittest.TestCase):
    def test_picks_closest_node(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        idx = _nearest_index(grid, np.array([-0.4, 0.4, 0.6, 2.9, 5.0]))
        np.testing.assert_array_equal(idx, [0, 0, 1, 3, 3])


class RelabelTests(unittest.TestCase):
    def setUp(self):
        self.Ns = self.Nh = 5
        self.soc_grid = np.linspace(0.2, 0.995, self.Ns)
        self.h2_grid = np.linspace(0.0, 200.0, self.Nh)
        self.u = np.array([-3000.0, 0.0, 1000.0])   # ELY, idle, FC

    def test_lookup_returns_grid_control(self):
        T = 3
        grid = np.ones((T, self.Ns, self.Nh, 2, 2), dtype=int)  # partout idle
        # Place un controle FC (idx 2) sur une cellule precise et l'ELY (idx 0)
        # sur une autre, pour deux etats visites bien identifies.
        i0 = _nearest_index(self.soc_grid, np.array([0.6]))[0]
        j0 = _nearest_index(self.h2_grid, np.array([100.0]))[0]
        grid[0, i0, j0, 1, 0] = 2          # t=0, fc_on=1, ely_on=0 -> FC
        i1 = _nearest_index(self.soc_grid, np.array([0.95]))[0]
        j1 = _nearest_index(self.h2_grid, np.array([20.0]))[0]
        grid[1, i1, j1, 0, 1] = 0          # t=1, fc_on=0, ely_on=1 -> ELY
        visited = {
            "SoC": np.array([0.6, 0.95]),
            "E_h2": np.array([100.0, 20.0]),
            "fc_on": np.array([1, 0]),
            "ely_on": np.array([0, 1]),
            "t": np.array([0, 1]),
        }
        y = relabel_with_policy_grid(visited, grid, self.u, self.soc_grid,
                                     self.h2_grid)
        np.testing.assert_allclose(y, [1000.0, -3000.0])

    def test_time_index_is_clamped_to_horizon(self):
        T = 2
        grid = np.full((T, self.Ns, self.Nh, 2, 2), 2, dtype=int)  # FC partout
        visited = {
            "SoC": np.array([0.6]), "E_h2": np.array([100.0]),
            "fc_on": np.array([0]), "ely_on": np.array([0]),
            "t": np.array([9]),    # au-dela de T -> ramene au dernier pas
        }
        y = relabel_with_policy_grid(visited, grid, self.u, self.soc_grid,
                                     self.h2_grid)
        np.testing.assert_allclose(y, [1000.0])


class AggregateTests(unittest.TestCase):
    def test_union_shapes_and_weights(self):
        base_X = np.zeros((3, 3))
        base_y = np.array([1.0, 2.0, 3.0])
        b1 = (np.ones((2, 3)), np.array([4.0, 5.0]))
        b2 = (2 * np.ones((1, 3)), np.array([6.0]))
        X, y, w = dagger_aggregate(base_X, base_y, [b1, b2],
                                   base_weight=1.0, visited_weight=2.0)
        self.assertEqual(X.shape, (6, 3))
        np.testing.assert_allclose(y, [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(w, [1, 1, 1, 2, 2, 2])

    def test_no_batches_returns_base(self):
        base_X = np.ones((2, 3))
        base_y = np.array([1.0, 2.0])
        X, y, w = dagger_aggregate(base_X, base_y, [])
        self.assertEqual(X.shape, (2, 3))
        np.testing.assert_allclose(w, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
