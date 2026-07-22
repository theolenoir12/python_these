"""Tests de la table et du chargeur de la synthèse Pareto (V11-p=2).

Vérifie que la table de campagne se parse, que ses familles sont connues du
style de tracé, que J3 est cohérent avec (dégradation, EENS), et que la
trajectoire DAgger est correctement reconstituée.
"""

import unittest

from FuzzyRules.plot_pareto_families_v11 import (
    DEFAULT_POINTS,
    FAMILY_STYLE,
    _dagger_path,
    load_points,
)


class ParetoPointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = load_points(DEFAULT_POINTS)

    def test_table_is_non_empty(self):
        self.assertGreaterEqual(len(self.points), 12)

    def test_all_families_have_a_style(self):
        for p in self.points:
            self.assertIn(p["famille"], FAMILY_STYLE)

    def test_j3_is_consistent_with_deg_and_eens(self):
        # j3 = C_deg[EUR] + 3*EENS[kWh] ; deg stocké en kEUR.
        for p in self.points:
            expected = p["deg"] * 1000.0 + 3.0 * p["eens"]
            self.assertAlmostEqual(p["j3"], expected, delta=0.6, msg=p["label"])

    def test_reference_and_learned_points_present(self):
        labels = {p["label"] for p in self.points}
        self.assertIn("RB1", labels)
        self.assertIn("arbre I0 d4", labels)
        self.assertIn("ANFIS I0 (mf3)", labels)

    def test_dagger_path_starts_at_tree_then_iterates(self):
        path = _dagger_path(self.points)
        self.assertEqual(path[0]["label"], "arbre I0 d4")
        self.assertEqual(len(path), 6)   # arbre I0 + 5 itérations
        self.assertTrue(all(p["augmentation"] == "dagger" for p in path[1:]))


if __name__ == "__main__":
    unittest.main()
