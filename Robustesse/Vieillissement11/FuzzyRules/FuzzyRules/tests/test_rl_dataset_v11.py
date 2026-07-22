"""Tests unitaires des briques du jeu d'apprentissage rule-learning V11-p=2.

Fonctions pures, sans cache enseignant : prevision d'energie nette (troncature de
queue), usure relative, zones de stratification, decoupage par blocs d'annees
contigus, poids par frequence inverse. Ces briques materialisent les regles du
protocole (PLAN_FUZZY_RULE_LEARNING_V11_P2 section 5) : jamais de decoupage
horaire aleatoire, stratification et surponderation des etats rares.
"""

import unittest

import numpy as np

from FuzzyRules.rl_dataset_v11 import (
    HORIZON_STEPS_IF,
    IF_FEATURES,
    SOH_EOL,
    STEPS_PER_YEAR,
    _normalize_features,
    _wear,
    _zone,
    build_split_mask,
    build_strata,
    inverse_frequency_weights,
    net_energy_forecast_kwh,
)


class NetEnergyForecastTests(unittest.TestCase):
    def test_cumulative_window_with_tail_truncation(self):
        # P_net constant 1000 W, horizon 2, dt=1 h -> 2 kWh sauf queue tronquee.
        e = net_energy_forecast_kwh(np.full(4, 1000.0), horizon=2)
        np.testing.assert_allclose(e, [2.0, 2.0, 2.0, 1.0])

    def test_signed_energy_follows_sign_of_power(self):
        e = net_energy_forecast_kwh(np.array([-1000.0, -1000.0]), horizon=2)
        self.assertLess(e[0], 0.0)

    def test_default_horizon_is_eighteen(self):
        self.assertEqual(HORIZON_STEPS_IF, 18)
        self.assertEqual(IF_FEATURES[-1], "E_net_h18_kwh")


class WearTests(unittest.TestCase):
    def test_fresh_component_has_zero_wear(self):
        self.assertEqual(float(_wear(1.0, SOH_EOL["bat"])), 0.0)

    def test_end_of_life_has_unit_wear(self):
        self.assertAlmostEqual(float(_wear(SOH_EOL["ely"], SOH_EOL["ely"])), 1.0)

    def test_midpoint_wear(self):
        eol = 0.7  # bat
        soh = 0.85  # a mi-chemin de la perte (1-0.85)/(1-0.7) = 0.5
        self.assertAlmostEqual(float(_wear(soh, eol)), 0.5)

    def test_wear_is_clamped_nonnegative(self):
        # SoH > 1 (impossible physiquement) ne doit pas produire d'usure < 0.
        self.assertEqual(float(_wear(1.2, 0.7)), 0.0)


class ZoneTests(unittest.TestCase):
    def test_digitize_low_mid_high(self):
        z = _zone(np.array([100.0, 1000.0, 5000.0]), (500.0, 3000.0))
        np.testing.assert_array_equal(z, [0, 1, 2])


class SplitMaskTests(unittest.TestCase):
    def test_contiguous_year_blocks_partition_without_overlap(self):
        n = 3 * STEPS_PER_YEAR
        split = build_split_mask(n, {"train": (0, 1), "val": (1, 2),
                                     "test": (2, 3)})
        self.assertEqual((split == "train").sum(), STEPS_PER_YEAR)
        self.assertEqual((split == "val").sum(), STEPS_PER_YEAR)
        self.assertEqual((split == "test").sum(), STEPS_PER_YEAR)
        # Blocs contigus : la frontiere tombe exactement sur l'annee.
        self.assertEqual(split[STEPS_PER_YEAR - 1], "train")
        self.assertEqual(split[STEPS_PER_YEAR], "val")

    def test_years_beyond_data_are_truncated(self):
        n = STEPS_PER_YEAR + 5
        split = build_split_mask(n, {"train": (0, 1), "val": (1, 20)})
        self.assertEqual((split == "train").sum(), STEPS_PER_YEAR)
        self.assertEqual((split == "val").sum(), 5)   # queue tronquee

    def test_unused_years_are_labelled_unused(self):
        n = 3 * STEPS_PER_YEAR
        split = build_split_mask(n, {"train": (0, 1)})
        self.assertEqual((split == "unused").sum(), 2 * STEPS_PER_YEAR)


class InverseFrequencyWeightTests(unittest.TestCase):
    def test_rare_strata_get_higher_weight(self):
        strata = np.array(["A", "A", "A", "B"])
        split = np.array(["train", "train", "train", "train"])
        w = inverse_frequency_weights(strata, split, "train")
        self.assertGreater(w[3], w[0])   # B (rare) > A (frequent)

    def test_weight_is_zero_outside_subset(self):
        strata = np.array(["A", "A", "B"])
        split = np.array(["train", "train", "val"])
        w = inverse_frequency_weights(strata, split, "train")
        self.assertEqual(w[2], 0.0)

    def test_uniform_strata_give_unit_weight(self):
        strata = np.array(["A", "A", "A", "A"])
        split = np.array(["train"] * 4)
        w = inverse_frequency_weights(strata, split, "train")
        np.testing.assert_allclose(w, 1.0)


class StrataTests(unittest.TestCase):
    def test_label_encodes_sign_amplitude_soc_h2_and_lol(self):
        teacher = {
            "P_net": np.array([2000.0, -4000.0]),
            "SoC": np.array([0.50, 0.10]),
            "lol_tab": np.array([0.0, 1.0]),
        }
        h2_n = np.array([0.30, 0.05])
        strata = build_strata(teacher, h2_n)
        self.assertEqual(strata[0], "def|p1|s1|h1|l0")
        self.assertEqual(strata[1], "sur|p2|s0|h0|l1")

    def test_idle_when_net_power_is_zero(self):
        teacher = {"P_net": np.array([0.0]), "SoC": np.array([0.5]),
                   "lol_tab": np.array([0.0])}
        strata = build_strata(teacher, np.array([0.5]))
        self.assertTrue(strata[0].startswith("idle|"))


class NormalizeFeaturesTests(unittest.TestCase):
    def test_soc_and_h2_are_clamped_to_unit_interval(self):
        p, soc_n, h2_n = _normalize_features(
            np.array([1000.0, 1000.0]),
            np.array([0.10, 1.10]),   # sous / au-dessus des bornes
            np.array([-10.0, 500.0]),  # sous / au-dessus des bornes
        )
        np.testing.assert_array_equal(p, [1000.0, 1000.0])
        self.assertTrue(np.all((soc_n >= 0.0) & (soc_n <= 1.0)))
        self.assertTrue(np.all((h2_n >= 0.0) & (h2_n <= 1.0)))
        self.assertEqual(soc_n[0], 0.0)
        self.assertEqual(h2_n[0], 0.0)


if __name__ == "__main__":
    unittest.main()
