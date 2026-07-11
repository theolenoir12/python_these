import importlib.util
import sys
import unittest
from pathlib import Path


ROBUSTESSE = Path(__file__).resolve().parents[1]
V8 = ROBUSTESSE / "Vieillissement8"
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import rb1_variants as core


class TestRB1Variants(unittest.TestCase):
    EXPECTED = {
        "rb1_hist_020_060": (0.20, 0.60),
        "rb1_failopt_040_075": (0.40, 0.75),
        "rb1_costopt_v8_020_035": (0.20, 0.35),
    }

    def test_registry_exact(self):
        self.assertEqual(set(core.VARIANTS), set(self.EXPECTED))
        for name, (low, high) in self.EXPECTED.items():
            self.assertEqual(core.VARIANTS[name]["soc_low"], low)
            self.assertEqual(core.VARIANTS[name]["soc_high"], high)

    def test_wrappers_match_registry(self):
        for variant_id, (low, high) in self.EXPECTED.items():
            path = V8 / ("RB1_" + variant_id.removeprefix("rb1_")) / "get_optimal_action_RB.py"
            spec = importlib.util.spec_from_file_location("test_" + variant_id, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.VARIANT_ID, variant_id)
            self.assertEqual(module.SOC_LOW, low)
            self.assertEqual(module.SOC_HIGH, high)

    def test_branch_values_and_boundaries(self):
        low, high = 0.20, 0.60
        # Deficit : FC seule sous le seuil, melange lineaire dans la bande,
        # batterie seule au-dessus.
        self.assertEqual(core.raw_rb1_action(low, high, low, 100.0), (0.0, 100.0, 0.0))
        self.assertEqual(core.raw_rb1_action(low, high, high, 100.0), (100.0, 0.0, 0.0))
        for actual, expected in zip(
            core.raw_rb1_action(low, high, 0.40, 100.0), (50.0, 50.0, 0.0)
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        # Surplus : batterie seule jusqu'au seuil haut, puis melange vers ELY.
        self.assertEqual(core.raw_rb1_action(low, high, 0.40, -100.0), (-100.0, 0.0, 0.0))
        for actual, expected in zip(
            core.raw_rb1_action(low, high, 0.80, -100.0), (-50.0, 0.0, -50.0)
        ):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_failures_preserve_legacy_pre_referee_semantics(self):
        action = core.raw_rb1_action(0.40, 0.75, 0.50, 100.0, failures=("FC",))
        self.assertEqual(action[0], 100.0)
        self.assertGreater(action[1], 0.0)  # le referee annule ensuite la FC
        action = core.raw_rb1_action(0.40, 0.75, 0.90, -100.0, failures=("ELY",))
        self.assertEqual(action[0], -100.0)
        self.assertLess(action[2], 0.0)

    def test_invalid_thresholds(self):
        for low, high in ((0, 0.5), (0.5, 0.5), (0.8, 0.2), (0.2, 1.0)):
            with self.assertRaises(ValueError):
                core.raw_rb1_action(low, high, 0.5, 1.0)

    def test_v94_costopt_adapter_is_explicit(self):
        path = (ROBUSTESSE / "Vieillissement9_4" / "RB1_costopt_v8_020_035"
                / "get_optimal_action_RB.py")
        spec = importlib.util.spec_from_file_location("test_v94_costopt", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.VARIANT_ID, "rb1_costopt_v8_020_035")
        self.assertEqual((module.SOC_LOW, module.SOC_HIGH), (0.20, 0.35))


if __name__ == "__main__":
    unittest.main()
