"""Tests du protocole de tuning MPC V11-p=2."""

from __future__ import annotations

import unittest

from benchmark_tuning_mpc_v11 import (
    BASE_PARAMETERS,
    DEFAULT_SCREEN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    MPCConfig,
    _baseline_source_label,
    _case_parameters,
    _screen_configs,
    _validation_configs,
)


class TestTuningMPCV11(unittest.TestCase):

    def test_screen_budget_and_one_factor_changes(self):
        cases = _case_parameters()
        self.assertEqual(len(cases), 13)
        self.assertEqual(cases["baseline"], BASE_PARAMETERS)
        for case, parameters in cases.items():
            differences = {
                key for key, value in parameters.items()
                if value != BASE_PARAMETERS[key]
            }
            self.assertEqual(len(differences), int(case != "baseline"), case)
        configs = _screen_configs(cases, list(DEFAULT_SCREEN_SEEDS))
        self.assertEqual(len(configs), 39)
        self.assertEqual(len({config["label"] for config in configs}), 39)

    def test_validation_budget_and_heldout_seeds(self):
        self.assertFalse(
            set(DEFAULT_SCREEN_SEEDS) & set(DEFAULT_VALIDATION_SEEDS))
        cases = _case_parameters()
        selected = [
            "baseline", "fc_dynamic_3", "fc_wear_2", "ely_wear_2",
            "combo_top2", "combo_top3",
        ]
        cases["combo_top2"] = dict(cases["baseline"])
        cases["combo_top2"].update({
            "fc_dynamic_scale": 3.0, "fc_wear_scale": 2.0})
        cases["combo_top3"] = dict(cases["combo_top2"])
        cases["combo_top3"]["ely_wear_scale"] = 2.0
        configs = _validation_configs(
            cases, selected, list(DEFAULT_VALIDATION_SEEDS))
        self.assertEqual(len(configs), 48)
        self.assertEqual(len({config["label"] for config in configs}), 48)

    def test_every_case_builds_a_valid_mpc_config(self):
        for case, parameters in _case_parameters().items():
            config = MPCConfig(**{
                key: value for key, value in parameters.items()
                if key in MPCConfig.__dataclass_fields__
            })
            self.assertEqual(config.horizon_steps, 24, case)
            self.assertEqual(config.health_mode, "no_soh", case)

    def test_baseline_cache_labels_match_forecast_benchmark(self):
        cases = _case_parameters()
        configs = _validation_configs(
            cases, ["baseline"], list(DEFAULT_VALIDATION_SEEDS))
        labels = {_baseline_source_label(config) for config in configs}
        self.assertIn("mpc_no_soh_h24_perfect", labels)
        self.assertIn("mpc_no_soh_h24_persistence", labels)
        self.assertIn("mpc_no_soh_h24_noisy_s0p5_r202604", labels)
        self.assertIn("mpc_no_soh_h24_noisy_s1p0_r202605", labels)
        self.assertIn("mpc_no_soh_h24_noisy_s1p5_r202604", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
