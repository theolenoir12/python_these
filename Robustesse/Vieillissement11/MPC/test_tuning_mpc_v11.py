"""Tests du protocole de tuning MPC V11-p=2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_tuning_mpc_v11 import (
    BASE_PARAMETERS,
    DEFAULT_SCREEN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    MPCConfig,
    _baseline_source_label,
    _case_parameters,
    _rank_screen,
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

    def test_screen_excludes_whole_case_if_one_seed_is_invalid(self):
        cases = {
            case: parameters for case, parameters in _case_parameters().items()
            if case in {"baseline", "battery_wear_0p5", "fc_wear_2",
                        "terminal_bat_1p2", "terminal_h2_1p25"}
        }
        configs = _screen_configs(cases, list(DEFAULT_SCREEN_SEEDS))
        results = {}
        for config in configs:
            offset = {
                "baseline": 0.0,
                "battery_wear_0p5": -0.04,
                "fc_wear_2": -0.03,
                "terminal_bat_1p2": -0.02,
                "terminal_h2_1p25": -0.05,
            }[config["tuning_case"]]
            results[config["label"]] = {
                "j_voll3_keur": 3.0 + offset,
                "lpsp_pct": 0.3,
                "degradation_keur": 2.0,
            }
        invalid_label = next(
            config["label"] for config in configs
            if config["tuning_case"] == "terminal_h2_1p25"
            and config["forecast_seed"] == DEFAULT_SCREEN_SEEDS[-1]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            selection = _rank_screen(
                output, configs, results, 3,
                {invalid_label: "deficit non ferme apres LOL"},
            )
            excluded = json.loads((output / "excluded_cases.json").read_text())
        self.assertEqual(
            selection["selected_single_factor_cases"],
            ["battery_wear_0p5", "fc_wear_2", "terminal_bat_1p2"],
        )
        self.assertNotIn(
            "terminal_h2_1p25",
            {row["tuning_case"] for row in selection["ranking"]},
        )
        self.assertEqual(
            excluded["terminal_h2_1p25"][invalid_label],
            "deficit non ferme apres LOL",
        )

    def test_screen_rejects_invalid_baseline(self):
        cases = {"baseline": _case_parameters()["baseline"]}
        configs = _screen_configs(cases, list(DEFAULT_SCREEN_SEEDS))
        results = {
            config["label"]: {
                "j_voll3_keur": 3.0,
                "lpsp_pct": 0.3,
                "degradation_keur": 2.0,
            }
            for config in configs
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "baseline"):
                _rank_screen(
                    Path(tmp), configs, results, 0,
                    {configs[0]["label"]: "invalide"},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
