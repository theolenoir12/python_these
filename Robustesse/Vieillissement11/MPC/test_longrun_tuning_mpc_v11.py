"""Tests du rejeu long du tuning MPC V11-p=2."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_longrun_tuning_mpc_v11 import _compare, _longrun_configs
from benchmark_tuning_mpc_v11 import BASE_PARAMETERS


class TestLongrunTuningMPCV11(unittest.TestCase):

    def test_configs_are_paired_and_only_change_retained_parameters(self):
        tuned = dict(BASE_PARAMETERS)
        tuned.update({
            "battery_wear_scale": 0.5,
            "terminal_bat_eur_per_kwh": 1.2,
            "fc_wear_scale": 2.0,
        })
        cases = {"baseline": dict(BASE_PARAMETERS), "combo_top3": tuned}
        configs = _longrun_configs(cases, "combo_top3", [202604, 202605])
        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {config["forecast_seed"] for config in configs},
            {202604, 202605},
        )
        self.assertTrue(all(config["forecast_mode"] == "noisy"
                            for config in configs))
        self.assertTrue(all(config["forecast_sigma_scale"] == 1.0
                            for config in configs))

    def test_comparison_is_paired_and_uses_ratio_of_means(self):
        cases = {"baseline": dict(BASE_PARAMETERS),
                 "combo_top3": dict(BASE_PARAMETERS)}
        configs = _longrun_configs(cases, "combo_top3", [1, 2])
        results = {}
        for config in configs:
            baseline = config["tuning_case"] == "baseline"
            seed = config["forecast_seed"]
            j3 = float(seed * (10.0 if baseline else 9.0))
            results[config["label"]] = {
                "j_voll3_keur": j3,
                "lpsp_pct": 1.0 if baseline else 0.9,
                "degradation_keur": 2.0 if baseline else 1.9,
                "eens_kwh": 100.0 if baseline else 90.0,
            }
        with tempfile.TemporaryDirectory() as tmp:
            comparison = _compare(
                Path(tmp), configs, results, "combo_top3", 1.0)
        self.assertAlmostEqual(comparison["aggregate_gain_j3_pct"], 10.0)
        self.assertEqual(comparison["tuned_wins"], 2)
        self.assertTrue(comparison["meets_inherited_materiality_threshold"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
