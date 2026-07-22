import unittest

from FuzzyRules.tune_flc_forecast_i0_v11 import (
    NOISE_SEEDS,
    STRENGTH_VALUES,
    _aggregate_noise,
    _selection,
    generate_jobs,
)


def _noise_result(strength, seed, lpsp, degradation, j3):
    return {
        "candidate_id": f"s{strength}_{seed}",
        "parameters": {"forecast_strength": strength, "noise_seed": seed},
        "metrics": {
            "lpsp_pct": lpsp,
            "eens_kwh": lpsp * 10.0,
            "degradation_eur": degradation,
            "j3_eur": j3,
        },
    }


class ForecastTuningDesignTests(unittest.TestCase):
    def test_job_budget_is_preannounced(self):
        jobs = generate_jobs()
        self.assertEqual(len(jobs), 22)
        self.assertEqual(sum(item["kind"] == "parent" for item in jobs), 1)
        self.assertEqual(sum(item["kind"] == "oracle" for item in jobs), 5)
        self.assertEqual(sum(item["kind"] == "gaussian_iid" for item in jobs), 16)

    def test_noise_grid_is_active_strengths_by_common_seeds(self):
        jobs = [item for item in generate_jobs() if item["kind"] == "gaussian_iid"]
        pairs = {
            (item["parameters"]["forecast_strength"], item["parameters"]["noise_seed"])
            for item in jobs
        }
        self.assertEqual(
            pairs,
            {(strength, seed) for strength in STRENGTH_VALUES[1:] for seed in NOISE_SEEDS},
        )

    def test_aggregate_and_selection_use_configuration_means(self):
        results = [
            _noise_result(0.25, 1, 0.7, 100.0, 120.0),
            _noise_result(0.25, 2, 0.7, 100.0, 122.0),
            _noise_result(0.50, 1, 0.6, 100.5, 119.0),
            _noise_result(0.50, 2, 0.6, 100.5, 119.0),
        ]
        aggregate = _aggregate_noise(results)
        selection = _selection(aggregate, {
            "metrics": {"lpsp_pct": 0.8, "degradation_eur": 100.0}
        })
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(
            selection["best_active_mean_j3"]["forecast_strength"], 0.50
        )


if __name__ == "__main__":
    unittest.main()
