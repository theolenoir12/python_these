import unittest

from FuzzyRules.tune_flc_i0_v11 import (
    COARSE_POINTS,
    PARAMETER_RANGES,
    generate_coarse_candidates,
    generate_local_candidates,
    nondominated,
)


def _result(candidate_id, lpsp, degradation, j3=None):
    parameters = {
        key: 0.5 * (lower + upper)
        for key, (lower, upper) in PARAMETER_RANGES.items()
    }
    return {
        "candidate_id": candidate_id,
        "parameters": parameters,
        "metrics": {
            "lpsp_pct": float(lpsp),
            "degradation_eur": float(degradation),
            "j3_eur": float(degradation if j3 is None else j3),
        },
    }


class TuningDesignTests(unittest.TestCase):
    def test_coarse_design_is_deterministic_and_has_announced_budget(self):
        first = generate_coarse_candidates()
        second = generate_coarse_candidates()
        self.assertEqual(first, second)
        self.assertEqual(len(first), COARSE_POINTS + 4)
        self.assertEqual(len({item["candidate_id"] for item in first}), len(first))

    def test_lhs_points_stay_inside_ranges(self):
        for candidate in generate_coarse_candidates():
            if candidate["stage"] != "coarse":
                continue
            for key, (lower, upper) in PARAMETER_RANGES.items():
                self.assertGreaterEqual(candidate["parameters"][key], lower)
                self.assertLessEqual(candidate["parameters"][key], upper)

    def test_nondominated_keeps_tradeoff_and_removes_dominated_point(self):
        results = [
            _result("reliable", 0.4, 3000.0),
            _result("durable", 0.8, 2000.0),
            _result("dominated", 0.9, 3100.0),
        ]
        ids = {item["candidate_id"] for item in nondominated(results)}
        self.assertEqual(ids, {"reliable", "durable"})

    def test_local_budget_is_at_most_thirty(self):
        parents = [
            _result("compromise", 0.6, 2500.0, 2700.0),
            _result("reliable", 0.4, 3000.0, 3100.0),
            _result("durable", 0.8, 2000.0, 3200.0),
        ]
        local = generate_local_candidates(parents)
        self.assertLessEqual(len(local), 30)
        self.assertEqual(len({item["candidate_id"] for item in local}), len(local))


if __name__ == "__main__":
    unittest.main()
