import unittest

from FuzzyRules.tune_flc_soh_i0_v11 import (
    STRENGTH_VALUES,
    _selection,
    generate_candidates,
)


def _result(candidate_id, fc, ely, lpsp, degradation, j3):
    return {
        "candidate_id": candidate_id,
        "parameters": {
            "soh_strength_fc": float(fc),
            "soh_strength_ely": float(ely),
        },
        "metrics": {
            "lpsp_pct": float(lpsp),
            "degradation_eur": float(degradation),
            "j3_eur": float(j3),
        },
    }


class SoHTuningDesignTests(unittest.TestCase):
    def test_grid_is_exact_cartesian_product_with_one_null(self):
        candidates = generate_candidates()
        self.assertEqual(len(candidates), len(STRENGTH_VALUES) ** 2)
        pairs = {
            (
                item["parameters"]["soh_strength_fc"],
                item["parameters"]["soh_strength_ely"],
            )
            for item in candidates
        }
        self.assertEqual(
            pairs,
            {(fc, ely) for fc in STRENGTH_VALUES for ely in STRENGTH_VALUES},
        )

    def test_selection_keeps_nonnull_and_constrained_roles(self):
        parent = {
            "metrics": {
                "lpsp_pct": 0.80,
                "degradation_eur": 100.0,
                "j3_eur": 130.0,
            }
        }
        candidates = [
            _result("null", 0.0, 0.0, 0.80, 100.0, 130.0),
            _result("best", 0.025, 0.0, 0.78, 100.5, 128.0),
            _result("durable", 0.0, 0.05, 0.84, 98.0, 131.0),
            _result("unreliable", 0.4, 0.4, 1.20, 80.0, 150.0),
        ]
        selection = _selection(candidates, parent)
        self.assertEqual(selection["minimum_j3"]["candidate_id"], "best")
        self.assertEqual(selection["best_nonnull_j3"]["candidate_id"], "best")
        self.assertEqual(
            selection["reliability_under_1pct_deg"]["candidate_id"], "best"
        )
        self.assertEqual(
            selection["durability_under_0p05_lpsp"]["candidate_id"], "durable"
        )


if __name__ == "__main__":
    unittest.main()
