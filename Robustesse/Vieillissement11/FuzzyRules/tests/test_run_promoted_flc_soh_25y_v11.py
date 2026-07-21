import unittest

from FuzzyRules.run_promoted_flc_soh_25y_v11 import (
    _decision,
    _deltas,
    _null_candidate,
    _selection_candidates,
)


class SoHPromotionDesignTests(unittest.TestCase):
    def test_selection_deduplicates_roles(self):
        selection = {
            "minimum_j3": {
                "candidate_id": "a", "parameters": {"soh_strength_fc": 0.0}
            },
            "best_nonnull_j3": {
                "candidate_id": "a", "parameters": {"soh_strength_fc": 0.0}
            },
            "reliability": {
                "candidate_id": "b", "parameters": {"soh_strength_fc": 0.1}
            },
        }
        candidates = _selection_candidates(selection)
        self.assertEqual([item["candidate_id"] for item in candidates], ["a", "b"])
        self.assertEqual(
            candidates[0]["roles"], ["minimum_j3", "best_nonnull_j3"]
        )

    def test_null_candidate_is_exactly_inactive(self):
        candidate = _null_candidate()
        self.assertEqual(candidate["parameters"], {
            "soh_strength_fc": 0.0,
            "soh_strength_ely": 0.0,
        })
        self.assertEqual(candidate["kind"], "null_control")

    def test_deltas_and_dominance(self):
        parent = {
            "lpsp_pct": 1.0, "eens_kwh": 100.0,
            "degradation_eur": 1000.0, "j3_eur": 1300.0,
        }
        child = {
            "lpsp_pct": 0.9, "eens_kwh": 90.0,
            "degradation_eur": 990.0, "j3_eur": 1260.0,
        }
        deltas = _deltas(child, parent)
        self.assertAlmostEqual(deltas["lpsp_pct_absolute"], -0.1)
        self.assertAlmostEqual(deltas["j3_eur_relative_pct"], -40 / 13)
        self.assertTrue(deltas["dominates_parent"])
        self.assertFalse(deltas["dominated_by_parent"])

    def test_decision_retains_parent_when_all_active_points_are_dominated(self):
        active = [{
            "candidate_id": "active",
            "metrics": {"j3_eur": 100.0},
            "deltas_vs_parent": {
                "dominated_by_parent": True,
                "j3_eur_relative_pct": 0.1,
            },
        }]
        decision = _decision(active)
        self.assertEqual(decision["status"], "retain_parent_i0")
        self.assertIsNone(decision["active_candidate_promoted"])


if __name__ == "__main__":
    unittest.main()
