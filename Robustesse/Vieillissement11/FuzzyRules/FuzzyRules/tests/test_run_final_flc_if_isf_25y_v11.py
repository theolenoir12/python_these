import unittest

from FuzzyRules.run_final_flc_if_isf_25y_v11 import (
    IID_SEEDS,
    _decision,
    _paired_statistics,
    generate_jobs,
)


def _result(kind, seed, lpsp, degradation, j3):
    return {
        "kind": kind,
        "parameters": {"noise_seed": seed},
        "metrics": {
            "lpsp_pct": lpsp,
            "eens_kwh": lpsp * 10,
            "degradation_eur": degradation,
            "j3_eur": j3,
        },
    }


class FinalForecastDesignTests(unittest.TestCase):
    def test_preannounced_job_budget_and_pairing(self):
        jobs = generate_jobs(1.0)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(item["kind"] == "iid_if" for item in jobs), 8)
        self.assertEqual(sum(item["kind"] == "iid_isf" for item in jobs), 8)
        self.assertEqual(
            {item["parameters"]["noise_seed"] for item in jobs if item["kind"] == "iid_if"},
            set(IID_SEEDS),
        )

    def test_paired_statistics_are_isf_minus_if(self):
        results = [
            _result("iid_if", 1, 0.7, 100.0, 120.0),
            _result("iid_isf", 1, 0.6, 99.0, 118.0),
            _result("iid_if", 2, 0.8, 101.0, 122.0),
            _result("iid_isf", 2, 0.7, 100.0, 120.0),
        ]
        paired = _paired_statistics(results)
        self.assertEqual(paired["metrics"]["j3_eur"]["mean"], -2.0)
        self.assertEqual(paired["metrics"]["j3_eur"]["gains"], 2)
        self.assertEqual(paired["both_primary_axes_improved"], 2)

    def test_decision_promotes_material_if_but_not_uncertain_isf(self):
        iid_if = {"metrics": {"j3_eur": {"mean": 980.0}}}
        paired = {
            "seeds": list(range(8)),
            "both_primary_axes_improved": 4,
            "metrics": {"j3_eur": {
                "mean": -1.0, "ci95_low": -3.0, "ci95_high": 1.0,
            }},
        }
        parent = {"metrics": {"j3_eur": 1000.0}}
        decision = _decision(iid_if, paired, parent)
        self.assertEqual(decision["if_decision"], "promote_if")
        self.assertEqual(decision["isf_decision"], "retain_if_over_isf")
        self.assertEqual(decision["selected_final_information_set"], "IF")


if __name__ == "__main__":
    unittest.main()
