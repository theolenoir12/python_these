import unittest

import numpy as np

from Robustesse.reproducibility import postprocess_corrected_p1_p3 as post


class TestCorrectedPostprocess(unittest.TestCase):
    def test_numeric_closures_accept_exact_decomposition(self):
        p1 = {
            "data": {
                "RB2": {
                    "lpsp": np.array([1.0]),
                    "deg": np.array([2.0]),
                    "eens": np.array([1000.0]),
                    "uni": np.array([5.0]),
                }
            }
        }
        p3 = {
            "T6m": {
                "data": {
                    "instant": {
                        "lpsp": np.array([1.0]),
                        "deg": np.array([2.0]),
                        "eens": np.array([1000.0]),
                        "uni0": np.array([5.0]),
                        "nint": np.array([1.0]),
                        "nprev": np.array([0.0]),
                        "waste": np.array([0.6]),
                        "wbat": np.array([0.1]),
                        "wfc": np.array([0.2]),
                        "wely": np.array([0.3]),
                        "outfc": np.array([0.0]),
                        "outely": np.array([0.0]),
                    }
                }
            }
        }
        post.verify_numeric_closures(p1, p3)

    def test_numeric_closures_reject_waste_mismatch(self):
        p1 = {"data": {}}
        p3 = {
            "T6m": {
                "data": {
                    "rul": {
                        "deg": np.array([2.0]), "eens": np.array([0.0]),
                        "uni0": np.array([2.0]), "waste": np.array([1.0]),
                        "wbat": np.array([0.0]), "wfc": np.array([0.0]),
                        "wely": np.array([0.0]),
                    }
                }
            }
        }
        with self.assertRaises(RuntimeError):
            post.verify_numeric_closures(p1, p3)

    def test_protocol_parameters_are_exact(self):
        dataset = {
            "provenance": {
                "experiment_id": "p3_rul_maintenance",
                "parameters": {"horizon_years": 25, "rul_margin": 1.0},
            }
        }
        post.require_protocol(
            dataset, "p3_rul_maintenance",
            {"horizon_years": 25, "rul_margin": 1.0},
        )
        with self.assertRaises(ValueError):
            post.require_protocol(
                dataset, "p3_rul_maintenance",
                {"horizon_years": 25, "rul_margin": 1.5},
            )


if __name__ == "__main__":
    unittest.main()
