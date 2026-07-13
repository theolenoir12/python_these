import sys
import unittest
from pathlib import Path

import numpy as np


ROBUSTESSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBUSTESSE))

from reproducibility.paired_stats import (
    bootstrap_mean_ci,
    cvar_high,
    exact_sign_test,
    summarize_difference,
)


class TestPairedStats(unittest.TestCase):
    def test_exact_sign_test_excludes_ties(self):
        result = exact_sign_test([-2, -1, 0, 1])
        self.assertEqual((result["negative"], result["ties"], result["positive"]), (2, 1, 1))
        self.assertAlmostEqual(result["pvalue"], 1.0)

    def test_es90_uses_exact_tail_size(self):
        x = np.arange(200.0)
        self.assertAlmostEqual(cvar_high(x, 0.90), np.arange(180.0, 200.0).mean())

    def test_bootstrap_is_deterministic(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
        self.assertEqual(
            bootstrap_mean_ci(x, n_resamples=2000, seed=7),
            bootstrap_mean_ci(x, n_resamples=2000, seed=7),
        )

    def test_summary_counts(self):
        result = summarize_difference(np.array([-2.0, -1.0, 0.0, 3.0]))
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (2, 1, 1))
        self.assertAlmostEqual(result["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
