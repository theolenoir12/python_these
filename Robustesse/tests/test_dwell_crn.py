import sys
import unittest
from pathlib import Path

import numpy as np


V94 = Path(__file__).resolve().parents[1] / "Vieillissement9_4"
if str(V94) not in sys.path:
    sys.path.insert(0, str(V94))

import bench_dwell_ely as dwell


class TestDwellCRN(unittest.TestCase):
    def test_noise_path_is_common_across_dwell_horizons(self):
        dwell.dwell_reset(2, "noisy", seed=3026)
        reference = dwell._DW["noise"].copy()
        dwell.dwell_reset(12, "noisy", seed=3026)
        self.assertTrue(np.array_equal(reference, dwell._DW["noise"]))

    def test_different_seed_changes_noise_path(self):
        dwell.dwell_reset(4, "noisy", seed=3026)
        reference = dwell._DW["noise"][:100].copy()
        dwell.dwell_reset(4, "noisy", seed=3027)
        self.assertFalse(np.array_equal(reference, dwell._DW["noise"][:100]))


if __name__ == "__main__":
    unittest.main()
