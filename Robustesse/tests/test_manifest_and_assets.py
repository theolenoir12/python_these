import hashlib
import sys
import unittest
from pathlib import Path


ROBUSTESSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBUSTESSE))

from reproducibility.validate_manifest import validate


class TestManifestAndAssets(unittest.TestCase):
    def test_manifest(self):
        errors = validate(ROBUSTESSE / "EXPERIMENTS_MANIFEST.json")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_v9_4_efficiency_tables_are_present_and_canonical(self):
        expected = {
            "FC_efficiency_LU_table_power.csv": "ebf6024f7526ba5ff1f25475e99ce4d729594eb9cae5b8f6284ebe0927684475",
            "ELY_efficiency_LU_table_power.csv": "5d21bf95314c14dcfbfabaf5215356dec40c1d6e0787983a3d1924676a7634d7",
        }
        common = ROBUSTESSE / "Vieillissement9_4" / "Common"
        for name, digest in expected.items():
            actual = hashlib.sha256((common / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
