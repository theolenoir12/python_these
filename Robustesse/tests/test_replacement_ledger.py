import sys
import unittest
from pathlib import Path


V94 = Path(__file__).resolve().parents[1] / "Vieillissement9_4"
if str(V94) not in sys.path:
    sys.path.insert(0, str(V94))

from Common.replacement_ledger import ReplacementLedger


class TestReplacementLedger(unittest.TestCase):
    def test_replacement_boundaries_are_disjoint(self):
        ledger = ReplacementLedger()
        first = ledger.retire("fc", 100.0, 7, "instant_eol", 0.89)
        second = ledger.retire("fc", 80.0, 11, "instant_eol", 0.89)
        self.assertEqual((first["start_step"], first["stop_step_exclusive"]), (0, 7))
        self.assertEqual((second["start_step"], second["stop_step_exclusive"]), (7, 11))
        snap = ledger.snapshot({"bat": 4.0, "fc": 3.0, "ely": 5.0}, 14)
        self.assertEqual(snap["current_start_step"]["fc"], 11)
        self.assertEqual(snap["retired_eur"]["fc"], 180.0)
        self.assertEqual(snap["total_eur"]["fc"], 183.0)

    def test_visit_replacement_starts_new_unit_on_visit_step(self):
        ledger = ReplacementLedger()
        event = ledger.retire("ely", 12.5, 20, "maintenance_prev", 0.94)
        self.assertEqual(event["stop_step_exclusive"], 20)
        snap = ledger.snapshot({"bat": 0.0, "fc": 0.0, "ely": 1.5}, 21)
        self.assertEqual(snap["current_start_step"]["ely"], 20)
        self.assertEqual(snap["total_eur"]["ely"], 14.0)

    def test_invalid_interval_is_rejected(self):
        ledger = ReplacementLedger()
        ledger.retire("bat", 1.0, 4, "test", 0.8)
        with self.assertRaises(ValueError):
            ledger.retire("bat", 1.0, 3, "test", 0.8)


if __name__ == "__main__":
    unittest.main()
