import sys
import unittest
from pathlib import Path


V94 = Path(__file__).resolve().parents[1] / "Vieillissement9_4"
if str(V94) not in sys.path:
    sys.path.insert(0, str(V94))

from Common.get_lol import get_lol
from Common.Init_EMR_MG_v16_python import FC, ELY
from Common.simulate_transition import simulate_transition


class TestGetLolFailures(unittest.TestCase):
    def test_fc_is_reduced_not_zeroed_at_hydrogen_floor(self):
        action, lol = get_lol(
            SoC_t=0.5,
            action=(0.0, 1000.0, 0.0),
            P_tot_ref_t=1000.0,
            defaillances=[],
            E_h2_t=0.5,
            E_h2_init=200.0,
            P_fc_max_t=FC["P_fc_max"],
            P_ely_max_t=ELY["P_ely_max"],
            SoH_bat_t=1.0,
        )
        self.assertGreater(action[1], 0.0)
        self.assertLess(action[1], 1000.0)
        self.assertGreater(lol, 0.0)

    def test_fc_hydrogen_floor_action_is_forward_feasible(self):
        """Regression for the long-run failure observed at hour 110254.

        The inverse H2 curve must agree with ``simulate_transition`` rather
        than overshoot the reservoir floor between two efficiency-LUT knots.
        """
        p_fc_max = 1630.01590182
        e_h2 = 0.442009697769
        action, lol = get_lol(
            SoC_t=0.20001,
            action=(0.0, 1000.0, 0.0),
            P_tot_ref_t=2585.55555556,
            defaillances=[],
            E_h2_t=e_h2,
            E_h2_init=200.0,
            P_fc_max_t=p_fc_max,
            P_ely_max_t=ELY["P_ely_max"],
            SoH_bat_t=1.0,
        )
        soc_tp1, sim_out = simulate_transition(
            0.20001, action, 2585.55555556, 0, lol, 0.0, 0.0, 1.0,
            e_h2, 200.0, p_fc_max, ELY["P_ely_max"],
        )
        self.assertGreaterEqual(soc_tp1, 0.0)
        self.assertIn("P_bat", sim_out)
        self.assertGreaterEqual(sim_out["E_h2_tp1"], 0.0)

    def test_failed_fc_creates_lol_even_without_battery_saturation(self):
        action, lol = get_lol(
            SoC_t=0.5,
            action=(100.0, 100.0, 0.0),
            P_tot_ref_t=200.0,
            defaillances=["FC"],
            E_h2_t=100.0,
            E_h2_init=200.0,
            P_fc_max_t=1e6,
            P_ely_max_t=1e6,
            SoH_bat_t=1.0,
        )
        self.assertEqual(action[1], 0.0)
        self.assertAlmostEqual(lol, 0.5, places=12)

    def test_failed_fc_is_not_credited_when_battery_hits_soc_floor(self):
        p_ref = 200.0
        action, lol = get_lol(
            SoC_t=0.20001,
            action=(100.0, 100.0, 0.0),
            P_tot_ref_t=p_ref,
            defaillances=["FC"],
            E_h2_t=100.0,
            E_h2_init=200.0,
            P_fc_max_t=1e6,
            P_ely_max_t=1e6,
            SoH_bat_t=1.0,
        )
        self.assertEqual(action[1], 0.0)
        expected = 1.0 - action[0] / p_ref
        self.assertAlmostEqual(lol, expected, places=12)
        self.assertGreater(lol, 0.99)


if __name__ == "__main__":
    unittest.main()
