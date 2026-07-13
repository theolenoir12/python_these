import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np


V94 = Path(__file__).resolve().parents[1] / "Vieillissement9_4"
if str(V94) not in sys.path:
    sys.path.insert(0, str(V94))

SIMULATION_DEPS = importlib.util.find_spec("scipy") is not None
if SIMULATION_DEPS:
    from Common import cost_fcn_total2 as C
    from Common.get_lol import get_lol
    from Common.main_init_and_loop import init_and_run_loop


def force_fc_strategy(
    SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
    SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
    RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    if P_tot_ref_t > 0:
        action = (0.0, P_tot_ref_t, 0.0)
    else:
        action = (P_tot_ref_t, 0.0, 0.0)
    return get_lol(
        SoC_t, action, P_tot_ref_t, defaillances, E_h2_t, E_h2_init,
        P_fc_max_t, P_ely_max_t, SoH_bat_t,
    )


def force_battery_strategy(
    SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
    SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
    RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    return get_lol(
        SoC_t, (P_tot_ref_t, 0.0, 0.0), P_tot_ref_t, defaillances,
        E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
    )


def force_hydrogen_strategy(
    SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
    SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
    RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    action = ((0.0, P_tot_ref_t, 0.0) if P_tot_ref_t > 0
              else (0.0, 0.0, P_tot_ref_t))
    return get_lol(
        SoC_t, action, P_tot_ref_t, defaillances, E_h2_t, E_h2_init,
        P_fc_max_t, P_ely_max_t, SoH_bat_t,
    )


class TestLoopReplacementBoundaries(unittest.TestCase):
    @unittest.skipUnless(SIMULATION_DEPS, "SciPy absent de l'environnement leger")
    def test_trigger_step_is_not_replayed_on_new_fc(self):
        original = dict(C.FC_REC)
        try:
            C.FC_REC.update(a_irr=1e7, b_rev=0.0, s=0.0, idle=0.0)
            data = init_and_run_loop(
                force_fc_strategy, n_years=0.005,
                replacement_accounting="corrected",
            )
            events = [event for event in data["degradation_ledger"]["events"]
                      if event["component"] == "fc"]
            self.assertGreater(len(events), 0)
            first = events[0]
            stop = first["stop_step_exclusive"]
            segment_cost = C.get_cost_fc(
                data["alpha_fc"][first["start_step"]:stop],
                data["P_fc"][first["start_step"]:stop],
            )[0]
        finally:
            C.FC_REC.clear()
            C.FC_REC.update(original)

        self.assertGreater(stop, first["start_step"])
        self.assertEqual(data["SoH_fc"][stop], 1.0)
        self.assertTrue(np.isclose(segment_cost, first["retired_eur"],
                                   rtol=1e-10, atol=1e-6))
        if len(events) > 1:
            self.assertEqual(events[1]["start_step"], stop)

    @unittest.skipUnless(SIMULATION_DEPS, "SciPy absent de l'environnement leger")
    def test_trigger_step_is_not_replayed_on_new_battery(self):
        original = C.deg_cumul2.copy()
        try:
            C.deg_cumul2 = original * 1e8
            data = init_and_run_loop(
                force_battery_strategy, n_years=0.005,
                replacement_accounting="corrected",
            )
            events = [event for event in data["degradation_ledger"]["events"]
                      if event["component"] == "bat"]
            self.assertGreater(len(events), 0)
            first = events[0]
            stop = first["stop_step_exclusive"]
            segment_cost = C.get_cost_bat(
                data["P_bat"][first["start_step"]:stop],
                data["SoC"][first["start_step"]:stop + 1],
                data["SoH_bat"][first["start_step"]:stop],
            )
        finally:
            C.deg_cumul2 = original

        self.assertEqual(data["SoH_bat"][stop], 1.0)
        self.assertTrue(np.isclose(segment_cost, first["retired_eur"],
                                   rtol=1e-10, atol=1e-6))

    @unittest.skipUnless(SIMULATION_DEPS, "SciPy absent de l'environnement leger")
    def test_trigger_step_is_not_replayed_on_new_electrolyser(self):
        original = dict(C.ELY_REC)
        try:
            C.ELY_REC.update(a2=0.0, b2=0.0, s=1e7, idle=0.0)
            data = init_and_run_loop(
                force_hydrogen_strategy, n_years=0.005,
                replacement_accounting="corrected",
            )
            events = [event for event in data["degradation_ledger"]["events"]
                      if event["component"] == "ely"]
            self.assertGreater(len(events), 0)
            first = events[0]
            stop = first["stop_step_exclusive"]
            segment_cost = C.get_cost_ely(
                data["alpha_ely"][first["start_step"]:stop],
                data["P_ely"][first["start_step"]:stop],
            )[0]
        finally:
            C.ELY_REC.clear()
            C.ELY_REC.update(original)

        self.assertEqual(data["SoH_ely"][stop], 1.0)
        self.assertTrue(np.isclose(segment_cost, first["retired_eur"],
                                   rtol=1e-10, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
