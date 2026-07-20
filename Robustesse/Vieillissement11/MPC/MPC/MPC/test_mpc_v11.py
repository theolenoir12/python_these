"""Tests courts du noyau MPC V11 ; aucune simulation longue."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Common import Init_EMR_MG_v16_python as I  # noqa: E402
from Common.degradation_v11 import (  # noqa: E402
    ELY_V11, MODEL_ID, aging_snapshot, new_ely_state, new_fc_state,
)
from Common.electrochemistry import ely_pmax, ely_power  # noqa: E402
from MPC.mpc_v11 import ETA, MPCConfig, MPCPolicyV11  # noqa: E402


class TestMPCV11(unittest.TestCase):
    def setUp(self):
        self.aging = {
            "fc": aging_snapshot("fc", new_fc_state()),
            "ely": aging_snapshot("ely", new_ely_state()),
            "dt_h": I.LOAD["Ts"] / 3600.0,
        }
        self.profile = np.array([3000.0, 4200.0, -4500.0, -8000.0, 900.0, 2400.0])

    def solve(self, policy: MPCPolicyV11):
        return policy.solve_horizon(
            self.profile, soc=0.50, h2_kwh=100.0, h2_capacity_kwh=200.0,
            soh_bat=1.0, soh_fc=1.0, soh_ely=1.0,
            alpha_fc=0.0, alpha_ely=0.0,
            p_fc_max_w=I.FC["P_fc_max"], p_ely_max_w=I.ELY["P_ely_max"],
            aging_context=self.aging,
        )

    def test_model_is_nominal_p2(self):
        self.assertEqual(MODEL_ID, "v11-doe-rakousky-mccay-colombo-2026-07-16")
        self.assertEqual(float(ELY_V11["stress_exponent"]), 2.0)

    def test_ely_piecewise_is_convex_and_contains_j_anchors(self):
        ecap = 0.999 * float(ely_pmax(0.0)) / ETA
        widths, slopes = MPCPolicyV11._ely_piecewise(0.0, ecap)
        breaks = np.r_[0.0, np.cumsum(widths)]
        self.assertTrue(np.all(widths > 0.0))
        self.assertTrue(np.all(np.diff(slopes) >= -1e-12))
        for density in (1.0, 2.0):
            anchor = float(ely_power(
                density * I.S * I.ELY["n_parallel"], 0.0)) / ETA
            self.assertLess(float(np.min(np.abs(breaks - anchor))), 1e-6)

    def test_milp_balance_modes_and_constraints(self):
        solution = self.solve(MPCPolicyV11(MPCConfig(horizon_steps=6)))
        self.assertTrue(solution["success"])
        self.assertLess(solution["constraint_residual"], 1e-6)
        self.assertLessEqual(int(np.max(solution["fc_on"] + solution["ely_on"])), 1)
        self.assertTrue(np.all(solution["shed_w"] >= -1e-8))
        self.assertTrue(np.all(solution["curtail_w"] >= -1e-8))
        self.assertLess(solution["solve_seconds"], 5.0)

    def test_no_soh_null_is_exact(self):
        base = MPCConfig(horizon_steps=6, health_mode="no_soh")
        null = replace(base, health_mode="soh", beta_fc=0.0, beta_ely=0.0)
        a = MPCPolicyV11(base)
        b = MPCPolicyV11(null)
        args = dict(
            forecast_w=self.profile, soc=0.50, h2_kwh=100.0,
            h2_capacity_kwh=200.0, soh_bat=0.82, soh_fc=0.93, soh_ely=0.94,
            alpha_fc=0.10, alpha_ely=0.10,
            p_fc_max_w=I.FC["P_fc_max"] * 0.90,
            p_ely_max_w=I.ELY["P_ely_max"] * 0.90,
            aging_context=self.aging,
        )
        sa = a.solve_horizon(**args)
        sb = b.solve_horizon(**args)
        self.assertEqual(sa["objective_eur"], sb["objective_eur"])
        for key in ("fc_w", "ely_w", "bat_discharge_w", "bat_charge_w",
                    "shed_w", "curtail_w"):
            self.assertTrue(np.array_equal(sa[key], sb[key]), key)
        self.assertEqual(sa["fc_wear_factor"], 1.0)
        self.assertEqual(sb["fc_wear_factor"], 1.0)
        self.assertEqual(sa["ely_wear_factor"], 1.0)
        self.assertEqual(sb["ely_wear_factor"], 1.0)

    def test_soh_weights_only_change_objective_weights(self):
        policy = MPCPolicyV11(MPCConfig(
            horizon_steps=6, health_mode="soh", beta_fc=1.0, beta_ely=1.0))
        solution = policy.solve_horizon(
            self.profile, 0.50, 100.0, 200.0, 0.82, 0.95, 0.95,
            0.05, 0.05, I.FC["P_fc_max"], I.ELY["P_ely_max"], self.aging)
        self.assertAlmostEqual(solution["fc_wear_factor"], 2.0)
        self.assertAlmostEqual(solution["ely_wear_factor"], 2.0)
        self.assertLess(solution["constraint_residual"], 1e-6)

    def test_noisy_forecast_is_paired_and_reproducible(self):
        common = dict(
            forecast_mode="noisy", forecast_seed=77,
            forecast_sigma_energy_kwh_18h=39.38,
            forecast_bias_energy_kwh_18h=-2.32,
        )
        p6 = MPCPolicyV11(MPCConfig(horizon_steps=6, **common))
        p24 = MPCPolicyV11(MPCConfig(horizon_steps=24, **common))
        error6 = p6._forecast_error_w(5)
        error24 = p24._forecast_error_w(23)
        self.assertTrue(np.array_equal(error6, error24[:5]))
        p6.reset()
        self.assertTrue(np.array_equal(error6, p6._forecast_error_w(5)))

        null = MPCPolicyV11(MPCConfig(
            horizon_steps=24, forecast_mode="noisy",
            forecast_sigma_energy_kwh_18h=0.0,
            forecast_bias_energy_kwh_18h=0.0,
        ))
        self.assertTrue(np.array_equal(null._forecast_error_w(23), np.zeros(23)))

    def test_noisy_forecast_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            MPCConfig(forecast_mode="unknown")
        with self.assertRaises(ValueError):
            MPCConfig(forecast_mode="noisy", forecast_error_rho=1.0)
        with self.assertRaises(ValueError):
            MPCConfig(forecast_mode="noisy", forecast_sigma_scale=-1.0)

    def test_first_executed_action_is_balanced(self):
        policy = MPCPolicyV11(MPCConfig(horizon_steps=6))
        action, lol = policy(
            0.50, self.profile[0], [], np.zeros(1), 0.0, 0.0,
            1.0, 100.0, 200.0, I.FC["P_fc_max"], I.ELY["P_ely_max"],
            np.inf, np.inf, 1.0, 1.0,
            P_tot_ref_future=self.profile, aging_context=self.aging,
        )
        delivered = float(sum(action))
        self.assertGreaterEqual(lol, 0.0)
        self.assertLessEqual(lol, 1.0 + 1e-9)
        self.assertAlmostEqual(delivered + lol * self.profile[0], self.profile[0], places=6)
        diagnostics = policy.diagnostics()
        self.assertEqual(diagnostics["failures"], 0)
        self.assertEqual(diagnostics["calls"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
