import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROBUSTESSE = Path(__file__).resolve().parents[1]
DEFAILLANCES = ROBUSTESSE / "Defaillances"
SIMULATION_DEPS = importlib.util.find_spec("scipy") is not None
if SIMULATION_DEPS:
    if str(DEFAILLANCES) not in sys.path:
        sys.path.insert(0, str(DEFAILLANCES))
    import rb1_pred_common as pred
    import robustesse_common as rc


class TestRB1PredNull(unittest.TestCase):
    @unittest.skipUnless(SIMULATION_DEPS, "SciPy absent de l'environnement leger")
    def test_disabled_prediction_equals_frozen_failopt(self):
        augmented = pred.make_rb1_pred(
            *rc.RB1_FAILOPT_THRESHOLDS, enable=False, noise=False, hyst=False
        )
        frozen = rc.load_strategy("rb1_failopt_040_075")
        base_args = dict(
            lol_tab=np.zeros(1), alpha_fc_t=0.0, alpha_ely_t=0.0,
            SoH_bat_t=1.0, E_h2_t=100.0, E_h2_init=rc.E_H2_INIT,
            P_fc_max_t=rc.I.FC["P_fc_max"], P_ely_max_t=rc.I.ELY["P_ely_max"],
            RUL_fc_t=rc.RUL_FC_DEFAULT, RUL_ely_t=rc.RUL_ELY_DEFAULT,
            SoH_fc_t=1.0, SoH_ely_t=1.0,
        )
        for soc in (0.2, 0.4, 0.6, 0.75, 0.9):
            for power in (-500.0, 500.0):
                for failures in ([], ["FC"], ["ELY"]):
                    args = (
                        soc, power, failures, base_args["lol_tab"],
                        base_args["alpha_fc_t"], base_args["alpha_ely_t"],
                        base_args["SoH_bat_t"], base_args["E_h2_t"],
                        base_args["E_h2_init"], base_args["P_fc_max_t"],
                        base_args["P_ely_max_t"], base_args["RUL_fc_t"],
                        base_args["RUL_ely_t"], base_args["SoH_fc_t"],
                        base_args["SoH_ely_t"],
                    )
                    expected = frozen(*args)
                    actual = augmented(*args, np.array([100.0, -100.0]))
                    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
                    self.assertEqual(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
