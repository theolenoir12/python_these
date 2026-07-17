"""Adaptateur V9_4 de la variante immuable ``rb1_costopt_v8_020_035``."""

import importlib.util
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[2] / "Vieillissement8" / "rb1_variants.py"
_SPEC = importlib.util.spec_from_file_location("_genial_rb1_variants_v8", _CORE_PATH)
_CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CORE)

VARIANT_ID = "rb1_costopt_v8_020_035"
SOC_LOW = 0.20
SOC_HIGH = 0.35


def get_optimal_action_RB(
    SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
    SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
    RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    return _CORE.run_rb1(
        SOC_LOW, SOC_HIGH, SoC_t, P_tot_ref_t, defaillances, lol_tab,
        alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init,
        P_fc_max_t, P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    )
