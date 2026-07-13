"""Variante immuable RB1 issue de l'optimisation nominale V8."""

from rb1_variants import run_rb1

VARIANT_ID = "rb1_costopt_v8_020_035"
SOC_LOW = 0.20
SOC_HIGH = 0.35


def get_optimal_action_RB(
    SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
    SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
    RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    return run_rb1(
        SOC_LOW, SOC_HIGH, SoC_t, P_tot_ref_t, defaillances, lol_tab,
        alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init,
        P_fc_max_t, P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    )
