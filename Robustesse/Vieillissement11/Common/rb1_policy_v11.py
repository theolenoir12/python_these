"""RB1 parametree locale a Vieillissement11.

Ce module evite l'adaptateur historique ``RB1_costopt_v8_020_035`` qui charge
explicitement le noyau de Vieillissement8. Les regles sont inchangees : seuls
les deux seuils de SoC sont exposes pour une optimisation sous V11.
"""

from __future__ import annotations

from .get_lol import get_lol


def make_rb1_policy_v11(soc_low, soc_high):
    """Construit RB1 pour ``0 <= soc_low < soc_high < 1``."""
    low, high = float(soc_low), float(soc_high)
    if not 0.0 <= low < high < 1.0:
        raise ValueError("seuils RB1 invalides")

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t
        del SoH_fc_t, SoH_ely_t
        if P_tot_ref_t > 0.0:
            if SoC_t <= low:
                fraction_battery = 0.0
            elif SoC_t >= high:
                fraction_battery = 1.0
            else:
                fraction_battery = (SoC_t - low) / (high - low)
            p_bat = P_tot_ref_t * fraction_battery
            p_fc = P_tot_ref_t - p_bat
            p_ely = 0.0
        else:
            if SoC_t <= high:
                fraction_battery = 1.0
            elif SoC_t >= 1.0:
                fraction_battery = 0.0
            else:
                fraction_battery = (1.0 - SoC_t) / (1.0 - high)
            p_bat = P_tot_ref_t * fraction_battery
            p_ely = P_tot_ref_t - p_bat
            p_fc = 0.0

        if "FC" in defaillances and P_tot_ref_t > 0.0:
            p_bat, p_fc = P_tot_ref_t, 0.0
        if "ELY" in defaillances and P_tot_ref_t < 0.0:
            p_bat, p_ely = P_tot_ref_t, 0.0

        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), P_tot_ref_t, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    rule.rb1_parameters = {"soc_low": low, "soc_high": high}
    return rule
