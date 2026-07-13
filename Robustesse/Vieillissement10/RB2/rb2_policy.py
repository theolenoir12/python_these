"""Politique RB2 avec consignes economiques et secours aux bornes de SoC.

Le regime normal reste la RB2 historique a puissances fixes. Si la puissance
batterie demandee ferait sortir le SoC de [0.2, 0.995], le reliquat est
reaffecte a la chaine H2, dans la limite de plafonds de secours distincts.
Aucun SoH n'est utilise dans la decision.
"""

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol


def make_rb2_policy(fc_base, ely_base, fc_emergency, ely_emergency, soc_low_reserve=0.2, soc_high_reserve=0.995):
    """Construit une RB2 statique avec garde-fou de faisabilite SoC.

    Les quatre parametres sont des fractions des Pmax nominaux.
    """

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        dt_h = I.LOAD["Ts"] / 3600.0
        fc_set = fc_base * I.FC["P_fc_max"]
        ely_set = ely_base * I.ELY["P_ely_max"]
        fc_cap = fc_emergency * I.FC["P_fc_max"]
        ely_cap = ely_emergency * I.ELY["P_ely_max"]

        fc_h2_max = (
            max(E_h2_t, 0.0) / dt_h * I.FC["eff"] * I.CONV["eta"] * 1000
        )
        ely_h2_max = (
            max(E_h2_init - E_h2_t, 0.0) / dt_h
            / (I.ELY["eff"] * I.CONV["eta"]) * 1000
        )

        p_fc = p_ely = 0.0
        if P_tot_ref_t > 0:
            available = min(fc_set, fc_h2_max)
            if P_tot_ref_t > available:
                p_fc = available
                p_bat = P_tot_ref_t - available
            else:
                p_bat = P_tot_ref_t
        elif P_tot_ref_t < 0:
            available = min(ely_set, ely_h2_max)
            if P_tot_ref_t < -available:
                p_ely = -available
                p_bat = P_tot_ref_t + available
            else:
                p_bat = P_tot_ref_t
        else:
            p_bat = 0.0

        # Prediction exacte du SoC avec les memes conventions que get_lol.
        capacity_wh = (
            I.BAT["parallel_num"] * I.BAT["series_num"] * I.BAT["Q_bat"]
            * I.BAT["v_cell_nom"] * SoH_bat_t
        )
        p_bat_cell = p_bat / I.CONV["eta"] ** np.sign(p_bat)
        soc_next = (
            SoC_t
            - p_bat_cell * dt_h * I.BAT["eff"] ** np.sign(-p_bat_cell)
            / capacity_wh
        )

        # Secours en deficit : la PEMFC couvre uniquement ce que la batterie
        # ne peut plus fournir sans franchir SoC_min.
        if soc_next < soc_low_reserve and "FC" not in defaillances:
            p_cell_limit = (
                (SoC_t - (soc_low_reserve + 0.00001)) * capacity_wh
                / (dt_h * I.BAT["eff"] ** -1)
            )
            p_bat_limit = max(0.0, p_cell_limit * I.CONV["eta"])
            required = max(0.0, P_tot_ref_t - p_bat_limit)
            p_fc = min(required, fc_cap, fc_h2_max)
            p_ely = 0.0
            p_bat = P_tot_ref_t - p_fc

        # Secours en surplus : le PEMWE absorbe uniquement ce que la batterie
        # ne peut plus stocker sans franchir SoC_max.
        elif soc_next > soc_high_reserve and "ELY" not in defaillances:
            p_cell_limit = (
                (SoC_t - (soc_high_reserve - 0.00001)) * capacity_wh
                / (dt_h * I.BAT["eff"])
            )
            p_bat_limit = min(0.0, p_cell_limit / I.CONV["eta"])
            required = max(0.0, -(P_tot_ref_t - p_bat_limit))
            p_ely = -min(required, ely_cap, ely_h2_max)
            p_fc = 0.0
            p_bat = P_tot_ref_t - p_ely

        if "FC" in defaillances and P_tot_ref_t > 0:
            p_fc = 0.0
            p_bat = P_tot_ref_t
        if "ELY" in defaillances and P_tot_ref_t < 0:
            p_ely = 0.0
            p_bat = P_tot_ref_t

        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), P_tot_ref_t, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    return rule

