"""Politique RB2 historique a deux consignes fixes de puissance H2.

La regle de gestion ne possede que deux parametres : une consigne PEMFC et une
consigne PEMWE, exprimees en fractions des puissances nominales. Les limites
appliquees ensuite sont exclusivement les contraintes physiques communes
(disponibilite du reservoir H2, defaillances et referee get_lol).
"""

from Common import Init_EMR_MG_v16_python as I
from Common.get_lol import get_lol


def make_rb2_policy(fc_setpoint, ely_setpoint):
    """Construit la RB2 statique historique a deux consignes de puissance."""

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        p_fc_set = fc_setpoint * I.FC["P_fc_max"]
        p_ely_set = ely_setpoint * I.ELY["P_ely_max"]
        dt_h = I.LOAD["Ts"] / 3600.0

        # Limites physiques du reservoir, communes a toutes les politiques.
        p_fc_h2_max = (
            max(E_h2_t, 0.0) / dt_h * I.FC["eff"] * I.CONV["eta"] * 1000
        )
        p_ely_h2_max = (
            max(E_h2_init - E_h2_t, 0.0) / dt_h
            / (I.ELY["eff"] * I.CONV["eta"]) * 1000
        )

        p_fc = 0.0
        p_ely = 0.0
        if P_tot_ref_t > 0.0:
            p_fc_available = min(p_fc_set, p_fc_h2_max)
            if P_tot_ref_t > p_fc_available:
                p_fc = p_fc_available
                p_bat = P_tot_ref_t - p_fc_available
            else:
                p_bat = P_tot_ref_t
        elif P_tot_ref_t < 0.0:
            p_ely_available = min(p_ely_set, p_ely_h2_max)
            if P_tot_ref_t < -p_ely_available:
                p_ely = -p_ely_available
                p_bat = P_tot_ref_t + p_ely_available
            else:
                p_bat = P_tot_ref_t
        else:
            p_bat = 0.0

        if "FC" in defaillances and P_tot_ref_t > 0.0:
            p_fc = 0.0
            p_bat = P_tot_ref_t
        if "ELY" in defaillances and P_tot_ref_t < 0.0:
            p_ely = 0.0
            p_bat = P_tot_ref_t

        return get_lol(
            SoC_t, (p_bat, p_fc, p_ely), P_tot_ref_t, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

    return rule
