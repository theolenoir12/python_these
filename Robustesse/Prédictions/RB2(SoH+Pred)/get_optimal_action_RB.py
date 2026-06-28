import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *

# RB2(SoH+Pred) -- le RB2 "ultime" : base RB2(SoH) (modulation du setpoint ELY
# par le SoH, conscience-degradation) + levier PREVISION = pre-charge batterie.
#   Levier prevision : si un deficit net est prevu sur l'horizon proche H_PRE et
#   que SoC < SOC_TARGET, on coupe l'ELY -> le surplus PV courant charge la
#   batterie (rendement ~95%) au lieu de la chaine H2 (ELY 0.7 x FC 0.5, lossy)
#   -> on entre dans le creux avec une marge de SoC -> moins de passages au
#   plancher -> LPSP plus faible, sans cout de degradation.
# Methodo : la strategie ne differe de RB2(SoH) QUE par une fonction de la
# prevision et RETOMBE EXACTEMENT sur RB2(SoH) si ENABLE=False ou prevision
# neutre (test nul) -> tout gain est attribuable a la prevision.
# Cale par sweep 25 ans (cout = deg + VoLL*EENS, VoLL=3) :
#   H_PRE=18 (echelle DIURNE), SOC_TARGET=0.90  -> total 78.69 kEUR
#   (LPSP 2.317 %, deg 59.69) : -1.57 kEUR / -1.96 % vs RB2(SoH) (80.26),
#   meilleur EMS de la famille. Test nul : ENABLE=False == RB2(SoH)=80.258 pile.
ENABLE       = True       # False -> RB2(SoH) exact (test nul)
USE_FORECAST = True
H_PRE        = 18         # horizon de pre-charge [pas = h]
SOC_TARGET   = 0.90       # on ne pre-charge que tant que SoC < cible


def _precharge(P_tot_ref_future, SoC_t):
    """True si un deficit net est prevu sur H_PRE et qu'il reste de la marge
    batterie a remplir. Prevision absente/neutre -> False -> RB2(SoH)."""
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= SOC_TARGET:
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h
    return net > 0.0


def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                          alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                          P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
                          P_tot_ref_future=None):
    ######################### RULES ##########################
    # Base RB2(SoH) : modulation du setpoint ELY par le SoH (FC non modulee)
    P_fc_set  = 0.440 * FC['P_fc_max'] * SoH_fc_t ** 0
    P_ely_set = 0.320 * ELY['P_ely_max'] * SoH_ely_t ** 0.5

    # Levier PREVISION : pre-charge batterie -> couper l'ELY
    if _precharge(P_tot_ref_future, SoC_t):
        P_ely_set = 0.0

    # Plafonds H2 (identiques a RB2(SoH))
    dt_h         = LOAD['Ts'] / 3600.0
    P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * FC['eff']  * CONV['eta'] * 1000
    P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (ELY['eff'] * CONV['eta']) * 1000

    if P_tot_ref_t > 0 :
        P_fc_avail = min(P_fc_set, P_fc_h2_max)
        if P_tot_ref_t > P_fc_avail :
            P_dc_fc_t  = P_fc_avail
            P_dc_bat_t = P_tot_ref_t - P_fc_avail
        else :
            P_dc_fc_t  = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_ely_t = 0
    if P_tot_ref_t < 0 :
        P_ely_avail = min(P_ely_set, P_ely_h2_max)
        if P_tot_ref_t < - P_ely_avail :
            P_dc_ely_t = - P_ely_avail
            P_dc_bat_t = P_tot_ref_t + P_ely_avail
        else :
            P_dc_ely_t = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_fc_t  = 0
    ##################################

    if 'FC' in defaillances :
        if P_tot_ref_t > 0 :
            P_dc_bat_t = P_tot_ref_t
    if 'ELY' in defaillances :
        if P_tot_ref_t < 0 :
            P_dc_bat_t = P_tot_ref_t

    action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
    action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                          E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
    return action, lol
