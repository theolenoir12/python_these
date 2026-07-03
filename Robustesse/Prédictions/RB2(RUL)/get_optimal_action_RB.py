import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *

# RB2(RUL) : analogue de RB2(SoH) ou la modulation du setpoint ELY utilise le RUL
# (RUL = extrapolation lineaire du SoH jusqu'a l'EoL, cf Common/get_rul) au lieu du
# SoH instantane. Le facteur remplace SoH_ely**p par (min(RUL_ely/RUL_REF, 1))**p.
#   - etat neuf / historique trop court : la boucle renvoie RUL = REF par defaut
#     -> facteur = 1 -> setpoint nominal (identique a RB2 neuf) ;
#   - quand la degradation s'accelere, le RUL extrapole chute sous REF -> setpoint
#     ELY reduit -> degradation ralentie -> duree de vie prolongee.
# Parametres cales par sweep 25 ans (cout = deg + VoLL*EENS, VoLL=3) :
#   RUL_ELY_REF = 1000 j, EXP_ELY = 0.1  -> total 81.05 kEUR
#   (LPSP 2.58 %, deg 59.92) : -5.3 % vs RB2 nu (85.55).
# La FC n'est pas modulee (EXP_FC = 0), comme dans RB2(SoH).
RUL_ELY_REF = 8000.0   # [jours] normalisation du RUL ELY
EXP_ELY     = 0.05      # exposant de modulation ELY


def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                          alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                          P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
    ######################### RULES ##########################
    # Modulation du setpoint ELY par le RUL normalise (borne a 1, comme le SoH)
    f_ely     = min(max(RUL_ely_t, 0.0) / RUL_ELY_REF, 1.0) ** EXP_ELY
    P_fc_set  = 0.440 * FC['P_fc_max']
    P_ely_set = 0.310 * ELY['P_ely_max'] * f_ely

    # Plafonds imposes par l'etat du reservoir H2 sur ce pas de temps
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
