import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t) :      
        
    # P_fc_max_t  = 20479.6126*(SoH_fc_t-0.9)**3 + 9735.2606*(SoH_fc_t-0.9)**2 + 1850.8154*(SoH_fc_t-0.9) + 1197.0693
    # P_ely_max_t = 1080197.5679*(SoH_ely_t-0.9)**3 + 178892.0120*(SoH_ely_t-0.9)**2 + 10227.1595*(SoH_ely_t-0.9) + 12145.1654
    ######################### RULES ##########################
    # Setpoints nominaux
    P_fc_set  = 0.440 * FC['P_fc_max'] * SoH_fc_t ** 0
    P_ely_set = 0.320 * ELY['P_ely_max'] * SoH_ely_t ** 0.5

    # Plafonds imposés par l'état du réservoir H2 sur ce pas de temps :
    #  - la FC ne peut sortir que l'H2 disponible        (E_h2_t)
    #  - l'ELY ne peut stocker que la place restante      (E_h2_init - E_h2_t)
    # Conversion énergie réservoir [kWh] -> puissance élec DC soutenable [W].
    # On garde le rendement nominal et eta du convertisseur pour rester
    # (légèrement) conservatif : la batterie couvre alors tout le reste.
    dt_h         = LOAD['Ts'] / 3600.0
    P_fc_h2_max  = max(E_h2_t, 0.0)               / dt_h * FC['eff']  * CONV['eta'] * 1000   # [W]
    P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0)   / dt_h / (ELY['eff'] * CONV['eta']) * 1000 # [W]

    if P_tot_ref_t > 0 :
        # Déficit : la FC est plafonnée par son setpoint ET par l'H2 disponible.
        # Si le réservoir est (presque) vide -> P_fc_avail ~ 0 -> la batterie prend tout.
        P_fc_avail = min(P_fc_set, P_fc_h2_max)
        if P_tot_ref_t > P_fc_avail :
            P_dc_fc_t  = P_fc_avail
            P_dc_bat_t = P_tot_ref_t - P_fc_avail
        else :
            P_dc_fc_t  = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_ely_t = 0
    if P_tot_ref_t < 0 :
        # Surplus : l'ELY est plafonné par son setpoint ET par la place restante.
        # Si le réservoir est (presque) plein -> P_ely_avail ~ 0 -> la batterie absorbe tout.
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

    action, lol = get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t)
    
    return action, lol