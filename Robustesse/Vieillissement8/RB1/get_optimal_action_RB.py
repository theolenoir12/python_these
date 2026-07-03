import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t) :      
        
    ######################### RULES ##########################
    # Seuils de SoC des deux genoux de la bande de mélange batterie/H2.
    # OPTIMISÉS pour la robustesse sous défaillance (sweep_rb1.py, juillet 2026) :
    #   SOC_LOW = 0.40, SOC_HIGH = 0.75   (ancien réglage : 0.20 / 0.60).
    # Les pentes ci-dessous en découlent : 1/(SOC_HIGH-SOC_LOW) en décharge,
    # 1/(1-SOC_HIGH) en charge (avant : 5/2 = 1/(0.6-0.2) = 1/(1-0.6)).
    SOC_LOW  = 0.40
    SOC_HIGH = 0.75

    if SOC_LOW < SoC_t < SOC_HIGH:
        if P_tot_ref_t <= 0:
            P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
            P_dc_fc_t = 0
        else:
            # décroissance linéaire de P_dc_bat_t : vaut 0 en SOC_LOW, 1 en SOC_HIGH
            P_dc_bat_t = P_tot_ref_t * (SoC_t - SOC_LOW) / (SOC_HIGH - SOC_LOW)
            P_dc_fc_t = P_tot_ref_t - P_dc_bat_t
            P_dc_ely_t = 0
    elif SoC_t <= SOC_LOW:
        if P_tot_ref_t > 0:
            P_dc_bat_t = 0
            P_dc_ely_t = 0
            P_dc_fc_t = P_tot_ref_t
        else:
            P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
            P_dc_fc_t = 0
    else:
        if P_tot_ref_t >= 0:
            P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
            P_dc_fc_t = 0
        else:
            # décroissance linéaire de P_dc_bat_t : vaut 1 en SOC_HIGH, 0 en 1.0
            P_dc_bat_t = P_tot_ref_t * (1.0 - SoC_t) / (1.0 - SOC_HIGH)
            P_dc_fc_t = 0
            P_dc_ely_t = P_tot_ref_t - P_dc_bat_t
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