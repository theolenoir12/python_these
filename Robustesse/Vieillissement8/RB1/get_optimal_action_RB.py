import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t) :      
        
    ######################### RULES ##########################    
    if 0.2 < SoC_t < 0.6:
        if P_tot_ref_t <= 0:
            P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
            P_dc_fc_t = 0
        else:
            P_dc_bat_t = P_tot_ref_t * (5/2 * SoC_t - 1/2)  # décroissance linéaire de P_dc_bat_t en fonction du SOC. Vaut 1 en 0.6 et 0 en 0.2
            P_dc_fc_t = P_tot_ref_t - P_dc_bat_t
            P_dc_ely_t = 0
    elif SoC_t <= 0.2:
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
            P_dc_bat_t = P_tot_ref_t * (-5/2 * SoC_t + 5/2)  # décroissance linéaire de P_dc_bat_t en fonction du SOC. Vaut 1 en 0.6 et 0 en 1
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