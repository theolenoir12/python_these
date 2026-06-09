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
    if P_tot_ref_t > 0 :
        if P_tot_ref_t > 950 * SoH_fc_t:#P_fc_max_t/FC['P_fc_max'] :
            P_dc_bat_t = P_tot_ref_t - 950 * SoH_fc_t#P_fc_max_t/FC['P_fc_max']
            P_dc_fc_t  = 950 * SoH_fc_t#P_fc_max_t/FC['P_fc_max']
            P_dc_ely_t = 0
        else :
            P_dc_bat_t = P_tot_ref_t 
            P_dc_fc_t  = 0
            P_dc_ely_t = 0
    if P_tot_ref_t < 0 :
        if P_tot_ref_t < - 9000 * SoH_ely_t:#P_ely_max_t/ELY['P_ely_max'] : 
            P_dc_bat_t = P_tot_ref_t + 9000 * SoH_ely_t#P_ely_max_t/ELY['P_ely_max']
            P_dc_fc_t  = 0
            P_dc_ely_t = - 9000 * SoH_ely_t#P_ely_max_t/ELY['P_ely_max']
        else :
            P_dc_bat_t = P_tot_ref_t 
            P_dc_fc_t  = 0
            P_dc_ely_t = 0
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