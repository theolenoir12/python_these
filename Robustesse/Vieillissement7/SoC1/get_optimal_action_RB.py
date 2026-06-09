import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t) :      
        
    ######################### RULES ##########################    
    if P_tot_ref_t >= 0 :
        P_dc_bat_t = 0
        P_dc_fc_t  = P_tot_ref_t
        P_dc_ely_t = 0
    else :
        P_dc_bat_t = P_tot_ref_t
        P_dc_fc_t  = 0
        P_dc_ely_t = 0
        
    P_bat_t = P_dc_bat_t * CONV['eta']**np.sign(P_dc_bat_t) 
    
    SoC_tp1 = SoC_t - P_bat_t / (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) 
    
    if SoC_tp1 > 0.995 or SoC_tp1 < 0.2 :
        if SoC_tp1 > 0.995 :
            P_bat_t    = (SoC_t - 0.99499) * (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) 
            P_dc_bat_t = P_bat_t / CONV['eta']**np.sign(P_bat_t) 
            P_dc_ely_t = P_tot_ref_t - P_dc_bat_t
            
        elif SoC_tp1 < 0.2 :
            P_bat_t   = (SoC_t - 0.20001) * (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) 
            P_dc_bat_t = P_bat_t / CONV['eta']**np.sign(P_bat_t)
            P_dc_fc_t  = P_tot_ref_t - P_dc_bat_t
            
    if P_dc_fc_t / CONV['eta'] > P_fc_max_t or abs(P_dc_ely_t) * CONV['eta'] > P_ely_max_t :
        if P_dc_fc_t / CONV['eta'] > P_fc_max_t :
            P_dc_fc_t  = P_fc_max_t * CONV['eta'] * 0.999
            P_dc_bat_t = P_tot_ref_t - P_dc_fc_t
            
        if  abs(P_dc_ely_t) * CONV['eta'] > P_ely_max_t :
            P_dc_ely_t = - P_ely_max_t / CONV['eta'] * 0.999
            P_dc_bat_t = P_tot_ref_t - P_dc_bat_t
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