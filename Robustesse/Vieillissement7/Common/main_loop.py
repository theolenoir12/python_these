import numpy as np
from .simulate_transition import simulate_transition
from .get_soh import get_soh
from .cost_fcn_total2 import get_cost_fc, get_cost_ely
from .physics import *
from .Init_EMR_MG_v16_python import *


def run_main_loop(data, SoC_t, get_action_func):

    for t in temps:
        
        # if int(t/T*100*10) == t/T*100*10 : 
        #     print("Temps global (%) :",t/T*100)
        
        j = int(t / LOAD['Ts'])
        P_load_t    = LOAD['P_ref'][j]
        P_dc_load_t = P_load_t / CONV['eta']
        P_pv_t      = PV['P'][j]
        P_dc_pv_t   = P_pv_t
        P_tot_ref_t   = (P_dc_load_t - P_dc_pv_t)
        
        
        alpha_fc_t  = alpha_fc[j]
        alpha_ely_t = alpha_ely[j]
        SoH_bat_t   = SoH_bat[j]
        E_h2_t      = E_h2[j]
    
        i_fc_max_t = (-194.3950 * alpha_fc_t + 196.5598)
        P_fc_max_t = i_fc_max_t * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc_t) * i_fc_max_t / FC['n_parallel'] 
                                                - A * FC['T'] * np.log((i_fc_max_t / S / FC['n_parallel'] + j_in) / j_0)
                                                - B * FC['T'] * np.log(1 - i_fc_max_t / S / FC['n_parallel'] / j_L / (1 - alpha_fc_t)))
        i_ely_max_t = (-219.9 * alpha_ely_t + 219.9)
        P_ely_max_t = i_ely_max_t * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely_t) * i_ely_max_t / ELY['n_parallel'] 
                                                   + A * ELY['T'] * np.log((i_ely_max_t / S / ELY['n_parallel'] + j_in) / j_0)
                                                   + B * ELY['T'] * np.log(1 - i_ely_max_t / S / ELY['n_parallel'] / j_L / (1 - alpha_ely_t)))
    
        action, lol = get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t)
        
        SoC_tp1, simOut = simulate_transition(SoC_t, action, P_tot_ref_t,plot,lol,alpha_fc_t,alpha_ely_t,SoH_bat_t, E_h2_t, E_h2_init,P_fc_max_t,P_ely_max_t)
        
        P_bat[j]     = simOut['P_bat']
        P_fc[j]      = simOut['P_fc']
        P_ely[j]     = simOut['P_ely']
        P_dc_load[j] = P_dc_load_t
        P_dc_pv[j]   = P_dc_pv_t
        P_dc_bat[j]  = simOut['P_dc_bat']
        P_dc_fc[j]   = simOut['P_dc_fc']
        P_dc_ely[j]  = simOut['P_dc_ely']
        lol_tab[j]   = lol
        E_h2[j+1]    = simOut['E_h2_tp1']
        SoC[j+1]     = SoC_tp1
        
        SoH_bat_tp1, SoH_fc_tp1, SoH_ely_tp1 = get_soh(alpha_fc[j_new_fc:j+1],P_fc[j_new_fc:j+1],alpha_ely[j_new_ely:j+1],P_ely[j_new_ely:j+1],P_bat[j_new_bat:j+1],SoC[j_new_bat:j+2],SoH_bat[j_new_bat:j+1])
        
        fcto, fcss, fci, fct, fch  = get_cost_fc(alpha_fc[j_new_fc:j+1], P_fc[j_new_fc:j+1])
        elto, elss, eli, elt, eltu = get_cost_ely(alpha_ely[j_new_ely:j+1], P_ely[j_new_ely:j+1])
        
        deg_fc['start-stop'][j] = fcss
        deg_fc['idling'][j]     = fci
        deg_fc['transient'][j]  = fct
        deg_fc['high'][j]       = fch
        deg_fc['total'][j]      = fcto*100/FC['cost']*((1 - FC['SoH_EoL'])*100)/100
        
        deg_ely['start-stop'][j]    = elss
        deg_ely['idling'][j]        = eli
        deg_ely['transient'][j]     = elt
        deg_ely['turning power'][j] = eltu
        deg_ely['total'][j]         = elto*100/ELY['cost']*((1 - ELY['SoH_EoL'])*100)/100
        
        if SoH_bat_tp1 < BAT['SoH_EoL'] : 
            SoH_bat_tp1 = 1
            j_new_bat = j
        if SoH_fc_tp1 < FC['SoH_EoL'] : 
            SoH_fc_tp1 = 1
            j_new_fc = j
        if SoH_ely_tp1 < ELY['SoH_EoL'] : 
            SoH_ely_tp1 = 1
            j_new_ely = j
        
        alpha_fc_tp1  = (1 - SoH_fc_tp1)/(1-FC['SoH_EoL'])*alpha_fc_eol
        alpha_ely_tp1 = (1 - SoH_ely_tp1)/(1-ELY['SoH_EoL'])*alpha_ely_eol
            
        SoH_bat[j+1]   = SoH_bat_tp1
        SoH_fc[j+1]    = SoH_fc_tp1
        SoH_ely[j+1]   = SoH_ely_tp1
        alpha_fc[j+1]  = alpha_fc_tp1
        alpha_ely[j+1] = alpha_ely_tp1   
        
        SoC_t = SoC_tp1
        
    return data