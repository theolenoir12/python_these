import numpy as np
from scipy.interpolate import interp1d
from sympy import symbols, Eq, solve
from scipy.optimize import root_scalar
from .Init_EMR_MG_v16_python import *

def simulate_transition(SoC_t, action, P_tot_ref_t,plot,lol,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t):
    
    P_dc_bat_t = action[0]
    P_dc_fc_t  = action[1]
    P_dc_ely_t = action[2]

    # Calcul de l'état de charge de la batterie au prochain pas de temps
    P_bat_t = P_dc_bat_t / CONV['eta']**np.sign(P_dc_bat_t) 
    SoC_tp1 = SoC_t - P_bat_t * LOAD['Ts'] / 3600 * BAT['eff']**np.sign(-P_bat_t)  / (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) 
    
    eff_fc  = np.interp((P_dc_fc_t / CONV['eta'] / P_fc_max_t) * 100, *FC['lut']) / 100
    eff_ely = np.interp(abs((P_dc_ely_t * CONV['eta'] / P_ely_max_t)) * 100, *ELY['lut']) / 100
    
    P_h2_t = (abs(P_dc_ely_t*CONV['eta'])*eff_ely-abs(P_dc_fc_t/CONV['eta'])/eff_fc)/1000 # [kW] puissance d'hydrogène arrivant dans le réservoir

    E_h2_tp1 = E_h2_t + P_h2_t * LOAD['Ts'] / 3600 # [kWh]
    if E_h2_tp1 > E_h2_init and E_h2_tp1 - E_h2_init < 0.001 :
        E_h2_tp1 = E_h2_init
    
    if SoC_tp1 < 0.2 or SoC_tp1 > 0.995:
        SoC_tp1 = -1
        simOut = {}
        return SoC_tp1, simOut
    
    if P_dc_fc_t / CONV['eta'] > P_fc_max_t or abs(P_dc_ely_t) * CONV['eta'] > P_ely_max_t: 
        SoC_tp1 = -1
        simOut = {}
        return SoC_tp1, simOut
    
    if E_h2_tp1 < 0 or E_h2_tp1 > E_h2_init:
        SoC_tp1 = -1
        simOut = {}
        return SoC_tp1, simOut

    # Stockage des résultats dans un dictionnaire
    simOut = {
        'P_bat': P_bat_t,
        'P_fc': P_dc_fc_t / CONV['eta'],
        'P_ely': P_dc_ely_t * CONV['eta'],
        'P_dc_bat': P_dc_bat_t,
        'P_dc_fc': P_dc_fc_t,
        'P_dc_ely': P_dc_ely_t,
        'E_h2_tp1': E_h2_tp1
    }

    return SoC_tp1, simOut