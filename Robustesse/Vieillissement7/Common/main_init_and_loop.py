import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from .Init_EMR_MG_v16_python import *
from .simulate_transition import simulate_transition
from .get_soh import get_soh
from .cost_fcn_total2 import *


def init_and_run_loop(get_optimal_action_RB):

    # Initialisation des variables
    T = (SIM['Tend'] / 365)*365*5  # horizon de temps
    SoC_init  = 0.5  # état initial
    E_h2_init = 200
    SoC_t = SoC_init
    
    plot = 1
    
    temps = np.arange(0, T - LOAD['Ts'], LOAD['Ts'])
    n     = len(temps)
    
    SoC       = np.zeros(n+1); SoC[0]  = SoC_init
    E_h2      = np.zeros(n+1); E_h2[0] = E_h2_init # [kWh] hydrogène stocké
    P_bat     = np.zeros(n)
    P_fc      = np.zeros(n)
    P_ely     = np.zeros(n)
    P_dc_load = np.zeros(n)
    P_dc_pv   = np.zeros(n)
    P_dc_bat  = np.zeros(n)
    P_dc_fc   = np.zeros(n)
    P_dc_ely  = np.zeros(n)
    P_fc      = np.zeros(n)
    P_ely     = np.zeros(n)
    alpha_fc  = np.zeros(n+1); alpha_fc[0]  = 0
    alpha_ely = np.zeros(n+1); alpha_ely[0] = 0
    lol_tab   = np.zeros(n)
    SoH_bat   = np.zeros(n+1); SoH_bat[0] = 1
    SoH_fc    = np.zeros(n+1); SoH_fc[0]  = 1
    SoH_ely   = np.zeros(n+1); SoH_ely[0] = 1
    
    deg_fc  = {'start-stop':np.zeros(n),'high':np.zeros(n), 'idling':np.zeros(n),
        'transient':np.zeros(n), 'total':np.zeros(n)}
    
    deg_ely = {'start-stop': np.zeros(n), 'turning power': np.zeros(n), 'idling': np.zeros(n),
        'transient':np.zeros(n), 'total':np.zeros(n)}
    
    defaillances = []
    
    if 'BAT' in defaillances :
        BAT['Ccapacity'] *= defaillances[defaillances.index('BAT')+1]
        BAT['Q_bat'] *= defaillances[defaillances.index('BAT')+1]

    ############################### GET ALPHAS AT EoL ########################################################
    def voltage_fc(alpha_fc,i_fc) : 
        voltage = FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc / FC['n_parallel'] 
                  - A * FC['T'] * np.log((i_fc / S / FC['n_parallel'] + j_in) / j_0)
                  - B * FC['T'] * np.log(1 - i_fc / S / FC['n_parallel'] / j_L / (1 - alpha_fc)))
        return voltage
    
    i_fc_nom    = 147.44120162016202 #75% du courant qui donne P_fc_max (voir modèles SOH)
    V_bol_fc    = voltage_fc(0.0, i_fc_nom)
    def residual_fc(alpha, SoH):
        return voltage_fc(alpha, i_fc_nom) / V_bol_fc - SoH

    def voltage_ely(alpha_ely,i_ely) : 
        voltage = ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely / ELY['n_parallel'] 
                  + A * ELY['T'] * np.log((i_ely / S / ELY['n_parallel'] + j_in) / j_0)
                  + B * ELY['T'] * np.log(1 - i_ely / S / ELY['n_parallel'] / j_L / (1 - alpha_ely)))
        return voltage        
    
    i_ely_nom = 164.925 #75% du courant qui donne P_ely_max
    V_bol_ely   = voltage_ely(0.0, i_ely_nom)
    def residual_ely(alpha, SoH):
        return V_bol_ely / voltage_ely(alpha, i_ely_nom) - SoH
    
    alpha_fc_eol  = brentq(residual_fc, 0.0, 0.32, args=(FC['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    alpha_ely_eol = brentq(residual_ely, 0.0, 0.2503, args=(ELY['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    ########################################################################################################
    # alpha_fc_eol  = (1 - FC['SoH_EoL'])*1873.6207/1766.1207 #avec l'équation P_fc_max = f(alpha_fc) pour EoL
    # alpha_ely_eol = (1 - ELY['SoH_EoL'])*60122.0238/61392.7339 #avec 3 stacks mais pas d'importance
    
    j_new_bat = 0
    j_new_fc  = 0
    j_new_ely = 0   
    
    # --- Initialisation des tableaux ---
    data = {
        "temps": temps,
        "n": n,
        "SoC": np.zeros(n+1),
        "E_h2": np.zeros(n+1),
        "P_bat": np.zeros(n),
        "P_fc": np.zeros(n),
        "P_ely": np.zeros(n),
        "P_dc_load": np.zeros(n),
        "P_dc_pv": np.zeros(n),
        "P_dc_bat": np.zeros(n),
        "P_dc_fc": np.zeros(n),
        "P_dc_ely": np.zeros(n),
        "alpha_fc": np.zeros(n+1),
        "alpha_ely": np.zeros(n+1),
        "lol_tab": np.zeros(n),
        "SoH_bat": np.zeros(n+1),
        "SoH_fc": np.zeros(n+1),
        "SoH_ely": np.zeros(n+1),
        "deg_fc": {
            'start-stop': np.zeros(n), 'high': np.zeros(n), 
            'idling': np.zeros(n), 'transient': np.zeros(n), 'total': np.zeros(n)
        },
        "deg_ely": {
            'start-stop': np.zeros(n), 'turning power': np.zeros(n), 
            'idling': np.zeros(n), 'transient': np.zeros(n), 'total': np.zeros(n)
        }
    }

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
        SoH_fc_t    = SoH_fc[j]
        SoH_ely_t   = SoH_ely[j]
        E_h2_t      = E_h2[j]
    
        i_fc_max_t = (-194.3950 * alpha_fc_t + 196.5598)
        P_fc_max_t = i_fc_max_t * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc_t) * i_fc_max_t / FC['n_parallel'] 
                                                - A * FC['T'] * np.log((i_fc_max_t / S / FC['n_parallel'] + j_in) / j_0)
                                                - B * FC['T'] * np.log(1 - i_fc_max_t / S / FC['n_parallel'] / j_L / (1 - alpha_fc_t)))
        i_ely_max_t = (-219.9 * alpha_ely_t + 219.9)
        P_ely_max_t = i_ely_max_t * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely_t) * i_ely_max_t / ELY['n_parallel'] 
                                                   + A * ELY['T'] * np.log((i_ely_max_t / S / ELY['n_parallel'] + j_in) / j_0)
                                                   + B * ELY['T'] * np.log(1 - i_ely_max_t / S / ELY['n_parallel'] / j_L / (1 - alpha_ely_t)))
    
        # CALCUL DU RUL OPTIMISÉ (sans get_rul)
        # Fuel Cell
        diff_j_fc = j - j_new_fc
        if diff_j_fc >= 20:
            delta_soh = SoH_fc[j_new_fc] - SoH_fc[j]
            if delta_soh > 1e-9: # Éviter division par zéro
                # Projection linéaire : (Temps_écoulé * Delta_total) / Delta_actuel - Temps_écoulé
                RUL_fc_t = (diff_j_fc * (SoH_fc[j_new_fc] - FC['SoH_EoL']) / delta_soh - diff_j_fc) / 24
            else: RUL_fc_t = 8000
        else: RUL_fc_t = 8000
    
        # Électrolyseur
        diff_j_ely = j - j_new_ely
        if diff_j_ely >= 20:
            delta_soh_ely = SoH_ely[j_new_ely] - SoH_ely[j]
            if delta_soh_ely > 1e-9:
                RUL_ely_t = (diff_j_ely * (SoH_ely[j_new_ely] - ELY['SoH_EoL']) / delta_soh_ely - diff_j_ely) / 24
            else: RUL_ely_t = 3000
        else: RUL_ely_t = 3000
    
        action, lol = get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t)
              
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
        
        alpha_fc_tp1  = brentq(residual_fc, 0.0, 0.32, args=(SoH_fc_tp1,), xtol=1e-10, rtol=1e-10)
        alpha_ely_tp1 = brentq(residual_ely, 0.0, 0.2503, args=(SoH_ely_tp1,), xtol=1e-10, rtol=1e-10)
        
        # alpha_fc_tp1  = (1 - SoH_fc_tp1)/(1-FC['SoH_EoL'])*alpha_fc_eol
        # alpha_ely_tp1 = (1 - SoH_ely_tp1)/(1-ELY['SoH_EoL'])*alpha_ely_eol
            
        SoH_bat[j+1]   = SoH_bat_tp1
        SoH_fc[j+1]    = SoH_fc_tp1
        SoH_ely[j+1]   = SoH_ely_tp1
        alpha_fc[j+1]  = alpha_fc_tp1
        alpha_ely[j+1] = alpha_ely_tp1   
        
        SoC_t = SoC_tp1
    
    data["temps"] = temps
    data["SoC"] = SoC
    data["E_h2"] = E_h2
    data["P_bat"] = P_bat
    data["P_fc"] = P_fc
    data["P_ely"] = P_ely
    data["P_dc_load"] = P_dc_load
    data["P_dc_pv"] = P_dc_pv
    data["P_dc_bat"] = P_dc_bat
    data["P_dc_fc"] = P_dc_fc
    data["P_dc_ely"] = P_dc_ely
    data["lol_tab"] = lol_tab
    data["alpha_fc"] = alpha_fc
    data["alpha_ely"] = alpha_ely
    data["SoH_bat"] = SoH_bat
    data["SoH_fc"] = SoH_fc
    data["SoH_ely"] = SoH_ely
    data["deg_fc"] = deg_fc
    data["deg_ely"] = deg_ely


    return data