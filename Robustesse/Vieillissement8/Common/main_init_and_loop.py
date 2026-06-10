import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from .Init_EMR_MG_v16_python import *
from .simulate_transition import simulate_transition
from .get_soh import get_soh
from .cost_fcn_total2 import *
from .cost_fcn_total2 import _ely_advance, UV_TO_PCT  # helpers prefixes _ non importes par *


def init_and_run_loop(get_optimal_action_RB):

    # Initialisation des variables
    T = (SIM['Tend'] / 365)*365*25  # horizon de temps
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
    
    deg_ely = {'start-stop': np.zeros(n), 'maintaining': np.zeros(n), 'reversible': np.zeros(n),
        'irreversible':np.zeros(n), 'total':np.zeros(n)}
    
    defaillances = []
    
    if 'BAT' in defaillances :
        BAT['Ccapacity'] *= defaillances[defaillances.index('BAT')+1]
        BAT['Q_bat'] *= defaillances[defaillances.index('BAT')+1]

    ############################### GET ALPHAS AT EoL ########################################################
    def voltage_fc(alpha_fc,i_fc) : 
        voltage = FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc / FC['n_parallel'] 
                  - A * FC['T'] * np.log((i_fc / S / FC['n_parallel'] + j_in) / FC['j_0'])
                  - B * FC['T'] * np.log(1 - i_fc / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))
        return voltage
    
    i_fc_nom    = 179.16718811881188 #75% du courant qui donne P_fc_max (voir modèles SOH)
    V_bol_fc    = voltage_fc(0.0, i_fc_nom)
    def residual_fc(alpha, SoH):
        return voltage_fc(alpha, i_fc_nom) / V_bol_fc - SoH

    def voltage_ely(alpha_ely,i_ely) : 
        voltage = ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely / ELY['n_parallel'] 
                  + A * ELY['T'] * np.log((i_ely / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                  + B * ELY['T'] * np.log(1 - i_ely / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))
        return voltage        
    
    i_ely_nom = 549.45 #75% du courant qui donne P_ely_max
    V_bol_ely   = voltage_ely(0.0, i_ely_nom)
    def residual_ely(alpha, SoH):
        return V_bol_ely / voltage_ely(alpha, i_ely_nom) - SoH
    
    alpha_fc_eol  = brentq(residual_fc, 0.0, 0.22223223, args=(FC['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    alpha_ely_eol = brentq(residual_ely, 0.0, 0.226330713, args=(ELY['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    
        # --- LUT alpha = f(SoH) ---
    _soh_grid      = np.linspace(1.0, min(FC['SoH_EoL'], ELY['SoH_EoL']), 2000)
    _alpha_fc_grid  = np.array([brentq(residual_fc,  0.0, 0.22223223,   args=(s,), xtol=1e-10) for s in _soh_grid])
    _alpha_ely_grid = np.array([brentq(residual_ely, 0.0, 0.226330713, args=(s,), xtol=1e-10) for s in _soh_grid])
    # soh_grid est décroissant (1.0 → SoH_EoL), on le retourne pour interp qui exige x croissant
    _soh_grid_flip      = _soh_grid[::-1]
    _alpha_fc_grid_flip  = _alpha_fc_grid[::-1]
    _alpha_ely_grid_flip = _alpha_ely_grid[::-1]
    ########################################################################################################
    # alpha_fc_eol  = (1 - FC['SoH_EoL'])*1873.6207/1766.1207 #avec l'équation P_fc_max = f(alpha_fc) pour EoL
    # alpha_ely_eol = (1 - ELY['SoH_EoL'])*60122.0238/61392.7339 #avec 3 stacks mais pas d'importance
    
    j_new_bat = 0
    j_new_fc  = 0
    j_new_ely = 0
    _rul_min_steps = int(20 * 3600 // LOAD['Ts'])

    # Accumulateurs de coût de dégradation (cumul depuis le dernier remplacement).
    # Remplacent le recalcul O(n^2) de get_soh/get_cost_* sur [j_new:j+1] par une
    # accumulation incrementale O(1)/pas, EXACTEMENT equivalente (telescopage verifie).
    cum_bat = 0.0
    cum_fc  = np.zeros(5)
    # ELY : modele de recuperation STATEFUL (pas de subtract-trick). Etats en uV,
    # reportes d'un pas a l'autre, remis a 0 au remplacement.
    V_irr_ely = 0.0
    V_rev_ely = 0.0
    V_ss_ely  = 0.0
    V_idle_ely = 0.0
    Ts_h = LOAD['Ts'] / 3600.0
    range_useful_ely = (1 - ELY['SoH_EoL']) * 100
    
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
            'start-stop': np.zeros(n), 'maintaining': np.zeros(n),
            'reversible': np.zeros(n), 'irreversible': np.zeros(n), 'total': np.zeros(n)
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
    
        i_fc_max_t = (-234.8032 * alpha_fc_t + 238.8252)
        P_fc_max_t = i_fc_max_t * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc_t) * i_fc_max_t / FC['n_parallel'] 
                                                - A * FC['T'] * np.log((i_fc_max_t / S / FC['n_parallel'] + j_in) / FC['j_0'])
                                                - B * FC['T'] * np.log(1 - i_fc_max_t / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc_t)))
        i_ely_max_t = (-732.6 * alpha_ely_t + 732.6)
        P_ely_max_t = i_ely_max_t * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely_t) * i_ely_max_t / ELY['n_parallel'] 
                                                   + A * ELY['T'] * np.log((i_ely_max_t / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                                                   + B * ELY['T'] * np.log(1 - i_ely_max_t / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely_t)))
    
        # CALCUL DU RUL OPTIMISÉ (sans get_rul)
        # Fuel Cell
        diff_j_fc = j - j_new_fc
        if diff_j_fc >= _rul_min_steps:
            delta_soh = SoH_fc[j_new_fc] - SoH_fc[j]
            if delta_soh > 1e-9: # Éviter division par zéro
                # Projection linéaire : (Temps_écoulé * Delta_total) / Delta_actuel - Temps_écoulé
                RUL_fc_t = (diff_j_fc * (SoH_fc[j_new_fc] - FC['SoH_EoL']) / delta_soh - diff_j_fc) * LOAD['Ts'] / 3600 / 24
            else: RUL_fc_t = 8000
        else: RUL_fc_t = 8000
    
        # Électrolyseur
        diff_j_ely = j - j_new_ely
        if diff_j_ely >= _rul_min_steps:
            delta_soh_ely = SoH_ely[j_new_ely] - SoH_ely[j]
            if delta_soh_ely > 1e-9:
                RUL_ely_t = (diff_j_ely * (SoH_ely[j_new_ely] - ELY['SoH_EoL']) / delta_soh_ely - diff_j_ely) * LOAD['Ts'] / 3600 / 24
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
        
        # --- Coûts de dégradation : accumulation incrementale (== recalcul sur [j_new:j+1]) ---
        # Marginal d'ajout du point j au segment courant : f([j-1,j]) - f([j-1]).
        if j == j_new_bat:
            m_bat = get_cost_bat(P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1])
        else:
            m_bat = (get_cost_bat(P_bat[j-1:j+1], SoC[j-1:j+2], SoH_bat[j-1:j+1])
                     - get_cost_bat(P_bat[j-1:j],   SoC[j-1:j+1], SoH_bat[j-1:j]))
        cum_bat += m_bat

        if j == j_new_fc:
            m_fc = np.array(get_cost_fc(alpha_fc[j:j+1], P_fc[j:j+1]))
        else:
            m_fc = (np.array(get_cost_fc(alpha_fc[j-1:j+1], P_fc[j-1:j+1]))
                    - np.array(get_cost_fc(alpha_fc[j-1:j], P_fc[j-1:j])))
        cum_fc += m_fc

        # ELY : avance stateful du modele de recuperation (V_rev/V_irr reportes).
        # P_prev = P_ely[j-1] sauf au 1er pas du segment (pas de start-stop, comme
        # l'ancien point unique).
        P_prev_ely = P_ely[j] if j == j_new_ely else P_ely[j-1]
        V_irr_ely, V_rev_ely, d_ss_ely, d_idle_ely = _ely_advance(
            V_irr_ely, V_rev_ely, P_ely[j], P_prev_ely, P_ely_max_t, Ts_h)
        V_ss_ely   += d_ss_ely
        V_idle_ely += d_idle_ely

        deg_irr_ely  = V_irr_ely  * UV_TO_PCT
        deg_rev_ely  = V_rev_ely  * UV_TO_PCT
        deg_ss_ely   = V_ss_ely   * UV_TO_PCT
        deg_idle_ely = V_idle_ely * UV_TO_PCT
        deg_pct_ely  = deg_irr_ely + deg_rev_ely + deg_ss_ely + deg_idle_ely

        SoH_bat_tp1 = 1 - cum_bat    / BAT['cost'] * (1 - BAT['SoH_EoL'])
        SoH_fc_tp1  = 1 - cum_fc[0]  / FC['cost']  * (1 - FC['SoH_EoL'])
        SoH_ely_tp1 = 1 - deg_pct_ely / 100   # = 1 - (V_irr+V_rev+ss+idle)/V_EoL ; peut remonter (recup)

        deg_fc['start-stop'][j] = cum_fc[1]
        deg_fc['idling'][j]     = cum_fc[2]
        deg_fc['transient'][j]  = cum_fc[3]
        deg_fc['high'][j]       = cum_fc[4]
        deg_fc['total'][j]      = cum_fc[0]*100/FC['cost']*((1 - FC['SoH_EoL'])*100)/100

        deg_ely['start-stop'][j]    = deg_ss_ely
        deg_ely['maintaining'][j]   = deg_idle_ely
        deg_ely['reversible'][j]    = deg_rev_ely
        deg_ely['irreversible'][j]  = deg_irr_ely
        deg_ely['total'][j]         = deg_pct_ely
        
        if SoH_bat_tp1 < BAT['SoH_EoL'] :
            SoH_bat_tp1 = 1
            j_new_bat = j
            cum_bat = get_cost_bat(P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1])
        if SoH_fc_tp1 < FC['SoH_EoL'] :
            SoH_fc_tp1 = 1
            j_new_fc = j
            cum_fc = np.array(get_cost_fc(alpha_fc[j:j+1], P_fc[j:j+1]))
        if SoH_ely_tp1 < ELY['SoH_EoL'] :
            SoH_ely_tp1 = 1
            j_new_ely = j
            # Unite neuve : etats remis a 0 puis 1er pas (P_prev = P[j] -> pas de start-stop)
            V_irr_ely, V_rev_ely, d_ss_ely, d_idle_ely = _ely_advance(
                0.0, 0.0, P_ely[j], P_ely[j], P_ely_max_t, Ts_h)
            V_ss_ely   = d_ss_ely
            V_idle_ely = d_idle_ely
        
        alpha_fc_tp1  = np.interp(SoH_fc_tp1,  _soh_grid_flip, _alpha_fc_grid_flip)
        alpha_ely_tp1 = np.interp(SoH_ely_tp1, _soh_grid_flip, _alpha_ely_grid_flip)
        
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