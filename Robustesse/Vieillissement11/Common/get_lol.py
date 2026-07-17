from .Init_EMR_MG_v16_python import *


def _invert_monotone_h2_power(target_kw, load_min_pct, load_max_pct, h2_from_load):
    """Return the largest load whose H2 power does not exceed target.

    Interpolating the inverse only at the efficiency-LUT knots is not
    consistent with the forward model, which interpolates efficiency and
    then evaluates power/efficiency.  Near an H2 boundary, the old inverse
    could therefore overshoot by a few Wh and make the next state
    infeasible.  Bisection uses the exact same forward expression and the
    lower bracket is deliberately returned to stay inside the reservoir.
    """
    lo = float(load_min_pct)
    hi = float(load_max_pct)
    target_kw = float(target_kw)
    if target_kw < h2_from_load(lo):
        return None
    if target_kw >= h2_from_load(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if h2_from_load(mid) <= target_kw:
            lo = mid
        else:
            hi = mid
    return lo


def get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t) :

        lol = 0
        
        P_dc_bat_t = action[0]
        P_dc_fc_t  = action[1]
        P_dc_ely_t = action[2]
        P_dc_h2_t = P_dc_fc_t + P_dc_ely_t

        
        if 'FC' in defaillances :
            P_dc_fc_t = 0
        if 'ELY' in defaillances :
            P_dc_ely_t = 0
        # Le bilan doit utiliser les puissances APRES application des pannes.
        # Sans ce recalcul, lol_soc creditait encore un composant hors service.
        P_dc_h2_t = P_dc_fc_t + P_dc_ely_t
        
        P_bat_t = P_dc_bat_t / CONV['eta']**np.sign(P_dc_bat_t) 
        
        SoC_tp1 = SoC_t - P_bat_t * LOAD['Ts'] / 3600 * BAT['eff']**np.sign(-P_bat_t) / (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) 
        
        if SoC_tp1 > 0.995 or SoC_tp1 < 0.2 :
            if SoC_tp1 > 0.995 :
                P_bat_t    = (SoC_t - 0.99499) * (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) / BAT['eff']**np.sign(-P_bat_t) * 3600 / LOAD['Ts']
                P_dc_bat_t = P_bat_t * CONV['eta']**np.sign(P_bat_t) 
                
            elif SoC_tp1 < 0.2 :
                P_bat_t   = (SoC_t - 0.20001) * (BAT['parallel_num']*BAT['series_num']*BAT['Q_bat']*BAT['v_cell_nom']*SoH_bat_t) / BAT['eff']**np.sign(-P_bat_t) * 3600 / LOAD['Ts']
                P_dc_bat_t = P_bat_t * CONV['eta']**np.sign(P_bat_t) 

            lol_soc = 1 - (P_dc_bat_t + P_dc_h2_t) / P_tot_ref_t 
        else :
            lol_soc = 0              
            
        if P_dc_fc_t / CONV['eta'] > P_fc_max_t or abs(P_dc_ely_t) * CONV['eta'] > P_ely_max_t :
            if P_dc_fc_t / CONV['eta'] > P_fc_max_t :
                P_dc_fc_t = P_fc_max_t * CONV['eta'] * 0.999
                
            if  abs(P_dc_ely_t) * CONV['eta'] > P_ely_max_t :
                P_dc_ely_t = - P_ely_max_t / CONV['eta'] * 0.999
                
            P_dc_h2_t = P_dc_fc_t + P_dc_ely_t
                                    
            lol_pmax = 1 - (P_dc_bat_t + P_dc_h2_t) / P_tot_ref_t
                
        else :
            lol_pmax = 0
        
        eff_fc  = np.interp((P_dc_fc_t / CONV['eta'] / P_fc_max_t) * 100, *FC['lut']) / 100
        eff_ely = np.interp(abs((P_dc_ely_t * CONV['eta'] / P_ely_max_t)) * 100, *ELY['lut']) / 100
        
        P_h2_t = (abs(P_dc_ely_t*CONV['eta'])*eff_ely-abs(P_dc_fc_t/CONV['eta'])/eff_fc)/1000 # [kW] puissance d'hydrogène arrivant dans le réservoir
        E_h2_tp1 = E_h2_t + P_h2_t * LOAD['Ts'] / 3600 # [kWh]
        
        if E_h2_tp1 > E_h2_init or E_h2_tp1 < 0 :   
            
            E_h2_tp1 = E_h2_init-0.001  if E_h2_tp1 > E_h2_init else 0.001
            P_h2_t = (E_h2_tp1 - E_h2_t) * 3600 / LOAD['Ts'] # [kW] (+ = ELY, - = FC)
        
            if abs(P_h2_t) < 0.1 :
                P_h2_t = 0
                P_dc_fc_t  = 0
                P_dc_ely_t = 0
        
            elif P_h2_t > 0: # Saturation ELY (On doit réduire P_ely)
                # --- Construction courbe : H2 produit [kW] vs Charge Stack [%] ---
                # P_stack [W] = (Load[%]/100) * P_max [W]
                # P_h2 [kW]   = P_stack [W] * (Eff[%]/100) / 1000
                stack_load_pct = ELY['lut'][0]
                stack_eff_pct  = ELY['lut'][1]
                
                def h2_from_load(load_pct):
                    eff = np.interp(load_pct, stack_load_pct, stack_eff_pct) / 100
                    return (load_pct / 100 * P_ely_max_t) * eff / 1000

                new_load = _invert_monotone_h2_power(
                    P_h2_t, stack_load_pct[0], stack_load_pct[-1], h2_from_load
                )
                if new_load is None: #tout en kW
                    P_dc_ely_t = 0
                else : 
                    # Mise à jour (P_dc coté réseau = P_stack / Rendement Convertisseur)
                    P_dc_ely_t = -(new_load / 100 * P_ely_max_t) / CONV['eta']
                    eff_ely    = np.interp(new_load, *ELY['lut']) / 100
                
            else: # Saturation FC (On doit réduire P_fc)
                # --- Construction courbe : H2 consommé [kW] vs Charge Stack [%] ---
                # P_h2 [kW] = (P_stack [W] / (Eff[%]/100)) / 1000
                stack_load_pct = FC['lut'][0]
                stack_eff_pct  = FC['lut'][1]
                
                def h2_from_load(load_pct):
                    eff = np.interp(load_pct, stack_load_pct, stack_eff_pct) / 100
                    return (load_pct / 100 * P_fc_max_t) / eff / 1000

                new_load = _invert_monotone_h2_power(
                    abs(P_h2_t), stack_load_pct[0], stack_load_pct[-1], h2_from_load
                )
                if new_load is None:
                    P_dc_fc_t = 0
                else :
                    # Mise à jour (P_dc coté réseau = P_stack * Rendement Convertisseur)
                    P_dc_fc_t = (new_load / 100 * P_fc_max_t) * CONV['eta']
                    eff_fc    = np.interp(new_load, *FC['lut']) / 100
                
            P_dc_h2_t = P_dc_fc_t + P_dc_ely_t
            
            lol_storage = 1 - (P_dc_bat_t + P_dc_h2_t) / P_tot_ref_t
        else : 
            lol_storage = 0
            
        # Garde-fou de bilan : meme sans saturation SoC/Pmax/H2, une puissance
        # annulee par une panne ne doit jamais disparaitre silencieusement.
        # En surplus (P_tot_ref_t <= 0), l'ecart est un ecretage et non une
        # energie de charge non servie.
        lol_balance = 0
        if P_tot_ref_t > 0:
            lol_balance = max(
                0.0, 1 - (P_dc_bat_t + P_dc_fc_t + P_dc_ely_t) / P_tot_ref_t
            )

        lol = max(lol_pmax, lol_storage, lol_soc, lol_balance)
        
        action = (P_dc_bat_t, P_dc_fc_t, P_dc_ely_t)
        
        return action, lol
