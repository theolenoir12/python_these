from .Init_EMR_MG_v16_python import *

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
                
                h2_curve_kw = (stack_load_pct / 100 * P_ely_max_t) * (stack_eff_pct / 100) / 1000
                if P_h2_t < h2_curve_kw[0] : #tout en kW
                    P_dc_ely_t = 0
                else : 
                    # Interpolation inverse : Quel % de charge donne ce P_h2_req ?
                    new_load = np.interp(P_h2_t, h2_curve_kw, stack_load_pct)
                    
                    # Mise à jour (P_dc coté réseau = P_stack / Rendement Convertisseur)
                    P_dc_ely_t = -(new_load / 100 * P_ely_max_t) / CONV['eta']
                    eff_ely    = np.interp(new_load, *ELY['lut']) / 100
                
            else: # Saturation FC (On doit réduire P_fc)
                # --- Construction courbe : H2 consommé [kW] vs Charge Stack [%] ---
                # P_h2 [kW] = (P_stack [W] / (Eff[%]/100)) / 1000
                stack_load_pct = FC['lut'][0]
                stack_eff_pct  = FC['lut'][1]
                
                # Sécurité +1e-6 pour éviter division par zéro
                h2_curve_kw = (stack_load_pct / 100 * P_fc_max_t) / ((stack_eff_pct + 1e-6) / 100) / 1000
                if P_h2_t < h2_curve_kw[0] :
                    P_dc_fc_t = 0
                else :
                    # Interpolation inverse (sur valeur absolue)
                    new_load = np.interp(abs(P_h2_t), h2_curve_kw, stack_load_pct)
                    
                    # Mise à jour (P_dc coté réseau = P_stack * Rendement Convertisseur)
                    P_dc_fc_t = (new_load / 100 * P_fc_max_t) * CONV['eta']
                    eff_fc    = np.interp(new_load, *FC['lut']) / 100
                
            P_dc_h2_t = P_dc_fc_t + P_dc_ely_t
            
            lol_storage = 1 - (P_dc_bat_t + P_dc_h2_t) / P_tot_ref_t
        else : 
            lol_storage = 0
            
        lol = max(lol_pmax,lol_storage,lol_soc)
        
        action = (P_dc_bat_t, P_dc_fc_t, P_dc_ely_t)
        
        return action, lol