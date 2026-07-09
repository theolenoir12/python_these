import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_low_prev) :      
        
    ######################### RULES ##########################    
    # Paramètre de lissage (à ajuster selon la dynamique souhaitée)
    # Proche de 0 = Très lent (ménage l'hydrogène)
    # Proche de 1 = Très réactif (proche du 100% hydrogène)
    alpha = 1e-6
    
    # 1. Calcul de la composante basse fréquence (pour l'hydrogène)
    # Note : P_low_prev doit être initialisé à 0 au début de votre simulation
    P_low_t = alpha * P_tot_ref_t + (1 - alpha) * P_low_prev
    
    # 2. La batterie prend tout le reste (Hautes Fréquences + reliquat)
    P_dc_bat_t = P_tot_ref_t - P_low_t
    
    # 3. Répartition de la composante lente vers l'hydrogène
    if P_low_t > 0:
        P_dc_fc_t  = P_low_t
        P_dc_ely_t = 0
    else:
        P_dc_fc_t  = 0
        P_dc_ely_t = P_low_t # La puissance est négative ici
    ##################################      
 
          
    if 'FC' in defaillances :
        if P_tot_ref_t > 0 : 
            P_dc_bat_t = P_tot_ref_t
    if 'ELY' in defaillances : 
        if P_tot_ref_t < 0 :
            P_dc_bat_t = P_tot_ref_t

    
    action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t

    action, lol = get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t)
    
    # 4. Mise à jour de la mémoire pour le prochain pas de temps
    if action[1] > 0 :        
        P_low_prev = action[1]
    else : 
        P_low_prev = action[2]
    
    return action, lol, P_low_prev