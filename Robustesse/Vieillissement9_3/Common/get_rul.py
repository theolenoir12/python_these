import numpy as np

def get_rul(SOH, SOH_EoL,Ts=3600):
    """
    Retourne un tableau du temps restant avant d'atteindre le seuil SOH_EoL.
    Estimation par régression linéaire sur tous les points depuis le dernier reset.
    """
    n = len(SOH)
    RUL = np.full(n, np.nan)  # initialisation avec NaN
    true_RUL = np.full(n, np.nan)
    _rul_min_steps = int(20 * 3600 // Ts)
    start_idx = 0  # début du segment courant
    
    for i in range(1, n):
        # Détection d'un reset
        if SOH[i] == 1 :
            for k in range(start_idx+1,i) :
                true_RUL[k] = (i - k) * Ts / 3600 / 24
            start_idx = i
            true_RUL[i] = np.nan
            continue
        
        if i - start_idx < _rul_min_steps :
            continue
        else :
            RUL[i] = (i-start_idx) * (SOH[start_idx] - SOH_EoL) / (SOH[start_idx] - SOH[i]) * Ts / 3600 / 24 - (i-start_idx) * Ts / 3600 / 24    
    
    return RUL, true_RUL