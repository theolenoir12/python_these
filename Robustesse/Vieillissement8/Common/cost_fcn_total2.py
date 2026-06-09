import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from .Init_EMR_MG_v16_python import *

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'Cumulative_degradation_bat.txt')
deg_cumul = np.loadtxt(file_path, delimiter=',')
deg_cumul1 = deg_cumul[:,0]/100
deg_cumul2 = deg_cumul[:,1]

def get_cost_bat(P_bat,SoC, SoH_bat):
    
    # Lecture des données de dégradation cumulative à partir d'un fichier CSV
    i_bat = P_bat / (BAT['v_cell_nom']*BAT['series_num'])

    # Interpolation pour trouver la dégradation cumulative asSoCiée aux SoC extrêmes
    cu_deg_SoC = np.interp(SoC, deg_cumul1, deg_cumul2)
    deg_SoC = np.abs(np.diff(cu_deg_SoC))

    # Mise à l'échelle de la dégradation en fonction de la capacité de la batterie
    deg_SoC = deg_SoC * (BAT['Q_bat'] * SoH_bat) * BAT['parallel_num'] / 2.15


    # Calcul des C-rates
    C_rates = np.abs(i_bat) / (BAT['Q_bat'] * SoH_bat * BAT['parallel_num'])

    # Facteurs de redimensionnement en fonction des C-rates
    #scaling_factors = np.where(C_rates > 1, 0.2956 * C_rates + (1 - 0.2956), 1)
    scaling_factors = np.where(C_rates > 1, 0.2956 * C_rates + (1 - 0.2956), 
                                np.where(C_rates >= 0, 1, 0))

    # Coût total en termes de dégradation (Ah)
    cost_tot = np.sum(deg_SoC * scaling_factors)

    # Calcul du coût de la batterie en pourcentage par rapport à la fin de vie
    cost_bat = cost_tot * 1e-6 / ((1 - BAT['SoH_EoL']) * BAT['Q_bat'] * BAT['parallel_num'])

    return cost_bat*BAT['cost']


def get_cost_ely(alpha_ely, P_ely):
    """
    Calcul des coûts de dégradation PEMWE basé sur la physique et les modes de fonctionnement.
    
    PHILOSOPHIE (basée sur CSV):
    - Tension cellule réf : 1.5V
    - Seuils Pmax : 1 A/cm² = 30%, 2 A/cm² = 60%
    
    TAUX DE DEGRADATION (Source CSV):
    - Start-stop : 44.4 µV/cycle
    - Idling (1% Pmax) : 1.5 µV/h
    - High Load Steady (>= 60% Pmax) : 196 µV/h
    - Low/Med Load Steady (< 60% Pmax) : 0 µV/h
    - Fluctuation Haute (Reste dans > 30% Pmax) : 66 µV/h
    - Fluctuation Complète (Croise le seuil bas) : 16 µV/h
    """
    
    P_ely = np.abs(P_ely)
    n_points = len(P_ely)
    
    # Early return
    if n_points == 0:
        return 0, 0, 0, 0, 0
    # NB: le cas n_points == 1 N'EST PLUS court-circuité (retour 0) : on calcule la
    # contribution "par point" du point unique (idle/steady), termes "par paire"
    # (start-stop, transient) nuls car P_prev = P[0]. Cela ne change rien au calcul
    # sur un tableau complet (n>=2) mais rend le coût additif -> accumulation O(n).

    # --- 1. CONSTANTES & CONVERSIONS ---
    
    # Temps d'échantillonnage en heures
    Ts_hours = LOAD['Ts'] / 3600
    
    # Facteur de conversion µV -> % de dégradation (pour une cellule de 1.5V)
    # 1 µV = 1e-6 V. Dégradation = (Delta_V / V_nom) * 100
    uv_to_pct = (1e-6 / 1.5) * 100 
    
    # Coefficients convertis en % / h (ou % / cycle)
    coeffs_pct = {
        'start_stop': 44.4 * uv_to_pct,      # par cycle
        'idle': 1.5 * uv_to_pct,             # par heure
        'steady_high': 196.0 * uv_to_pct,    # par heure (si P >= 60%)
        'fluc_high': 66.0 * uv_to_pct,       # par heure (fluctuation 1-2 A/cm²)
        'fluc_full': 16.0 * uv_to_pct        # par heure (fluctuation 0-2 A/cm²)
    }

    # Calcul de la P_max actuelle (vieillissement inclus)
    
    i_ely_max = (-732.6 * alpha_ely + 732.6)
    P_ely_max = i_ely_max * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely_max / ELY['n_parallel'] 
                                               + A * ELY['T'] * np.log((i_ely_max / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                                               + B * ELY['T'] * np.log(1 - i_ely_max / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))

    # Pour simplifier les masques, on utilise P_max du pas de temps courant
    # (Note: P_max change peu sur un horizon court, mais on garde la vectorisation)
    P_ely_max = P_ely_max if isinstance(P_ely_max, np.ndarray) else np.full(n_points, P_ely_max)

    # --- 2. DEFINITION DES SEUILS (PHILOSOPHIE UTILISATEUR) ---
    
    # Seuils de puissance en valeur absolue
    th_start = 0.0005 * P_ely_max     # 0.05% Pmax
    th_idle  = 0.01 * P_ely_max       # 1% Pmax
    th_30    = 0.30 * P_ely_max       # 30% Pmax (1 A/cm²)
    th_60    = 0.60 * P_ely_max       # 60% Pmax (2 A/cm²)

    # --- 3. ANALYSE VECTORISEE DES ETATS ---
    
    P_curr = P_ely
    # On crée une version décalée pour comparer t et t-1 (padding avec le premier point)
    P_prev = np.concatenate(([P_ely[0]], P_ely[:-1]))
    
    # A. DETECTION START-STOP
    # On considère un cycle si on passe de P < th_start à P >= th_start
    is_off_prev = P_prev < th_start
    is_on_curr = P_curr >= th_start
    # Un start est une transition OFF -> ON. 
    # (Note: ton CSV donne 44.4µV/cycle. On compte ici les démarrages.
    # Si le coût est par "cycle complet" (ON+OFF), on applique à chaque démarrage).
    starts = is_off_prev & is_on_curr
    cost_start_stop = np.sum(starts) * coeffs_pct['start_stop']

    # B. ANALYSE DES MODES OPÉRATIONNELS (POUR LES HEURES ACTIVES)
    # On ne dégrade "à l'heure" que si le système est ON (au moins en Idle)
    is_running = P_curr >= th_start
    
    # Calcul de la variation (Fluctuation vs Steady)
    # On définit un seuil de variation pour distinguer "Steady" de "Transient"
    # Disons 5% de Pmax. En dessous, c'est du bruit/steady. Au dessus, c'est une rampe/fluctuation.
    delta_P = np.abs(P_curr - P_prev)
    rate_of_change_per_hour = delta_P / Ts_hours
    is_transient = rate_of_change_per_hour > (0.05 * P_ely_max /(Ts_hours/1))
    is_steady = ~is_transient
    
    # --- C. CALCUL DES COUTS PAR CATEGORIE ---
    
    # 1. IDLING (Maintenance/Standby)
    # Mode : Running MAIS puissance très faible (< 1% mais > 0 ou juste seuil idle)
    # Ici ta définition "Idling (1% Pmax)" suggère la zone [0, 1%].
    # Attention: "starts" compte l'événement, ici on compte le temps passé.
    # On exclut les points qui sont purement OFF (0).
    mask_idle = (P_curr > 0) & (P_curr <= th_idle) 
    # Coût = nb_heures * taux * Ts
    cost_maint = np.sum(mask_idle) * coeffs_pct['idle'] * Ts_hours

    # 2. STEADY STATE (Régime établi)
    # Condition : Running + Steady (pas de variation majeure)
    # Zone High : >= 60% Pmax
    mask_steady_high = is_running & is_steady & (P_curr >= th_60)
    # Zone Low/Med : < 60% Pmax (Coût = 0 selon ta consigne "Constant 1A/cm² -> 0")
    # mask_steady_low = is_running & is_steady & (P_curr < th_60) # Coût 0, implicite.
    
    cost_steady = np.sum(mask_steady_high) * coeffs_pct['steady_high'] * Ts_hours

    # 3. TRANSIENTS (Fluctuations)
    # Condition : Running + Transient
    mask_fluc_candidates = is_running & is_transient
    
    # Distinction High Power Fluc vs Full Fluc
    # High Fluc : La variation se fait entièrement dans la zone haute (> 30% Pmax)
    # C'est à dire : P_curr > 30% ET P_prev > 30%
    # (Ta consigne : "66 µV/h entre 1 et 2 A/cm²")
    is_in_high_zone = (P_curr >= th_30) & (P_prev >= th_30)
    
    mask_fluc_high = mask_fluc_candidates & is_in_high_zone
    
    # Full Fluc : La variation traverse la zone basse (0-2 A/cm² mixé)
    # Donc candidat transient MAIS pas resté purement dans la zone haute
    mask_fluc_full = mask_fluc_candidates & (~is_in_high_zone)
    
    cost_shift = (np.sum(mask_fluc_high) * coeffs_pct['fluc_high'] * Ts_hours +
                  np.sum(mask_fluc_full) * coeffs_pct['fluc_full'] * Ts_hours)

    # --- 4. TOTAL ET NORMALISATION ---
    
    # Somme des % de dégradation
    degradation_total_pct = cost_start_stop + cost_maint + cost_steady + cost_shift
    
    # Calcul du coût financier
    # L'indicateur est normalisé par la plage de SoH utilisable (SoH_EoL).
    # Si SoH_EoL = 0.8 (fin de vie à 80%), la plage utile est 20% (0.2).
    # Le coût est la fraction de vie consommée * CAPEX.
    
    range_useful = (1 - ELY['SoH_EoL']) * 100 # Ex: (1 - 0.8)*100 = 20%
    fraction_consumed = degradation_total_pct / range_useful 
    
    cost_financial = fraction_consumed * ELY['cost']

    return cost_financial, cost_start_stop, cost_maint, cost_shift, cost_steady


def get_cost_fc(alpha_fc, P_fc):
    # Récupération des données de puissance et de temps d'échantillonnage
    Ts = LOAD['Ts']  # Intervalle de temps (secondes)

    # Calcul des puissances maximales et des seuils haut/bas
    i_fc_max = (-234.8032 * alpha_fc + 238.8252)
    P_fc_max = i_fc_max* FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc_max / FC['n_parallel'] 
                                            - A * FC['T'] * np.log((i_fc_max / S / FC['n_parallel'] + j_in) / FC['j_0'])
                                            - B * FC['T'] * np.log(1 - i_fc_max / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))

    #P_fc_max = 1e3 * (6.249 - 5.619 * alpha_fc) * FC['n_parallel'] * FC['n_series'] / 50 #empirique en fonction de mon modèle
    P_high = 0.8 * P_fc_max
    P_low  = 0.01 * P_fc_max

    # Coefficients de dégradation
    alpha_on_off = 1.96e-3  / 4 # (% de tension / cycle)
    alpha_high   = 1.47e-3  / 4 # (% de tension / heure)
    alpha_low    = 1.26e-3  / 4 # (% de tension / heure)
    alpha_shift  = 5.93e-5  / 4 # (% de tension / cycle)

    # Comptage des cycles ON/OFF et calcul du coût asSoCié
    counter_on_off = np.sum(np.diff(P_fc < 1) == 1)/2 #pour éviter de compter en double
    cost_on_off = counter_on_off * alpha_on_off

    # Comptage et calcul du coût pour haute et basse puissance
    counter_high = np.sum(P_fc > P_high)
    cost_high = counter_high * Ts * alpha_high / 3600  # Conversion du temps en heures

    counter_low = np.sum((P_fc < P_low) & (P_fc > 1*np.ones(len(P_fc))))
    cost_low = counter_low * Ts * alpha_low / 3600  # Conversion du temps en heures

    ####################### EXTRAPOLATION LINÉAIRE ###############
    # mask_const = (P_fc >= P_low) & (P_fc <= P_high)
    # alpha_const_points = P_fc[mask_const] / P_high[mask_const] * alpha_high    
    # cost_const = np.sum(alpha_const_points) * Ts / 3600
    # cost_high = cost_high + cost_const
    #############################################################
    
     ####################### EXTRAPOLATION LINÉAIRE ###############
    mask_const = (P_fc >= P_low) & (P_fc <= P_high)
    alpha_const_points = alpha_low + (alpha_high - alpha_low) * (P_fc[mask_const] - P_low[mask_const]) / (P_high[mask_const] - P_low[mask_const])
    cost_const = np.sum(alpha_const_points) * Ts / 3600
    cost_high = cost_high + cost_const
    ############################################################# 

    # Calcul du coût des transitions de puissance
    cost_shift = alpha_shift * np.sum(np.abs(np.diff(P_fc)) / (P_high[:-1] - P_low[:-1]))

    # Coût total
    cost_tot = cost_on_off + cost_high + cost_low + cost_shift

    # Normalisation du coût en pourcentage par rapport à la fin de vie (10% de perte de tension)
    cost_fc = cost_tot / ((1 - FC['SoH_EoL'])*100) * 100

    return cost_fc/100*FC['cost'], cost_on_off, cost_low, cost_shift, cost_high

def get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat):
    cost_bat = get_cost_bat(P_bat, SoC, SoH_bat)
    cost_fc  = get_cost_fc(alpha_fc, P_fc)[0]
    cost_ely = get_cost_ely(alpha_ely, P_ely)[0] 
    
    return cost_bat + cost_fc + cost_ely