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


# ============================================================================
#  MODELE DE DEGRADATION PEMWE (electrolyseur) : RECUPERATION reversible/irreversible
# ----------------------------------------------------------------------------
#  Calibre sur les 5 modes de durabilite de Rakousky et al. (J. Power Sources
#  342, 2017), Table 2 : delta_chrono = hausse de tension sur le test de 1009 h
#  divisee par 1009 h (PAS un regime asymptotique). Valeurs cibles (uV/h /cell) :
#     A const 1 A/cm2 = 0 ; B const 2 = 194 ; C dyn 2<->1 6h = 65 ;
#     D dyn 2<->0 6h = 16 ; E dyn 2<->0 10min = 50.
#
#  Physique (Rakousky 3.1) : la degradation PEMWE a une part IRREVERSIBLE
#  (permanente : corrosion Ti-PTL, hausse resistance ohmique) et une part
#  REVERSIBLE qui se construit en fonctionnement et se RECUPERE quand le
#  courant est interrompu :
#     dV_irr/dt = a(i)                  [permanent]
#     dV_rev/dt = b(i) - k(i) * V_rev   [se construit puis recupere]
#     + s par demarrage (transition OFF -> ON)
#  - a, b nuls sous 1 A/cm2 (=30% Pmax) ; rampe lineaire 30%->60% Pmax
#    (1->2 A/cm2) ; saturation au "rated" au-dela de 60% (consigne utilisateur).
#  - Recuperation k(i) fortement decroissante : ~instantanee a i~0 (reset a
#    l'arret), ~nulle a i=1 A/cm2 (pas de recup en operation, conforme cellule
#    C qui degrade presque comme du constant). Le terme idle/maintaining
#    (1.5 uV/h, Lu et al. Table 4) est conserve, applique a tres faible P.
#
#  Le modele est INVARIANT en Ts (integration temporelle, V_rev close-form) et
#  O(n) (etat reporte) -> remplace l'ancien modele "classification par regime"
#  qui dependait du pas de temps. Reproduit les 5 modes a ~1e-20 pres.
#
#  Parametres /cellule (fit moindres carres) :
ELY_REC = {
    'a2': 30.057,    # uV/h    generation irreversible a 2 A/cm2 (60% Pmax)
    'b2': 163.943,   # uV/h    generation reversible   a 2 A/cm2
    'k0': 213.206,   # 1/h     recuperation a i=0   (tau ~ 0.3 min)
    'k1': 0.0021,    # 1/h     recuperation a i=1   (~nulle, tau ~ 470 h)
    's' : 5.829,     # uV/cycle demarrage (OFF -> ON)
    'idle': 1.5,     # uV/h    maintien a tres faible puissance (Lu et al.)
}
ELY_F30 = 0.30                  # 1 A/cm2 = 30% Pmax
ELY_F60 = 0.60                  # 2 A/cm2 = 60% Pmax
UV_TO_PCT = (1e-6 / 1.5) * 100  # uV (sur cellule 1.5 V) -> % de tension


def _ely_pmax(alpha_ely):
    """P_max de l'electrolyseur (vieillissement inclus) ; meme formule que la boucle."""
    i_ely_max = (-732.6 * alpha_ely + 732.6)
    return i_ely_max * ELY['n_parallel'] * ELY['n_series'] * (
        ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely_max / ELY['n_parallel']
        + A * ELY['T'] * np.log((i_ely_max / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
        + B * ELY['T'] * np.log(1 - i_ely_max / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))


def _ely_rates(f):
    """Taux a(f), b(f) (uV/h) et k(f) (1/h) en fonction de f = P/Pmax."""
    p = ELY_REC
    if f <= ELY_F30:
        a = 0.0
        b = 0.0
        k = p['k0'] + (p['k1'] - p['k0']) * (f / ELY_F30) if ELY_F30 > 0 else p['k1']
    elif f <= ELY_F60:
        frac = (f - ELY_F30) / (ELY_F60 - ELY_F30)   # 0 a 30% Pmax -> 1 a 60%
        a = p['a2'] * frac
        b = p['b2'] * frac
        k = p['k1'] * (1.0 - frac)                    # k1 a 30% -> 0 a 60%
    else:
        a = p['a2']                                   # saturation au rated
        b = p['b2']
        k = 0.0
    return a, b, k


def _ely_advance(V_irr, V_rev, P_curr, P_prev, P_max, Ts_h):
    """Avance le modele de recuperation PEMWE d'un pas de temps Ts (etats en uV).
    Retourne (V_irr, V_rev, d_startstop, d_idle) ; increments en uV."""
    Pc = abs(P_curr)
    Pp = abs(P_prev)
    f = Pc / P_max if P_max > 0 else 0.0
    a, b, k = _ely_rates(f)
    V_irr = V_irr + a * Ts_h
    if k > 1e-12:
        Veq = b / k
        V_rev = Veq + (V_rev - Veq) * np.exp(-k * Ts_h)   # solution exacte sur le pas
    else:
        V_rev = V_rev + b * Ts_h
    th_start = 0.0005 * P_max
    d_ss = ELY_REC['s'] if (Pp < th_start and Pc >= th_start) else 0.0
    th_idle = 0.01 * P_max
    d_idle = ELY_REC['idle'] * Ts_h if (0.0 < Pc <= th_idle) else 0.0
    return V_irr, V_rev, d_ss, d_idle


def get_cost_ely(alpha_ely, P_ely):
    """Cout de degradation PEMWE (modele recuperation reversible/irreversible).

    Integre l'etat (V_irr, V_rev) depuis 0 sur le tableau fourni. Utilise tel
    quel sur tableau complet par get_cost_total ; la BOUCLE n'appelle pas cette
    fonction (etat reporte via _ely_advance) car le modele est stateful.

    Retourne : (cost_financial_EUR, deg_startstop_%, deg_maintaining_%,
                deg_reversible_%, deg_irreversible_%).
    """
    P_ely = np.atleast_1d(np.abs(P_ely)).astype(float)
    alpha_ely = np.atleast_1d(alpha_ely).astype(float)
    n = len(P_ely)
    if n == 0:
        return 0, 0, 0, 0, 0
    Ts_h = LOAD['Ts'] / 3600.0
    Pmax = _ely_pmax(alpha_ely)
    Pmax = np.full(n, Pmax) if np.ndim(Pmax) == 0 else np.asarray(Pmax)
    if len(Pmax) != n:
        Pmax = np.full(n, Pmax.flat[0])

    V_irr = 0.0
    V_rev = 0.0
    V_ss = 0.0
    V_idle = 0.0
    P_prev = P_ely[0]
    for idx in range(n):
        V_irr, V_rev, d_ss, d_idle = _ely_advance(V_irr, V_rev, P_ely[idx], P_prev, Pmax[idx], Ts_h)
        V_ss += d_ss
        V_idle += d_idle
        P_prev = P_ely[idx]

    deg_irr  = V_irr  * UV_TO_PCT
    deg_rev  = V_rev  * UV_TO_PCT
    deg_ss   = V_ss   * UV_TO_PCT
    deg_idle = V_idle * UV_TO_PCT
    deg_pct  = deg_irr + deg_rev + deg_ss + deg_idle

    cost_financial = deg_pct / ((1 - ELY['SoH_EoL']) * 100) * ELY['cost']
    return cost_financial, deg_ss, deg_idle, deg_rev, deg_irr


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
    alpha_on_off = 1.96e-3   # (% de tension / cycle)
    alpha_high   = 1.47e-3   # (% de tension / heure)
    alpha_low    = 1.26e-3   # (% de tension / heure)
    alpha_shift  = 5.93e-5   # (% de tension / cycle)

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
    
    #  ####################### EXTRAPOLATION LINÉAIRE ###############
    # mask_const = (P_fc >= P_low) & (P_fc <= P_high)
    # alpha_const_points = alpha_low + (alpha_high - alpha_low) * (P_fc[mask_const] - P_low[mask_const]) / (P_high[mask_const] - P_low[mask_const])
    # cost_const = np.sum(alpha_const_points) * Ts / 3600
    # cost_high = cost_high + cost_const
    # ############################################################# 

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