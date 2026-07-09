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

# --- Vieillissement CALENDAIRE batterie (optionnel, OFF par defaut) ---
# None ou 0 -> terme calendaire NUL -> comportement IDENTIQUE a l'original
# (modele de base purement cyclage). Active par l'analyse de sensibilite
# (R3-major3-ii). BAT_CAL_TCAL_Y = vie calendaire [ans] a SoC=100% constant pour
# atteindre l'EoL ; forme lineaire g(SoC)=SoC (taux ~ temps de residence pondere
# par le SoC, variable controlee par l'EMS).
BAT_CAL_TCAL_Y = None

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

    # --- Terme CALENDAIRE optionnel (OFF si BAT_CAL_TCAL_Y est None/0) ---
    # Perte de capacite calendaire = somme_t k_cal(SoC_t)*dt, ADDITIVE par pas ->
    # compatible avec l'accumulation incrementale de la boucle (telescopage OK
    # sur SoC[:-1] = SoC en debut de pas). cost_tot est en micro-Ah (x1e-6 -> Ah).
    if BAT_CAL_TCAL_Y:
        Ts_h = LOAD['Ts'] / 3600.0
        soc_step = np.atleast_1d(SoC).astype(float)[:-1]   # SoC au debut de chaque pas
        # k_cal(SoC=1) [Ah/h] : SoC=1 constant -> EoL (perte 1-SoH_EoL) en T_cal ans
        k1 = ((1 - BAT['SoH_EoL']) * BAT['Q_bat'] * BAT['parallel_num']
              / (BAT_CAL_TCAL_Y * 8760.0))
        q_cal_Ah = float(np.sum(k1 * soc_step * Ts_h))      # g(SoC) = SoC (lineaire)
        cost_tot = cost_tot + q_cal_Ah * 1e6                # remise en micro-Ah

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
    's' : 11.7,      # uV/cycle demarrage (OFF -> ON)
    'idle': 1.5,     # uV/h    maintien a tres faible puissance (Lu et al.)
}

ELY_F30 = 0.30                  # 1 A/cm2 = 30% Pmax
ELY_F60 = 0.60                  # 2 A/cm2 = 60% Pmax
UV_TO_PCT = (1e-6 / 1.5) * 100  # uV (sur cellule 1.5 V) -> % de tension

# ============================================================================
#  [LEGACY / Pei et al. 2008]  Ancien modele PEMFC "classification par regime"
# ----------------------------------------------------------------------------
#  CONSERVE uniquement pour compatibilite : les modules DP (DP/, DP2/) importent
#  encore ces constantes et reimplementent le modele de Pei inline (vectorise).
#  Le nouveau modele reversible/irreversible (ci-dessous, get_cost_fc) NE les
#  utilise PLUS. Pour migrer le DP vers le nouveau modele il faudra reecrire les
#  dp_core.py / dp_aging.py -> hors scope de cette version (focalisee rule-based).
FC_FHIGH = 0.80                 # seuil haute puissance (R3 : "80%")
FC_FLOW  = 0.01                 # seuil idling / basse puissance (R3 : "1%")
FC_ALPHA_ON_OFF = 1.96e-3       # / cycle  (start-stop)
FC_ALPHA_HIGH   = 1.47e-3       # / heure  (haute puissance)
FC_ALPHA_LOW    = 1.26e-3       # / heure  (idling)
FC_ALPHA_SHIFT  = 5.93e-5       # / cycle  (transient)


# ============================================================================
#  MODELE DE DEGRADATION PEMFC : REVERSIBLE / IRREVERSIBLE + dependance au courant
# ----------------------------------------------------------------------------
#  Remplace le modele de Pei (4 regimes a coefficients fixes) par un modele
#  a etats, structurellement SYMETRIQUE de celui de l'electrolyseur (_ely_advance)
#  et calibre sur deux etudes recentes :
#
#   [McCay et al., J. Power Sources 665 (2026) 239011]  short-stack 10 cellules,
#     profils maritimes (charge lente, load-levelling) = cas d'usage le plus
#     proche d'un micro-reseau stationnaire. Quantifie la SEPARATION reversible /
#     irreversible a j_moyen = 0.5 A/cm2 (par cellule) :
#        - reversible  : 52 uV/h (charge constante)  vs 22 uV/h (dynamique)
#                        -> composante DOMINANTE, recuperable au repos/arret.
#        - irreversible: 1.2 uV/h (statique) -> 4.8 uV/h (dynamique)
#                        -> perte PERMANENTE = ~20% de perte d'ECSA cathode / 1500 h
#                        (murissement d'Ostwald), croit avec la densite de courant.
#
#   [Colombo et al., J. Power Sources 553 (2023) 232246]  cellule segmentee,
#     1000 h de cycle automobile realiste (ID-FAST) avec stops de recuperation.
#     Table 4 : taux de perte de tension operando CROISSANT avec j (MEA CCM B,
#     etat de l'art, la plus durable) [uV/h] :
#        j = 0.095 -> 2.4 ; 0.589 -> 13.5 ; 1.273 -> 21.9 ; 1.748 -> 31.7
#     -> donne la FORME de la dependance au courant (remplace les seuils 1%/80%).
#     Reversible ~ "few to 20 mV", recupere aux long-stops (CV a bas potentiel).
#
#  Physique retenue (par cellule, etats en uV) avec f = |P_fc| / P_fc_max :
#     dV_irr/dt = a(f)                    [permanent : Ostwald / ECSA, ~ f^2]
#     dV_rev/dt = b(f) - k(f) * V_rev     [se construit en charge, RECUPERE au repos]
#     V_ss  += s   a chaque demarrage (OFF -> ON)        [permanent, severe]
#     V_idle+= idle * dt  a tres basse puissance (haut potentiel, dissolution Pt)
#  Hypothese de modelisation (documentee) : la recuperation du reversible est
#  activee quand la FC est A L'ARRET (f ~ 0) -- ce qui, dans un systeme bien gere,
#  correspond a une sequence d'arret avec purge/etape reductrice (McCay, Colombo
#  recuperent par excursion a bas potentiel). C'est le levier naturel d'un
#  micro-reseau (les periodes OFF de la FC sont abondantes). La recuperation par
#  excursion a FORT courant (mecanisme McCay constant>dynamique) n'est PAS
#  modelisee dans cette version -> a activer via k(f) si besoin (cf. note README).
#
#  Modele INVARIANT en Ts (V_rev en forme close sur le pas) et O(n) (etat reporte)
#  -> comme l'ELY, la BOUCLE utilise _fc_advance (stateful), pas le subtract-trick.
#
#  Coefficients /cellule (ajustables ; defauts calibres McCay + Colombo) :
FC_REC = {
    'a_irr': 6.0,     # uV/h  irreversible a f=1 (loi a(f)=a_irr*f^2 ; a(0.46)~1.27
                      #        ~ 1.2 uV/h McCay statique a j=0.5). Convexe : le fort
                      #        courant est penalise (Colombo : taux croit avec j).
    'b_rev': 45.0,    # uV/h  generation reversible a f=1 (b(f)=b_rev*f ; ordre de
                      #        grandeur McCay 22-52 uV/h, Colombo few-20 mV).
    'k_rest': 2.0,    # 1/h   recuperation du reversible a l'ARRET (tau ~ 0.5 h).
    'k_op'  : 0.002,  # 1/h   recuperation en fonctionnement (~nulle -> accumulation).
    's'     : 20.0,   # uV/cycle demarrage OFF->ON (start-stop, ~ Fletcher 24 uV).
    'idle'  : 3.0,    # uV/h  maintien a tres basse puissance (haut potentiel).
}
FC_F_OFF  = 0.01   # f en dessous duquel la FC est consideree "a l'arret" (recup).
FC_F_IDLE = 0.05   # f en dessous duquel s'applique la penalite idle/haut potentiel.

# Reference de tension /cellule (BoL, au courant nominal) pour convertir uV -> %.
# Calculee sur le meme modele de polarisation que la boucle, robuste aux params.
_i_fc_nom_ref = 179.16718811881188   # 75% du courant de P_fc_max (cf. modeles SOH)
_V_cell_ref_fc = (FC['E_0']
                  - FC['R'] * _i_fc_nom_ref / FC['n_parallel']
                  - A * FC['T'] * np.log((_i_fc_nom_ref / S / FC['n_parallel'] + j_in) / FC['j_0'])
                  - B * FC['T'] * np.log(1 - _i_fc_nom_ref / S / FC['n_parallel'] / FC['j_L']))
UV_TO_PCT_FC = (1e-6 / _V_cell_ref_fc) * 100   # uV (sur cellule ~0.86 V) -> % de tension


def _fc_pmax(alpha_fc):
    """P_max de la PEMFC (vieillissement inclus) ; meme formule que la boucle."""
    i_fc_max = (-234.8032 * alpha_fc + 238.8252)
    return i_fc_max * FC['n_parallel'] * FC['n_series'] * (
        FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc_max / FC['n_parallel']
        - A * FC['T'] * np.log((i_fc_max / S / FC['n_parallel'] + j_in) / FC['j_0'])
        - B * FC['T'] * np.log(1 - i_fc_max / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))


def _fc_rates(f):
    """Taux a(f), b(f) (uV/h) et k(f) (1/h) en fonction de f = |P_fc|/P_fc_max.
    a : generation irreversible (Ostwald/ECSA, convexe en courant, Colombo/McCay).
    b : generation reversible (croit avec la charge).
    k : recuperation du reversible (rapide a l'arret, ~nulle en fonctionnement)."""
    p = FC_REC
    fc = min(max(f, 0.0), 1.0)
    a = p['a_irr'] * fc * fc
    b = p['b_rev'] * fc
    k = p['k_rest'] if fc <= FC_F_OFF else p['k_op']
    return a, b, k


def _fc_advance(V_irr, V_rev, P_curr, P_prev, P_max, Ts_h):
    """Avance le modele reversible/irreversible PEMFC d'un pas Ts (etats en uV).
    Retourne (V_irr, V_rev, d_startstop, d_idle) ; increments en uV.
    Symetrique de _ely_advance (meme conventions de seuils/telescopage)."""
    Pc = abs(P_curr)
    Pp = abs(P_prev)
    f = Pc / P_max if P_max > 0 else 0.0
    a, b, k = _fc_rates(f)
    V_irr = V_irr + a * Ts_h
    if k > 1e-12:
        Veq = b / k
        V_rev = Veq + (V_rev - Veq) * np.exp(-k * Ts_h)   # solution exacte sur le pas
    else:
        V_rev = V_rev + b * Ts_h
    th_start = 0.0005 * P_max
    d_ss = FC_REC['s'] if (Pp < th_start and Pc >= th_start) else 0.0
    th_idle = FC_F_IDLE * P_max
    th_off  = FC_F_OFF * P_max
    d_idle = FC_REC['idle'] * Ts_h if (th_off < Pc <= th_idle) else 0.0
    return V_irr, V_rev, d_ss, d_idle


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
    """Cout de degradation PEMFC (modele reversible/irreversible, cf. en-tete).

    Integre l'etat (V_irr, V_rev) depuis 0 sur le tableau fourni -- utilise tel
    quel par get_cost_total et les scripts batch. La BOUCLE n'appelle PAS cette
    fonction (etat reporte via _fc_advance) car le modele est stateful, comme
    l'ELY.

    Retourne : (cost_financial_EUR, deg_startstop_%, deg_idle_%,
                deg_reversible_%, deg_irreversible_%).
    Positions symetriques de get_cost_ely (le 4e/5e element = reversible/irrev).
    """
    P_fc = np.atleast_1d(np.abs(P_fc)).astype(float)
    alpha_fc = np.atleast_1d(alpha_fc).astype(float)
    n = len(P_fc)
    if n == 0:
        return 0, 0, 0, 0, 0
    Ts_h = LOAD['Ts'] / 3600.0
    Pmax = _fc_pmax(alpha_fc)
    Pmax = np.full(n, Pmax) if np.ndim(Pmax) == 0 else np.asarray(Pmax)
    if len(Pmax) != n:
        Pmax = np.full(n, Pmax.flat[0])

    V_irr = 0.0
    V_rev = 0.0
    V_ss = 0.0
    V_idle = 0.0
    P_prev = P_fc[0]
    for idx in range(n):
        V_irr, V_rev, d_ss, d_idle = _fc_advance(V_irr, V_rev, P_fc[idx], P_prev, Pmax[idx], Ts_h)
        V_ss += d_ss
        V_idle += d_idle
        P_prev = P_fc[idx]

    deg_irr  = V_irr  * UV_TO_PCT_FC
    deg_rev  = V_rev  * UV_TO_PCT_FC
    deg_ss   = V_ss   * UV_TO_PCT_FC
    deg_idle = V_idle * UV_TO_PCT_FC
    deg_pct  = deg_irr + deg_rev + deg_ss + deg_idle

    cost_financial = deg_pct / ((1 - FC['SoH_EoL']) * 100) * FC['cost']
    return cost_financial, deg_ss, deg_idle, deg_rev, deg_irr

def get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat):
    cost_bat = get_cost_bat(P_bat, SoC, SoH_bat)
    cost_fc  = get_cost_fc(alpha_fc, P_fc)[0]
    cost_ely = get_cost_ely(alpha_ely, P_ely)[0] 
    
    return cost_bat + cost_fc + cost_ely