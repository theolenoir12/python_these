import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

# ==========================================================================
#  RB2(SoH)(Pred) : RB2(SoH) augmentee par la PREVISION 48h, EXACTEMENT comme
#  RB2(Pred) augmente RB2 nu.
#
#  BASELINE = RB2(SoH) : RB2 dont les setpoints sont MODULES PAR L'ETAT DE SANTE
#    P_fc_set  = 0.440 * FC_max * SoH_fc^0      (FC : pas de modulation)
#    P_ely_set = 0.320 * ELY_max * SoH_ely^0.5  (ELY : on baisse le setpoint
#                                                quand l'electrolyseur vieillit)
#
#  AUGMENTATION (identique a RB2(Pred)), UN SEUL LEVIER 100% PREVISIONNEL :
#    la PRE-CHARGE BATTERIE. Si la prevision annonce un DEFICIT NET sur l'horizon
#    proche H_PRE et que le SoC a de la marge -> on COUPE l'electrolyseur
#    (P_ely_set=0) pour que le surplus PV courant charge la BATTERIE (~95%
#    aller-retour) au lieu de partir dans la chaine H2 (lossy). On aborde le
#    creux a venir avec une marge de SoC -> moins de passages au plancher ->
#    moins de LPSP.
#
#  PROPRIETE METHODO : RB2(SoH)(Pred) ne differe de RB2(SoH) que par une fonction
#  de P_tot_ref_future. Prevision NEUTRE ou ENABLE/USE_FORECAST=False -> on
#  retombe EXACTEMENT sur RB2(SoH) (test nul) -> tout gain est ATTRIBUABLE a la
#  prevision, en plus de la modulation SoH deja presente dans la baseline.
#
#  Bruit + anti-clignotement : memes mecanismes/parametres que RB2(Pred)
#  (sigma=39.38 kWh @18h du backtest LSTM ; hysteresis +-M_SIGMA*sigma + gel
#  MIN_DWELL). cf. ../robustesse_bruit_prevision.txt.
# ==========================================================================

# --- Reglages ---
ENABLE       = True     # False -> RB2(SoH) pur a l'identique (test nul)
USE_FORECAST = True     # False -> idem (pas de levier sans prevision)
H_PRE        = 18       # horizon de pre-charge [pas = h], optimum diurne (RB2(Pred))
SOC_TARGET   = 0.99     # on ne pre-charge que si SoC < cette cible

# --- Bruit de prevision (incertitude REALISTE sur l'energie nette prevue) -----
# Memes parametres que RB2(Pred) : backtest LSTM a l'horizon H_PRE=18h
#     biais ~ -2.3 kWh (negligeable) | sigma ~ 39.4 kWh.
# net_pred = net_vrai + N(biais, sigma). NOISE_ENABLE=False -> omniscient (test nul).
NOISE_ENABLE = True
BIAS_E_KWH   = -2.32    # biais du backtest a 18h [kWh]
SIGMA_E_KWH  = 39.38    # ecart-type du backtest a 18h [kWh] (valeur de DESIGN)
# Sigma du bruit REELLEMENT INJECTE. None -> = SIGMA_E_KWH (cas nominal). Le
# DECOUPLER permet de tester la robustesse a une MISESTIMATION de sigma : la bande
# d'hysteresis reste calee sur SIGMA_E_KWH (design) tandis que le vrai bruit varie
# (cf. sens_pred_noise.py, ellipses de sensibilite).
SIGMA_INJECT_KWH = None

# --- Anti-clignotement (robustesse au bruit) : hysteresis a deux seuils -------
# Bande +-M_SIGMA*sigma sur l'energie nette prevue + maintien minimal MIN_DWELL.
# Optimum robuste herite de RB2(Pred) : M_SIGMA=1.0 / MIN_DWELL=12.
# HYST_ENABLE=False ou (M_SIGMA=0 et MIN_DWELL=0) -> decision binaire net>0.
HYST_ENABLE = True
M_SIGMA     = 1.0       # demi-largeur de bande = M_SIGMA * sigma [-]
MIN_DWELL   = 12        # duree minimale de maintien d'un etat [pas/h]

_rng      = np.random.default_rng(0)
_state_on = False       # etat courant de la pre-charge (ELY coupe ?)
_dwell    = 0           # compteur de maintien restant [pas]


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision. A appeler avant chaque run
    Monte-Carlo pour une realisation independante et reproductible."""
    global _rng
    _rng = np.random.default_rng(seed)


def reset():
    """Reinitialise l'etat de l'hysteresis. A APPELER avant chaque run (les
    workers d'un pool sont reutilises -> sinon l'etat fuit d'un run a l'autre)."""
    global _state_on, _dwell
    _state_on = False
    _dwell    = 0


def _precharge(P_tot_ref_future, SoC_t):
    """True s'il faut pre-charger la batterie : un deficit net est prevu sur
    l'horizon H_PRE et le SoC a de la marge. 100% fonction de la prevision."""
    global _state_on, _dwell
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= SOC_TARGET:
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    # Bruit de prevision : net_pred = net_vrai + N(biais, sigma_inject) (kWh -> Wh).
    if NOISE_ENABLE:
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        net += (BIAS_E_KWH + sig_inj * _rng.standard_normal()) * 1000.0

    if not HYST_ENABLE:
        return net > 0.0  # decision binaire d'origine (P_tot_ref>0 = deficit)

    # --- Hysteresis a deux seuils + maintien minimal -------------------------
    th = M_SIGMA * SIGMA_E_KWH * 1000.0  # demi-largeur de bande [Wh]
    if _dwell > 0:
        _dwell -= 1                       # etat gele
    elif (not _state_on) and net > th:
        _state_on = True;  _dwell = MIN_DWELL
    elif _state_on and net < -th:
        _state_on = False; _dwell = MIN_DWELL
    return _state_on


def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_tot_ref_future=None) :

    ######################### RULES ##########################
    # Setpoints RB2(SoH) : modules par l'etat de sante (ancrage baseline).
    P_fc_set  = 0.440 * FC['P_fc_max'] * SoH_fc_t ** 0
    P_ely_set = 0.320 * ELY['P_ely_max'] * SoH_ely_t ** 0.5

    # --- AUGMENTATION PREVISION : pre-charge batterie (seule modif vs RB2(SoH)) ---
    # On coupe l'ELY pour rediriger le surplus courant vers la batterie.
    if _precharge(P_tot_ref_future, SoC_t):
        P_ely_set = 0.0
    # ------------------------------------------------------------------------

    dt_h         = LOAD['Ts'] / 3600.0
    P_fc_h2_max  = max(E_h2_t, 0.0)               / dt_h * FC['eff']  * CONV['eta'] * 1000   # [W]
    P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0)   / dt_h / (ELY['eff'] * CONV['eta']) * 1000 # [W]

    if P_tot_ref_t > 0 :
        P_fc_avail = min(P_fc_set, P_fc_h2_max)
        if P_tot_ref_t > P_fc_avail :
            P_dc_fc_t  = P_fc_avail
            P_dc_bat_t = P_tot_ref_t - P_fc_avail
        else :
            P_dc_fc_t  = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_ely_t = 0
    if P_tot_ref_t < 0 :
        P_ely_avail = min(P_ely_set, P_ely_h2_max)
        if P_tot_ref_t < - P_ely_avail :
            P_dc_ely_t = - P_ely_avail
            P_dc_bat_t = P_tot_ref_t + P_ely_avail
        else :
            P_dc_ely_t = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_fc_t  = 0
    ##################################

    if 'FC' in defaillances :
        if P_tot_ref_t > 0 :
            P_dc_bat_t = P_tot_ref_t
    if 'ELY' in defaillances :
        if P_tot_ref_t < 0 :
            P_dc_bat_t = P_tot_ref_t

    action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t

    action, lol = get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t)

    return action, lol
