import math
import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *

# ==========================================================================
#  RB2(Proba) : pre-charge previsionnelle declenchee par la PROBABILITE de
#  deficit, au lieu d'une bande d'energie fixe.
#
#  IDEE. RB2(Pred) filtre le bruit par une hysteresis en ENERGIE de demi-
#  largeur M_SIGMA*sigma. Or on CONNAIT sigma (backtest LSTM) : on peut donc
#  raisonner directement en PROBABILITE. Avec net_pred ~ N(net_vrai, sigma) :
#      P(deficit) = P(net_vrai > 0 | net_pred) = Phi(net_pred / sigma)
#  et on declenche/relache la pre-charge par une hysteresis SUR CETTE
#  PROBABILITE :
#      - ENTRER en pre-charge (ELY coupe)  si  P(deficit) > P_HI
#      - SORTIR (ELY rallume)              si  P(deficit) < P_LO
#      - entre les deux : garder l'etat courant (zone morte).
#
#  EQUIVALENCE / DIFFERENCE avec l'hysteresis en energie :
#    P_HI/P_LO symetriques (p, 1-p) <=> bande +-Phi^-1(p)*sigma. Par ex.
#    P_HI=0.84 / P_LO=0.16 == M_SIGMA=1.0. L'interet est triple :
#      1. la bande se REGLE en unites d'erreur de prevision (probabilite),
#         donc si sigma change (autre modele, autre horizon, sigma dependant
#         du regime meteo), le reglage reste valable SANS re-sweep ;
#      2. on peut REDUIRE la bande (P_HI=0.60/0.70) : l'hypothese testee est
#         qu'une bande plus etroite SANS gel recupere les declenchements que
#         la bande +-1sigma bloquait (cf. caveat "plateau omniscient" du
#         README de Predictions), tout en gardant assez de rejet du bruit ;
#      3. les seuils peuvent etre ASYMETRIQUES (confiance pour agir differente
#         de la confiance pour arreter).
#    Si sigma -> 0 (prevision parfaite), Phi(net/sigma) -> echelon : on
#    RETOMBE sur la decision omnisciente binaire (le caveat disparait par
#    construction).
#
#  PROPRIETE METHODO (identique a RB2(Pred)) : ENABLE=False ou prevision
#  absente -> RB2 socle EXACT (test nul, tout gain attribuable a la prevision).
# ==========================================================================

# --- Reglages ---
ENABLE       = True     # False -> RB2 socle a l'identique (test nul)
USE_FORECAST = True     # False -> idem
H_PRE        = 18       # horizon de pre-charge [pas = h] (optimum diurne, herite)
SOC_TARGET   = 0.99     # on ne pre-charge que si SoC < cette cible

# --- Bruit de prevision (memes conventions que RB2(Pred)) ---------------------
NOISE_ENABLE = True
BIAS_E_KWH   = -2.32    # biais du backtest a 18h [kWh]
SIGMA_E_KWH  = 39.38    # ecart-type backtest a 18h [kWh] (valeur de DESIGN)
SIGMA_INJECT_KWH = None # None -> = SIGMA_E_KWH ; sinon test de misestimation
# Correlation temporelle du bruit (AR(1)) : eps_t = rho*eps_{t-1} + sqrt(1-rho^2)*xi_t.
# rho=0.0 (defaut) -> iid strictement identique ; les fenetres 18 h consecutives se
# recouvrant a 17/18, l'erreur reelle est tres autocorrelee (rho ~0.9+ plausible).
NOISE_RHO = 0.0

# --- Socle RB2 cost-min (comparaison honnete, cf reopt_pred.txt) --------------
C_FC_BASE   = 0.440
C_ELY_BASE  = 0.310
GAMMA_FC    = 0.0       # exposant SoH_fc  (0 = socle nu ; >0 = variante SoH)
GAMMA_ELY   = 0.0       # exposant SoH_ely (0 = socle nu ; >0 = variante SoH)

# --- Hooks SoH_bat (cross-modulation, OFF par defaut : test nul preserve) -----
# Quand la batterie vieillit (SoH_bat < 1), on DEPLACE du flux vers la chaine
# H2 en remontant les setpoints FC/ELY d'un facteur SoH_bat^(-BETA). BETA=0 ->
# facteur 1 -> socle exact. Cf. README_fable.txt section 3.
BETA_FC_BAT  = 0.0
BETA_ELY_BAT = 0.0

# --- Declencheur probabiliste --------------------------------------------------
P_HI      = 0.70        # on ENTRE en pre-charge si P(deficit) > P_HI
P_LO      = 0.30        # on SORT si P(deficit) < P_LO
MIN_DWELL = 0           # duree minimale de maintien [pas/h] (0 = pas de gel ;
                        # hypothese : la zone morte probabiliste suffit)

_rng      = np.random.default_rng(0)
_state_on = False       # etat courant de la pre-charge (ELY coupe ?)
_dwell    = 0           # compteur de maintien restant [pas]
_eps      = 0.0         # etat AR(1) du bruit (utilise si NOISE_RHO > 0)


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision (1 seed / run Monte-Carlo)."""
    global _rng, _eps
    _rng = np.random.default_rng(seed)
    _eps = 0.0


def reset():
    """Reinitialise l'etat de l'hysteresis. A APPELER avant chaque run (les
    workers d'un pool sont reutilises -> sinon l'etat fuit d'un run a l'autre)."""
    global _state_on, _dwell, _eps
    _state_on = False
    _dwell    = 0
    _eps      = 0.0


def _phi(x):
    """CDF gaussienne standard."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _precharge(P_tot_ref_future, SoC_t):
    """True s'il faut pre-charger : hysteresis sur P(deficit) = Phi(net/sigma)."""
    global _state_on, _dwell
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= SOC_TARGET:
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    if NOISE_ENABLE:
        global _eps
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        xi = _rng.standard_normal()
        _eps = NOISE_RHO * _eps + math.sqrt(1.0 - NOISE_RHO ** 2) * xi if NOISE_RHO > 0.0 else xi
        net += (BIAS_E_KWH + sig_inj * _eps) * 1000.0

    sig_wh = SIGMA_E_KWH * 1000.0            # incertitude de DESIGN [Wh]
    if sig_wh <= 0.0:
        p_def = 1.0 if net > 0.0 else 0.0    # sigma nul -> decision binaire exacte
    else:
        p_def = _phi(net / sig_wh)

    if _dwell > 0:
        _dwell -= 1                          # etat gele
    elif (not _state_on) and p_def > P_HI:
        _state_on = True;  _dwell = MIN_DWELL
    elif _state_on and p_def < P_LO:
        _state_on = False; _dwell = MIN_DWELL
    return _state_on


def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_tot_ref_future=None) :

    ######################### RULES ##########################
    # Setpoints socle (+ modulations SoH optionnelles, OFF par defaut)
    P_fc_set  = C_FC_BASE  * FC['P_fc_max']   * SoH_fc_t  ** GAMMA_FC
    P_ely_set = C_ELY_BASE * ELY['P_ely_max'] * SoH_ely_t ** GAMMA_ELY
    if BETA_FC_BAT != 0.0 or BETA_ELY_BAT != 0.0:
        # Cross-modulation SoH_bat : report de flux vers la chaine H2 quand la
        # batterie vieillit. Cap au max physique vieilli (cote DC) pour ne pas
        # generer de lol_pmax artificiel dans get_lol.
        P_fc_set  = min(P_fc_set  * SoH_bat_t ** (-BETA_FC_BAT),  P_fc_max_t  * CONV['eta'])
        P_ely_set = min(P_ely_set * SoH_bat_t ** (-BETA_ELY_BAT), P_ely_max_t / CONV['eta'])

    # --- AUGMENTATION PREVISION : pre-charge batterie (seule modif vs socle) ---
    if _precharge(P_tot_ref_future, SoC_t):
        P_ely_set = 0.0
    # ---------------------------------------------------------------------------

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
