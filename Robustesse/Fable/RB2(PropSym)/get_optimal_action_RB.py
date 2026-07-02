import math
import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *

# ==========================================================================
#  RB2(PropSym) : RB2(Prop) + levier previsionnel SYMETRIQUE (pre-decharge).
#
#  RB2(Prop) ne traite que les DEFICITS prevus (pre-charge batterie en coupant
#  l'ELY). Le miroir manque : avant un GROS SURPLUS prevu, une batterie deja
#  haute sature a SOC_MAX -> le surplus est ecrete (lol de curtailment) et
#  l'energie est perdue pour la chaine H2. Le levier symetrique FAIT DE LA
#  PLACE : pendant les heures de deficit qui PRECEDENT un surplus confiant,
#  on reduit le setpoint FC pour que la BATTERIE couvre une part plus grande
#  du deficit -> on aborde le surplus avec un SoC plus bas -> moins de
#  saturation haute, plus de surplus absorbe (batterie + ELY), H2 economise
#  (la FC tourne moins, le rendement batterie ~95% >> chaine H2).
#
#  MECANISME (meme machinerie probabiliste que la pre-charge) :
#      w_pre = Phi( +net_pred / (TAU     * sigma) )   -> P_ely_set *= (1 - w_pre)
#      w_sym = Phi( -net_pred / (TAU_SYM * sigma) )   -> P_fc_set  *= (1 - w_sym)
#  avec les gardes :
#      w_pre actif seulement si SoC < SOC_TARGET      (comme RB2(Prop))
#      w_sym actif seulement si SoC > SOC_SYM_FLOOR   (marge avant le plancher :
#            couper la FC a SoC bas creuserait le LPSP, exactement ce qu'on
#            veut eviter).
#  net_pred > 0 => w_sym ~ 0 et net_pred < 0 => w_pre ~ 0 : les deux leviers
#  sont naturellement exclusifs (un seul agit a la fois).
#
#  ATTRIBUTION : SYM_ENABLE=False -> RB2(Prop) exact (meme tirage de bruit) ;
#  ENABLE=False -> RB2 socle exact (test nul complet). La difference
#  RB2(PropSym) - RB2(Prop) mesure donc le levier symetrique SEUL.
#  Balayage (TAU_SYM, SOC_SYM_FLOOR) : bench_fable.py --sweep sym.
# ==========================================================================

# --- Reglages ---
ENABLE       = True     # False -> RB2 socle a l'identique (test nul)
USE_FORECAST = True     # False -> idem
H_PRE        = 18       # horizon de decision [pas = h] (optimum diurne, herite)
SOC_TARGET   = 0.99     # pre-charge seulement si SoC < cette cible

# --- Levier symetrique ----------------------------------------------------------
SYM_ENABLE    = True    # False -> RB2(Prop) exact
TAU_SYM       = 1.0     # temperature de la sigmoide de pre-decharge
SOC_SYM_FLOOR = 0.50    # pre-decharge seulement si SoC > ce plancher (marge LPSP)

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
BETA_FC_BAT  = 0.0
BETA_ELY_BAT = 0.0

# --- Modulation proportionnelle (pre-charge, heritee de RB2(Prop)) -------------
TAU = 1.0               # temperature de la sigmoide de pre-charge

_rng = np.random.default_rng(0)
_eps = 0.0              # etat AR(1) du bruit (utilise si NOISE_RHO > 0)


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision (1 seed / run Monte-Carlo)."""
    global _rng, _eps
    _rng = np.random.default_rng(seed)
    _eps = 0.0


def reset():
    """Reinitialise l'etat AR(1) du bruit (les modulations sont sans memoire)."""
    global _eps
    _eps = 0.0


def _phi(x):
    """CDF gaussienne standard."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _weights(P_tot_ref_future, SoC_t):
    """(w_pre, w_sym) dans [0,1] : confiance en deficit / en surplus sur H_PRE.
    UN SEUL tirage de bruit par pas (meme prevision pour les deux leviers) ->
    avec SYM_ENABLE=False la sequence aleatoire est identique a RB2(Prop)."""
    if not (ENABLE and USE_FORECAST):
        return 0.0, 0.0
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return 0.0, 0.0
    if (not SYM_ENABLE) and SoC_t >= SOC_TARGET:
        # RB2(Prop) ne TIRE PAS de bruit quand SoC >= SOC_TARGET (retour avant
        # tirage) : on reproduit ce chemin pour que SYM_ENABLE=False donne une
        # sequence aleatoire STRICTEMENT identique a RB2(Prop).
        return 0.0, 0.0
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    if NOISE_ENABLE:
        global _eps
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        xi = _rng.standard_normal()
        _eps = NOISE_RHO * _eps + math.sqrt(1.0 - NOISE_RHO ** 2) * xi if NOISE_RHO > 0.0 else xi
        net += (BIAS_E_KWH + sig_inj * _eps) * 1000.0

    sig_wh = SIGMA_E_KWH * 1000.0
    # Pre-charge (deficit prevu), gardee par SOC_TARGET
    if SoC_t >= SOC_TARGET or sig_wh <= 0.0 or TAU <= 0.0:
        w_pre = (1.0 if net > 0.0 else 0.0) if SoC_t < SOC_TARGET else 0.0
    else:
        w_pre = _phi(net / (TAU * sig_wh))
    # Pre-decharge (surplus prevu), gardee par SOC_SYM_FLOOR
    if not SYM_ENABLE or SoC_t <= SOC_SYM_FLOOR:
        w_sym = 0.0
    elif sig_wh <= 0.0 or TAU_SYM <= 0.0:
        w_sym = 1.0 if net < 0.0 else 0.0
    else:
        w_sym = _phi(-net / (TAU_SYM * sig_wh))
    return w_pre, w_sym


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

    # --- AUGMENTATION PREVISION : modulations CONTINUES (seules modifs vs socle) -
    w_pre, w_sym = _weights(P_tot_ref_future, SoC_t)
    P_ely_set = P_ely_set * (1.0 - w_pre)   # pre-charge  : freiner l'ELY
    P_fc_set  = P_fc_set  * (1.0 - w_sym)   # pre-decharge : freiner la FC
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
