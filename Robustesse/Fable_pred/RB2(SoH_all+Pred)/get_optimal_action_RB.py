import math
import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
import Common.get_lol as _gl

# ==========================================================================
#  RB2(SoH_all+Pred) -- "l'ultime" : empilement complet des leviers valides.
#  (nomenclature : SoH_bat=plafond SoC ; SoH_H2=gammas ; SoH_all=les deux)
#
#      niveau 0 (socle)   : RB2 cost-min          (0.440 / 0.310)
#      niveau 1 (SoH H2)  : setpoints modules par l'etat de sante
#                           P_fc_set  = 0.440*Pmax*SoH_fc^1
#                           P_ely_set = 0.310*Pmax*SoH_ely^2
#      niveau 2 (SoH bat) : plafond de SoC vieillissant (Common/get_lol)
#                           soc_max(t) = 0.995 - g*(1 - SoH_bat), g = 0.2
#      niveau 3 (Pred)    : pre-charge batterie declenchee par une hysteresis
#                           probabiliste a +-1sigma sur l'energie nette prevue
#                           (P_HI = 0.84 / P_LO = 0.16 == M_SIGMA = 1.0),
#                           gel MIN_DWELL = 12 h (convention production).
#
#  ADAPTATION SPECIFIQUE (interaction niveaux 2 x 3) : la cible de pre-charge
#  suit le PLAFOND VIEILLI. Avec un SOC_TARGET fixe (0.99), la pre-charge
#  viserait une zone devenue interdite par le plafond abaisse -> ELY coupe
#  pour rien (la batterie ne peut plus absorber). SOC_TARGET_MODE="ceiling"
#  cale la cible sur soc_max(t) - marge, lue dans Common.get_lol pour rester
#  strictement coherente avec la contrainte appliquee par la boucle.
#
#  REFERENCES / CONTROLES (25 ans, VoLL=3, cf Fable/sweep_fable_unified.txt
#  et Predictions/reopt_sohpred.txt) :
#      ENABLE=False + g=0.2 ........ RB2(SoH) unifiee = 78.336  (test nul)
#      ENABLE=True  + g=0.0 ........ ~RB2(SoH+Pred) reopt = 77.667 (reference)
#      ENABLE=True  + g=0.2 ........ RB2 ULTIME : cible < 77.67, attendu
#                                    ~77.2-77.4 si la pre-charge conserve sa
#                                    valeur (-1.10 sur base RB2(SoH)).
#  PROPRIETE METHODO : chaque niveau se desactive proprement (test nul en
#  cascade) -> attribution de chaque increment.
# ==========================================================================

# --- Levier previsionnel (niveau 3) ---
ENABLE       = True     # False -> base unifiee EXACTE (test nul niveau 3)
USE_FORECAST = True
H_PRE        = 18       # horizon de pre-charge [pas = h] (optimum diurne herite)

# Cible de pre-charge :
#   "ceiling" -> cible = soc_max_vieilli(t) - SOC_TARGET_MARGIN  (defaut)
#   "fixed"   -> cible = SOC_TARGET
SOC_TARGET_MODE   = "ceiling"
SOC_TARGET_MARGIN = 0.005
SOC_TARGET        = 0.99

# --- Bruit de prevision (conventions RB2(Pred)) --------------------------------
NOISE_ENABLE = True
BIAS_E_KWH   = -2.32    # biais backtest a 18h [kWh]
SIGMA_E_KWH  = 39.38    # ecart-type backtest a 18h [kWh] (valeur de DESIGN)
SIGMA_INJECT_KWH = None # None -> = SIGMA_E_KWH ; sinon test de misestimation
NOISE_RHO    = 0.0      # correlation AR(1) (0 = iid, retro-compatible)

# --- Declencheur probabiliste (equivalent bande +-1sigma) ----------------------
P_HI      = 0.84        # entrer en pre-charge si P(deficit) > P_HI  (== +1sigma)
P_LO      = 0.16        # sortir si P(deficit) < P_LO                (== -1sigma)
MIN_DWELL = 12          # gel apres bascule [pas/h] (0 teste equivalent, cf
                        # sweep_fable_proba ; 12 = convention production,
                        # -150 demarrages ELY)

# --- Base unifiee (niveaux 0-2) -------------------------------------------------
C_FC_BASE   = 0.440
C_ELY_BASE  = 0.310
GAMMA_FC    = 1.0       # SoH_fc  -> setpoint FC   (optimum sweep_soh_attribution)
GAMMA_ELY   = 2.0       # SoH_ely -> setpoint ELY  (idem)
# Plafond SoC vieillissant : APPLIQUE PAR Common/get_lol (SOC_MAX_AGED_GAIN).
# Le bench le fixe par tache via "_lol:SOC_MAX_AGED_GAIN" ; en run standalone,
# main.py copie SOC_WIN_GAIN ci-dessous dans Common.get_lol. La strategie LIT
# toujours la valeur effective dans Common.get_lol (jamais celle-ci) pour que
# cible de pre-charge et contrainte restent coherentes.
SOC_WIN_GAIN = 0.2

_rng      = np.random.default_rng(0)
_state_on = False       # etat courant de la pre-charge (ELY coupe ?)
_dwell    = 0           # compteur de maintien restant [pas]
_eps      = 0.0         # etat AR(1) du bruit


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision (1 seed / run Monte-Carlo)."""
    global _rng, _eps
    _rng = np.random.default_rng(seed)
    _eps = 0.0


def reset():
    """Reinitialise l'etat de l'hysteresis + AR(1). A appeler avant chaque run."""
    global _state_on, _dwell, _eps
    _state_on = False
    _dwell    = 0
    _eps      = 0.0


def _phi(x):
    """CDF gaussienne standard."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _soc_target(SoH_bat_t):
    """Cible de pre-charge : le plafond vieilli EFFECTIF (lu dans Common.get_lol,
    donc coherent avec la contrainte), moins une marge -- ou une valeur fixe."""
    if SOC_TARGET_MODE == "fixed":
        return SOC_TARGET
    soc_max_t = _gl.SOC_MAX - _gl.SOC_MAX_AGED_GAIN * (1.0 - SoH_bat_t)
    if soc_max_t < _gl.SOC_MIN + 0.1:
        soc_max_t = _gl.SOC_MIN + 0.1
    return soc_max_t - SOC_TARGET_MARGIN


def _precharge(P_tot_ref_future, SoC_t, SoH_bat_t):
    """True s'il faut pre-charger : hysteresis probabiliste +-1sigma sur
    l'energie nette prevue, cible calee sur le plafond vieilli."""
    global _state_on, _dwell
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= _soc_target(SoH_bat_t):
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    if NOISE_ENABLE:
        global _eps
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        xi = _rng.standard_normal()
        _eps = NOISE_RHO * _eps + math.sqrt(1.0 - NOISE_RHO ** 2) * xi if NOISE_RHO > 0.0 else xi
        net += (BIAS_E_KWH + sig_inj * _eps) * 1000.0

    sig_wh = SIGMA_E_KWH * 1000.0
    if sig_wh <= 0.0:
        p_def = 1.0 if net > 0.0 else 0.0    # sigma nul -> omniscient binaire
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
    # Base unifiee : setpoints H2 modules par l'etat de sante (niveau 1) ;
    # le plafond SoC vieillissant (niveau 2) est applique par get_lol.
    P_fc_set  = C_FC_BASE  * FC['P_fc_max']   * SoH_fc_t  ** GAMMA_FC
    P_ely_set = C_ELY_BASE * ELY['P_ely_max'] * SoH_ely_t ** GAMMA_ELY

    # --- Niveau 3 : pre-charge previsionnelle (seule modif vs base unifiee) ---
    if _precharge(P_tot_ref_future, SoC_t, SoH_bat_t):
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
