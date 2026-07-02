import math
import numpy as np
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *

# ==========================================================================
#  RB2(Prop) : pre-charge previsionnelle PROPORTIONNELLE (modulation continue
#  de l'electrolyseur par la confiance dans le deficit prevu).
#
#  IDEE. Les variantes existantes coupent l'ELY en TOUT-OU-RIEN (P_ely_set=0
#  ou nominal). C'est cette bascule binaire qui rend la decision fragile au
#  bruit (clignotement -> start-stop) et qui a impose hysteresis + gel. Ici on
#  supprime la bascule : le setpoint ELY est module CONTINUMENT par le poids
#      w = Phi( net_pred / (TAU * sigma) )  dans (0,1)
#      P_ely_set = C_ELY * P_ely_max * (1 - w)
#  ou net_pred est l'energie nette prevue sur H_PRE et sigma l'incertitude de
#  prevision (backtest). w s'interprete comme P(deficit) temperee par TAU.
#
#  CE QUE CA CHANGE :
#   1. ANTI-CLIGNOTEMENT PAR CONSTRUCTION : le bruit sur net_pred fait varier
#      w de maniere lisse ; l'ELY ne passe par OFF que si w ~ 1 (deficit
#      quasi certain). Plus besoin de MIN_DWELL ni de zone morte.
#   2. AMPLITUDE UTILISEE, PAS SEULEMENT LE SIGNE : un petit deficit prevu
#      (< sigma) donne une pre-charge PARTIELLE (l'ELY est reduit, pas coupe).
#      Ces declenchements "doux" sont precisement ceux que la bande +-1sigma
#      de l'hysteresis bloquait (caveat du README de Predictions).
#   3. CONVERGENCE OMNISCIENTE : si sigma -> 0, w -> echelon -> on retombe
#      sur la pre-charge binaire omnisciente. Le levier ne peut pas "bloquer"
#      la valeur de la prevision quand celle-ci devient bonne.
#
#  TAU regle la douceur : TAU petit -> quasi binaire (reactif, moins robuste),
#  TAU grand -> tres doux (robuste, levier plus faible). Balayage via
#  bench_fable.py --sweep prop.
#
#  PROPRIETE METHODO : ENABLE=False ou prevision absente -> w force a 0 ->
#  RB2 socle EXACT (test nul). Base = socle cost-min 0.440/0.310 (reopt).
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
BETA_FC_BAT  = 0.0
BETA_ELY_BAT = 0.0

# --- Modulation proportionnelle ------------------------------------------------
TAU = 1.0               # temperature de la sigmoide (en unites de sigma)

_rng = np.random.default_rng(0)
_eps = 0.0              # etat AR(1) du bruit (utilise si NOISE_RHO > 0)


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision (1 seed / run Monte-Carlo)."""
    global _rng, _eps
    _rng = np.random.default_rng(seed)
    _eps = 0.0


def reset():
    """Reinitialise l'etat AR(1) du bruit (la modulation elle-meme est sans memoire)."""
    global _eps
    _eps = 0.0


def _phi(x):
    """CDF gaussienne standard."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _precharge_weight(P_tot_ref_future, SoC_t):
    """Poids de pre-charge w dans [0,1] : confiance dans un deficit net sur
    H_PRE. 0 = prevision neutre/absente (RB2 socle exact)."""
    if not (ENABLE and USE_FORECAST):
        return 0.0
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return 0.0
    if SoC_t >= SOC_TARGET:
        return 0.0
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    if NOISE_ENABLE:
        global _eps
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        xi = _rng.standard_normal()
        _eps = NOISE_RHO * _eps + math.sqrt(1.0 - NOISE_RHO ** 2) * xi if NOISE_RHO > 0.0 else xi
        net += (BIAS_E_KWH + sig_inj * _eps) * 1000.0

    sig_wh = TAU * SIGMA_E_KWH * 1000.0      # echelle de la sigmoide [Wh]
    if sig_wh <= 0.0:
        return 1.0 if net > 0.0 else 0.0     # TAU/sigma nul -> binaire exact
    return _phi(net / sig_wh)


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

    # --- AUGMENTATION PREVISION : modulation CONTINUE (seule modif vs socle) ---
    w = _precharge_weight(P_tot_ref_future, SoC_t)
    P_ely_set = P_ely_set * (1.0 - w)
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
