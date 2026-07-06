import math
import numpy as np
from scipy.optimize import linprog
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
import Common.get_lol as _gl
from Common.cost_fcn_total2 import (deg_cumul1, deg_cumul2, ELY_REC, UV_TO_PCT,
                                    FC_ALPHA_HIGH, FC_ALPHA_SHIFT, FC_ALPHA_ON_OFF,
                                    FC_FHIGH, ELY_F30, ELY_F60)

# ==========================================================================
#  MPC(LP) -- Commande predictive a horizon glissant (LP resolu a chaque pas).
#
#  PRINCIPE : a chaque pas, un programme lineaire optimise la trajectoire
#  (P_fc, P_ely, P_bat, delestage) sur les MPC_H prochaines heures a partir
#  de la prevision du profil net (fenetre P_tot_ref_future de la boucle
#  forecast), puis SEULE la premiere action est appliquee (horizon glissant).
#  La batterie reste la variable d'ajustement a l'execution, comme pour les
#  RB : l'action retournee est toujours equilibree et passe par get_lol, qui
#  ecrete et compte le lol EXACTEMENT comme pour les autres strategies
#  (aucune possibilite de sous-servir "en silence").
#
#  MODELE INTERNE (surrogat lineaire des modeles de degradation, calibre a
#  l'import sur les modeles REELS de Common/cost_fcn_total2) :
#    - batterie  : cout lineaire au throughput, pente moyenne de la table
#                  Cumulative_degradation_bat sur [SOC_MIN, SOC_MAX]
#                  (~85 EUR/MWh cycle) ; C-rate < 1 sur ce systeme -> pas de
#                  surcout C-rate.
#    - ELY       : usure nulle sous le genou 30 % Pmax, charniere (hinge)
#                  lineaire au-dela, calee sur le taux (a2+b2) a 60 % Pmax
#                  (~12 EUR/h) ; demarrages via cout sur |Delta P_ely|.
#                  NB : le vrai taux SATURE a 60 % ; la charniere continue de
#                  croitre -> le LP sur-evalue legerement > 60 % (conservatif,
#                  et coherent avec l'optimum reel <= genou).
#    - PEMFC     : charniere haute puissance au-dela de 80 % Pmax (calee pour
#                  retrouver le taux plein a 100 %), transitoires + on/off via
#                  cout sur |Delta P_fc| ; l'idling est traite a l'execution
#                  (bande morte -> 0, pas de fonctionnement < 1 % Pmax).
#    - rendements H2 constants (FC['eff'], ELY['eff']) comme pour les plafonds
#      RB2 ; la centrale applique les VRAIES LUT -> l'ecart modele/reel est
#      absorbe par get_lol et paye dans les metriques.
#    - valeurs TERMINALES : l'energie restante en fin d'horizon est creditee
#      (MPC_V_BAT, MPC_V_H2 EUR/kWh) pour eviter la vidange de fin d'horizon.
#      Garde anti-arbitrage verifiee a la construction : il ne doit pas etre
#      rentable de cycler batterie -> ELY juste pour le credit terminal.
#
#  PREVISION : fenetre omnisciente de la boucle + bruit AR(1) par pas (rho
#  MPC_NOISE_RHO), re-tire a chaque appel, calibre pour que l'erreur AGREGEE
#  a 18 h retrouve le backtest de RB2(Pred) : sigma = 39.38 kWh, biais =
#  -2.32 kWh (memes constantes de design que Fable_pred). Le pas courant
#  (k=0) est connu exactement, le bruit ne s'applique qu'a k >= 1.
#
#  CONVENTIONS BANC (identiques aux autres strategies) :
#      set_noise_seed(seed)  puis  reset()  avant chaque run.
#  Reglages surchargables par setattr (pattern bench_ultime).
# ==========================================================================

# --- Horizon et solveur ---------------------------------------------------
MPC_H            = 24      # horizon [pas = h] (<= H_forecast de la boucle)
MPC_VOLL         = 3.0     # penalite delestage INTERNE au LP [EUR/kWh]
                           # (l'evaluation reste VoLL=3 dans le banc ; ce
                           # bouton trace le front du MPC, cf --sweep pareto)

# --- Valeurs terminales [EUR/kWh] ------------------------------------------
MPC_V_BAT        = 0.60    # credit energie batterie en fin d'horizon
MPC_V_H2         = 1.00    # credit H2 en fin d'horizon (< VoLL*eta_chaine=1.35 :
                           # pas de thesaurisation pendant un delestage)

# --- Surrogats de degradation (echelles, 1.0 = calibration modele reel) ----
MPC_ELY_WEAR_SCALE = 1.0   # pente charniere ELY > genou 30 %
MPC_FC_WEAR_SCALE  = 1.0   # pente charniere FC > 80 %
MPC_SW_SCALE       = 1.0   # couts de commutation |Delta P| (FC et ELY)
MPC_C_BAT_SCALE    = 1.0   # cout throughput batterie

# --- Marges de securite (evitent l'ecretage systematique par get_lol) ------
MPC_SOC_MARGIN   = 0.005   # marge sur [SOC_MIN, soc_max_vieilli]
MPC_H2_MARGIN    = 0.5     # [kWh] marge sur [0, E_h2_init]

# --- Execution --------------------------------------------------------------
MPC_FC_MIN_FRAC  = 0.01    # bande morte FC  : f < 1 % Pmax (stack) -> 0
MPC_ELY_MIN_FRAC = 0.01    # bande morte ELY : e < 1 % Pmax (stack) -> 0
                           # (supprime idling FC/ELY et faux demarrages)
MPC_ELY_MIN_DWELL = 0      # gel on/off ELY [pas] (0 = off ; 12 = convention
                           # production si les demarrages explosent)

# --- Bruit de prevision (calibration backtest RB2(Pred), Fable_pred) --------
MPC_NOISE_ENABLE = True
MPC_NOISE_RHO    = 0.8     # correlation AR(1) horaire des erreurs par pas
BIAS_E_KWH       = -2.32   # biais AGREGE a 18 h [kWh]   (backtest)
SIGMA_E_KWH      = 39.38   # ecart-type AGREGE a 18 h [kWh] (backtest)
_SIGMA_AGG_H     = 18      # horizon d'agregation de la calibration [pas]

# --- Etat module (reinitialise par reset()) ---------------------------------
_rng        = np.random.default_rng(0)
_f_prev     = 0.0          # derniere P_dc_fc executee  [W] (continuite |Delta|)
_e_prev     = 0.0          # derniere |P_dc_ely| executee [W]
_ely_on     = False        # etat on/off ELY (gel optionnel)
_ely_dwell  = 0
LP_FAILURES = 0            # compteur d'echecs solveur (fallback RB2)


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision (1 seed / run Monte-Carlo)."""
    global _rng
    _rng = np.random.default_rng(seed)


def reset():
    """Reinitialise l'etat interne. A appeler avant chaque run."""
    global _f_prev, _e_prev, _ely_on, _ely_dwell, LP_FAILURES
    _f_prev = 0.0
    _e_prev = 0.0
    _ely_on = False
    _ely_dwell = 0
    LP_FAILURES = 0


# ==========================================================================
#  CALIBRATION (a l'import, sur les modeles reels)
# ==========================================================================
_ETA   = CONV['eta']                    # 0.9
_EFB   = BAT['eff']                     # 0.95
_KD    = 1.0 / (_ETA * _EFB)            # W DC decharge -> W "SoC"
_KC    = _ETA * _EFB                    # W DC charge   -> W "SoC"
_E_NOM = BAT['series_num'] * BAT['parallel_num'] * BAT['Q_bat'] * BAT['v_cell_nom']  # [Wh]

# Batterie : pente moyenne de la table cumulative sur la fenetre utile ->
# EUR par Wh d'energie cyclee (comptage cote SoC ; le SoH se simplifie).
_cu = np.interp([0.2, 0.995], deg_cumul1, deg_cumul2)
_C_BAT_EUR_WH = (abs(_cu[1] - _cu[0]) / 2.15 * 1e-6 / (1 - BAT['SoH_EoL'])
                 * BAT['cost']) / ((0.995 - 0.2) * _E_NOM)

# ELY : taux plein (a2+b2) en EUR/h, atteint a 60 % Pmax (rampe depuis 30 %).
_ELY_RATE_EUR_H = ((ELY_REC['a2'] + ELY_REC['b2']) * UV_TO_PCT
                   / ((1 - ELY['SoH_EoL']) * 100) * ELY['cost'])
_ELY_START_EUR  = ELY_REC['s'] * UV_TO_PCT / ((1 - ELY['SoH_EoL']) * 100) * ELY['cost']

# FC : taux haute puissance en EUR/h ; transitoire + on/off en EUR/cycle.
_FC_HIGH_EUR_H  = FC_ALPHA_HIGH / ((1 - FC['SoH_EoL']) * 100) * FC['cost']
_FC_SHIFT_EUR   = FC_ALPHA_SHIFT / ((1 - FC['SoH_EoL']) * 100) * FC['cost']
_FC_ONOFF_EUR   = FC_ALPHA_ON_OFF / ((1 - FC['SoH_EoL']) * 100) * FC['cost']

_EFF_FC_M  = FC['eff']    # rendements constants du MODELE interne
_EFF_ELY_M = ELY['eff']   # (la centrale applique les vraies LUT)

_TS_H = LOAD['Ts'] / 3600.0

# Garde anti-arbitrage batterie -> ELY -> credit H2 (voir en-tete).
def _check_no_arbitrage():
    gain_h2  = _ETA * _EFF_ELY_M * MPC_V_H2 / 1000.0          # EUR/Wh DC vers ELY
    cout_bat = _KD * (_C_BAT_EUR_WH * MPC_C_BAT_SCALE + MPC_V_BAT / 1000.0)
    if gain_h2 > cout_bat:
        import warnings
        warnings.warn(f"MPC(LP) : arbitrage batterie->ELY rentable "
                      f"(gain {gain_h2:.2e} > cout {cout_bat:.2e} EUR/Wh) ; "
                      f"baisser MPC_V_H2 ou monter MPC_V_BAT.")
_check_no_arbitrage()


_sig_cache = {}

def _noise_sigma_w():
    """Ecart-type PAR PAS [W] tel que l'erreur agregee sur _SIGMA_AGG_H pas
    (somme AR(1) de correlation MPC_NOISE_RHO) retrouve SIGMA_E_KWH."""
    key = (MPC_NOISE_RHO, SIGMA_E_KWH, _SIGMA_AGG_H)
    if key not in _sig_cache:
        m, rho = _SIGMA_AGG_H, MPC_NOISE_RHO
        coeff = m + 2.0 * sum((m - l) * rho ** l for l in range(1, m))
        _sig_cache[key] = SIGMA_E_KWH * 1000.0 / math.sqrt(coeff) / _TS_H
    return _sig_cache[key]


def _forecast(window):
    """Fenetre de prevision : verite + biais + bruit AR(1) par pas (k>=1)."""
    p = np.asarray(window, dtype=float).copy()
    if MPC_NOISE_ENABLE and len(p) > 1:
        sig = _noise_sigma_w()
        rho = MPC_NOISE_RHO
        eps = np.empty(len(p) - 1)
        e = _rng.standard_normal()
        eps[0] = e
        for k in range(1, len(eps)):
            e = rho * e + math.sqrt(1.0 - rho * rho) * _rng.standard_normal()
            eps[k] = e
        p[1:] += BIAS_E_KWH * 1000.0 / _SIGMA_AGG_H / _TS_H + sig * eps
    return p


# ==========================================================================
#  LP : variables par pas k (toutes cote DC, en W)
#     f  = P_dc_fc >= 0            e  = |P_dc_ely| >= 0
#     bd = decharge batterie       bc = charge batterie
#     s  = delestage (slack)       c  = ecretage PV (slack)
#     zf = depassement FC 80 %     ze = depassement ELY genou 30 %
#     df = |Delta f|               de = |Delta e|
#  Layout : x = [f | e | bd | bc | s | c | zf | ze | df | de], blocs de H.
# ==========================================================================
def _solve_lp(p, SoC_t, E_h2_t, E_h2_init, SoH_bat_t, P_fc_max_t, P_ely_max_t,
              soc_max_t):
    H = len(p)
    nv = 10 * H
    iF, iE, iBD, iBC, iS, iC, iZF, iZE, iDF, iDE = (np.arange(H) + i * H
                                                    for i in range(10))
    E_soh = _E_NOM * SoH_bat_t                       # [Wh]
    fcap  = 0.999 * _ETA * P_fc_max_t                # f : cote DC
    ecap  = 0.999 * P_ely_max_t / _ETA
    f80   = FC_FHIGH * _ETA * P_fc_max_t             # seuils charnieres (DC)
    e30   = ELY_F30 * P_ely_max_t / _ETA
    g_e   = _ETA * _EFF_ELY_M                        # W DC -> W H2 (ELY)
    g_f   = 1.0 / (_ETA * _EFF_FC_M)                 # W DC -> W H2 (FC)

    # --- Objectif [EUR] ------------------------------------------------------
    cthr = _C_BAT_EUR_WH * MPC_C_BAT_SCALE
    c = np.full(nv, 1e-9)                            # epsilon anti-degenerescence
    c[iBD] = (cthr + MPC_V_BAT / 1000.0) * _KD * _TS_H
    c[iBC] = (cthr - MPC_V_BAT / 1000.0) * _KC * _TS_H
    c[iS]  = MPC_VOLL / 1000.0 * _TS_H
    c[iF] += MPC_V_H2 / 1000.0 * g_f * _TS_H         # H2 consomme = credit perdu
    c[iE] -= MPC_V_H2 / 1000.0 * g_e * _TS_H         # H2 produit  = credit gagne
    c[iZF] = MPC_FC_WEAR_SCALE * _FC_HIGH_EUR_H * _TS_H / ((1 - FC_FHIGH) * _ETA * P_fc_max_t)
    c[iZE] = MPC_ELY_WEAR_SCALE * _ELY_RATE_EUR_H * _TS_H / ((ELY_F60 - ELY_F30) * P_ely_max_t / _ETA)
    c[iDF] = MPC_SW_SCALE * (_FC_SHIFT_EUR / (FC_FHIGH - 0.01) + _FC_ONOFF_EUR) / (_ETA * P_fc_max_t)
    c[iDE] = MPC_SW_SCALE * _ELY_START_EUR / (P_ely_max_t / _ETA)

    # --- Egalites : equilibre de puissance ------------------------------------
    A_eq = np.zeros((H, nv))
    r = np.arange(H)
    A_eq[r, iBD] = 1.0; A_eq[r, iBC] = -1.0
    A_eq[r, iF]  = 1.0; A_eq[r, iE]  = -1.0
    A_eq[r, iS]  = 1.0; A_eq[r, iC]  = -1.0
    b_eq = p.copy()

    # --- Inegalites ------------------------------------------------------------
    tril = np.tril(np.ones((H, H)))
    A_ub = np.zeros((8 * H, nv))
    b_ub = np.zeros(8 * H)

    # SoC : SoC_k+1 = SoC0 - cumsum(bd*KD - bc*KC)*Ts/E_soh dans [floor, ceil]
    ceil_soc  = soc_max_t - MPC_SOC_MARGIN
    floor_soc = _gl.SOC_MIN + MPC_SOC_MARGIN
    A_ub[0:H, iBC[0]:iBC[0]+H] = tril * (_KC * _TS_H / E_soh)
    A_ub[0:H, iBD[0]:iBD[0]+H] = -tril * (_KD * _TS_H / E_soh)
    b_ub[0:H] = ceil_soc - SoC_t
    A_ub[H:2*H] = -A_ub[0:H]
    b_ub[H:2*H] = SoC_t - floor_soc

    # H2 : E_k+1 = E0 + cumsum(e*g_e - f*g_f)*Ts/1000 dans [marge, Emax-marge]
    A_ub[2*H:3*H, iE[0]:iE[0]+H] = tril * (g_e * _TS_H / 1000.0)
    A_ub[2*H:3*H, iF[0]:iF[0]+H] = -tril * (g_f * _TS_H / 1000.0)
    b_ub[2*H:3*H] = (E_h2_init - MPC_H2_MARGIN) - E_h2_t
    A_ub[3*H:4*H] = -A_ub[2*H:3*H]
    b_ub[3*H:4*H] = E_h2_t - MPC_H2_MARGIN

    # Charnieres zf >= f - f80, ze >= e - e30
    A_ub[4*H:5*H, iF[0]:iF[0]+H]   = np.eye(H)
    A_ub[4*H:5*H, iZF[0]:iZF[0]+H] = -np.eye(H)
    b_ub[4*H:5*H] = f80
    A_ub[5*H:6*H, iE[0]:iE[0]+H]   = np.eye(H)
    A_ub[5*H:6*H, iZE[0]:iZE[0]+H] = -np.eye(H)
    b_ub[5*H:6*H] = e30

    # |Delta| : df_k >= +/-(f_k - f_km1), idem de (k=0 reference _f/_e_prev)
    D = np.eye(H) - np.eye(H, k=-1)
    A_ub[6*H:7*H, iF[0]:iF[0]+H]   = D
    A_ub[6*H:7*H, iDF[0]:iDF[0]+H] = -np.eye(H)
    b_ub[6*H] = _f_prev
    A_ub[7*H:8*H, iE[0]:iE[0]+H]   = D
    A_ub[7*H:8*H, iDE[0]:iDE[0]+H] = -np.eye(H)
    b_ub[7*H] = _e_prev
    # (le sens descendant est borne par le meme df via -D : on empile)
    A_dn = np.zeros((2 * H, nv))
    b_dn = np.zeros(2 * H)
    A_dn[0:H, iF[0]:iF[0]+H]   = -D
    A_dn[0:H, iDF[0]:iDF[0]+H] = -np.eye(H)
    b_dn[0] = -_f_prev
    A_dn[H:2*H, iE[0]:iE[0]+H]   = -D
    A_dn[H:2*H, iDE[0]:iDE[0]+H] = -np.eye(H)
    b_dn[H] = -_e_prev
    A_ub = np.vstack([A_ub, A_dn])
    b_ub = np.concatenate([b_ub, b_dn])

    bounds = [(0.0, None)] * nv
    for k in range(H):
        bounds[iF[k]] = (0.0, fcap)
        bounds[iE[k]] = (0.0, ecap)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if not res.success:
        return None
    return float(res.x[iF[0]]), float(res.x[iE[0]])


def _fallback_rb2(SoC_t, P_tot_ref_t, E_h2_t, E_h2_init):
    """Regles RB2 socle (0.440/0.310) si le LP echoue -- ne doit pas arriver."""
    P_fc_set  = 0.440 * FC['P_fc_max']
    P_ely_set = 0.310 * ELY['P_ely_max']
    P_fc_h2  = max(E_h2_t, 0.0) / _TS_H * FC['eff'] * _ETA * 1000
    P_ely_h2 = max(E_h2_init - E_h2_t, 0.0) / _TS_H / (ELY['eff'] * _ETA) * 1000
    if P_tot_ref_t > 0:
        f = min(P_fc_set, P_fc_h2)
        f = f if P_tot_ref_t > f else 0.0
        return f, 0.0
    e = min(P_ely_set, P_ely_h2)
    e = e if P_tot_ref_t < -e else 0.0
    return 0.0, e


def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_tot_ref_future=None) :

    global _f_prev, _e_prev, _ely_on, _ely_dwell, LP_FAILURES

    # Plafond SoC courant, lu dans Common.get_lol (coherence avec la centrale).
    soc_max_t = _gl.SOC_MAX - _gl.SOC_MAX_AGED_GAIN * (1.0 - SoH_bat_t)
    if soc_max_t < _gl.SOC_MIN + 0.1:
        soc_max_t = _gl.SOC_MIN + 0.1

    # --- Prevision + LP -------------------------------------------------------
    f0 = e0 = None
    if P_tot_ref_future is not None and len(P_tot_ref_future) >= 2:
        H = min(MPC_H, len(P_tot_ref_future))
        p = _forecast(P_tot_ref_future[:H])
        p[0] = P_tot_ref_t                       # pas courant connu exactement
        sol = _solve_lp(p, SoC_t, E_h2_t, E_h2_init, SoH_bat_t,
                        P_fc_max_t, P_ely_max_t, soc_max_t)
        if sol is not None:
            f0, e0 = sol
    if f0 is None:
        LP_FAILURES += 1
        f0, e0 = _fallback_rb2(SoC_t, P_tot_ref_t, E_h2_t, E_h2_init)

    # --- Execution -------------------------------------------------------------
    # Netting FC/ELY simultanes (garde ; le LP ne le fait pas a l'optimum)
    if f0 > 0.0 and e0 > 0.0:
        net = f0 - e0
        f0, e0 = (net, 0.0) if net >= 0.0 else (0.0, -net)

    # Bandes mortes (suppriment idling et micro-demarrages du modele reel)
    if f0 * 1.0 / _ETA < max(1.0, MPC_FC_MIN_FRAC * P_fc_max_t):
        f0 = 0.0
    if e0 * _ETA < MPC_ELY_MIN_FRAC * P_ely_max_t:
        e0 = 0.0

    # Gel on/off ELY optionnel (convention MIN_DWELL production)
    if MPC_ELY_MIN_DWELL > 0:
        want_on = e0 > 0.0
        if _ely_dwell > 0:
            _ely_dwell -= 1
            if want_on != _ely_on:
                e0 = 0.0 if not _ely_on else e0   # etat gele
                want_on = _ely_on
        elif want_on != _ely_on:
            _ely_on = want_on
            _ely_dwell = MPC_ELY_MIN_DWELL

    # Plafonds H2 du pas (RB2-style, conservatifs avec rendements nominaux)
    P_fc_h2_max  = max(E_h2_t, 0.0) / _TS_H * FC['eff'] * _ETA * 1000
    P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / _TS_H / (ELY['eff'] * _ETA) * 1000
    f0 = min(f0, P_fc_h2_max)
    e0 = min(e0, P_ely_h2_max)

    # La batterie prend TOUT le residu (meme surface d'action que les RB) :
    # get_lol ecrete et compte le lol exactement comme pour les autres.
    P_dc_fc_t  = f0
    P_dc_ely_t = -e0
    P_dc_bat_t = P_tot_ref_t - P_dc_fc_t - P_dc_ely_t

    if 'FC' in defaillances :
        if P_tot_ref_t > 0 :
            P_dc_bat_t = P_tot_ref_t
    if 'ELY' in defaillances :
        if P_tot_ref_t < 0 :
            P_dc_bat_t = P_tot_ref_t

    action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t

    action, lol = get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t)

    _f_prev = action[1]
    _e_prev = abs(action[2])

    return action, lol
