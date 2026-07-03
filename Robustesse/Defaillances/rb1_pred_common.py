"""
rb1_pred_common.py -- ETAPE 2 : RB1 augmentee par la PREVISION de profils de
puissance, evaluee sous DEFAILLANCE.
=======================================================================
SOURCE 100% ASCII (cf. robustesse_common.py).

Ce module fournit :
  1. make_rb1_pred(a, b, ...) : RB1(a,b) [seuils optimises a l'etape 1] AUGMENTEE
     d'un UNIQUE levier 100% previsionnel -- la PRE-CHARGE BATTERIE -- porte a
     l'identique de RB2(Pred)/get_optimal_action_RB.py :
        si la prevision annonce un DEFICIT NET sur l'horizon proche H_PRE, on
        COUPE l'ELY (on force tout le surplus courant vers la BATTERIE au lieu
        de la chaine H2, tres lossy) -> on aborde le creux a venir avec une
        marge de SoC -> moins de passages au plancher -> moins de LPSP.
     Bruit de prevision LSTM empirique (biais/sigma du backtest 18h) + hysteresis
     anti-clignotement (M_SIGMA, MIN_DWELL), exactement comme RB2(Pred).
     TEST NUL : prevision neutre / ENABLE=False -> retombe EXACTEMENT sur RB1(a,b),
     donc tout gain est ATTRIBUABLE a la prevision.

  2. run_week_pred(...) : clone de robustesse_common._run_week qui, en plus,
     CONSTRUIT la fenetre de prevision P_tot_ref_future = net[j:j+H] et la passe
     a la strategie, et gere l'etat (reset + seed du bruit) par semaine.

Le harness de defaillance (baseline RB2 figee, plafonnement panne + reroutage
batterie, refere get_lol) est IDENTIQUE a l'etude principale : seule la reaction
de la strategie change.
"""
import numpy as np
import robustesse_common as rc

I   = rc.I
eta = rc.eta
get_lol            = rc.get_lol
simulate_transition = rc.simulate_transition

# --- Parametres du bruit / hysteresis (== RB2(Pred), backtest LSTM 18h) -------
H_PRE_DEFAULT   = 18       # horizon de pre-charge [pas = h] (optimum diurne)
SOC_TARGET      = 0.99     # pas de pre-charge si SoC deja au plafond
BIAS_E_KWH      = -2.32    # biais backtest 18h [kWh]
SIGMA_E_KWH     = 39.38    # sigma backtest 18h [kWh] (valeur de DESIGN, cale l'hyst)
M_SIGMA_DEFAULT = 1.0      # demi-largeur de bande = M_SIGMA * sigma
MIN_DWELL_DEF   = 12       # duree minimale de maintien [pas/h]


# =============================================================================
# RB1(Pred) PARAMETREE : factory avec etat de bruit/hysteresis ENCAPSULE
# =============================================================================
# DEUX leviers pilotes par le MEME signal previsionnel `net_future` = energie
# nette voulue sur l'horizon H_PRE (== l'entree de RB2(Pred)) :
#
#   (1) PRE-CHARGE (charge)   -- le port direct de RB2(Pred). Si un deficit net
#       est prevu, on COUPE l'ELY -> le surplus courant reste en BATTERIE.
#       NB : redondant avec l'adaptativite-SoC de RB1 (RB1 garde deja tout le
#       surplus en batterie sous SoC_high) -> marge faible, garde pour memoire.
#
#   (2) RESERVE ANTICIPEE (decharge) -- LEVIER ORTHOGONAL, propre a RB1. RB1
#       decide le partage FC/batterie selon le SEUL SoC instantane : au-dessus de
#       SoC_high il vide la batterie (FC coupee). Si un deficit SOUTENU est prevu,
#       vider la batterie parce que le SoC est haut est imprudent -> on ELARGIT la
#       bande vers le haut (b -> b_reserve) pour ENGAGER LA FC plus tot et
#       PRESERVER une reserve batterie pour le creux a venir. Info absente du SoC.
#
# Les deux partagent l'etat d'hysteresis (deficit_ahead) : robuste au bruit LSTM,
# et si la prevision est neutre / enable=False -> RB1(a,b) nu (test nul preserve).
def make_rb1_pred(a, b, precharge=True, reserve=False, b_reserve=0.95,
                  h2_gate=0.0, enable=True, noise=True, hyst=True,
                  h_pre=H_PRE_DEFAULT, m_sigma=M_SIGMA_DEFAULT,
                  min_dwell=MIN_DWELL_DEF, sigma_inject=None):
    """RB1(a,b) augmentee des leviers previsionnels. Renvoie une fonction d'action
    acceptant P_tot_ref_future en 16e argument, DOTEE de deux methodes :
        .reset()            -> reinitialise l'hysteresis (a appeler par semaine) ;
        .set_noise_seed(s)  -> reseede le bruit (par tirage, reproductible).
    precharge/reserve : active chaque levier. enable=False (ou les deux off)
    -> RB1(a,b) nu (test nul). noise=False -> prevision OMNISCIENTE.
    hyst=False -> decision binaire net>0. b_reserve : seuil haut en mode reserve.
    h2_gate : le levier RESERVE (qui preserve la batterie en brulant du H2) ne
    s'active que si le reservoir est au-dessus de h2_gate*E_h2_init -> evite la
    famine FC sous panne ELY (H2 non reconstitue) et la fragilite au bruit."""
    a = float(a); b = float(b); b_res = float(b_reserve); h2g = float(h2_gate)
    dt_h   = I.LOAD['Ts'] / 3600.0
    sig_inj = SIGMA_E_KWH if sigma_inject is None else float(sigma_inject)
    th = m_sigma * SIGMA_E_KWH * 1000.0            # demi-bande hysteresis [Wh]

    st = {"rng": np.random.default_rng(0), "on": False, "dwell": 0}

    def reset():
        st["on"] = False; st["dwell"] = 0

    def set_noise_seed(s):
        st["rng"] = np.random.default_rng(s)

    def _deficit_ahead(P_future):
        """True si un DEFICIT net est prevu (confiant) sur l'horizon h_pre.
        Signal previsionnel commun aux deux leviers (independant du SoC)."""
        if not enable or P_future is None or len(P_future) == 0:
            return False
        net = float(np.sum(np.asarray(P_future[:h_pre], dtype=float))) * dt_h  # [Wh]
        if noise:
            net += (BIAS_E_KWH + sig_inj * st["rng"].standard_normal()) * 1000.0
        if not hyst:
            return net > 0.0
        if st["dwell"] > 0:
            st["dwell"] -= 1
        elif (not st["on"]) and net > th:
            st["on"] = True;  st["dwell"] = min_dwell
        elif st["on"] and net < -th:
            st["on"] = False; st["dwell"] = min_dwell
        return st["on"]

    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
                              P_tot_ref_future=None):
        deficit_ahead = _deficit_ahead(P_tot_ref_future)

        if P_tot_ref_t > 0:                         # --- DEFICIT (decharge) ---
            # LEVIER RESERVE : deficit prevu -> bande elargie (b -> b_res) donc FC
            # engagee plus tot et batterie preservee. GARDE H2 : on ne rationne la
            # batterie via la FC que si le reservoir a de la marge (sinon sous
            # panne ELY on affamerait la FC). Sinon RB1(a,b) nominal.
            reserve_on = (reserve and deficit_ahead and E_h2_t >= h2g * E_h2_init)
            b_eff = b_res if reserve_on else b
            if SoC_t <= a:
                frac = 0.0
            elif SoC_t >= b_eff:
                frac = 1.0
            else:
                frac = (SoC_t - a) / (b_eff - a)
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_fc_t  = P_tot_ref_t - P_dc_bat_t
            P_dc_ely_t = 0.0
        else:                                        # --- SURPLUS (charge) ---
            # LEVIER PRE-CHARGE : deficit prevu (et marge SoC) -> on COUPE l'ELY,
            # tout le surplus reste en batterie. Sinon RB1(a,b) nominal.
            if precharge and deficit_ahead and SoC_t < SOC_TARGET:
                frac = 1.0
            elif SoC_t <= b:
                frac = 1.0
            elif SoC_t >= 1.0:
                frac = 0.0
            else:
                frac = (1.0 - SoC_t) / (1.0 - b)
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_ely_t = P_tot_ref_t - P_dc_bat_t
            P_dc_fc_t  = 0.0

        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t

        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol

    get_optimal_action_RB.reset = reset
    get_optimal_action_RB.set_noise_seed = set_noise_seed
    return get_optimal_action_RB


# =============================================================================
# SIMULATION D'UNE SEMAINE AVEC PREVISION (clone de rc._run_week + forecast)
# =============================================================================
_P_REF = I.LOAD["P_ref"]
_P_PV  = I.PV["P"]
_NET   = _P_REF / eta - _P_PV                       # profil net global [W] (== _profile_net)


def run_week_pred(strat_func, baseline, t0, fc_derate, ely_derate, h_pre=H_PRE_DEFAULT):
    """Identique a rc._run_week mais construit P_tot_ref_future = net[j:j+h_pre]
    et le passe a la strategie (16e arg). Panne pendant WEEK_HOURS puis reparation
    jusqu'a EVAL_HOURS. fc_derate=ely_derate=1.0 -> contrefactuel SANS panne."""
    SoC_t   = float(baseline["SoC"][t0])
    E_h2_t  = float(baseline["E_h2"][t0])
    a_fc    = float(baseline["alpha_fc"][t0])
    a_ely   = float(baseline["alpha_ely"][t0])
    SoH_bat = float(baseline["SoH_bat"][t0])
    SoH_fc  = float(baseline["SoH_fc"][t0])
    SoH_ely = float(baseline["SoH_ely"][t0])

    P_fc_max0  = rc.p_fc_max_of_alpha(a_fc)
    P_ely_max0 = rc.p_ely_max_of_alpha(a_ely)
    lol_dummy = np.zeros(1)
    L = len(_NET)

    planned = 0.0
    unserved = 0.0
    for k in range(rc.EVAL_HOURS):
        j = t0 + k
        fc_d  = fc_derate  if k < rc.WEEK_HOURS else 1.0
        ely_d = ely_derate if k < rc.WEEK_HOURS else 1.0
        fc_cap  = fc_d  * P_fc_max0
        ely_cap = ely_d * P_ely_max0

        P_dc_load_t = _P_REF[j] / eta
        P_dc_pv_t   = _P_PV[j]
        P_tot_ref_t = P_dc_load_t - P_dc_pv_t

        P_future = _NET[j:min(j + h_pre, L)]        # fenetre de prevision (vrai futur)

        action_nom, _ = strat_func(
            SoC_t, P_tot_ref_t, [], lol_dummy, a_fc, a_ely, SoH_bat,
            E_h2_t, rc.E_H2_INIT, P_fc_max0, P_ely_max0,
            rc.RUL_FC_DEFAULT, rc.RUL_ELY_DEFAULT, SoH_fc, SoH_ely, P_future)

        P_dc_fc_t  = min(action_nom[1], fc_cap  * eta * 0.999) if action_nom[1] > 0 else 0.0
        P_dc_ely_t = max(action_nom[2], -ely_cap / eta * 0.999) if action_nom[2] < 0 else 0.0
        P_dc_bat_t = P_tot_ref_t - P_dc_fc_t - P_dc_ely_t

        action, lol = get_lol(
            SoC_t, (P_dc_bat_t, P_dc_fc_t, P_dc_ely_t), P_tot_ref_t, [],
            E_h2_t, rc.E_H2_INIT, P_fc_max0, P_ely_max0, SoH_bat)

        SoC_tp1, simOut = simulate_transition(
            SoC_t, action, P_tot_ref_t, rc.PLOT_FLAG, lol, a_fc, a_ely, SoH_bat,
            E_h2_t, rc.E_H2_INIT, P_fc_max0, P_ely_max0)
        if simOut:
            SoC_t = SoC_tp1
            E_h2_t = simOut["E_h2_tp1"]
        else:
            SoC_t = min(max(SoC_t, 0.2), 0.995)

        net = P_dc_load_t - P_dc_pv_t
        p = max(net, 0.0) / 1000.0
        r = max(net * (1.0 - lol), 0.0) / 1000.0
        planned  += p
        unserved += max(p - r, 0.0)

    lpsp = (unserved / planned * 100.0) if planned > 0 else 0.0
    ens  = unserved * rc.Ts_h
    return float(lpsp), float(ens)
