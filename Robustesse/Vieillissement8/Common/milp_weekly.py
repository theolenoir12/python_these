"""
=============================================================================
MILP PAR BLOCS HEBDOMADAIRES — Simulation 5 ans, Front de Pareto
=============================================================================

PRINCIPE
--------
L'horizon de 5 ans est découpé en blocs de N_week heures (défaut 168h = 1 semaine).
Pour chaque bloc :
  1. Les SoH (FC, ELY, BAT) sont FIGÉS à leur valeur de début de bloc.
  2. Un MILP GLOBAL est résolu sur les N_week heures → trajectoire OPTIMALE.
  3. Les SoH sont mis à jour en fin de bloc via get_soh() (code existant).
  4. L'état (SoC, E_h2, u_fc, u_ely) est propagé exactement au bloc suivant.

CORRECTIONS v2
--------------
  - Contrainte terminale SoC transformée en slack pénalisé (plus d'infaisabilité)
  - build_deg_costs : toutes les divisions protégées (SoH=0, P_fc_max=0)
  - SoH_bat initialisé à 1 partout (évite division par zéro dans get_cost_bat)
  - P_dc_load stocké côté réseau (cohérent avec main_plot)
  - Fallback 50/50 : état final (u_fc_prev, etc.) correctement propagé

FRONT DE PARETO
---------------
Pour chaque epsilon ∈ [0, 1] :  min  ε·J_deg + (1-ε)·J_lol
Chaque point = 1 simulation complète 5 ans (260 résolutions MILP 168h).

FIGURES PRODUITES PAR POINT DU FRONT
--------------------------------------
  all_aging.pdf, everything_combined_v2.pdf, pareto_front.pdf
=============================================================================
"""

import numpy as np
import os
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix, vstack as sp_vstack

# ---------------------------------------------------------------------------
# Chemin vers Common/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, '..')))

from Common.Init_EMR_MG_v16_python import (
    LOAD, PV, FC, ELY, BAT, CONV,
    A, B, S, j_0, j_L, j_in,
)
from Common.cost_fcn_total2 import get_cost_bat, get_cost_fc, get_cost_ely, get_cost_total
from Common.get_soh import get_soh


# =============================================================================
# SECTION 1 — PARAMÈTRES PHYSIQUES ET COÛTS DE DÉGRADATION
# =============================================================================

def compute_P_fc_max(alpha):
    """P_fc_max [W] en fonction du vieillissement alpha."""
    alpha = float(np.clip(alpha, 0.0, 0.9999))
    i_max = (-194.3950 * alpha + 196.5598) * FC['n_parallel']
    i_max = max(i_max, 0.0)
    if i_max < 1e-6:
        return 0.0
    return float(i_max * FC['n_series'] * (
        FC['E_0']
        - FC['R'] * (1 + alpha) * i_max / FC['n_parallel']
        - A * FC['T'] * np.log((i_max / S / FC['n_parallel'] + j_in) / j_0)
        - B * FC['T'] * np.log(1 - i_max / S / FC['n_parallel'] / j_L / (1 - alpha))
    ))


def compute_P_ely_max(alpha):
    """P_ely_max [W] en fonction du vieillissement alpha."""
    alpha = float(np.clip(alpha, 0.0, 0.9999))
    i_max = (-219.9 * alpha + 219.9) * ELY['n_parallel']
    i_max = max(i_max, 0.0)
    if i_max < 1e-6:
        return 0.0
    return float(i_max * ELY['n_series'] * (
        ELY['E_0']
        + ELY['R'] * (1 + alpha) * i_max / ELY['n_parallel']
        + A * ELY['T'] * np.log((i_max / S / ELY['n_parallel'] + j_in) / j_0)
        + B * ELY['T'] * np.log(1 - i_max / S / ELY['n_parallel'] / j_L / (1 - alpha))
    ))


def _safe_coef(v, name=''):
    """Retourne v si fini, sinon 0 avec avertissement."""
    if not np.isfinite(float(v)):
        warnings.warn(f"build_deg_costs: {name}={v} → remplacé par 0")
        return 0.0
    return float(v)


def build_deg_costs(P_fc_max, P_ely_max, SoH_bat, Ts_h=1.0):
    """
    Coûts unitaires de dégradation cohérents avec cost_fcn_total2.py.
    Toutes les divisions sont protégées contre les valeurs nulles ou négatives.
    """
    # ---- BATTERIE ----
    Q_bat  = BAT['Q_bat'];  v_nom = BAT['v_cell_nom']
    n_ser  = BAT['series_num']; n_par = BAT['parallel_num']
    SoH_b  = max(float(SoH_bat), BAT['SoH_EoL'] + 0.01)
    C_Wh   = max(Q_bat * v_nom * n_ser * n_par * SoH_b, 1.0)
    # coût en €/Wh échangé (stack)
    c_bat  = _safe_coef(
        (Q_bat * SoH_b * n_par / 2.15) * 1e-6
        / (0.4 * Q_bat * n_par) * BAT['cost'] / C_Wh, 'c_bat')

    # ---- FC ----
    pfc    = max(float(P_fc_max), 1.0)
    pg_fc  = (1 - FC['SoH_EoL']) * 100          # 20 %
    Phi    = 0.80 * pfc;  Plo = 0.01 * pfc
    den_fc = max(Phi - Plo, 1.0)
    c_fc_ss = _safe_coef(1.96e-3 / pg_fc * FC['cost'],              'c_fc_ss')
    c_fc_hi = _safe_coef(1.47e-3 * Ts_h / pg_fc * FC['cost'],       'c_fc_hi')
    c_fc_lo = _safe_coef(1.26e-3 * Ts_h / pg_fc * FC['cost'],       'c_fc_lo')
    c_fc_tr = _safe_coef(5.93e-5 / pg_fc * FC['cost'] / den_fc,     'c_fc_tr')

    # ---- ELY ----
    pely   = max(float(P_ely_max), 1.0)
    pg_ely = (1 - ELY['SoH_EoL']) * 100         # 20 %
    uv2pct = 1e-6 / 1.5 * 100
    c_ely_ss = _safe_coef(44.4  * uv2pct / pg_ely * ELY['cost'],              'c_ely_ss')
    c_ely_hi = _safe_coef(196.0 * uv2pct * Ts_h / pg_ely * ELY['cost'],       'c_ely_hi')
    c_ely_id = _safe_coef(1.5   * uv2pct * Ts_h / pg_ely * ELY['cost'],       'c_ely_id')
    c_ely_tr = _safe_coef((66.0+16.0)/2 * uv2pct * Ts_h / pg_ely * ELY['cost'], 'c_ely_tr')

    return {
        'C_bat_Wh' : C_Wh,
        'c_bat_Wh' : c_bat,
        'c_fc_ss'  : c_fc_ss,  'c_fc_hi' : c_fc_hi,
        'c_fc_lo'  : c_fc_lo,  'c_fc_tr' : c_fc_tr,
        'P_fc_hi'  : 0.80 * pfc,  'P_fc_lo' : 0.01 * pfc,
        'c_ely_ss' : c_ely_ss, 'c_ely_hi': c_ely_hi,
        'c_ely_id' : c_ely_id, 'c_ely_tr': c_ely_tr,
        'P_ely_hi' : 0.60 * pely, 'P_ely_lo': 0.01 * pely,
    }


# =============================================================================
# SECTION 2 — CONSTRUCTION ET RÉSOLUTION DU MILP HEBDOMADAIRE
# =============================================================================

def solve_week_milp(
    P_load_w, P_pv_w,
    P_fc_max, P_ely_max, C_bat_Wh,
    E_h2_max, soc_init, e_h2_init,
    costs, epsilon,
    u_fc_prev=0, u_ely_prev=0,
    p_fc_prev=0.0, p_ely_prev=0.0,
    time_limit=120.0,
):
    """
    Résout le MILP global sur N heures (typiquement 168h = 1 semaine).

    CONVENTION : toutes les puissances internes (p_fc, p_ely, p_bd, p_bc)
    sont côté STACK [W]. Les conversions vers le bus DC se font dans le
    bilan de puissance via les rendements des convertisseurs.

    Variables continues (N points chacune) :
      p_fc    : puissance stack FC [W], ≥ 0
      p_ely   : puissance stack ELY [W], ≥ 0  (convention positive = consomme)
      p_bd    : puissance stack décharge batterie [W], ≥ 0
      p_bc    : puissance stack charge batterie [W], ≥ 0
      s_lol   : puissance non servie côté réseau [W], ≥ 0
               (slack du bilan — pénalisée dans l'objectif)
      s_sur   : surplus PV non utilisable [W], ≥ 0
               (slack du bilan côté excès — gratuit dans l'objectif)
      dfc_p, dfc_n  : composantes positives/négatives de Δp_fc
      dely_p, dely_n: composantes positives/négatives de Δp_ely

    Variables d'état (N+1 points) :
      soc   : état de charge batterie [0..1]
      e_h2  : énergie H2 stockée [kWh]

    Variable slack terminale (scalaire) :
      s_soc_term : écart SoC terminal [0..1], pénalisée dans l'objectif

    Variables binaires (N points chacune) :
      u_fc, u_ely   : on/off
      u_bat         : 1=décharge, 0=charge (exclusion mutuelle p_bd/p_bc)
      st_fc, st_ely : démarrage (transition 0→1)
      hi_fc, lo_fc  : indicateurs de régime FC
      hi_ely, id_ely: indicateurs de régime ELY

    BILAN DE PUISSANCE (côté réseau DC) :
      p_fc·η_c + p_bd·η_b·η_c - p_bc/(η_b·η_c)
      - p_ely/η_c - P_pv[t] + s_lol - s_sur = P_load[t]
    → Le bilan est une ÉGALITÉ. s_lol absorbe le déficit (≥0),
      s_sur absorbe le surplus (≥0). Les deux ne peuvent pas être
      simultanément positifs car ils ont des signes opposés, mais le
      solveur MILP choisit le minimum grâce aux coûts.

    NORMALISATION de l'objectif :
    Les deux termes (LoL et dégradation) sont exprimés en EUROS
    pour assurer une commensurabilité réelle :
      - Terme LoL    : s_lol [W] × Ts [h] × c_lol [€/Wh]
                       c_lol = 0.5 €/kWh = 5e-4 €/Wh (coût de délestage typique)
      - Terme dégradation : coûts unitaires déjà en €/[unité]
    L'objectif mixte devient :
      min  (1-ε) × c_lol × Ts × Σ s_lol
         + ε × [ dégradations en € ]
         + w_term × s_soc_term
    Quand ε=0 : seul le LoL est minimisé → s_lol → 0 → FC et BAT
    travaillent au maximum pour satisfaire la demande.
    Quand ε=1 : seules les dégradations sont minimisées → machines
    au repos → s_lol → max (tout le déficit non servi).
    """
    N  = len(P_load_w)
    Ts = 1.0                           # pas de temps [h]
    eta_c   = CONV['eta']
    eta_b   = BAT['eff']
    eta_fc  = max(FC['eff'],  0.01)
    eta_ely = max(ELY['eff'], 0.01)
    P_bat_max = max(C_bat_Wh, 1.0)    # 1C-rate comme borne de puissance

    # ---- index des variables ------------------------------------------------
    V = {}
    off = 0
    for name in ['p_fc','p_ely','p_bd','p_bc','s_lol','s_sur',
                 'dfc_p','dfc_n','dely_p','dely_n']:
        V[name] = np.arange(off, off + N); off += N
    for name in ['soc', 'e_h2']:
        V[name] = np.arange(off, off + N + 1); off += N + 1
    V['s_soc_term'] = np.array([off]); off += 1
    for name in ['u_fc','u_ely','u_bat','st_fc','st_ely',
                 'hi_fc','lo_fc','hi_ely','id_ely']:
        V[name] = np.arange(off, off + N); off += N
    n_x = off

    # ---- vecteur objectif ---------------------------------------------------
    # NORMALISATION : les deux termes (LoL et dégradation) sont exprimés en €
    # et normalisés dynamiquement pour être commensurables à tout ε.
    #
    # Terme LoL   : (1-ε) × c_lol × Σ s_lol [W·h]
    #   c_lol est calculé tel que "100% LoL sur la semaine" ≈ REF_DEG_COST €
    #   REF_DEG_COST = référence représentative du coût de dégradation/semaine
    #
    # Terme dégradation : ε × Σ coûts_unitaires × quantités
    #
    # Pour ε=0 : les binaires de dégradation (hi_fc, lo_fc, hi_ely, id_ely,
    # st_fc, st_ely) ont un coeff nul et deviennent libres → résolution LP
    # triviale. On leur ajoute une régularisation infinitésimale pour forcer
    # le solveur à les traiter comme des entiers.

    REF_DEG_COST = 120.0  # € : coût dégradation de référence par semaine
    E_net_week   = max(float(np.sum(np.maximum(P_load_w - P_pv_w, 0.0))), 1.0)  # Wh
    c_lol = REF_DEG_COST / E_net_week   # €/Wh : normalisé par l'énergie à servir

    # Poids de pénalité SoC terminal (doit dominer la dégradation d'un cycle)
    w_soc_term = REF_DEG_COST * 5.0

    # Régularisation ε=0 : coeff minimal sur les binaires pour forcer intégralité
    EPS_REG = 1e-4   # € : négligeable devant REF_DEG_COST mais suffit au solveur

    c_obj = np.zeros(n_x)
    w_lol = (1.0 - epsilon) * c_lol * Ts   # €/W par pas de temps
    w_deg = epsilon / 1000

    c_obj[V['s_lol']]      += w_lol
    c_obj[V['s_soc_term']] += w_soc_term
    # Dégradation batterie (toujours présente même à ε=0 si on échange de l'énergie)
    c_obj[V['p_bd']]       += w_deg * costs['c_bat_Wh'] * Ts
    c_obj[V['p_bc']]       += w_deg * costs['c_bat_Wh'] * Ts
    # FC
    c_obj[V['st_fc']]      += max(w_deg * costs['c_fc_ss'],         EPS_REG)
    c_obj[V['hi_fc']]      += max(w_deg * costs['c_fc_hi'] * Ts,   EPS_REG)
    c_obj[V['lo_fc']]      += max(w_deg * costs['c_fc_lo'] * Ts,   EPS_REG)
    c_obj[V['dfc_p']]      += w_deg * costs['c_fc_tr']
    c_obj[V['dfc_n']]      += w_deg * costs['c_fc_tr']
    # ELY
    c_obj[V['st_ely']]     += max(w_deg * costs['c_ely_ss'],        EPS_REG)
    c_obj[V['hi_ely']]     += max(w_deg * costs['c_ely_hi'] * Ts,  EPS_REG)
    c_obj[V['id_ely']]     += max(w_deg * costs['c_ely_id'] * Ts,  EPS_REG)
    c_obj[V['dely_p']]     += w_deg * costs['c_ely_tr']
    c_obj[V['dely_n']]     += w_deg * costs['c_ely_tr']

    if not np.all(np.isfinite(c_obj)):
        bad = np.where(~np.isfinite(c_obj))[0]
        raise ValueError(f"c_obj non fini aux indices {bad}")

    # ---- bornes -------------------------------------------------------------
    lb = np.zeros(n_x)
    ub = np.full(n_x, np.inf)

    ub[V['p_fc']]  = P_fc_max
    ub[V['p_ely']] = P_ely_max
    ub[V['p_bd']]  = P_bat_max
    ub[V['p_bc']]  = P_bat_max
    # s_lol : borné par la demande maximale (évite dual-unbounded)
    ub[V['s_lol']] = max(np.max(P_load_w), 1.0)
    # s_sur : borné par la demande (surplus max = PV total disponible sur l'horizon)
    ub[V['s_sur']] = max(np.max(P_pv_w), np.max(P_load_w), 1.0)

    lb[V['soc']] = 0.20      # borne basse opérationnelle (profondeur de décharge max 80%)
    ub[V['soc']] = 0.995
    lb[V['soc'][0]] = soc_init;  ub[V['soc'][0]] = soc_init
    lb[V['e_h2']] = 0.0;         ub[V['e_h2']] = E_h2_max
    lb[V['e_h2'][0]] = e_h2_init; ub[V['e_h2'][0]] = e_h2_init

    ub[V['dfc_p']]  = P_fc_max;   ub[V['dfc_n']]  = P_fc_max
    ub[V['dely_p']] = P_ely_max;  ub[V['dely_n']] = P_ely_max
    # Slack terminal SoC borné par l'écart maximal possible
    ub[V['s_soc_term'][0]] = max(soc_init - lb[V['soc'][1]], 1.0)

    for name in ['u_fc','u_ely','u_bat','st_fc','st_ely',
                 'hi_fc','lo_fc','hi_ely','id_ely']:
        ub[V[name]] = 1.0

    # ---- intégralité --------------------------------------------------------
    integ = np.zeros(n_x)
    for name in ['u_fc','u_ely','u_bat','st_fc','st_ely',
                 'hi_fc','lo_fc','hi_ely','id_ely']:
        integ[V[name]] = 1.0

    # ---- contraintes --------------------------------------------------------
    r_u, c_u, v_u, rhs_u = [], [], [], []
    r_e, c_e, v_e, rhs_e = [], [], [], []
    ru = [0]; re = [0]

    def ineq(cols, vals, rhs):
        for col, val in zip(cols, vals):
            r_u.append(ru[0]); c_u.append(int(col)); v_u.append(float(val))
        rhs_u.append(float(rhs)); ru[0] += 1

    def eq(cols, vals, rhs):
        for col, val in zip(cols, vals):
            r_e.append(re[0]); c_e.append(int(col)); v_e.append(float(val))
        rhs_e.append(float(rhs)); re[0] += 1

    for t in range(N):
        # ------------------------------------------------------------------
        # BILAN DE PUISSANCE (côté réseau DC) :
        #
        #   [FC → réseau]    + [bat_décharge → réseau]
        # - [réseau → ELY]  - [réseau → bat_charge]
        # - [PV → réseau]   + s_lol - s_sur = P_load_réseau[t]
        #
        # p_fc [stack] → réseau : p_fc × η_c
        # p_ely [stack consomme] ← réseau : p_ely / η_c
        # p_bd [stack] → réseau : p_bd × η_b × η_c
        # p_bc [réseau → stack] : p_bc / (η_b × η_c)
        # P_pv est déjà côté réseau
        # ------------------------------------------------------------------
        eq(
            [V['p_fc'][t],    V['p_ely'][t],
             V['p_bd'][t],    V['p_bc'][t],
             V['s_lol'][t],   V['s_sur'][t]],
            [eta_c,           -1.0/eta_c,
             eta_b*eta_c,     -1.0/(eta_b*eta_c),
             1.0,             -1.0],
            P_load_w[t] - P_pv_w[t],
        )

        # ------------------------------------------------------------------
        # DYNAMIQUE SoC (en fraction, variables stack)
        # SoC[t+1] = SoC[t] - (p_bd - p_bc) × Ts / C_bat_Wh
        # Note : C_bat_Wh est la capacité en Wh (énergie côté stack)
        # ------------------------------------------------------------------
        eq(
            [V['soc'][t+1], V['soc'][t], V['p_bd'][t], V['p_bc'][t]],
            [1.0, -1.0, Ts/C_bat_Wh, -Ts/C_bat_Wh],
            0.0,
        )

        # ------------------------------------------------------------------
        # DYNAMIQUE H2 [kWh]
        # Physique : ELY produit H2 (p_ely > 0 → e_h2 augmente)
        #            FC consomme H2 (p_fc  > 0 → e_h2 diminue)
        #
        # Convention eq(cols, vals, rhs) :
        #   sum(vals[i] * x[cols[i]]) = rhs
        # → e_h2[t+1] = e_h2[t] - coeff_ely*p_ely - coeff_fc*p_fc
        #
        # Pour e_h2 augmente quand p_ely > 0 : coeff_ely < 0 → -eta_ely  ✓
        # Pour e_h2 diminue  quand p_fc  > 0 : coeff_fc  > 0 → +1/eta_fc ✓
        # ------------------------------------------------------------------
        eq(
            [V['e_h2'][t+1], V['e_h2'][t], V['p_ely'][t], V['p_fc'][t]],
            [1.0, -1.0, -eta_ely*Ts/1000.0, +Ts/eta_fc/1000.0],
            0.0,
        )

        # ------------------------------------------------------------------
        # ON/OFF (big-M)
        # ------------------------------------------------------------------
        ineq([V['p_fc'][t],  V['u_fc'][t]],  [1.0, -P_fc_max],  0.0)
        ineq([V['p_ely'][t], V['u_ely'][t]], [1.0, -P_ely_max], 0.0)

        # ------------------------------------------------------------------
        # EXCLUSION MUTUELLE charge/décharge batterie via u_bat binaire
        # u_bat=1 → décharge (p_bd libre, p_bc=0)
        # u_bat=0 → charge   (p_bc libre, p_bd=0)
        # ------------------------------------------------------------------
        ineq([V['p_bd'][t], V['u_bat'][t]], [1.0, -P_bat_max], 0.0)
        ineq([V['p_bc'][t], V['u_bat'][t]], [1.0,  P_bat_max], P_bat_max)

        # ------------------------------------------------------------------
        # DÉMARRAGE FC : st_fc[t] ≥ u_fc[t] − u_fc[t−1]  (capture 0→1)
        # ------------------------------------------------------------------
        if t == 0:
            ineq([V['u_fc'][0],  V['st_fc'][0]],  [1.0, -1.0], float(u_fc_prev))
            ineq([V['u_ely'][0], V['st_ely'][0]], [1.0, -1.0], float(u_ely_prev))
        else:
            ineq([V['u_fc'][t],  V['st_fc'][t],  V['u_fc'][t-1]],  [1.0, -1.0, -1.0], 0.0)
            ineq([V['u_ely'][t], V['st_ely'][t], V['u_ely'][t-1]], [1.0, -1.0, -1.0], 0.0)
        ineq([V['st_fc'][t],  V['u_fc'][t]],  [1.0, -1.0], 0.0)
        ineq([V['st_ely'][t], V['u_ely'][t]], [1.0, -1.0], 0.0)

        # ------------------------------------------------------------------
        # INDICATEURS DE RÉGIME FC/ELY (indicateurs souples, big-M unilatéral)
        #
        # PRINCIPE : ces indicateurs servent UNIQUEMENT à calculer les coûts
        # de dégradation dans l'objectif. On ne force PAS leur valeur avec
        # des contraintes bilatérales (qui créeraient de l'infaisabilité).
        # Le solveur les activera naturellement pour minimiser le coût.
        #
        # FC haute puissance : hi_fc = 1 autorisé seulement si p_fc ≥ P_fc_hi
        #   → p_fc ≤ P_fc_hi + (P_fc_max - P_fc_hi)·hi_fc  [big-M borne sup]
        #   → si hi_fc=0 : p_fc ≤ P_fc_hi  (pas de haute puissance)
        #   → si hi_fc=1 : p_fc ≤ P_fc_max (borne naturelle)
        # Note : comme hi_fc est pénalisé dans l'objectif (coût dégradation),
        # le solveur mettra hi_fc=1 uniquement si p_fc > P_fc_hi est nécessaire.
        # Mais comme hi_fc=0 contraint p_fc ≤ P_fc_hi, le solveur DOIT mettre
        # hi_fc=1 quand la demande l'exige. C'est cohérent et non-infaisable.
        #
        # FC basse puissance/idling : lo_fc = 1 autorisé si u_fc=1 et p_fc ≤ P_fc_lo
        #   → lo_fc ≤ u_fc
        #   → p_fc ≤ P_fc_lo + P_fc_max·(1 - lo_fc)  [quand lo_fc=1 : p_fc ≤ P_fc_lo]
        # ------------------------------------------------------------------
        Phi_fc  = costs['P_fc_hi']
        Plo_fc  = costs['P_fc_lo']
        # hi_fc : borne sup de p_fc dépend de hi_fc (big-M)
        ineq([V['p_fc'][t], V['hi_fc'][t]], [1.0, -(P_fc_max - Phi_fc)], Phi_fc)
        # lo_fc : lié à u_fc, borne la puissance minimale
        ineq([V['lo_fc'][t], V['u_fc'][t]], [1.0, -1.0], 0.0)  # lo_fc ≤ u_fc
        ineq([V['p_fc'][t], V['lo_fc'][t]], [1.0, P_fc_max], P_fc_max + Plo_fc)  # si lo=1: p≤Plo
        ineq([V['hi_fc'][t], V['lo_fc'][t]], [1.0, 1.0], 1.0)  # mutuellement exclusifs

        Phi_ely = costs['P_ely_hi']
        Plo_ely = costs['P_ely_lo']
        # hi_ely
        ineq([V['p_ely'][t], V['hi_ely'][t]], [1.0, -(P_ely_max - Phi_ely)], Phi_ely)
        # id_ely (idling)
        ineq([V['id_ely'][t], V['u_ely'][t]], [1.0, -1.0], 0.0)
        ineq([V['p_ely'][t], V['id_ely'][t]], [1.0, P_ely_max], P_ely_max + Plo_ely)
        ineq([V['hi_ely'][t], V['id_ely'][t]], [1.0, 1.0], 1.0)

        # ------------------------------------------------------------------
        # LINÉARISATION |Δp| = dp_pos + dp_neg (splittage)
        # p[t] − p[t−1] = dp_pos[t] − dp_neg[t]
        # ------------------------------------------------------------------
        if t == 0:
            eq([V['p_fc'][0],  V['dfc_p'][0],  V['dfc_n'][0]],
               [1.0, -1.0, 1.0], float(p_fc_prev))
            eq([V['p_ely'][0], V['dely_p'][0], V['dely_n'][0]],
               [1.0, -1.0, 1.0], float(p_ely_prev))
        else:
            eq([V['p_fc'][t],  V['p_fc'][t-1],  V['dfc_p'][t],  V['dfc_n'][t]],
               [1.0, -1.0, -1.0, 1.0], 0.0)
            eq([V['p_ely'][t], V['p_ely'][t-1], V['dely_p'][t], V['dely_n'][t]],
               [1.0, -1.0, -1.0, 1.0], 0.0)

    # ------------------------------------------------------------------
    # CONTRAINTE TERMINALE SOUPLE SoC
    # On exige SoC[N] + s_soc_term ≥ soc_init
    # → −SoC[N] − s_soc_term ≤ −soc_init
    # ------------------------------------------------------------------
    ineq([V['soc'][N], V['s_soc_term'][0]], [-1.0, -1.0], -soc_init)

    # ---- assemblage sparse -------------------------------------------------
    n_ineq = ru[0]; n_eq = re[0]
    A_ineq = csc_matrix((v_u, (r_u, c_u)), shape=(n_ineq, n_x))
    A_eq   = csc_matrix((v_e, (r_e, c_e)), shape=(n_eq,   n_x))
    A_all  = sp_vstack([A_ineq, A_eq], format='csc')
    lb_all = np.concatenate([-np.inf * np.ones(n_ineq), np.array(rhs_e)])
    ub_all = np.concatenate([np.array(rhs_u),           np.array(rhs_e)])

    constraints = LinearConstraint(A_all, lb_all, ub_all)
    bounds      = Bounds(lb, ub)

    sol = milp(
        c_obj,
        constraints=constraints,
        integrality=integ,
        bounds=bounds,
        options={'disp': False, 'time_limit': time_limit, 'mip_rel_gap': 5e-3},
    )

    if sol.status in (2, 3, 4) or sol.x is None:
        return None

    x = sol.x
    return {
        'p_fc'   : x[V['p_fc']],
        'p_ely'  : x[V['p_ely']],
        'p_bd'   : x[V['p_bd']],
        'p_bc'   : x[V['p_bc']],
        's_lol'  : x[V['s_lol']],
        'soc'    : x[V['soc']],
        'e_h2'   : x[V['e_h2']],
        'u_fc'   : np.round(x[V['u_fc']]).astype(int),
        'u_ely'  : np.round(x[V['u_ely']]).astype(int),
        'st_fc'  : np.round(x[V['st_fc']]).astype(int),
        'st_ely' : np.round(x[V['st_ely']]).astype(int),
        'hi_fc'  : np.round(x[V['hi_fc']]).astype(int),
        'lo_fc'  : np.round(x[V['lo_fc']]).astype(int),
        'hi_ely' : np.round(x[V['hi_ely']]).astype(int),
        'id_ely' : np.round(x[V['id_ely']]).astype(int),
        'obj'    : sol.fun,
        'status' : sol.status,
    }


# =============================================================================
# SECTION 3 — SIMULATION 5 ANS PAR BLOCS HEBDOMADAIRES
# =============================================================================

def run_simulation_5y(epsilon, N_week=168, verbose=True):
    """
    Simulation complète 5 ans avec MILP par blocs hebdomadaires.
    Retourne un dict `data` compatible avec run_main_plot().
    """
    Ts    = LOAD['Ts']          # 3600 s
    Ts_h  = Ts / 3600.0         # 1 h
    eta_c = CONV['eta']
    eta_b = BAT['eff']

    P_ref_full = np.array(LOAD['P_ref'])
    P_pv_full  = np.array(PV['P'])

    # Horizon 5 ans
    T_total    = int(3600 * 24 * 365 * 5)
    temps_full = np.arange(0, T_total, Ts, dtype=float)
    n_total    = len(temps_full)

    def get_profile(arr, idx_start, length):
        L = len(arr)
        return arr[np.arange(idx_start, idx_start + length) % L]

    # ---- Initialisation ----
    SoC_init  = 0.5;   E_h2_init = 200.0
    SoC       = np.zeros(n_total + 1); SoC[0]  = SoC_init
    E_h2      = np.zeros(n_total + 1); E_h2[0] = E_h2_init
    P_bat     = np.zeros(n_total)
    P_fc      = np.zeros(n_total)
    P_ely     = np.zeros(n_total)
    P_dc_load = np.zeros(n_total)
    P_dc_pv   = np.zeros(n_total)
    P_dc_bat  = np.zeros(n_total)
    P_dc_fc   = np.zeros(n_total)
    P_dc_ely  = np.zeros(n_total)
    lol_tab   = np.zeros(n_total)

    alpha_fc  = np.zeros(n_total + 1)
    alpha_ely = np.zeros(n_total + 1)
    # Initialiser TOUS les SoH à 1 (évite division par zéro dans get_cost_bat)
    SoH_bat   = np.ones(n_total + 1)
    SoH_fc    = np.ones(n_total + 1)
    SoH_ely   = np.ones(n_total + 1)

    alpha_fc_eol  = (1 - FC['SoH_EoL'])  * 1873.6207  / 1766.1207
    alpha_ely_eol = (1 - ELY['SoH_EoL']) * 60122.0238 / 61392.7339

    deg_fc  = {k: np.zeros(n_total)
               for k in ['start-stop','high','idling','transient','total']}
    deg_ely = {k: np.zeros(n_total)
               for k in ['start-stop','turning power','idling','transient','total']}

    j_new_bat = j_new_fc = j_new_ely = 0
    u_fc_prev = 0;    u_ely_prev = 0
    p_fc_prev = 0.0;  p_ely_prev = 0.0

    n_weeks = int(np.ceil(n_total / N_week))
    fallback_count = 0

    for w in range(n_weeks):
        j0 = w * N_week
        j1 = min(j0 + N_week, n_total)
        Nw = j1 - j0

        if verbose and w % 10 == 0:
            print(f"  Semaine {w+1}/{n_weeks} ({w/n_weeks*100:.0f}%)  "
                  f"SoH_fc={SoH_fc[j0]:.3f}  SoH_ely={SoH_ely[j0]:.3f}  "
                  f"SoH_bat={SoH_bat[j0]:.3f}")

        # Profils côté réseau [W] — le bilan MILP est écrit côté réseau
        # Les rendements convertisseurs sont inclus dans les coefficients du bilan
        P_load_net_w = get_profile(P_ref_full, j0, Nw)
        P_load_w     = P_load_net_w   # même chose : bilan réseau
        P_pv_w       = get_profile(P_pv_full,  j0, Nw)

        # Paramètres courants
        alpha_fc_w  = float(alpha_fc[j0])
        alpha_ely_w = float(alpha_ely[j0])
        SoH_bat_w   = float(SoH_bat[j0])
        P_fc_max_w  = float(compute_P_fc_max(alpha_fc_w))
        P_ely_max_w = float(compute_P_ely_max(alpha_ely_w))

        costs    = build_deg_costs(P_fc_max_w, P_ely_max_w, SoH_bat_w, Ts_h)
        C_bat_Wh = costs['C_bat_Wh']
        soc_w    = float(SoC[j0])
        e_h2_w   = float(E_h2[j0])

        # ---- Résolution MILP ------------------------------------------------
        res = solve_week_milp(
            P_load_w, P_pv_w,
            P_fc_max_w, P_ely_max_w, C_bat_Wh,
            E_h2_init, soc_w, e_h2_w,
            costs, epsilon,
            u_fc_prev=u_fc_prev, u_ely_prev=u_ely_prev,
            p_fc_prev=p_fc_prev, p_ely_prev=p_ely_prev,
            time_limit=120.0,
        )

        # ---- Fallback 50/50 si MILP échoue ----------------------------------
        if res is None:
            fallback_count += 1
            if verbose:
                print(f"    [!] Fallback 50/50 semaine {w+1}")
            for t_local in range(Nw):
                t = j0 + t_local
                pnet = P_load_w[t_local] - P_pv_w[t_local]
                if pnet > 0:
                    p_fc_t  = min(pnet * 0.5, P_fc_max_w)
                    p_bat_t = pnet - p_fc_t * eta_c
                    p_ely_t = 0.0
                else:
                    p_ely_t = min(-pnet * 0.5 * eta_c, P_ely_max_w)
                    p_bat_t = pnet + p_ely_t / eta_c
                    p_fc_t  = 0.0

                P_fc[t]      = p_fc_t
                P_ely[t]     = -p_ely_t
                P_bat[t]     = p_bat_t * eta_c if p_bat_t > 0 else p_bat_t / eta_c
                P_dc_fc[t]   = p_fc_t  * eta_c
                P_dc_ely[t]  = -p_ely_t / eta_c
                P_dc_bat[t]  = P_bat[t]
                P_dc_load[t] = P_load_net_w[t_local]   # côté réseau
                P_dc_pv[t]   = P_pv_w[t_local]

                pnet_res = P_load_net_w[t_local] - P_pv_w[t_local]
                pwr_avail = P_dc_fc[t] + max(P_dc_bat[t], 0.0)
                lol_tab[t] = max(0.0, pnet_res - pwr_avail) / max(pnet_res, 1.0)
                lol_tab[t] = float(np.clip(lol_tab[t], 0.0, 1.0))

                SoC[t+1]  = float(np.clip(SoC[t] - P_bat[t] / C_bat_Wh, 0.20, 0.995))
                eff_ely_t = np.interp(abs(p_ely_t / max(P_ely_max_w, 1.0)) * 100,
                                      *ELY['lut']) / 100
                eff_fc_t  = np.interp(abs(p_fc_t  / max(P_fc_max_w,  1.0)) * 100,
                                      *FC['lut'])  / 100
                P_h2_t    = (p_ely_t * eff_ely_t - p_fc_t / max(eff_fc_t, 0.01)) / 1000
                E_h2[t+1] = float(np.clip(E_h2[t] + P_h2_t * Ts_h, 0.0, E_h2_init))

            # État final pour le prochain bloc
            u_fc_prev  = 1 if P_fc[j1-1] > 1.0 else 0
            u_ely_prev = 1 if P_ely[j1-1] < -1.0 else 0
            p_fc_prev  = float(P_fc[j1-1])
            p_ely_prev = float(-P_ely[j1-1])

        # ---- Trajectoire MILP réussie ---------------------------------------
        else:
            for t_local in range(Nw):
                t       = j0 + t_local
                p_fc_t  = float(res['p_fc'][t_local])
                p_ely_t = float(res['p_ely'][t_local])
                p_bd_t  = float(res['p_bd'][t_local])
                p_bc_t  = float(res['p_bc'][t_local])
                s_lol_t = float(res['s_lol'][t_local])

                P_fc[t]      = p_fc_t
                P_ely[t]     = -p_ely_t            # convention : ELY < 0
                P_bat[t]     = p_bd_t - p_bc_t     # stack, + = décharge
                P_dc_fc[t]   = p_fc_t  * eta_c
                P_dc_ely[t]  = -p_ely_t / eta_c
                # p_bd/p_bc sont côté stack ; côté réseau :
                # décharge stack → réseau : p_bd × η_bat × η_conv
                # réseau → charge stack   : p_bc / (η_bat × η_conv)
                P_dc_bat[t]  = p_bd_t * eta_b * eta_c - p_bc_t / (eta_b * eta_c)
                P_dc_load[t] = P_load_net_w[t_local]   # côté réseau
                P_dc_pv[t]   = P_pv_w[t_local]

                pnet_res   = P_load_net_w[t_local] - P_pv_w[t_local]
                lol_tab[t] = s_lol_t / max(pnet_res, 1.0) if pnet_res > 0 else 0.0
                lol_tab[t] = float(np.clip(lol_tab[t], 0.0, 1.0))

                SoC[t+1]  = float(res['soc'][t_local + 1])
                E_h2[t+1] = float(res['e_h2'][t_local + 1])

            u_fc_prev  = int(res['u_fc'][-1])
            u_ely_prev = int(res['u_ely'][-1])
            p_fc_prev  = float(res['p_fc'][-1])
            p_ely_prev = float(res['p_ely'][-1])

        # ---- Mise à jour des SoH via get_soh() ------------------------------
        j_end = j1
        # SoC passé à get_soh : de j_new_bat à j_end INCLUS (N+1 éléments)
        SoH_bat_new, SoH_fc_new, SoH_ely_new = get_soh(
            alpha_fc [j_new_fc  : j_end],   P_fc [j_new_fc  : j_end],
            alpha_ely[j_new_ely : j_end],   P_ely[j_new_ely : j_end],
            P_bat    [j_new_bat : j_end],   SoC  [j_new_bat : j_end + 1],
            SoH_bat  [j_new_bat : j_end],
        )

        # ---- Dégradations par composant (valeur cumulée sur le bloc) --------
        fcto, fcss, fci, fct, fch   = get_cost_fc(
            alpha_fc [j_new_fc  : j_end], P_fc [j_new_fc  : j_end])
        elto, elss, eli, elt, eltu  = get_cost_ely(
            alpha_ely[j_new_ely : j_end], P_ely[j_new_ely : j_end])

        t_last = j_end - 1
        deg_fc['start-stop'][t_last] = fcss
        deg_fc['idling'][t_last]     = fci
        deg_fc['transient'][t_last]  = fct
        deg_fc['high'][t_last]       = fch
        deg_fc['total'][t_last]      = (fcto * 100 / FC['cost']
                                        * (1 - FC['SoH_EoL']) * 100 / 100)

        deg_ely['start-stop'][t_last]    = elss
        deg_ely['idling'][t_last]        = eli
        deg_ely['transient'][t_last]     = elt
        deg_ely['turning power'][t_last] = eltu
        deg_ely['total'][t_last]         = (elto * 100 / ELY['cost']
                                            * (1 - ELY['SoH_EoL']) * 100 / 100)

        # ---- Reset SoH si EoL -----------------------------------------------
        if SoH_bat_new < BAT['SoH_EoL']:
            SoH_bat_new = 1.0; j_new_bat = j_end
        if SoH_fc_new  < FC['SoH_EoL']:
            SoH_fc_new  = 1.0; j_new_fc  = j_end
        if SoH_ely_new < ELY['SoH_EoL']:
            SoH_ely_new = 1.0; j_new_ely = j_end

        # ---- Propagation SoH et alpha sur tous les pas du bloc --------------
        for t in range(j0, j_end):
            SoH_bat[t+1]   = SoH_bat_new
            SoH_fc[t+1]    = SoH_fc_new
            SoH_ely[t+1]   = SoH_ely_new
            alpha_fc[t+1]  = ((1 - SoH_fc_new)  / (1 - FC['SoH_EoL'])
                              * alpha_fc_eol)
            alpha_ely[t+1] = ((1 - SoH_ely_new) / (1 - ELY['SoH_EoL'])
                              * alpha_ely_eol)

    if verbose:
        print(f"\n  Simulation terminée. Fallbacks: {fallback_count}/{n_weeks}")

    return {
        'temps'    : temps_full, 'n'        : n_total,
        'SoC'      : SoC,        'E_h2'     : E_h2,
        'P_bat'    : P_bat,      'P_fc'     : P_fc,
        'P_ely'    : P_ely,
        'P_dc_load': P_dc_load,  'P_dc_pv'  : P_dc_pv,
        'P_dc_bat' : P_dc_bat,   'P_dc_fc'  : P_dc_fc,
        'P_dc_ely' : P_dc_ely,   'lol_tab'  : lol_tab,
        'alpha_fc' : alpha_fc,   'alpha_ely': alpha_ely,
        'SoH_bat'  : SoH_bat,    'SoH_fc'   : SoH_fc,
        'SoH_ely'  : SoH_ely,    'deg_fc'   : deg_fc,
        'deg_ely'  : deg_ely,
    }


# =============================================================================
# SECTION 4 — CALCUL DES KPIs
# =============================================================================

def compute_kpis(data):
    """Calcule LPSP (%) et coût de dégradation total (€)."""
    P_load = np.array(data['P_dc_load'])
    P_pv   = np.array(data['P_dc_pv'])
    lol    = np.array(data['lol_tab'])

    pnet = np.maximum(P_load - P_pv, 0.0)
    lpsp = (lol * pnet).sum() / max(pnet.sum(), 1.0) * 100.0

    soh_clean = np.where(np.isnan(data['SoH_bat']), 1.0, data['SoH_bat'])
    cost_deg = get_cost_total(
        data['alpha_fc'][:-1], data['P_fc'],
        data['alpha_ely'][:-1], data['P_ely'],
        data['P_bat'], data['SoC'],
        LOAD, BAT, FC, ELY, soh_clean[:-1],
    )
    return lpsp, cost_deg


# =============================================================================
# SECTION 5 — FIGURES
# =============================================================================

def plot_all_aging(data, savedir, epsilon):
    """all_aging.pdf : SoH BAT/FC/ELY + camemberts dégradation."""
    plt.rcParams.update({
        "font.family":"serif", "font.size":22, "axes.titlesize":28,
        "axes.labelsize":26, "xtick.labelsize":20, "ytick.labelsize":20,
        "legend.fontsize":20,
    })
    temps   = data['temps']
    t_years = temps / (3600 * 24 * 365)
    n       = data['n']
    SoH_b   = data['SoH_bat'][:-1].copy()
    SoH_f   = data['SoH_fc'][:-1].copy()
    SoH_e   = data['SoH_ely'][:-1].copy()
    deg_fc  = data['deg_fc']
    deg_ely = data['deg_ely']

    # Masquage des resets
    for k in range(1, n):
        if SoH_b[k] == 1.0: SoH_b[k-1] = np.nan
        if SoH_f[k] == 1.0: SoH_f[k-1] = np.nan
        if SoH_e[k] == 1.0: SoH_e[k-1] = np.nan

    colors = {'bat':'#1f77b4', 'fc':'#d62728', 'ely':'#2ca02c'}
    fig = plt.figure(figsize=(24, 16))
    gspec = gs.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.65,
                        height_ratios=[1, 1.4])

    for i, (k, soh_v, eol, title) in enumerate([
        ('bat', SoH_b, BAT['SoH_EoL'], r'\mathbf{Battery}'),
        ('fc',  SoH_f, FC['SoH_EoL'],  r'\mathbf{PEMFC}'),
        ('ely', SoH_e, ELY['SoH_EoL'], r'\mathbf{PEMWE}'),
    ]):
        ax = fig.add_subplot(gspec[0, i*2:(i+1)*2])
        e_pct = eol * 100; v_pct = soh_v * 100
        ax.plot(t_years, v_pct, lw=5, color=colors[k], label=r'$SoH$')
        ax.fill_between(t_years, v_pct, e_pct, color=colors[k], alpha=0.12)
        ax.axhline(y=e_pct, color='r', ls='--', lw=3, label=r'$EoL$')
        ax.set_title(rf'${title}$', pad=20)
        ax.set_xlabel(r'$\mathbf{Time\ (years)}$')
        if i == 0: ax.set_ylabel(r'$SoH\ [\%]$')
        ax.set_yticks([int(e_pct), 100])
        ax.grid(True, alpha=0.3, ls=':')
        ax.legend(loc='upper right', framealpha=0.9)

    colors_pie = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    for d, keys, labels, title, pos in [
        (deg_fc,  ['start-stop','idling','transient','high'],
         ['Start-stop','Idling','Transient','High power'],
         r'\mathbf{PEMFC\ Degradation}', gspec[1, 0:3]),
        (deg_ely, ['start-stop','idling','transient','turning power'],
         ['Start-stop','Idling','Transient','High power'],
         r'\mathbf{PEMWE\ Degradation}', gspec[1, 3:6]),
    ]:
        ax = fig.add_subplot(pos)
        tot = np.where(np.isnan(d['total']), 0.0, d['total'])
        idx = np.argmax(tot)
        vals = np.array([d[k][max(idx-1, 0)] for k in keys])
        mask = (vals > 0) & ~np.isnan(vals)
        if mask.sum() > 0:
            w_p, t_p, at_p = ax.pie(
                vals[mask], labels=np.array(labels)[mask],
                autopct='%1.1f%%', startangle=140,
                colors=np.array(colors_pie)[mask],
                pctdistance=0.75, explode=[0.06]*mask.sum(),
            )
            plt.setp(t_p, fontsize=22); plt.setp(at_p, fontsize=20, weight='bold')
        ax.set_title(rf'${title}$', pad=12)

    fig.suptitle(rf'$\mathbf{{Aging\ summary\ —\ \varepsilon={epsilon:.2f}}}$',
                 fontsize=30, y=0.98)
    fpath = os.path.join(savedir, 'all_aging.pdf')
    plt.savefig(fpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_everything_combined(data, savedir, epsilon):
    """everything_combined_v2.pdf : 7 lignes × 2 colonnes (global + zoom)."""
    plt.rcParams.update({
        "font.family":"serif","font.size":18,"axes.titlesize":24,
        "axes.labelsize":20,"xtick.labelsize":16,"ytick.labelsize":16,
        "legend.fontsize":16,"lines.linewidth":1.4,
    })
    temps    = data['temps']
    t_days   = temps / 3600 / 24
    n        = data['n']
    P_dc_load = np.array(data['P_dc_load'])
    P_dc_pv   = np.array(data['P_dc_pv'])
    P_dc_bat  = np.array(data['P_dc_bat'])
    P_dc_fc   = np.array(data['P_dc_fc'])
    P_dc_ely  = np.array(data['P_dc_ely'])
    lol_tab   = np.array(data['lol_tab'])
    SoC       = np.array(data['SoC'])
    E_h2      = np.array(data['E_h2'])

    P_planned = (P_dc_load - P_dc_pv) / 1000
    P_real    = P_planned * (1 - lol_tab)
    lps_pct   = lol_tab * 100

    zoom_s, zoom_e = 210, 217
    mask_z = (t_days >= zoom_s) & (t_days <= zoom_e)
    stride = max(1, n // 8000)

    rows = [
        (P_planned, P_real,   'b','y','Planned','Real',
         r'P_{\mathrm{bus}}\ [kW]',   r'$\mathbf{Power\ demand}$'),
        (P_dc_bat/1000, None, 'b',None,None,None,
         r'P_{\mathrm{bat}}\ [kW]',   r'$\mathbf{Battery\ power}$'),
        (P_dc_fc/1000,  None, 'r',None,None,None,
         r'P_{\mathrm{FC}}\ [kW]',    r'$\mathbf{PEMFC\ power}$'),
        (P_dc_ely/1000, None, 'g',None,None,None,
         r'P_{\mathrm{ELY}}\ [kW]',   r'$\mathbf{PEMWE\ power}$'),
        (SoC[:-1]*100,  None, 'b',None,None,None,
         r'SoC_{\mathrm{bat}}\ [\%]', r'$\mathbf{Battery\ SoC}$'),
        (E_h2[:-1],     None, 'g',None,None,None,
         r'E_{H2}\ [kWh]',            r'$\mathbf{H2\ stored}$'),
        (lps_pct,       None, 'r',None,None,None,
         r'LPS\ [\%]',                r'$\mathbf{Loss\ of\ power\ supply}$'),
    ]
    fig, axs = plt.subplots(7, 2, figsize=(22, 20), sharex='col')
    plt.subplots_adjust(wspace=0.08, hspace=0.38)

    for i, (y1, y2, c1, c2, l1, l2, ylabel, title) in enumerate(rows):
        ax_g, ax_z = axs[i, 0], axs[i, 1]
        ax_g.plot(t_days[::stride], y1[::stride], color=c1, lw=1.4, label=l1)
        if y2 is not None:
            ax_g.plot(t_days[::stride], y2[::stride], color=c2, lw=1.4, label=l2)
        if l1 is not None:
            ax_g.legend(loc='upper right', fontsize=14)
        ax_g.set_ylabel(rf'${ylabel}$', fontsize=18)
        ax_g.grid(True, alpha=0.35); ax_g.margins(x=0.01)
        ax_g.text(1.02, 1.02, title, transform=ax_g.transAxes,
                  ha='center', va='bottom', fontsize=22)
        ymin, ymax = ax_g.get_ylim()
        ax_z.plot(t_days[mask_z], y1[mask_z], color=c1, marker='.', lw=1.4)
        if y2 is not None:
            ax_z.plot(t_days[mask_z], y2[mask_z], color=c2, marker='.', lw=1.4)
        ax_z.set_xlim(zoom_s, zoom_e); ax_z.set_ylim(ymin, ymax)
        ax_z.grid(True, alpha=0.35)

    axs[6, 0].set_xlabel(r'$\mathbf{Time\ (days)}$', fontsize=20)
    axs[6, 1].set_xlabel(r'$\mathbf{Time\ (days)}$', fontsize=20)
    axs[0, 0].set_title(r'$\mathbf{Full\ simulation\ (5\ years)}$', fontsize=22)
    axs[0, 1].set_title(r'$\mathbf{Zoom:\ week\ 30}$', fontsize=22)
    fig.suptitle(rf'$\mathbf{{MILP\ weekly\ —\ \varepsilon={epsilon:.2f}}}$',
                 fontsize=28, y=1.004)

    fpath = os.path.join(savedir, 'everything_combined_v2.pdf')
    plt.savefig(fpath, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fpath


def plot_pareto_summary(pareto_results, savedir):
    """pareto_front.pdf : front de Pareto LPSP vs coût dégradation."""
    plt.rcParams.update({"font.family":"serif","font.size":14})
    eps_v  = [p['epsilon']  for p in pareto_results]
    lpsp_v = [p['lpsp']     for p in pareto_results]
    cost_v = [p['cost_deg'] for p in pareto_results]

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(lpsp_v, cost_v, c=eps_v, cmap='plasma',
                    s=120, zorder=4, edgecolors='k', linewidths=0.7)
    ax.plot(lpsp_v, cost_v, 'k--', lw=1.5, alpha=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'$\varepsilon$', fontsize=14)
    for e, l, c in zip(eps_v, lpsp_v, cost_v):
        ax.annotate(f'ε={e:.2f}', (l, c),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=9, alpha=0.85)
    ax.set_xlabel('LPSP (%)', fontsize=15)
    ax.set_ylabel('Degradation cost (€)', fontsize=15)
    ax.set_title(r'$\mathbf{Pareto\ front\ —\ MILP\ weekly\ (5\ years)}$',
                 fontsize=16)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fpath = os.path.join(savedir, 'pareto_front.pdf')
    plt.savefig(fpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return fpath


# =============================================================================
# SECTION 6 — POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def run_pareto_milp(
    epsilons=None,
    output_root='Results_MILP',
    N_week=168,
    verbose=True,
):
    """
    Lance la simulation 5 ans pour chaque epsilon et sauvegarde les figures.

    Paramètres :
      epsilons    : liste de ε ∈ [0, 1]  (défaut : 11 points)
      output_root : dossier racine des résultats
      N_week      : taille d'un bloc (défaut 168h)
      verbose     : affichage progression

    Retourne :
      pareto_results : liste de dicts {'epsilon','lpsp','cost_deg'}
    """
    if epsilons is None:
        epsilons = np.linspace(0, 1, 11)

    os.makedirs(output_root, exist_ok=True)
    pareto_results = []

    for eps in epsilons:
        eps_str = f'{eps:.2f}'
        savedir = os.path.join(output_root, f'eps_{eps_str}')
        os.makedirs(savedir, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'  ε = {eps_str}  →  {savedir}')
        print(f'{"="*60}')

        data = run_simulation_5y(epsilon=eps, N_week=N_week, verbose=verbose)

        lpsp, cost_deg = compute_kpis(data)
        print(f'  KPIs : LPSP = {lpsp:.2f}%   Cost_deg = {cost_deg:.1f} €')

        f1 = plot_all_aging(data, savedir, eps)
        f2 = plot_everything_combined(data, savedir, eps)
        print(f'  Figures : {os.path.basename(f1)}  {os.path.basename(f2)}')

        pareto_results.append({
            'epsilon' : eps,
            'lpsp'    : lpsp,
            'cost_deg': cost_deg,
        })

    fpareto = plot_pareto_summary(pareto_results, output_root)
    print(f'\nPareto front : {fpareto}')

    csv_path = os.path.join(output_root, 'pareto_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('epsilon,lpsp_pct,cost_deg_eur\n')
        for p in pareto_results:
            f.write(f"{p['epsilon']:.4f},{p['lpsp']:.4f},{p['cost_deg']:.2f}\n")
    print(f'CSV : {csv_path}')

    return pareto_results


# =============================================================================
# EXÉCUTION DIRECTE
# =============================================================================
if __name__ == '__main__':
    import time
    t0 = time.time()
    pareto = run_pareto_milp(
        epsilons=np.array([0.01, 0.1, 0.99]),
        output_root='Results_MILP',
        N_week=168,
        verbose=True,
    )
    print(f'\nTotal : {(time.time()-t0)/3600:.2f} h')
    print(f'\n{"ε":>6}  {"LPSP (%)":>10}  {"Cost_deg (€)":>14}')
    print('─' * 36)
    for p in pareto:
        print(f"  {p['epsilon']:.2f}  {p['lpsp']:>10.2f}  {p['cost_deg']:>14.1f}")
