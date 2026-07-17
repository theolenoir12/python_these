"""
main_init_and_loop_maintenance.py -- BOUCLE 25 ANS AVEC FENETRES DE MAINTENANCE
================================================================================
SOURCE 100% ASCII (convention mesocentre).

Variante de main_init_and_loop.py pour le contexte INSULAIRE (proposition P3 de
Robustesse/ANALYSE_CRITIQUE_integration_vieillissement.txt) : les remplacements
ne sont plus instantanes mais n'ont lieu qu'aux VISITES de maintenance
periodiques. La boucle de base n'est PAS modifiee.

REGLES DU MODELE
----------------
- Visites tous les `visit_period_months` mois (730 h/mois). Un remplacement ne
  peut avoir lieu QU'A une visite.
- FC / ELY : a la traversee du seuil EoL, le composant est declare HORS SERVICE
  (stack non exploitable : tension trop basse). Il est retire du service
  (mecanisme `defaillances` : la strategie reporte sur la batterie, get_lol
  force sa puissance a zero), son etat de vieillissement est GELE, jusqu'au
  remplacement a la visite suivante.
- BATTERIE : pas de mort brutale ; sous l'EoL elle continue de fonctionner a
  capacite degradee (SoH_bat continue de baisser) jusqu'au remplacement a la
  visite suivante.
- POLITIQUES de remplacement a chaque visite :
    'instant'    : PAS de fenetres -- remplacement immediat a l'EoL, comme la
                   boucle de base (reference "continentale"). Chaque
                   remplacement compte 1 intervention (pas de groupage).
    'corrective' : remplace uniquement les composants en attente (morts).
    'calendar'   : + remplacement PREVENTIF si l'age du composant depassera
                   son age calendaire de reference avant la visite suivante
                   (ages de reference = durees de vie NOMINALES x marge,
                   fournis par l'appelant -- politique "sans pronostic").
    'rul'        : + remplacement PREVENTIF si la RUL ESTIMEE en ligne
                   (extrapolation lineaire du SoH, meme estimateur que la
                   boucle de base + ajout batterie) est inferieure a
                   l'intervalle jusqu'a la visite suivante x rul_margin.
- PERIMETRE DU PREVENTIF (prev_scope, defaut ('fc','ely')) : le preventif ne
  s'applique qu'aux composants a PANNE DURE (hors service a l'EoL -> outage).
  La batterie est une panne MOLLE par construction (elle continue a capacite
  degradee) : la remplacer preventivement n'evite aucun outage et ne fait que
  jeter de la vie residuelle -- diagnostic du run 214867 : waste rul 5.8 kEUR
  en moyenne, corr(waste, m_bat)=+0.76, entierement batterie. Le contrefactuel
  (batterie incluse) reste accessible via prev_scope=('bat','fc','ely').
- COMPTABILITE : une INTERVENTION = une visite ou au moins un remplacement a
  lieu (le cout fixe C_visite s'applique par intervention, en post-traitement:
  il n'influence pas la simulation). Le remplacement preventif jette la vie
  residuelle : waste = (SoH - EoL)/(1 - EoL) x cout_composant (0 si mort).

SORTIE : dict `data` identique a la boucle de base + cle 'maintenance' :
    n_visits, n_interventions, n_repl {bat,fc,ely}, n_prev {bat,fc,ely},
    outage_h {fc,ely}, waste_eur, waste_comp {bat,fc,ely},
    repl_log (liste (jour, comp, 'corr'/'prev')),
    failure_log (frontieres d'arret dur). Le dict principal contient aussi
    `degradation_ledger` en comptabilite corrigee.

NOTE deg : la boucle fournit un ledger des couts par unite physique. Pendant
un gel, l'etat et son cout restent donc geles ; un pas n'est jamais attribue
a la fois a l'unite retiree et a sa remplacante.
"""
import numpy as np
from scipy.optimize import brentq
from .Init_EMR_MG_v16_python import *
from .simulate_transition import simulate_transition
from .cost_fcn_total2 import *
from .cost_fcn_total2 import _ely_advance, UV_TO_PCT
from .cost_fcn_total2 import _fc_advance, UV_TO_PCT_FC
from .cost_fcn_total2 import cost_fc_state_eur, cost_ely_state_eur
from .replacement_ledger import ReplacementLedger
from .electrochemistry import fc_pmax, ely_pmax
from .lifetime_metrics import compute_first_life_metrics


def init_and_run_loop_maintenance(get_optimal_action_RB, n_years=25,
                                  visit_period_months=6.0, policy='corrective',
                                  rul_margin=1.0, calendar_ages_y=None,
                                  prev_scope=('fc', 'ely'),
                                  replacement_accounting='corrected'):
    """Cf. docstring module. calendar_ages_y = dict(bat=, fc=, ely=) en annees
    (None pour un composant = jamais de preventif calendaire). prev_scope =
    composants eligibles au remplacement PREVENTIF (defaut : pannes dures)."""
    assert policy in ('instant', 'corrective', 'calendar', 'rul')
    if replacement_accounting not in ('corrected', 'legacy_overlap'):
        raise ValueError('replacement_accounting inconnu : %s' % replacement_accounting)
    calendar_ages_y = calendar_ages_y or {}

    T = (SIM['Tend'] / 365) * 365 * n_years
    SoC_init  = 0.5
    E_h2_init = 200
    SoC_t = SoC_init
    plot = 1

    temps = np.arange(0, T - LOAD['Ts'], LOAD['Ts'])
    n     = len(temps)

    SoC       = np.zeros(n+1); SoC[0]  = SoC_init
    E_h2      = np.zeros(n+1); E_h2[0] = E_h2_init
    P_bat     = np.zeros(n)
    P_fc      = np.zeros(n)
    P_ely     = np.zeros(n)
    P_dc_load = np.zeros(n)
    P_dc_pv   = np.zeros(n)
    P_dc_bat  = np.zeros(n)
    P_dc_fc   = np.zeros(n)
    P_dc_ely  = np.zeros(n)
    alpha_fc  = np.zeros(n+1)
    alpha_ely = np.zeros(n+1)
    lol_tab   = np.zeros(n)
    SoH_bat   = np.zeros(n+1); SoH_bat[0] = 1
    SoH_fc    = np.zeros(n+1); SoH_fc[0]  = 1
    SoH_ely   = np.zeros(n+1); SoH_ely[0] = 1

    deg_fc  = {'start-stop': np.zeros(n), 'idling': np.zeros(n), 'reversible': np.zeros(n),
               'irreversible': np.zeros(n), 'total': np.zeros(n)}
    deg_ely = {'start-stop': np.zeros(n), 'maintaining': np.zeros(n), 'reversible': np.zeros(n),
               'irreversible': np.zeros(n), 'total': np.zeros(n)}

    ############ alphas EoL + LUT alpha = f(SoH) (identique boucle de base) ############
    def voltage_fc(a, i_fc):
        return FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + a) * i_fc / FC['n_parallel']
               - A * FC['T'] * np.log((i_fc / S / FC['n_parallel'] + j_in) / FC['j_0'])
               - B * FC['T'] * np.log(1 - i_fc / S / FC['n_parallel'] / FC['j_L'] / (1 - a)))

    i_fc_nom = 179.16718811881188
    V_bol_fc = voltage_fc(0.0, i_fc_nom)

    def residual_fc(a, SoH):
        return voltage_fc(a, i_fc_nom) / V_bol_fc - SoH

    def voltage_ely(a, i_ely):
        return ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + a) * i_ely / ELY['n_parallel']
               + A * ELY['T'] * np.log((i_ely / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
               + B * ELY['T'] * np.log(1 - i_ely / S / ELY['n_parallel'] / ELY['j_L'] / (1 - a)))

    i_ely_nom = 549.45
    V_bol_ely = voltage_ely(0.0, i_ely_nom)

    def residual_ely(a, SoH):
        return V_bol_ely / voltage_ely(a, i_ely_nom) - SoH

    _soh_grid = np.linspace(1.0, min(FC['SoH_EoL'], ELY['SoH_EoL']), 2000)
    _alpha_fc_grid  = np.array([brentq(residual_fc,  0.0, 0.274215, args=(s,), xtol=1e-10) for s in _soh_grid])
    _alpha_ely_grid = np.array([brentq(residual_ely, 0.0, 0.248679, args=(s,), xtol=1e-10) for s in _soh_grid])
    _soh_grid_flip       = _soh_grid[::-1]
    _alpha_fc_grid_flip  = _alpha_fc_grid[::-1]
    _alpha_ely_grid_flip = _alpha_ely_grid[::-1]
    #####################################################################################

    j_new_bat = 0
    j_new_fc  = 0
    j_new_ely = 0
    _rul_min_steps = int(20 * 3600 // LOAD['Ts'])
    j_rul_bat = 0
    j_rul_fc  = 0
    j_rul_ely = 0

    cum_bat = 0.0
    V_irr_fc = V_rev_fc = V_ss_fc = V_idle_fc = 0.0
    V_irr_ely = V_rev_ely = V_ss_ely = V_idle_ely = 0.0
    Ts_h = LOAD['Ts'] / 3600.0
    day_per_step = LOAD['Ts'] / 3600.0 / 24.0
    ledger = ReplacementLedger() if replacement_accounting == 'corrected' else None

    # --- etat maintenance ---
    visit_steps = max(1, int(round(visit_period_months * 730.0 * 3600.0 / LOAD['Ts'])))
    next_visit  = visit_steps
    avail_fc  = True
    avail_ely = True
    pending   = {'bat': False, 'fc': False, 'ely': False}
    maint = dict(n_visits=0, n_interventions=0,
                 n_repl={'bat': 0, 'fc': 0, 'ely': 0},
                 n_prev={'bat': 0, 'fc': 0, 'ely': 0},
                 outage_h={'fc': 0.0, 'ely': 0.0},
                 waste_eur=0.0, waste_comp={'bat': 0.0, 'fc': 0.0, 'ely': 0.0},
                 repl_log=[], failure_log=[])
    comp_cost = {'bat': BAT['cost'], 'fc': FC['cost'], 'ely': ELY['cost']}
    comp_eol  = {'bat': BAT['SoH_EoL'], 'fc': FC['SoH_EoL'], 'ely': ELY['SoH_EoL']}

    def _rul_days(j, j_anchor, soh_arr, eol):
        """Extrapolation lineaire du SoH depuis l'ancre (meme formule que la
        boucle de base) ; renvoie None si historique insuffisant ou SoH plat."""
        diff = j - j_anchor
        if diff < _rul_min_steps:
            return None
        delta = soh_arr[j_anchor] - soh_arr[j]
        if delta <= 1e-9:
            return None
        return (diff * (soh_arr[j_anchor] - eol) / delta - diff) * day_per_step

    for t in temps:
        j = int(t / LOAD['Ts'])

        # ============ VISITE DE MAINTENANCE (avant la decision du pas) ============
        if policy != 'instant' and j == next_visit:
            maint['n_visits'] += 1
            horizon_days = visit_steps * day_per_step
            to_replace = []   # (comp, 'corr'/'prev', soh_avant)
            for comp in ('bat', 'fc', 'ely'):
                soh_now = {'bat': SoH_bat, 'fc': SoH_fc, 'ely': SoH_ely}[comp][j]
                if pending[comp]:
                    to_replace.append((comp, 'corr', soh_now))
                    continue
                if comp not in prev_scope:
                    continue           # preventif reserve aux pannes dures
                if policy == 'rul':
                    anchor = {'bat': j_rul_bat, 'fc': j_rul_fc, 'ely': j_rul_ely}[comp]
                    arr    = {'bat': SoH_bat, 'fc': SoH_fc, 'ely': SoH_ely}[comp]
                    rul = _rul_days(j, anchor, arr, comp_eol[comp])
                    if rul is not None and rul < horizon_days * rul_margin:
                        to_replace.append((comp, 'prev', soh_now))
                elif policy == 'calendar':
                    age_ref = calendar_ages_y.get(comp)
                    j_new = {'bat': j_new_bat, 'fc': j_new_fc, 'ely': j_new_ely}[comp]
                    if age_ref is not None and (j - j_new + visit_steps) * day_per_step / 365.0 > age_ref:
                        to_replace.append((comp, 'prev', soh_now))
            for comp, kind, soh_before in to_replace:
                if ledger is not None:
                    current_cost = {
                        'bat': cum_bat,
                        'fc': cost_fc_state_eur(V_irr_fc, V_rev_fc, V_ss_fc, V_idle_fc),
                        'ely': cost_ely_state_eur(V_irr_ely, V_rev_ely, V_ss_ely, V_idle_ely),
                    }[comp]
                    ledger.retire(
                        comp, current_cost, j, 'maintenance_' + kind, soh_before
                    )
                if kind == 'prev':
                    maint['n_prev'][comp] += 1
                    waste = max(0.0, (soh_before - comp_eol[comp]) / (1 - comp_eol[comp])) * comp_cost[comp]
                    maint['waste_eur'] += waste
                    maint['waste_comp'][comp] += waste
                maint['n_repl'][comp] += 1
                maint['repl_log'].append((round(j * day_per_step, 1), comp, kind))
                pending[comp] = False
                if comp == 'bat':
                    SoH_bat[j] = 1.0
                    j_new_bat = j
                    j_rul_bat = j
                    cum_bat = 0.0
                elif comp == 'fc':
                    SoH_fc[j] = 1.0
                    alpha_fc[j] = 0.0
                    j_new_fc = j
                    j_rul_fc = j
                    V_irr_fc = V_rev_fc = V_ss_fc = V_idle_fc = 0.0
                    avail_fc = True
                elif comp == 'ely':
                    SoH_ely[j] = 1.0
                    alpha_ely[j] = 0.0
                    j_new_ely = j
                    j_rul_ely = j
                    V_irr_ely = V_rev_ely = V_ss_ely = V_idle_ely = 0.0
                    avail_ely = True
            if to_replace:
                maint['n_interventions'] += 1
            next_visit += visit_steps
        # ===========================================================================

        P_load_t    = LOAD['P_ref'][j]
        P_dc_load_t = P_load_t / CONV['eta']
        P_pv_t      = PV['P'][j]
        P_dc_pv_t   = P_pv_t
        P_tot_ref_t = (P_dc_load_t - P_dc_pv_t)

        alpha_fc_t  = alpha_fc[j]
        alpha_ely_t = alpha_ely[j]
        SoH_bat_t   = SoH_bat[j]
        SoH_fc_t    = SoH_fc[j]
        SoH_ely_t   = SoH_ely[j]
        E_h2_t      = E_h2[j]

        P_fc_max_t = float(fc_pmax(alpha_fc_t))
        P_ely_max_t = float(ely_pmax(alpha_ely_t))

        # RUL en ligne (exposees a la strategie, memes formules que la boucle de base)
        rul = _rul_days(j, j_rul_fc, SoH_fc, FC['SoH_EoL'])
        RUL_fc_t = rul if rul is not None else 8000
        rul = _rul_days(j, j_rul_ely, SoH_ely, ELY['SoH_EoL'])
        RUL_ely_t = rul if rul is not None else 3000

        # composants HS -> mecanisme defaillances (strategie + get_lol)
        defaillances_t = []
        if not avail_fc:
            defaillances_t.append('FC')
            maint['outage_h']['fc'] += Ts_h
        if not avail_ely:
            defaillances_t.append('ELY')
            maint['outage_h']['ely'] += Ts_h

        action, lol = get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances_t, lol_tab,
                                            alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t,
                                            E_h2_init, P_fc_max_t, P_ely_max_t,
                                            RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t)

        SoC_tp1, simOut = simulate_transition(SoC_t, action, P_tot_ref_t, plot, lol,
                                              alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t,
                                              E_h2_init, P_fc_max_t, P_ely_max_t)

        if SoC_tp1 < 0 or "P_bat" not in simOut:
            raise RuntimeError(
                "transition physique infaisable au pas %d (t=%.3f h): "
                "SoC=%.12g E_h2=%.12g P_ref=%.12g action=%r "
                "P_fc_max=%.12g P_ely_max=%.12g lol=%.12g policy=%s"
                % (
                    j, float(t) / 3600.0, SoC_t, E_h2_t, P_tot_ref_t,
                    tuple(float(x) for x in action), P_fc_max_t,
                    P_ely_max_t, lol, policy,
                )
            )

        P_bat[j]     = simOut['P_bat']
        P_fc[j]      = simOut['P_fc']
        P_ely[j]     = simOut['P_ely']
        P_dc_load[j] = P_dc_load_t
        P_dc_pv[j]   = P_dc_pv_t
        P_dc_bat[j]  = simOut['P_dc_bat']
        P_dc_fc[j]   = simOut['P_dc_fc']
        P_dc_ely[j]  = simOut['P_dc_ely']
        lol_tab[j]   = lol
        E_h2[j+1]    = simOut['E_h2_tp1']
        SoC[j+1]     = SoC_tp1

        # --- batterie : accumulation incrementale (toujours en service) ---
        if j == j_new_bat:
            m_bat = get_cost_bat(P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1])
        else:
            m_bat = (get_cost_bat(P_bat[j-1:j+1], SoC[j-1:j+2], SoH_bat[j-1:j+1])
                     - get_cost_bat(P_bat[j-1:j], SoC[j-1:j+1], SoH_bat[j-1:j]))
        cum_bat += m_bat

        # --- FC : avance stateful si en service, GEL sinon ---
        if avail_fc:
            P_prev_fc = P_fc[j] if j == j_new_fc else P_fc[j-1]
            V_irr_fc, V_rev_fc, d_ss_fc, d_idle_fc = _fc_advance(
                V_irr_fc, V_rev_fc, P_fc[j], P_prev_fc, P_fc_max_t, Ts_h, alpha_fc_t)
            V_ss_fc   += d_ss_fc
            V_idle_fc += d_idle_fc
        deg_irr_fc  = V_irr_fc  * UV_TO_PCT_FC
        deg_rev_fc  = V_rev_fc  * UV_TO_PCT_FC
        deg_ss_fc   = V_ss_fc   * UV_TO_PCT_FC
        deg_idle_fc = V_idle_fc * UV_TO_PCT_FC
        deg_pct_fc  = deg_irr_fc + deg_rev_fc + deg_ss_fc + deg_idle_fc

        # --- ELY : avance stateful si en service, GEL sinon ---
        if avail_ely:
            P_prev_ely = P_ely[j] if j == j_new_ely else P_ely[j-1]
            V_irr_ely, V_rev_ely, d_ss_ely, d_idle_ely = _ely_advance(
                V_irr_ely, V_rev_ely, P_ely[j], P_prev_ely, P_ely_max_t, Ts_h, alpha_ely_t)
            V_ss_ely   += d_ss_ely
            V_idle_ely += d_idle_ely
        deg_irr_ely  = V_irr_ely  * UV_TO_PCT
        deg_rev_ely  = V_rev_ely  * UV_TO_PCT
        deg_ss_ely   = V_ss_ely   * UV_TO_PCT
        deg_idle_ely = V_idle_ely * UV_TO_PCT
        deg_pct_ely  = deg_irr_ely + deg_rev_ely + deg_ss_ely + deg_idle_ely

        SoH_bat_tp1 = 1 - cum_bat / BAT['cost'] * (1 - BAT['SoH_EoL'])
        SoH_fc_tp1  = 1 - deg_pct_fc / 100
        SoH_ely_tp1 = 1 - deg_pct_ely / 100

        deg_fc['start-stop'][j]   = deg_ss_fc
        deg_fc['idling'][j]       = deg_idle_fc
        deg_fc['reversible'][j]   = deg_rev_fc
        deg_fc['irreversible'][j] = deg_irr_fc
        deg_fc['total'][j]        = deg_pct_fc
        deg_ely['start-stop'][j]   = deg_ss_ely
        deg_ely['maintaining'][j]  = deg_idle_ely
        deg_ely['reversible'][j]   = deg_rev_ely
        deg_ely['irreversible'][j] = deg_irr_ely
        deg_ely['total'][j]        = deg_pct_ely

        # ============ TRAVERSEE DE L'EoL ============
        if SoH_bat_tp1 < BAT['SoH_EoL']:
            if policy == 'instant':
                soh_retired = SoH_bat_tp1
                SoH_bat_tp1 = 1
                j_rul_bat = j + 1
                if replacement_accounting == 'corrected':
                    ledger.retire('bat', cum_bat, j + 1, 'instant_eol', soh_retired)
                    j_new_bat = j + 1
                    cum_bat = 0.0
                else:
                    j_new_bat = j
                    cum_bat = get_cost_bat(
                        P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1]
                    )
                maint['n_repl']['bat'] += 1
                maint['n_interventions'] += 1
                maint['repl_log'].append((round((j + 1) * day_per_step, 1), 'bat', 'inst'))
            elif not pending['bat']:
                pending['bat'] = True   # continue a capacite degradee jusqu'a la visite
        if SoH_fc_tp1 < FC['SoH_EoL']:
            if policy == 'instant':
                soh_retired = SoH_fc_tp1
                SoH_fc_tp1 = 1
                j_rul_fc = j + 1
                if replacement_accounting == 'corrected':
                    ledger.retire(
                        'fc', cost_fc_state_eur(V_irr_fc, V_rev_fc, V_ss_fc, V_idle_fc),
                        j + 1, 'instant_eol', soh_retired,
                    )
                    j_new_fc = j + 1
                    V_irr_fc = V_rev_fc = V_ss_fc = V_idle_fc = 0.0
                else:
                    j_new_fc = j
                    V_irr_fc, V_rev_fc, d_ss_fc, d_idle_fc = _fc_advance(
                        0.0, 0.0, P_fc[j], P_fc[j], P_fc_max_t, Ts_h, alpha_fc_t
                    )
                    V_ss_fc = d_ss_fc
                    V_idle_fc = d_idle_fc
                maint['n_repl']['fc'] += 1
                maint['n_interventions'] += 1
                maint['repl_log'].append((round((j + 1) * day_per_step, 1), 'fc', 'inst'))
            elif avail_fc:
                pending['fc'] = True
                avail_fc = False        # HS + gel jusqu'a la visite
                maint['failure_log'].append({
                    'component': 'fc', 'state_index': j + 1,
                    'day': (j + 1) * day_per_step, 'soh': float(SoH_fc_tp1),
                })
        if SoH_ely_tp1 < ELY['SoH_EoL']:
            if policy == 'instant':
                soh_retired = SoH_ely_tp1
                SoH_ely_tp1 = 1
                j_rul_ely = j + 1
                if replacement_accounting == 'corrected':
                    ledger.retire(
                        'ely', cost_ely_state_eur(V_irr_ely, V_rev_ely, V_ss_ely, V_idle_ely),
                        j + 1, 'instant_eol', soh_retired,
                    )
                    j_new_ely = j + 1
                    V_irr_ely = V_rev_ely = V_ss_ely = V_idle_ely = 0.0
                else:
                    j_new_ely = j
                    V_irr_ely, V_rev_ely, d_ss_ely, d_idle_ely = _ely_advance(
                        0.0, 0.0, P_ely[j], P_ely[j], P_ely_max_t, Ts_h, alpha_ely_t
                    )
                    V_ss_ely = d_ss_ely
                    V_idle_ely = d_idle_ely
                maint['n_repl']['ely'] += 1
                maint['n_interventions'] += 1
                maint['repl_log'].append((round((j + 1) * day_per_step, 1), 'ely', 'inst'))
            elif avail_ely:
                pending['ely'] = True
                avail_ely = False
                maint['failure_log'].append({
                    'component': 'ely', 'state_index': j + 1,
                    'day': (j + 1) * day_per_step, 'soh': float(SoH_ely_tp1),
                })

        # alphas : bornes de la LUT = EoL ; sous l'EoL (gel FC/ELY) on borne a
        # l'alpha EoL (np.interp sature en bord de grille -> OK).
        alpha_fc_tp1  = np.interp(SoH_fc_tp1,  _soh_grid_flip, _alpha_fc_grid_flip)
        alpha_ely_tp1 = np.interp(SoH_ely_tp1, _soh_grid_flip, _alpha_ely_grid_flip)

        SoH_bat[j+1]   = SoH_bat_tp1
        SoH_fc[j+1]    = SoH_fc_tp1
        SoH_ely[j+1]   = SoH_ely_tp1
        alpha_fc[j+1]  = alpha_fc_tp1
        alpha_ely[j+1] = alpha_ely_tp1

        SoC_t = SoC_tp1

    data = {
        "temps": temps, "n": n,
        "SoC": SoC, "E_h2": E_h2,
        "P_bat": P_bat, "P_fc": P_fc, "P_ely": P_ely,
        "P_dc_load": P_dc_load, "P_dc_pv": P_dc_pv,
        "P_dc_bat": P_dc_bat, "P_dc_fc": P_dc_fc, "P_dc_ely": P_dc_ely,
        "lol_tab": lol_tab,
        "alpha_fc": alpha_fc, "alpha_ely": alpha_ely,
        "SoH_bat": SoH_bat, "SoH_fc": SoH_fc, "SoH_ely": SoH_ely,
        "deg_fc": deg_fc, "deg_ely": deg_ely,
        "maintenance": maint,
        "replacement_accounting": replacement_accounting,
    }
    data['first_life_metrics'] = compute_first_life_metrics(data, LOAD['Ts'])
    if ledger is not None:
        current_eur = {
            'bat': cum_bat,
            'fc': cost_fc_state_eur(V_irr_fc, V_rev_fc, V_ss_fc, V_idle_fc),
            'ely': cost_ely_state_eur(V_irr_ely, V_rev_ely, V_ss_ely, V_idle_ely),
        }
        data['degradation_ledger'] = ledger.snapshot(current_eur, n)
    return data
