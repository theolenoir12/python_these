import inspect

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from .Init_EMR_MG_v16_python import *
from .simulate_transition import simulate_transition
from .get_soh import get_soh
from .cost_fcn_total2 import *
from .cost_fcn_total2 import UV_TO_PCT, UV_TO_PCT_FC
from .degradation_v11 import (
    advance_ely_power, advance_fc_power, aging_snapshot,
    new_ely_state, new_fc_state, reversible_uv,
    soh_operando, soh_permanent, state_cost_eur,
)
from .replacement_ledger import ReplacementLedger
from .electrochemistry import (
    FC_I_NOMINAL, ELY_I_NOMINAL, fc_pmax, ely_pmax,
    fc_voltage_cell, ely_voltage_cell,
)
from .lifetime_metrics import compute_first_life_metrics


def init_and_run_loop(get_optimal_action_RB, n_years=25,
                      replacement_accounting="corrected"):
    """Simule l'EMS avec une comptabilite explicite des remplacements.

    ``corrected`` (defaut) attribue chaque pas a une seule unite physique et
    fournit ``data['degradation_ledger']``. ``legacy_overlap`` reproduit le
    rejeu historique du pas de franchissement sur l'unite neuve ; il n'est
    conserve que pour diagnostiquer les anciennes sorties.
    """
    if replacement_accounting not in ("corrected", "legacy_overlap"):
        raise ValueError("replacement_accounting inconnu : %s" % replacement_accounting)

    # Les politiques previsionnelles portent un petit etat (hysteresis/bruit).
    # Chaque simulation doit repartir du meme etat, notamment dans les workers
    # reutilises par les balayages d'optimisation.
    reset_policy = getattr(get_optimal_action_RB, "reset", None)
    if callable(reset_policy):
        reset_policy()
    policy_parameters = inspect.signature(get_optimal_action_RB).parameters
    accepts_forecast = "P_tot_ref_future" in policy_parameters
    accepts_aging = "aging_context" in policy_parameters
    forecast_horizon_steps = getattr(
        get_optimal_action_RB, "forecast_horizon_steps", None
    )
    if accepts_forecast and forecast_horizon_steps is None:
        forecast_horizon_steps = max(1, int(round(48 * 3600 / LOAD["Ts"])))
    forecast_horizon_steps = int(forecast_horizon_steps or 0)

    # Initialisation des variables
    T = (SIM['Tend'] / 365)*365*n_years  # horizon de temps (defaut 25 ans)
    SoC_init  = 0.5  # état initial
    E_h2_init = 200
    SoC_t = SoC_init
    
    plot = 1
    
    temps = np.arange(0, T - LOAD['Ts'], LOAD['Ts'])
    n     = len(temps)
    
    SoC       = np.zeros(n+1); SoC[0]  = SoC_init
    E_h2      = np.zeros(n+1); E_h2[0] = E_h2_init # [kWh] hydrogène stocké
    P_bat     = np.zeros(n)
    P_fc      = np.zeros(n)
    P_ely     = np.zeros(n)
    P_dc_load = np.zeros(n)
    P_dc_pv   = np.zeros(n)
    P_dc_bat  = np.zeros(n)
    P_dc_fc   = np.zeros(n)
    P_dc_ely  = np.zeros(n)
    P_fc      = np.zeros(n)
    P_ely     = np.zeros(n)
    alpha_fc  = np.zeros(n+1); alpha_fc[0]  = 0
    alpha_ely = np.zeros(n+1); alpha_ely[0] = 0
    lol_tab   = np.zeros(n)
    SoH_bat   = np.zeros(n+1); SoH_bat[0] = 1
    SoH_fc    = np.zeros(n+1); SoH_fc[0]  = 1
    SoH_ely   = np.zeros(n+1); SoH_ely[0] = 1
    SoH_fc_operando = np.zeros(n+1); SoH_fc_operando[0] = 1
    SoH_ely_operando = np.zeros(n+1); SoH_ely_operando[0] = 1
    RUL_fc    = np.full(n, np.inf)
    RUL_ely   = np.full(n, np.inf)

    # Profil net reel mis a disposition des couches de prediction. La politique
    # ne recoit qu'une fenetre future et ne peut donc pas agir sur une autre
    # grandeur que ses deux consignes H2.
    profile_net = (
        np.asarray(LOAD["P_ref"][:n], dtype=float) / CONV["eta"]
        - np.asarray(PV["P"][:n], dtype=float)
    )
    
    deg_fc  = {'start-stop':np.zeros(n),'idling':np.zeros(n), 'reversible':np.zeros(n),
        'irreversible':np.zeros(n), 'total':np.zeros(n)}
    
    deg_ely = {'start-stop': np.zeros(n), 'maintaining': np.zeros(n), 'reversible': np.zeros(n),
        'irreversible':np.zeros(n), 'total':np.zeros(n)}
    
    defaillances = []
    
    if 'BAT' in defaillances :
        BAT['Ccapacity'] *= defaillances[defaillances.index('BAT')+1]
        BAT['Q_bat'] *= defaillances[defaillances.index('BAT')+1]

    ############################### GET ALPHAS AT EoL ########################
    i_fc_nom = FC_I_NOMINAL
    i_ely_nom = ELY_I_NOMINAL
    V_bol_fc = float(fc_voltage_cell(i_fc_nom, 0.0))
    V_bol_ely = float(ely_voltage_cell(i_ely_nom, 0.0))

    def residual_fc(alpha, SoH):
        return float(fc_voltage_cell(i_fc_nom, alpha)) / V_bol_fc - SoH

    def residual_ely(alpha, SoH):
        return V_bol_ely / float(ely_voltage_cell(i_ely_nom, alpha)) - SoH

    alpha_fc_eol  = brentq(residual_fc, 0.0, 0.274215, args=(FC['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    alpha_ely_eol = brentq(residual_ely, 0.0, 0.248679, args=(ELY['SoH_EoL'],), xtol=1e-10, rtol=1e-10)
    
        # --- LUT alpha = f(SoH) ---
    _soh_grid      = np.linspace(1.0, min(FC['SoH_EoL'], ELY['SoH_EoL']), 2000)
    _alpha_fc_grid  = np.array([brentq(residual_fc,  0.0, 0.274215,   args=(s,), xtol=1e-10) for s in _soh_grid])
    _alpha_ely_grid = np.array([brentq(residual_ely, 0.0, 0.248679, args=(s,), xtol=1e-10) for s in _soh_grid])
    # soh_grid est décroissant (1.0 → SoH_EoL), on le retourne pour interp qui exige x croissant
    _soh_grid_flip      = _soh_grid[::-1]
    _alpha_fc_grid_flip  = _alpha_fc_grid[::-1]
    _alpha_ely_grid_flip = _alpha_ely_grid[::-1]
    ########################################################################################################
    # alpha_fc_eol  = (1 - FC['SoH_EoL'])*1873.6207/1766.1207 #avec l'équation P_fc_max = f(alpha_fc) pour EoL
    # alpha_ely_eol = (1 - ELY['SoH_EoL'])*60122.0238/61392.7339 #avec 3 stacks mais pas d'importance
    
    j_new_bat = 0
    j_new_fc  = 0
    j_new_ely = 0
    _rul_min_steps = int(20 * 3600 // LOAD['Ts'])

    # Ancres RUL : index ou le SoH de l'unite COURANTE vaut 1 (unite neuve).
    # Distinctes de j_new_* (ancres de telescopage des couts) : un remplacement
    # decide au pas j ne met le SoH a 1 qu'a l'index j+1. Utiliser SoH[j_new]
    # (= SoH[j], valeur EoL ~0.9 de l'ANCIENNE unite) rendait delta_soh negatif
    # pendant toute la vie des unites suivantes -> RUL figee a sa valeur par
    # defaut apres le 1er remplacement (levier RB2(RUL) desactive en silence).
    j_rul_fc  = 0
    j_rul_ely = 0

    # Accumulateurs de coût de dégradation (cumul depuis le dernier remplacement).
    # Remplacent le recalcul O(n^2) de get_soh/get_cost_* sur [j_new:j+1] par une
    # accumulation incrementale O(1)/pas, EXACTEMENT equivalente (telescopage verifie).
    cum_bat = 0.0
    # V11 separe l'etat permanent (cout, remplacement) de l'etat reversible
    # (performance operando et information disponible pour l'EMS).
    fc_state = new_fc_state()
    ely_state = new_ely_state()
    Ts_h = LOAD['Ts'] / 3600.0
    range_useful_ely = (1 - ELY['SoH_EoL']) * 100
    ledger = ReplacementLedger() if replacement_accounting == "corrected" else None
    
    # --- Initialisation des tableaux ---
    data = {
        "temps": temps,
        "n": n,
        "SoC": np.zeros(n+1),
        "E_h2": np.zeros(n+1),
        "P_bat": np.zeros(n),
        "P_fc": np.zeros(n),
        "P_ely": np.zeros(n),
        "P_dc_load": np.zeros(n),
        "P_dc_pv": np.zeros(n),
        "P_dc_bat": np.zeros(n),
        "P_dc_fc": np.zeros(n),
        "P_dc_ely": np.zeros(n),
        "alpha_fc": np.zeros(n+1),
        "alpha_ely": np.zeros(n+1),
        "lol_tab": np.zeros(n),
        "SoH_bat": np.zeros(n+1),
        "SoH_fc": np.zeros(n+1),
        "SoH_ely": np.zeros(n+1),
        "RUL_fc": np.full(n, np.inf),
        "RUL_ely": np.full(n, np.inf),
        "deg_fc": {
            'start-stop': np.zeros(n), 'idling': np.zeros(n),
            'reversible': np.zeros(n), 'irreversible': np.zeros(n), 'total': np.zeros(n)
        },
        "deg_ely": {
            'start-stop': np.zeros(n), 'maintaining': np.zeros(n),
            'reversible': np.zeros(n), 'irreversible': np.zeros(n), 'total': np.zeros(n)
        }
    }

    for t in temps:
        
        # if int(t/T*100*10) == t/T*100*10 : 
        #     print("Temps global (%) :",t/T*100)
        
        j = int(t / LOAD['Ts'])
        P_load_t    = LOAD['P_ref'][j]
        P_dc_load_t = P_load_t / CONV['eta']
        P_pv_t      = PV['P'][j]
        P_dc_pv_t   = P_pv_t
        P_tot_ref_t   = (P_dc_load_t - P_dc_pv_t)
        
        
        alpha_fc_t  = alpha_fc[j]
        alpha_ely_t = alpha_ely[j]
        SoH_bat_t   = SoH_bat[j]
        SoH_fc_t    = SoH_fc[j]
        SoH_ely_t   = SoH_ely[j]
        E_h2_t      = E_h2[j]
    
        P_fc_max_t = float(fc_pmax(alpha_fc_t))
        P_ely_max_t = float(ely_pmax(alpha_ely_t))
    
        # CALCUL DU RUL OPTIMISÉ (sans get_rul)
        # Fuel Cell (ancre j_rul_fc : SoH[j_rul_fc] = 1 par construction)
        diff_j_fc = j - j_rul_fc
        if diff_j_fc >= _rul_min_steps:
            delta_soh = SoH_fc[j_rul_fc] - SoH_fc[j]
            if delta_soh > 1e-9: # Éviter division par zéro
                # Projection linéaire : (Temps_écoulé * Delta_total) / Delta_actuel - Temps_écoulé
                RUL_fc_t = (diff_j_fc * (SoH_fc[j_rul_fc] - FC['SoH_EoL']) / delta_soh - diff_j_fc) * LOAD['Ts'] / 3600 / 24
            else: RUL_fc_t = np.inf
        else: RUL_fc_t = np.inf

        # Électrolyseur (ancre j_rul_ely : SoH[j_rul_ely] = 1 par construction)
        diff_j_ely = j - j_rul_ely
        if diff_j_ely >= _rul_min_steps:
            delta_soh_ely = SoH_ely[j_rul_ely] - SoH_ely[j]
            if delta_soh_ely > 1e-9:
                RUL_ely_t = (diff_j_ely * (SoH_ely[j_rul_ely] - ELY['SoH_EoL']) / delta_soh_ely - diff_j_ely) * LOAD['Ts'] / 3600 / 24
            else: RUL_ely_t = np.inf
        else: RUL_ely_t = np.inf
        RUL_fc[j] = RUL_fc_t
        RUL_ely[j] = RUL_ely_t

        policy_args = (
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
            alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
            P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )
        aging_context = {
            "fc": aging_snapshot("fc", fc_state),
            "ely": aging_snapshot("ely", ely_state),
            "dt_h": Ts_h,
        }
        if accepts_forecast:
            if forecast_horizon_steps > 0:
                end_forecast = min(j + forecast_horizon_steps, n)
                future_net = profile_net[j:end_forecast]
            else:
                future_net = None
            if accepts_aging:
                action, lol = get_optimal_action_RB(
                    *policy_args, P_tot_ref_future=future_net,
                    aging_context=aging_context,
                )
            else:
                action, lol = get_optimal_action_RB(
                    *policy_args, P_tot_ref_future=future_net
                )
        elif accepts_aging:
            action, lol = get_optimal_action_RB(
                *policy_args, aging_context=aging_context
            )
        else:
            action, lol = get_optimal_action_RB(*policy_args)
              
        SoC_tp1, simOut = simulate_transition(SoC_t, action, P_tot_ref_t,plot,lol,alpha_fc_t,alpha_ely_t,SoH_bat_t, E_h2_t, E_h2_init,P_fc_max_t,P_ely_max_t)

        if SoC_tp1 < 0 or "P_bat" not in simOut:
            raise RuntimeError(
                "transition physique infaisable au pas %d (t=%.3f h): "
                "SoC=%.12g E_h2=%.12g P_ref=%.12g action=%r "
                "P_fc_max=%.12g P_ely_max=%.12g lol=%.12g"
                % (
                    j, float(t) / 3600.0, SoC_t, E_h2_t, P_tot_ref_t,
                    tuple(float(x) for x in action), P_fc_max_t,
                    P_ely_max_t, lol,
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
        
        # --- Coûts de dégradation : accumulation incrementale (== recalcul sur [j_new:j+1]) ---
        # Marginal d'ajout du point j au segment courant : f([j-1,j]) - f([j-1]).
        if j == j_new_bat:
            m_bat = get_cost_bat(P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1])
        else:
            m_bat = (get_cost_bat(P_bat[j-1:j+1], SoC[j-1:j+2], SoH_bat[j-1:j+1])
                     - get_cost_bat(P_bat[j-1:j],   SoC[j-1:j+1], SoH_bat[j-1:j]))
        cum_bat += m_bat

        # FC/ELY : avance des etats hybrides V11. Au premier pas d'une unite,
        # P_prev=P_curr conserve la convention historique sans faux demarrage.
        P_prev_fc = P_fc[j] if j == j_new_fc else P_fc[j-1]
        fc_state = advance_fc_power(
            fc_state, P_fc[j], P_prev_fc, alpha_fc_t, Ts_h
        )
        deg_irr_fc = fc_state["irreversible_uv"] * UV_TO_PCT_FC
        deg_rev_fc = reversible_uv(fc_state) * UV_TO_PCT_FC
        deg_ss_fc = fc_state["start_uv"] * UV_TO_PCT_FC
        deg_idle_fc = fc_state["idle_uv"] * UV_TO_PCT_FC
        deg_pct_fc  = deg_irr_fc + deg_rev_fc + deg_ss_fc + deg_idle_fc

        P_prev_ely = P_ely[j] if j == j_new_ely else P_ely[j-1]
        ely_state = advance_ely_power(
            ely_state, P_ely[j], P_prev_ely, alpha_ely_t, Ts_h
        )
        deg_irr_ely = ely_state["irreversible_uv"] * UV_TO_PCT
        # Le canal non permanent regroupe le reversible et le conditionnement
        # fini : tous deux affectent l'operando, aucun n'est capitalise.
        deg_rev_ely = (
            reversible_uv(ely_state) + ely_state["breakin_uv"]
        ) * UV_TO_PCT
        deg_ss_ely = ely_state["start_uv"] * UV_TO_PCT
        deg_idle_ely = ely_state["idle_uv"] * UV_TO_PCT
        deg_pct_ely  = deg_irr_ely + deg_rev_ely + deg_ss_ely + deg_idle_ely

        SoH_bat_tp1 = 1 - cum_bat    / BAT['cost'] * (1 - BAT['SoH_EoL'])
        SoH_fc_tp1 = soh_permanent("fc", fc_state)
        SoH_ely_tp1 = soh_permanent("ely", ely_state)
        SoH_fc_operando_tp1 = soh_operando("fc", fc_state)
        SoH_ely_operando_tp1 = soh_operando("ely", ely_state)

        deg_fc['start-stop'][j]    = deg_ss_fc
        deg_fc['idling'][j]        = deg_idle_fc
        deg_fc['reversible'][j]    = deg_rev_fc
        deg_fc['irreversible'][j]  = deg_irr_fc
        deg_fc['total'][j]         = deg_pct_fc

        deg_ely['start-stop'][j]    = deg_ss_ely
        deg_ely['maintaining'][j]   = deg_idle_ely
        deg_ely['reversible'][j]    = deg_rev_ely
        deg_ely['irreversible'][j]  = deg_irr_ely
        deg_ely['total'][j]         = deg_pct_ely
        
        if SoH_bat_tp1 < BAT['SoH_EoL']:
            soh_retired = SoH_bat_tp1
            SoH_bat_tp1 = 1
            if replacement_accounting == "corrected":
                ledger.retire("bat", cum_bat, j + 1, "instant_eol", soh_retired)
                # Le pas j appartient a l'ancienne unite ; la neuve commence a j+1.
                j_new_bat = j + 1
                cum_bat = 0.0
            else:
                # Convention historique : le pas j est rejoue sur l'unite neuve.
                j_new_bat = j
                cum_bat = get_cost_bat(
                    P_bat[j:j+1], SoC[j:j+2], SoH_bat[j:j+1]
                )
        if SoH_fc_tp1 < FC['SoH_EoL']:
            soh_retired = SoH_fc_tp1
            SoH_fc_tp1 = 1
            j_rul_fc = j + 1
            if replacement_accounting == "corrected":
                ledger.retire(
                    "fc", state_cost_eur("fc", fc_state),
                    j + 1, "instant_eol", soh_retired,
                )
                j_new_fc = j + 1
                fc_state = new_fc_state()
            else:
                j_new_fc = j
                fc_state = advance_fc_power(
                    new_fc_state(), P_fc[j], P_fc[j], alpha_fc_t, Ts_h
                )
        if SoH_ely_tp1 < ELY['SoH_EoL']:
            soh_retired = SoH_ely_tp1
            SoH_ely_tp1 = 1
            j_rul_ely = j + 1
            if replacement_accounting == "corrected":
                ledger.retire(
                    "ely", state_cost_eur("ely", ely_state),
                    j + 1, "instant_eol", soh_retired,
                )
                j_new_ely = j + 1
                ely_state = new_ely_state()
            else:
                j_new_ely = j
                ely_state = advance_ely_power(
                    new_ely_state(), P_ely[j], P_ely[j], alpha_ely_t, Ts_h
                )
        
        alpha_fc_tp1  = np.interp(SoH_fc_tp1,  _soh_grid_flip, _alpha_fc_grid_flip)
        alpha_ely_tp1 = np.interp(SoH_ely_tp1, _soh_grid_flip, _alpha_ely_grid_flip)
        
        # alpha_fc_tp1  = (1 - SoH_fc_tp1)/(1-FC['SoH_EoL'])*alpha_fc_eol
        # alpha_ely_tp1 = (1 - SoH_ely_tp1)/(1-ELY['SoH_EoL'])*alpha_ely_eol
            
        SoH_bat[j+1]   = SoH_bat_tp1
        SoH_fc[j+1]    = SoH_fc_tp1
        SoH_ely[j+1]   = SoH_ely_tp1
        SoH_fc_operando[j+1] = SoH_fc_operando_tp1
        SoH_ely_operando[j+1] = SoH_ely_operando_tp1
        alpha_fc[j+1]  = alpha_fc_tp1
        alpha_ely[j+1] = alpha_ely_tp1   
        
        SoC_t = SoC_tp1
    
    data["temps"] = temps
    data["SoC"] = SoC
    data["E_h2"] = E_h2
    data["P_bat"] = P_bat
    data["P_fc"] = P_fc
    data["P_ely"] = P_ely
    data["P_dc_load"] = P_dc_load
    data["P_dc_pv"] = P_dc_pv
    data["P_dc_bat"] = P_dc_bat
    data["P_dc_fc"] = P_dc_fc
    data["P_dc_ely"] = P_dc_ely
    data["lol_tab"] = lol_tab
    data["alpha_fc"] = alpha_fc
    data["alpha_ely"] = alpha_ely
    data["SoH_bat"] = SoH_bat
    data["SoH_fc"] = SoH_fc
    data["SoH_ely"] = SoH_ely
    data["SoH_fc_operando"] = SoH_fc_operando
    data["SoH_ely_operando"] = SoH_ely_operando
    data["RUL_fc"] = RUL_fc
    data["RUL_ely"] = RUL_ely
    data["deg_fc"] = deg_fc
    data["deg_ely"] = deg_ely
    data["replacement_accounting"] = replacement_accounting
    data["first_life_metrics"] = compute_first_life_metrics(data, LOAD["Ts"])
    if ledger is not None:
        current_eur = {
            "bat": cum_bat,
            "fc": state_cost_eur("fc", fc_state),
            "ely": state_cost_eur("ely", ely_state),
        }
        data["degradation_ledger"] = ledger.snapshot(current_eur, n)


    return data
