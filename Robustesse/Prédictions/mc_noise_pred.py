# -*- coding: utf-8 -*-
"""
Monte-Carlo du BRUIT DE PREVISION sur RB2(Pred).
=================================================
La pre-charge batterie de RB2(Pred) decide a partir de l'ENERGIE NETTE prevue
sur l'horizon H_PRE. On remplace la prevision OMNISCIENTE par une prevision
REALISTE = vrai futur + N(biais, sigma), (biais, sigma) issus du backtest LSTM
mesure a 18h (cf. Predictions profils/pv_profils_backtest_h18.py).

Ce script :
  1) TEST NUL : NOISE_ENABLE=False -> doit reproduire RB2(Pred) omniscient (83.49).
  2) MONTE-CARLO : NOISE_ENABLE=True, N graines independantes -> distribution du
     cout unifie (moyenne +/- ecart-type), comparee a l'omniscient et a RB2 nu.

Scoring = boucle FORECAST de Predictions + metrics() (copie exacte de
Analyse_sensibilite/sens_common.metrics) + cout unifie (VoLL=3, E_REF 25 ans).

Usage : python mc_noise_pred.py [N_SEEDS]   (defaut N=24)
"""
import os, sys, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RB2PRED_DIR = os.path.join(HERE, "RB2(Pred)")
# Predictions/ dans le path -> 'Common' resout vers Predictions/Common (coherent).
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY            # noqa: E402
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast  # noqa: E402
from Common.cost_fcn_total2 import get_cost_total                       # noqa: E402

# --- Constantes de cout unifie (cf. voll_common) ---
E_REF_KWH = 273380.731444   # energie nette planifiee 25 ans [kWh]
VOLL      = 3.0             # EUR/kWh, constant

# --- Strategie RB2(Pred) chargee sous un nom unique (evite tout conflit) ---
_spec = importlib.util.spec_from_file_location(
    "strat_rb2pred", os.path.join(RB2PRED_DIR, "get_optimal_action_RB.py"))
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)


def metrics(data):
    """LPSP [%] et cout de degradation [kEUR] -- COPIE EXACTE de sens_common.metrics."""
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc = data["alpha_fc"][:-1]; alpha_ely = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)), SoH_bat[~np.isnan(SoH_bat)])
    P_planned = (P_dc_load - P_dc_pv) / 1000.0
    P_real    = (P_dc_load - P_dc_pv) * (1 - lol) / 1000.0
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    cost = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
                          LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(cost)


def total_keur(lpsp, deg):
    return deg + VOLL * (lpsp / 100.0) * E_REF_KWH / 1000.0


def _score():
    data = init_and_run_loop_forecast(strat.get_optimal_action_RB, H_forecast=48, n_years=25)
    lpsp, deg = metrics(data)
    return lpsp, deg, total_keur(lpsp, deg)


def run_noise_seed(seed):
    """Un run MC : prevision bruitee, graine `seed`. Fonction top-level (picklable)."""
    strat.NOISE_ENABLE = True
    strat.set_noise_seed(seed)
    return (seed,) + _score()


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 24

    print("=" * 70)
    print("RB2(Pred) -- bruit de prevision  (sigma=%.2f kWh, biais=%.2f kWh @18h)"
          % (strat.SIGMA_E_KWH, strat.BIAS_E_KWH))
    print("=" * 70)

    # 1) TEST NUL : omniscient (NOISE_ENABLE=False)
    strat.NOISE_ENABLE = False
    t0 = time.time()
    lpsp0, deg0, tot0 = _score()
    print("\n[NUL] omniscient (NOISE_ENABLE=False) : "
          "total=%.3f kEUR | LPSP=%.4f%% | deg=%.4f  (%.0fs)"
          % (tot0, lpsp0, deg0, time.time() - t0))
    print("      (reference attendue RB2(Pred) omniscient ~ 83.49)")

    # 2) MONTE-CARLO : prevision bruitee
    print("\n[MC] %d graines, prevision bruitee ..." % n_seeds)
    t0 = time.time()
    workers = max(1, (os.cpu_count() or 2) - 1)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run_noise_seed, range(n_seeds)):
            res.append(r)
            s_, lpsp_, deg_, tot_ = r
            print("   seed %3d : total=%.3f | LPSP=%.4f%% | deg=%.4f" % (s_, tot_, lpsp_, deg_))
    print("   (%.0fs, %d workers)" % (time.time() - t0, workers))

    arr = np.array([[a, b, c] for (_, a, b, c) in res])  # lpsp, deg, total
    lpsp_m, deg_m, tot_m = arr.mean(axis=0)
    lpsp_s, deg_s, tot_s = arr.std(axis=0, ddof=1)

    print("\n" + "-" * 70)
    print("RESULTATS (N=%d)" % n_seeds)
    print("-" * 70)
    print("  RB2 nu (reference)            : 85.548 kEUR  (LPSP 2.454%)")
    print("  RB2(Pred) omniscient [NUL]    : %.3f kEUR  (LPSP %.4f%%)" % (tot0, lpsp0))
    print("  RB2(Pred) bruite  [MC moyenne]: %.3f +/- %.3f kEUR" % (tot_m, tot_s))
    print("                       LPSP     : %.4f +/- %.4f %%" % (lpsp_m, lpsp_s))
    print("                       deg      : %.4f +/- %.4f kEUR" % (deg_m, deg_s))
    print("  min / max total sur les runs  : %.3f / %.3f" % (arr[:, 2].min(), arr[:, 2].max()))
    print("\n  Degradation du gain due au bruit :")
    print("    gain omniscient vs RB2 nu  : %.3f kEUR" % (85.548 - tot0))
    print("    gain bruite    vs RB2 nu  : %.3f kEUR (moyenne)" % (85.548 - tot_m))
    frac = (85.548 - tot_m) / (85.548 - tot0) * 100 if (85.548 - tot0) != 0 else float('nan')
    print("    -> le bruit conserve %.1f%% du gain de l'omniscient" % frac)


if __name__ == "__main__":
    main()
