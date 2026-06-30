# -*- coding: utf-8 -*-
"""
RB2(Pred) robuste au bruit : ANTI-CLIGNOTEMENT par HYSTERESIS.
==============================================================
Le bruit de prevision fait basculer la decision binaire net>0 d'un pas a
l'autre -> l'electrolyseur clignote (marche/arret) -> degradation start-stop.
On stabilise la coupure ELY par une hysteresis a deux seuils (bande +-M_SIGMA*
sigma sur l'energie nette prevue) + une duree minimale de maintien MIN_DWELL.

Ce script :
  1) References : omniscient binaire (~83.49), bruite binaire (~85.88).
  2) Sweep (M_SIGMA, MIN_DWELL) en BRUITE-HYSTERESIS, Monte-Carlo N graines.
  3) Plafond : omniscient-hysteresis au meilleur (M_SIGMA, MIN_DWELL).
Compte aussi les DEMARRAGES ELY (off->on) pour confirmer le mecanisme.

Scoring identique a mc_noise_pred.py (boucle forecast + metrics + VoLL=3).
Usage : python mc_noise_hyst.py [N_SEEDS]   (defaut N=4 ; std inter-graines ~0.13)
"""
import os, sys, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RB2PRED_DIR = os.path.join(HERE, "RB2(Pred)")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY            # noqa: E402
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast  # noqa: E402
from Common.cost_fcn_total2 import get_cost_total                       # noqa: E402

E_REF_KWH = 273380.731444
VOLL      = 3.0

_spec = importlib.util.spec_from_file_location(
    "strat_rb2pred", os.path.join(RB2PRED_DIR, "get_optimal_action_RB.py"))
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)


def metrics(data):
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


def count_ely_starts(P_ely, tol=1e-6):
    """Nombre de demarrages ELY (off -> on)."""
    on = np.abs(np.asarray(P_ely)) > tol
    return int(np.sum(on[1:] & ~on[:-1]))


def total_keur(lpsp, deg):
    return deg + VOLL * (lpsp / 100.0) * E_REF_KWH / 1000.0


def evaluate(args):
    """(noise_on, hyst_on, m_sigma, min_dwell, seed) -> dict de resultats.
    Fonction top-level (picklable). Reseed + reset AVANT chaque run."""
    noise_on, hyst_on, m_sigma, min_dwell, seed = args
    strat.NOISE_ENABLE = noise_on
    strat.HYST_ENABLE  = hyst_on
    strat.M_SIGMA      = m_sigma
    strat.MIN_DWELL    = min_dwell
    strat.set_noise_seed(seed)
    strat.reset()
    data = init_and_run_loop_forecast(strat.get_optimal_action_RB, H_forecast=48, n_years=25)
    lpsp, deg = metrics(data)
    return dict(noise=noise_on, hyst=hyst_on, m=m_sigma, dwell=min_dwell, seed=seed,
                lpsp=lpsp, deg=deg, total=total_keur(lpsp, deg),
                starts=count_ely_starts(data["P_ely"]))


def run_many(arglist, workers):
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, arglist):
            res.append(r)
    return res


def agg(rows):
    """Moyenne sur les graines : (total_m, total_s, lpsp_m, deg_m, starts_m)."""
    a = np.array([[r["total"], r["lpsp"], r["deg"], r["starts"]] for r in rows], float)
    m = a.mean(axis=0); s = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(4)
    return m[0], s[0], m[1], m[2], m[3]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    workers = max(1, (os.cpu_count() or 2) - 1)
    seeds = list(range(N))

    print("=" * 78)
    print("RB2(Pred) ANTI-CLIGNOTEMENT  (sigma=%.2f kWh @18h, N=%d graines)"
          % (strat.SIGMA_E_KWH, N))
    print("=" * 78)

    # --- References (1 run chacune, deterministes ou bruit moyenne) ---
    t0 = time.time()
    omni = evaluate((False, False, 0.0, 0, 0))                 # omniscient binaire
    print("\n[REF] omniscient binaire : total=%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%d"
          % (omni["total"], omni["lpsp"], omni["deg"], omni["starts"]))
    noisy_bin = agg(run_many([(True, False, 0.0, 0, s) for s in seeds], workers))
    print("[REF] bruite binaire     : total=%.3f+/-%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%.0f"
          % (noisy_bin[0], noisy_bin[1], noisy_bin[2], noisy_bin[3], noisy_bin[4]))
    print("      RB2 nu = 85.548 (LPSP 2.454%)")

    # --- Sweep hysteresis (bruite) ---
    M_GRID     = [0.5, 1.0, 1.5]
    DWELL_GRID = [0, 6, 12]
    print("\n[SWEEP] bruite-hysteresis  M_SIGMA x MIN_DWELL  (moyenne sur %d graines)" % N)
    print("  %-7s %-7s | %-9s %-7s | %-8s %-8s | %s"
          % ("M_SIG", "DWELL", "total", "+/-", "LPSP%", "deg", "ELYstarts"))
    results = {}
    for m in M_GRID:
        for d in DWELL_GRID:
            rows = run_many([(True, True, m, d, s) for s in seeds], workers)
            tm, ts, lp, dg, st = agg(rows)
            results[(m, d)] = (tm, ts, lp, dg, st)
            print("  %-7.2f %-7d | %-9.3f %-7.3f | %-8.4f %-8.4f | %.0f"
                  % (m, d, tm, ts, lp, dg, st))

    # --- Meilleur + plafond omniscient-hysteresis ---
    best = min(results, key=lambda k: results[k][0])
    bm, bs, blp, bdg, bst = results[best]
    print("\n[BEST bruite] M_SIGMA=%.2f MIN_DWELL=%d -> total=%.3f+/-%.3f (LPSP=%.4f%%, deg=%.4f, starts=%.0f)"
          % (best[0], best[1], bm, bs, blp, bdg, bst))
    omni_h = evaluate((False, True, best[0], best[1], 0))
    print("[PLAFOND] omniscient-hysteresis (memes params) : total=%.3f | LPSP=%.4f%% | deg=%.4f | starts=%d"
          % (omni_h["total"], omni_h["lpsp"], omni_h["deg"], omni_h["starts"]))

    print("\n" + "-" * 78)
    print("BILAN  (plus bas = mieux)")
    print("-" * 78)
    print("  RB2 nu                         : 85.548")
    print("  omniscient binaire             : %.3f" % omni["total"])
    print("  bruite binaire (fragile)       : %.3f" % noisy_bin[0])
    print("  bruite + hysteresis (best)     : %.3f   <-- robustesse" % bm)
    print("  gain bruite-hyst vs RB2 nu     : %+.3f kEUR" % (85.548 - bm))
    print("  gain bruite-bin  vs RB2 nu     : %+.3f kEUR" % (85.548 - noisy_bin[0]))
    print("  (%.0fs total)" % (time.time() - t0))


if __name__ == "__main__":
    main()
