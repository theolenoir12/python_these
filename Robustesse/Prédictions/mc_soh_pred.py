# -*- coding: utf-8 -*-
"""
RB2(SoH)(Pred) : augmenter RB2(SoH) par la PREVISION, comme RB2(Pred) augmente
RB2 nu, et verifier que le levier (pre-charge) + sa robustesse au bruit (hyst.)
TRANSFERENT a la baseline modulee par l'etat de sante.

SIMULATION COURTE par defaut (n_years reduit) pour un temps de calcul raisonnable
-- la mesure 25 ans complete reste possible via l'argument N_YEARS.

Compare, a horizon fixe :
  [0] RB2(SoH) pur (ENABLE=False)         -> baseline (test nul)
  [1] RB2(SoH)+Pred OMNISCIENT (binaire)  -> borne sup. du levier previsionnel
  [2] RB2(SoH)+Pred BRUITE binaire        -> fragilite (clignotement ELY)
  [3] RB2(SoH)+Pred BRUITE + HYSTERESIS   -> robustesse (M_SIGMA=1.0/MIN_DWELL=12)

Scoring identique a mc_noise_hyst.py (boucle forecast + metrics + VoLL=3).
Usage : python mc_soh_pred.py [N_SEEDS] [N_YEARS]   (defaut N=8, n_years=5)
"""
import os, sys, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RB2SOH_DIR = os.path.join(HERE, "RB2(SoH)")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY            # noqa: E402
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast  # noqa: E402
from Common.cost_fcn_total2 import get_cost_total                       # noqa: E402

E_REF_KWH = 273380.731444
VOLL      = 3.0

_spec = importlib.util.spec_from_file_location(
    "strat_rb2soh", os.path.join(RB2SOH_DIR, "get_optimal_action_RB.py"))
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)

N_YEARS = 5  # ecrase par argv[2] dans main()


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
    on = np.abs(np.asarray(P_ely)) > tol
    return int(np.sum(on[1:] & ~on[:-1]))


def total_keur(lpsp, deg):
    return deg + VOLL * (lpsp / 100.0) * E_REF_KWH / 1000.0


def evaluate(args):
    """(enable, noise_on, hyst_on, m_sigma, min_dwell, seed, n_years) -> dict."""
    enable, noise_on, hyst_on, m_sigma, min_dwell, seed, n_years = args
    strat.ENABLE       = enable
    strat.NOISE_ENABLE = noise_on
    strat.HYST_ENABLE  = hyst_on
    strat.M_SIGMA      = m_sigma
    strat.MIN_DWELL    = min_dwell
    strat.set_noise_seed(seed)
    strat.reset()
    data = init_and_run_loop_forecast(strat.get_optimal_action_RB,
                                      H_forecast=48, n_years=n_years)
    lpsp, deg = metrics(data)
    return dict(lpsp=lpsp, deg=deg, total=total_keur(lpsp, deg),
                starts=count_ely_starts(data["P_ely"]))


def agg(rows):
    a = np.array([[r["total"], r["lpsp"], r["deg"], r["starts"]] for r in rows], float)
    m = a.mean(axis=0); s = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(4)
    return m[0], s[0], m[1], m[2], m[3]


def run_many(arglist, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evaluate, arglist))


def main():
    N  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else N_YEARS
    workers = max(1, (os.cpu_count() or 2) - 1)
    seeds = list(range(N))

    print("=" * 78)
    print("RB2(SoH)(Pred)  (sigma=%.2f kWh @18h, N=%d graines, simu %d ans)"
          % (strat.SIGMA_E_KWH, N, ny))
    print("=" * 78)
    t0 = time.time()

    # [0] baseline RB2(SoH) pur (ENABLE=False) : deterministe -> 1 run
    base = evaluate((False, False, False, 0.0, 0, 0, ny))
    print("\n[0] RB2(SoH) pur (baseline)      : total=%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%d"
          % (base["total"], base["lpsp"], base["deg"], base["starts"]))

    # [1] +Pred OMNISCIENT binaire : deterministe -> 1 run
    omni = evaluate((True, False, False, 0.0, 0, 0, ny))
    print("[1] +Pred OMNISCIENT (binaire)   : total=%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%d  (%+.3f vs base)"
          % (omni["total"], omni["lpsp"], omni["deg"], omni["starts"], omni["total"] - base["total"]))

    # [2] +Pred BRUITE binaire : Monte-Carlo
    nb = agg(run_many([(True, True, False, 0.0, 0, s, ny) for s in seeds], workers))
    print("[2] +Pred BRUITE binaire         : total=%.3f+/-%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%.0f  (%+.3f vs base)"
          % (nb[0], nb[1], nb[2], nb[3], nb[4], nb[0] - base["total"]))

    # [3] +Pred BRUITE + HYSTERESIS (optimum herite M=1.0/dwell=12) : Monte-Carlo
    nh = agg(run_many([(True, True, True, 1.0, 12, s, ny) for s in seeds], workers))
    print("[3] +Pred BRUITE + HYST (1.0/12) : total=%.3f+/-%.3f | LPSP=%.4f%% | deg=%.4f | ELYstarts=%.0f  (%+.3f vs base)"
          % (nh[0], nh[1], nh[2], nh[3], nh[4], nh[0] - base["total"]))

    print("\n" + "-" * 78)
    print("BILAN  (plus bas = mieux ; gain = baseline - variante)")
    print("-" * 78)
    print("  RB2(SoH) pur (baseline)        : %.3f" % base["total"])
    print("  +Pred omniscient (borne sup.)  : %.3f   (%+.3f)" % (omni["total"], base["total"] - omni["total"]))
    print("  +Pred bruite binaire (fragile) : %.3f   (%+.3f)" % (nb[0], base["total"] - nb[0]))
    print("  +Pred bruite + hysteresis      : %.3f   (%+.3f)" % (nh[0], base["total"] - nh[0]))
    print("  (%.0fs total)" % (time.time() - t0))


if __name__ == "__main__":
    main()
