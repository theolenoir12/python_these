"""
reopt_pred.py -- RE-OPTIMISATION de RB2(Pred) sur le socle cost-min de RB2.
==========================================================================
RB2(Pred) = RB2 (socle 0.440/0.310) + pre-charge batterie previsionnelle.
On re-regle le levier previsionnel sur ce socle (les setpoints restent figes,
cf. decision Theo). Metrique = cout unifie (VoLL=3, voll_common), identique au
reste des figures.

Phase 1 (rapide, deterministe) : sweep H_PRE en prevision OMNISCIENTE (binaire),
  + baseline RB2 nu (pre-charge OFF) sur le socle -> doit retomber sur ~80.11.
  -> meilleur H_PRE (cout unifie min).
Phase 2 (Monte-Carlo) : au meilleur H_PRE, version BRUITEE + HYSTERESIS (la seule
  legitime a tracer, cf README), point = MOYENNE sur N graines.

Lancement (depuis Predictions/) : python reopt_pred.py [N_MC]   (N_MC defaut 12)
"""
import os, sys, time, importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
from Common.cost_fcn_total2 import get_cost_total
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Analyse_sensibilite")))
import voll_common as V

_spec = importlib.util.spec_from_file_location(
    "rb2pred", os.path.join(HERE, "RB2(Pred)", "get_optimal_action_RB.py"))
strat = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(strat)

C_FC, C_ELY = 0.440, 0.310          # socle cost-min (fige)
# mode via argv[2] : 'pred' (gammas 0,0) ou 'sohpred' (gammas 1,2 = cost-min RB2(SoH))
MODE = sys.argv[2] if len(sys.argv) > 2 else 'pred'
GAMMA_FC, GAMMA_ELY = (1.0, 2.0) if MODE == 'sohpred' else (0.0, 0.0)
H_GRID  = [6, 12, 18, 24, 36]       # horizons de pre-charge a tester [h]
M_SIGMA_BEST, MIN_DWELL_BEST = 1.0, 12   # hysteresis (re-confirmee du travail anterieur)
OUT_TXT = os.path.join(HERE, f"reopt_{MODE}.txt")


def metrics(data):
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol = data["lol_tab"]
    SoC = data["SoC"]; af = data["alpha_fc"][:-1]; ae = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1: SoH_bat[k-1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)), SoH_bat[~np.isnan(SoH_bat)])
    p = np.clip((P_dc_load - P_dc_pv) / 1000.0, 0, None)
    r = np.clip((P_dc_load - P_dc_pv) * (1 - lol) / 1000.0, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    deg = get_cost_total(af, P_fc, ae, P_ely, P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(deg)


def _setup(enable, noise, hyst, h_pre, m_sigma, min_dwell, seed):
    strat.C_FC_BASE = C_FC; strat.C_ELY_BASE = C_ELY
    strat.GAMMA_FC = GAMMA_FC; strat.GAMMA_ELY = GAMMA_ELY
    strat.ENABLE = enable; strat.USE_FORECAST = enable
    strat.NOISE_ENABLE = noise; strat.HYST_ENABLE = hyst
    strat.H_PRE = h_pre; strat.M_SIGMA = m_sigma; strat.MIN_DWELL = min_dwell
    strat.set_noise_seed(seed); strat.reset()


def eval_omni(args):
    """(kind, h_pre) -> deterministe. kind='base' -> RB2 nu ; 'omni' -> pre-charge."""
    kind, h_pre = args
    _setup(enable=(kind != 'base'), noise=False, hyst=False, h_pre=h_pre,
           m_sigma=0.0, min_dwell=0, seed=0)
    d = init_and_run_loop_forecast(strat.get_optimal_action_RB, H_forecast=48, n_years=25)
    lp, dg = metrics(d)
    return (kind, h_pre, lp, dg, V.total_cost_keur(lp, dg))


def eval_mc(args):
    """(h_pre, seed) -> bruite+hysteresis, une graine."""
    h_pre, seed = args
    _setup(enable=True, noise=True, hyst=True, h_pre=h_pre,
           m_sigma=M_SIGMA_BEST, min_dwell=MIN_DWELL_BEST, seed=seed)
    d = init_and_run_loop_forecast(strat.get_optimal_action_RB, H_forecast=48, n_years=25)
    lp, dg = metrics(d)
    return (lp, dg, V.total_cost_keur(lp, dg))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    nw = max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    # --- Phase 1 : baseline + sweep H_PRE omniscient ---
    tasks = [('base', H_GRID[0])] + [('omni', h) for h in H_GRID]
    print(f"[Phase 1] baseline + sweep H_PRE omniscient ({len(tasks)} sims, {nw} workers)", flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        res1 = list(ex.map(eval_omni, tasks))
    base = next(r for r in res1 if r[0] == 'base')
    omnis = [r for r in res1 if r[0] == 'omni']
    for r in sorted(omnis, key=lambda r: r[1]):
        print(f"   H_PRE={r[1]:2d}h -> LPSP={r[2]:.4f}%  deg={r[3]:.3f}  UNIF={r[4]:.3f} k€", flush=True)
    print(f"   [baseline RB2 nu socle 0.440/0.310] LPSP={base[2]:.4f}% deg={base[3]:.3f} UNIF={base[4]:.3f} k€", flush=True)
    best = min(omnis, key=lambda r: r[4]); best_h = best[1]
    print(f"   >>> meilleur H_PRE (omniscient) = {best_h} h -> {best[4]:.3f} k€", flush=True)

    # --- Phase 2 : point bruite+hysteresis MC au meilleur H_PRE ---
    print(f"[Phase 2] bruite+hysteresis (M_SIGMA={M_SIGMA_BEST}, MIN_DWELL={MIN_DWELL_BEST}), "
          f"H_PRE={best_h}h, N={N} graines", flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        res2 = list(ex.map(eval_mc, [(best_h, s) for s in range(N)]))
    arr = np.array(res2)  # [lpsp, deg, total]
    mean = arr.mean(axis=0); std = arr.std(axis=0, ddof=1) if N > 1 else np.zeros(3)
    print(f"   RB2(Pred) bruite+hyst (moyenne MC N={N}) : "
          f"LPSP={mean[0]:.4f}+/-{std[0]:.4f}%  deg={mean[1]:.4f}+/-{std[1]:.4f}  "
          f"UNIF={mean[2]:.4f}+/-{std[2]:.4f} k€", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# RB2(Pred/SoH+Pred) re-optimise ; mode=%s gammas=(%.0f,%.0f) socle 0.440/0.310 ; VoLL=3\n" % (MODE, GAMMA_FC, GAMMA_ELY))
        f.write("# Phase 1 omniscient (deterministe) :\n")
        f.write("kind;H_PRE;LPSP(%);deg(kEUR);total(kEUR)\n")
        for r in [base] + sorted(omnis, key=lambda r: r[1]):
            f.write(f"{r[0]};{r[1]};{r[2]:.4f};{r[3]:.4f};{r[4]:.4f}\n")
        f.write(f"# best H_PRE omniscient = {best_h} h\n")
        f.write(f"# Phase 2 bruite+hyst MC (N={N}, M_SIGMA={M_SIGMA_BEST}, MIN_DWELL={MIN_DWELL_BEST}, H_PRE={best_h}) :\n")
        f.write(f"RB2(Pred);{best_h};{mean[0]:.4f};{mean[1]:.4f};{mean[2]:.4f}"
                f"  # +/- {std[0]:.4f} ; {std[1]:.4f} ; {std[2]:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
