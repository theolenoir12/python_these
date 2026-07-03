"""
run_rb1_pred.py -- ETAPE 2 : la PREVISION de profils de puissance ameliore-t-elle
la robustesse sous defaillance de RB1-opt ?
=======================================================================
On part de RB1-opt (seuils optimises a l'etape 1, sweep_rb1.py) et on compare,
sur les MEMES 200 tirages / 4 scenarios de panne que l'etude principale :

  - RB1-opt (nu)            : reference (levier previsionnel desactive = test nul) ;
  - RB1-opt(Pred) OMNI      : pre-charge sur prevision PARFAITE (borne superieure) ;
  - RB1-opt(Pred) HYST      : pre-charge sur prevision BRUITEE (bruit LSTM 18h) +
                              hysteresis anti-clignotement -> variante DEPLOYABLE.

Le gain de robustesse attribuable a la prevision = LPSP-panne(nu) - LPSP-panne(Pred).
Test nul garanti : prevision neutre -> RB1-opt(Pred) == RB1-opt nu (cf. maxdiff=0).

LANCER (env conda simu_env, depuis ce dossier) :
    python run_rb1_pred.py
"""
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import robustesse_common as rc
import rb1_pred_common as rp

# ======================= CONFIGURATION =======================
RB1_OPT   = (0.40, 0.75)          # seuils optimises (sweep_rb1.py, plateau 1.481%)
N_DRAWS   = 200
SEED      = 0
VARIANTS  = ["nu", "omni", "hyst"]
OUT_TXT   = "run_rb1_pred.txt"
N_WORKERS = rc.N_WORKERS
# =============================================================


def make_variant(name):
    a, b = RB1_OPT
    if name == "nu":
        return rp.make_rb1_pred(a, b, enable=False)
    if name == "omni":
        return rp.make_rb1_pred(a, b, enable=True, noise=False, hyst=False)
    if name == "hyst":
        return rp.make_rb1_pred(a, b, enable=True, noise=True, hyst=True)
    raise ValueError(name)


def _eval_over_draws(strat, times, derates):
    lp = np.empty(len(times))
    for i, t0 in enumerate(times):
        strat.reset()
        strat.set_noise_seed(int(t0))               # bruit reproductible par tirage
        lp[i], _ = rp.run_week_pred(strat, rc._BL, int(t0), *derates)
    return lp


def evaluate(task):
    """task = (variant, scenario_key|'__nom__'). Renvoie (task, lpsp_array)."""
    variant, sk = task
    rc._ensure_loaded()
    strat = make_variant(variant)
    times = rc._TIMES
    if sk == "__nom__":
        derates = (1.0, 1.0)
    else:
        comp, sev = rc.SCENARIOS[sk]
        derates = rc.derates_of(comp, sev)
    return task, _eval_over_draws(strat, times, derates)


def main():
    # Baseline + tirages (memes que l'etude principale) -----------------------
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)
    times = rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED)
    np.savez_compressed(rc.MC_SETUP_CACHE, t=times)

    scen_keys = list(rc.SCENARIOS.keys())
    tasks = [(v, sk) for v in VARIANTS for sk in scen_keys]
    tasks += [(v, "__nom__") for v in VARIANTS]

    print("--- RB1(Pred) sous defaillance : RB1-opt=(%.2f, %.2f), %d tirages, "
          "%d taches (%d workers) ---" % (RB1_OPT[0], RB1_OPT[1], N_DRAWS, len(tasks), N_WORKERS),
          flush=True)
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for (task, lp) in ex.map(evaluate, tasks):
            res[task] = lp
            print("  %-5s %-9s : LPSP moy=%.3f%%" % (task[0], task[1], lp.mean()), flush=True)
    print("--- %.0fs ---\n" % (time.time() - t0), flush=True)

    # Tableau de synthese ------------------------------------------------------
    lines = []
    lines.append("RB1(Pred) sous defaillance -- RB1-opt=(%.2f, %.2f)  N_DRAWS=%d  EVAL_HOURS=%d"
                 % (RB1_OPT[0], RB1_OPT[1], N_DRAWS, rc.EVAL_HOURS))
    lines.append("LPSP-panne moyenne [%] par scenario ; gain = nu - variante (positif = mieux)")
    lines.append("")
    header = "  scenario     " + "".join("%-10s" % v for v in VARIANTS) \
             + "  gain_omni  gain_hyst  %recup"
    lines.append(header)
    score = {v: [] for v in VARIANTS}
    for sk in scen_keys:
        row = "  %-11s  " % sk
        vals = {v: res[(v, sk)].mean() for v in VARIANTS}
        for v in VARIANTS:
            row += "%-10.3f" % vals[v]
            score[v].append(vals[v])
        g_omni = vals["nu"] - vals["omni"]
        g_hyst = vals["nu"] - vals["hyst"]
        recup = (100.0 * g_hyst / g_omni) if abs(g_omni) > 1e-6 else float("nan")
        row += "  %+8.3f  %+8.3f  %5.0f%%" % (g_omni, g_hyst, recup)
        lines.append(row)
    # Moyenne 4 scenarios (== score de l'etape 1) ------------------------------
    m = {v: float(np.mean(score[v])) for v in VARIANTS}
    g_omni = m["nu"] - m["omni"]; g_hyst = m["nu"] - m["hyst"]
    recup = (100.0 * g_hyst / g_omni) if abs(g_omni) > 1e-6 else float("nan")
    lines.append("  " + "-" * 76)
    lines.append("  %-11s  " % "MOYENNE" + "".join("%-10.3f" % m[v] for v in VARIANTS)
                 + "  %+8.3f  %+8.3f  %5.0f%%" % (g_omni, g_hyst, recup))
    # Rappel nominal (contrefactuel sans panne) --------------------------------
    lines.append("")
    lines.append("  nominal (sans panne) : " +
                 "  ".join("%s=%.3f%%" % (v, res[(v, "__nom__")].mean()) for v in VARIANTS))

    txt = "\n".join(lines)
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print("\nTexte -> %s" % OUT_TXT)


if __name__ == "__main__":
    main()
