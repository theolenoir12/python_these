"""
mc_rb1_pred_ellipse.py -- ETAPE 2 (b.2) : dispersion Monte-Carlo (ellipse) et
sensibilite a une MISESTIMATION de sigma, sur la config DEPLOYABLE gagnante.
=======================================================================
SOURCE 100% ASCII.

Config gagnante (mc_rb1_pred_noise.py) : levier RESERVE
    b_reserve=0.99, h_pre=18, M_SIGMA=2.0, MIN_DWELL=12, h2_gate=0.5.

PHASE A -- ELLIPSE. On repete R fois l'evaluation (200 tirages de panne), chaque
repetition tirant des realisations de bruit INDEPENDANTES (seed = rep*K + t0).
-> distribution du score deployable (moyenne LPSP-panne 4 scenarios) : moyenne et
ecart-type. Comme les 200 tirages moyennent deja 200 realisations, l'agregat
auto-moyenne le bruit -> dispersion attendue faible (robustesse).

PHASE B -- MISESTIMATION DE SIGMA. La bande d'hysteresis reste calee sur le sigma
de DESIGN (39.38 kWh) tandis qu'on INJECTE un bruit de sigma different
(sigma_inject in [0.5..1.5]*39.38). Montre si le gain tient quand on a mal estime
le niveau de bruit reel. (cf. sens_pred_noise.py de RB2(Pred).)

LANCER (env conda simu_env) :  python mc_rb1_pred_ellipse.py
"""
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import robustesse_common as rc
import rb1_pred_common as rp

# ======================= CONFIGURATION =======================
RB1_OPT   = (0.40, 0.75)
WIN = dict(b_reserve=0.99, h_pre=18, h2_gate=0.5, m_sigma=2.0, min_dwell=12)
N_DRAWS   = 200
SEED      = 0
R_REPS    = 30                    # repetitions Monte-Carlo (realisations de bruit)
KSEED     = 100000                # decalage de graine entre repetitions
SIGMA_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]   # sigma_inject / sigma_design
OUT_TXT   = "mc_rb1_pred_ellipse.txt"
N_WORKERS = rc.N_WORKERS
# =============================================================
H_PRE = WIN["h_pre"]


def _mk(sigma_inject=None):
    a, b = RB1_OPT
    return rp.make_rb1_pred(a, b, reserve=True, precharge=False,
                            b_reserve=WIN["b_reserve"], h2_gate=WIN["h2_gate"],
                            h_pre=WIN["h_pre"], noise=True, hyst=True,
                            m_sigma=WIN["m_sigma"], min_dwell=WIN["min_dwell"],
                            sigma_inject=sigma_inject)


def _eval(strat, times, derates, seed_base):
    lp = np.empty(len(times))
    for i, t0 in enumerate(times):
        strat.reset()
        strat.set_noise_seed(int(seed_base + t0))
        lp[i], _ = rp.run_week_pred(strat, rc._BL, int(t0), *derates, h_pre=H_PRE)
    return lp


def eval_rep(rep):
    """Une repetition Monte-Carlo (realisation de bruit rep). Renvoie
    (rep, {scenario: mean_LPSP})."""
    rc._ensure_loaded()
    strat = _mk()
    times = rc._TIMES
    sb = rep * KSEED
    return rep, {sk: _eval(strat, times, rc.derates_of(*rc.SCENARIOS[sk]), sb).mean()
                 for sk in rc.SCENARIOS}


def eval_sigma(fac):
    """Une valeur de sigma_inject (misestimation). Passe unique (seed=t0)."""
    rc._ensure_loaded()
    strat = _mk(sigma_inject=fac * rp.SIGMA_E_KWH)
    times = rc._TIMES
    return fac, {sk: _eval(strat, times, rc.derates_of(*rc.SCENARIOS[sk]), 0).mean()
                 for sk in rc.SCENARIOS}


def eval_nu():
    rc._ensure_loaded()
    a, b = RB1_OPT
    strat = rp.make_rb1_pred(a, b, enable=False)
    times = rc._TIMES
    return {sk: _eval(strat, times, rc.derates_of(*rc.SCENARIOS[sk]), 0).mean()
            for sk in rc.SCENARIOS}


def main():
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)
    np.savez_compressed(rc.MC_SETUP_CACHE,
                        t=rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED))
    scen_keys = list(rc.SCENARIOS.keys())

    nu = eval_nu()
    nu_score = float(np.mean(list(nu.values())))

    # PHASE A : ellipse -------------------------------------------------------
    print("--- Ellipse : %d repetitions x %d tirages (%d workers) ---"
          % (R_REPS, N_DRAWS, N_WORKERS), flush=True)
    t0 = time.time()
    scores = []; per_scen = {sk: [] for sk in scen_keys}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for rep, scen in ex.map(eval_rep, range(R_REPS)):
            scores.append(float(np.mean(list(scen.values()))))
            for sk in scen_keys:
                per_scen[sk].append(scen[sk])
    scores = np.array(scores)
    print("  (%.0fs)" % (time.time() - t0), flush=True)

    # PHASE B : misestimation de sigma ----------------------------------------
    print("--- Sensibilite sigma : %d valeurs ---" % len(SIGMA_FACTORS), flush=True)
    sig_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for fac, scen in ex.map(eval_sigma, SIGMA_FACTORS):
            sig_rows.append((fac, float(np.mean(list(scen.values()))), scen))
    sig_rows.sort()

    # Sortie ------------------------------------------------------------------
    lines = []
    lines.append("MC bruit RB1(Pred) -- ELLIPSE + sensibilite sigma")
    lines.append("Config deployable : b_reserve=%.2f h_pre=%d M_SIGMA=%.1f MIN_DWELL=%d h2_gate=%.1f"
                 % (WIN["b_reserve"], WIN["h_pre"], WIN["m_sigma"], WIN["min_dwell"], WIN["h2_gate"]))
    lines.append("sigma_design=%.2f kWh ; %d repetitions de bruit" % (rp.SIGMA_E_KWH, R_REPS))
    lines.append("")
    lines.append("REFERENCE nu       : score=%.3f  [%s]"
                 % (nu_score, "  ".join("%s=%.3f" % (k, nu[k]) for k in scen_keys)))
    lines.append("REFERENCE omni     : score=1.433  (gain omniscient +0.047)")
    lines.append("")
    lines.append("--- ELLIPSE (moyenne +/- ecart-type sur %d realisations) ---" % R_REPS)
    lines.append("  score deployable = %.3f +/- %.3f %%   (gain = %+.3f +/- %.3f vs nu)"
                 % (scores.mean(), scores.std(), nu_score - scores.mean(), scores.std()))
    frac = 100.0 * (nu_score - scores.mean()) / 0.047
    lines.append("  => %.0f%% du gain omniscient recupere en moyenne ; dispersion du bruit tres faible"
                 % frac)
    for sk in scen_keys:
        arr = np.array(per_scen[sk])
        lines.append("    %-10s : %.3f +/- %.3f   (nu %.3f, gain %+.3f)"
                     % (sk, arr.mean(), arr.std(), nu[sk], nu[sk] - arr.mean()))
    lines.append("")
    lines.append("--- MISESTIMATION DE SIGMA (bande figee a sigma_design) ---")
    lines.append("  sig/sig0  score    gain     " + "  ".join("%-9s" % k for k in scen_keys))
    for fac, score, scen in sig_rows:
        lines.append("  %.2f      %7.3f  %+6.3f   " % (fac, score, nu_score - score)
                     + "  ".join("%9.3f" % scen[k] for k in scen_keys))
    lines.append("")
    gains = [nu_score - s for _, s, _ in sig_rows]
    lines.append("  gain min sur la plage sigma = %+.3f -> le levier reste %s."
                 % (min(gains), "benefique" if min(gains) > 0 else "fragile"))

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print("\nTexte -> %s" % OUT_TXT)


if __name__ == "__main__":
    main()
