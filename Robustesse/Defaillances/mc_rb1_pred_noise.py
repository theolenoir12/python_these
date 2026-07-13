"""
mc_rb1_pred_noise.py -- ETAPE 2 (b.1) : reglage des DISPOSITIFS ANTI-BRUIT du
levier RESERVE de RB1(Pred), sous defaillance, avec prevision BRUITEE.
=======================================================================
SOURCE 100% ASCII.

CADRE
-----
La GEOMETRIE du levier est FIGEE a son optimum OMNISCIENT (sweep_rb1_pred.py) :
    b_reserve = 0.99, h_pre = 18 h.
On DEGRADE maintenant la prevision par le bruit LSTM empirique
    net_pred = net_vrai + N(biais=-2.32 kWh, sigma=39.38 kWh)   (backtest 18h)
et on regle les DEUX dispositifs de robustesse :
    - hysteresis  : demi-bande = M_SIGMA * sigma, maintien MIN_DWELL ;
    - garde H2    : le levier ne s'active que si E_h2 > h2_gate * E_h2_init
                    (evite la famine FC sous panne ELY quand le bruit declenche
                     a tort la reserve, qui rationne la batterie en brulant du H2).

But : trouver la config (h2_gate, M_SIGMA) qui MAXIMISE le gain DEPLOYABLE (score
= LPSP-panne moyenne sur 4 scenarios) tout en EVITANT une regression sur un
scenario (typiquement ELY_total). Passe unique : 1 realisation de bruit par
tirage (seed = t0). L'ellipse (Monte-Carlo multi-realisations) + la sensibilite a
une misestimation de sigma sont dans mc_rb1_pred_ellipse.py, sur le gagnant.

LANCER (env conda simu_env, depuis ce dossier) :  python mc_rb1_pred_noise.py
"""
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import robustesse_common as rc
import rb1_pred_common as rp

# ======================= CONFIGURATION =======================
RB1_OPT   = rc.RB1_FAILOPT_THRESHOLDS
B_RESERVE = 0.99          # geometrie figee (optimum omniscient)
H_PRE     = 18
MIN_DWELL = 12            # maintien hysteresis (production RB2(Pred))
N_DRAWS   = 200
SEED      = 0
GRID_GATE   = [0.0, 0.5, 0.7, 0.8, 0.9]
GRID_MSIGMA = [0.5, 1.0, 1.5, 2.0]
OUT_TXT   = "mc_rb1_pred_noise.txt"
N_WORKERS = rc.N_WORKERS
# =============================================================


def _eval_over_draws(strat, times, derates):
    lp = np.empty(len(times))
    for i, t0 in enumerate(times):
        strat.reset()
        strat.set_noise_seed(int(t0))
        lp[i], _ = rp.run_week_pred(strat, rc._BL, int(t0), *derates, h_pre=H_PRE)
    return lp


def evaluate(task):
    """task = ('nu', None) ou (gate, m_sigma). Renvoie (task, lpsp_nom, {sk: arr})."""
    a, b = RB1_OPT
    rc._ensure_loaded()
    if task[0] == "nu":
        strat = rp.make_rb1_pred(a, b, enable=False)
    else:
        gate, ms = task
        strat = rp.make_rb1_pred(a, b, reserve=True, precharge=False,
                                 b_reserve=B_RESERVE, h2_gate=gate, h_pre=H_PRE,
                                 noise=True, hyst=True, m_sigma=ms, min_dwell=MIN_DWELL)
    times = rc._TIMES
    nom = _eval_over_draws(strat, times, (1.0, 1.0))
    per = {}
    for sk, (comp, sev) in rc.SCENARIOS.items():
        per[sk] = _eval_over_draws(strat, times, rc.derates_of(comp, sev))
    return task, nom, per


def main():
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)
    times = rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED)
    np.savez_compressed(rc.MC_SETUP_CACHE, t=times)

    scen_keys = list(rc.SCENARIOS.keys())
    tasks = [("nu", None)] + [(g, m) for g in GRID_GATE for m in GRID_MSIGMA]
    print("--- MC bruit RB1(Pred) : levier fige (b_res=%.2f, h_pre=%d), %d configs "
          "x %d tirages (%d workers) ---" % (B_RESERVE, H_PRE, len(tasks) - 1, N_DRAWS, N_WORKERS),
          flush=True)

    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for task, nom, per in ex.map(evaluate, tasks):
            res[task] = (nom, per)
            score = float(np.mean([per[sk].mean() for sk in scen_keys]))
            print("  %-16s score=%.3f%%  nom=%.3f%%"
                  % (str(task), score, nom.mean()), flush=True)
    print("--- %.0fs ---\n" % (time.time() - t0), flush=True)

    # Reference nu -------------------------------------------------------------
    nu_nom, nu_per = res[("nu", None)]
    nu_scen = {sk: nu_per[sk].mean() for sk in scen_keys}
    nu_score = float(np.mean(list(nu_scen.values())))

    # Classement des configs bruitees -----------------------------------------
    rows = []
    for task in tasks:
        if task[0] == "nu":
            continue
        gate, ms = task
        nom, per = res[task]
        scen = {sk: per[sk].mean() for sk in scen_keys}
        score = float(np.mean(list(scen.values())))
        # nb de scenarios sans regression (variante <= nu + eps)
        n_ok = sum(1 for sk in scen_keys if scen[sk] <= nu_scen[sk] + 1e-3)
        rows.append((score, gate, ms, nom.mean(), n_ok, scen))
    rows.sort(key=lambda x: x[0])

    lines = []
    lines.append("MC bruit RB1(Pred) -- levier RESERVE fige (b_reserve=%.2f, h_pre=%d, MIN_DWELL=%d)"
                 % (B_RESERVE, H_PRE, MIN_DWELL))
    lines.append("prevision BRUITEE : biais=%.2f kWh, sigma=%.2f kWh (backtest 18h) ; passe unique"
                 % (rp.BIAS_E_KWH, rp.SIGMA_E_KWH))
    lines.append("score = moyenne LPSP-panne 4 scenarios ; gain = nu - config ; n_ok = # scen sans regression")
    lines.append("")
    lines.append("REFERENCE RB1-opt nu : score=%.3f  nom=%.3f   [%s]"
                 % (nu_score, nu_nom.mean(),
                    "  ".join("%s=%.3f" % (k, nu_scen[k]) for k in scen_keys)))
    lines.append("REFERENCE omniscient (0.99/18, sweep) : score=1.433  gain_omni=+0.047")
    lines.append("")
    lines.append("  gate  M_sig  score    nom     gain    n_ok  " + "  ".join("%-9s" % k for k in scen_keys))
    for score, gate, ms, nom_m, n_ok, scen in rows:
        lines.append("  %.1f   %.1f    %7.3f  %6.3f  %+6.3f   %d/4  "
                     % (gate, ms, score, nom_m, nu_score - score, n_ok)
                     + "  ".join("%9.3f" % scen[k] for k in scen_keys))
    best = rows[0]
    # meilleur SANS regression (n_ok == 4), si existe
    best_clean = next((r for r in rows if r[4] == 4), None)
    lines.append("")
    lines.append("=== MEILLEUR score : gate=%.1f M_sigma=%.1f -> score=%.3f (gain %+.3f, %d/4 sans regression) ==="
                 % (best[1], best[2], best[0], nu_score - best[0], best[4]))
    if best_clean:
        lines.append("=== MEILLEUR SANS regression : gate=%.1f M_sigma=%.1f -> score=%.3f (gain %+.3f) ==="
                     % (best_clean[1], best_clean[2], best_clean[0], nu_score - best_clean[0]))
        frac = 100.0 * (nu_score - best_clean[0]) / 0.047
        lines.append("    => %.0f%% du gain omniscient (+0.047) recupere, sans degrader aucun scenario." % frac)

    txt = "\n".join(lines)
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print("\nTexte -> %s" % OUT_TXT)


if __name__ == "__main__":
    main()
