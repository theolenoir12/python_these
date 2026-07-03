"""
sweep_rb1_pred.py -- ETAPE 2 (a) : optimisation OMNISCIENTE de la geometrie du
levier RESERVE de RB1(Pred), sous defaillance.
=======================================================================
SOURCE 100% ASCII.

CADRE (valide avec l'auteur, juillet 2026)
------------------------------------------
On optimise le levier RESERVE en supposant la prevision PARFAITE (omnisciente) :
la strategie lit le VRAI futur `net_future = somme(charge - PV)` sur l'horizon
h_pre. Le bruit LSTM + Monte-Carlo de robustesse viendront DANS UN SECOND TEMPS ;
ici on cherche la MEILLEURE GEOMETRIE du levier (borne superieure du potentiel).

LEVIER RESERVE (cote decharge, orthogonal au SoC) : si un deficit net est prevu
sur h_pre, on ELARGIT la bande de melange (b=SOC_HIGH -> b_reserve) pour engager
la FC plus tot et PRESERVER une reserve batterie. b_reserve = SOC_HIGH (0.75) =
levier OFF = RB1-opt nu (reference incluse dans la grille).

La garde H2 (`h2_gate`) est un dispositif ANTI-BRUIT (evite la famine FC sous
panne ELY quand la prevision bruitee declenche a tort) -> mise a 0 ici (omniscient).

SWEEP : grille b_reserve x h_pre, evaluee a travers le harness MC de defaillance
(memes 200 tirages, baseline RB2, 4 scenarios). Score = moyenne des LPSP-panne
sur les 4 scenarios (== metrique de l'etape 1) ; garde-fou LPSP nominale.

SORTIES : sweep_rb1_pred.txt + results/sweep_rb1_pred_heatmap.pdf

LANCER (env conda simu_env, depuis ce dossier) :  python sweep_rb1_pred.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import robustesse_common as rc
import rb1_pred_common as rp

# ======================= CONFIGURATION =======================
RB1_OPT   = (0.40, 0.75)                            # seuils optimises (etape 1)
N_DRAWS   = 200
SEED      = 0
GRID_BRES = np.round(np.array([0.75, 0.80, 0.85, 0.90, 0.95, 0.99]), 3)  # 0.75 = nu
GRID_HPRE = np.array([6, 12, 18, 24, 48, 72])       # horizon de prevision [h]
TOL_NOM   = 0.02                                    # garde-fou nominal (points de %)

OUT_TXT   = "sweep_rb1_pred.txt"
OUT_PDF   = os.path.join(rc.RESULTS_DIR, "sweep_rb1_pred_heatmap.pdf")
N_WORKERS = rc.N_WORKERS
# =============================================================


def evaluate(args):
    """task = (b_reserve, h_pre). Levier RESERVE OMNISCIENT (noise=False, gate=0).
    Renvoie (b_reserve, h_pre, lpsp_nom, {scenario: lpsp_array})."""
    b_res, h_pre = args
    a, b = RB1_OPT
    rc._ensure_loaded()
    strat = rp.make_rb1_pred(a, b, reserve=True, precharge=False,
                             b_reserve=float(b_res), h2_gate=0.0,
                             noise=False, hyst=False, h_pre=int(h_pre))
    bl, times = rc._BL, rc._TIMES
    n = len(times)

    lpsp_nom = np.empty(n)
    for i, t0 in enumerate(times):
        strat.reset()
        lpsp_nom[i], _ = rp.run_week_pred(strat, bl, int(t0), 1.0, 1.0, h_pre=int(h_pre))

    per = {}
    for sk, (comp, sev) in rc.SCENARIOS.items():
        lp = np.empty(n)
        d = rc.derates_of(comp, sev)
        for i, t0 in enumerate(times):
            strat.reset()
            lp[i], _ = rp.run_week_pred(strat, bl, int(t0), d[0], d[1], h_pre=int(h_pre))
        per[sk] = lp
    return float(b_res), int(h_pre), lpsp_nom, per


def main():
    # Baseline + tirages (memes que l'etude principale) -----------------------
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)
    times = rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED)
    np.savez_compressed(rc.MC_SETUP_CACHE, t=times)

    combos = [(br, h) for br in GRID_BRES for h in GRID_HPRE]
    print("--- Sweep RB1(Pred) OMNISCIENT : %d couples (b_reserve, h_pre) x %d tirages "
          "x (%d scenarios + nominal) (%d workers) ---"
          % (len(combos), N_DRAWS, len(rc.SCENARIOS), N_WORKERS), flush=True)
    print("    b_reserve=%s  (0.75 = levier OFF = RB1-opt nu)" % GRID_BRES, flush=True)
    print("    h_pre=%s\n" % GRID_HPRE, flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, combos), 1):
            br, h, nom, per = r
            score = float(np.mean([per[sk].mean() for sk in rc.SCENARIOS]))
            results.append((br, h, nom, per, score))
            print("  [%2d/%d] b_res=%.2f h_pre=%2d -> score=%.3f%%  LPSP_nom=%.3f%%"
                  % (i, len(combos), br, h, score, nom.mean()), flush=True)
    print("--- %d sims en %.0fs ---\n" % (len(combos), time.time() - t0), flush=True)

    # Reference nu (b_reserve = SOC_HIGH = 0.75, tout h_pre equivalent) --------
    ref = next(r for r in results if abs(r[0] - RB1_OPT[1]) < 1e-9)
    ref_nom, ref_score = ref[2].mean(), ref[4]
    print("Reference RB1-opt nu (b_reserve=%.2f) : score=%.3f%%  LPSP_nom=%.3f%%"
          % (RB1_OPT[1], ref_score, ref_nom), flush=True)

    # Classement + garde-fou nominal ------------------------------------------
    scen_keys = list(rc.SCENARIOS.keys())
    rows = []
    for br, h, nom, per, score in results:
        nom_m = nom.mean()
        feas = nom_m <= ref_nom + TOL_NOM
        rows.append((score, br, h, nom_m, feas, {sk: per[sk].mean() for sk in scen_keys}))
    rows.sort(key=lambda x: x[0])
    feasibles = [r for r in rows if r[4]]
    best = feasibles[0] if feasibles else rows[0]

    # Sortie texte -------------------------------------------------------------
    with open(OUT_TXT, "w") as f:
        f.write("Sweep RB1(Pred) OMNISCIENT -- levier RESERVE, RB1-opt=(%.2f, %.2f)\n"
                % RB1_OPT)
        f.write("N_DRAWS=%d  SEED=%d  EVAL_HOURS=%d  gate=0 (omniscient)  TOL_NOM=%.2f\n"
                % (N_DRAWS, SEED, rc.EVAL_HOURS, TOL_NOM))
        f.write("score = moyenne des LPSP-panne moyennes sur les 4 scenarios [%]\n")
        f.write("b_reserve=%.2f (=SOC_HIGH) => levier OFF = RB1-opt nu : score=%.3f  nom=%.3f\n\n"
                % (RB1_OPT[1], ref_score, ref_nom))
        f.write("  b_res  h_pre   score    LPSP_nom  feas  gain    "
                + "  ".join("%-9s" % k for k in scen_keys) + "\n")
        for score, br, h, nom_m, feas, per in rows:
            f.write("  %.2f   %3d    %7.3f  %7.3f   %s  %+6.3f  "
                    % (br, h, score, nom_m, "ok" if feas else " -", ref_score - score)
                    + "  ".join("%9.3f" % per[k] for k in scen_keys) + "\n")
        f.write("\n=== MEILLEUR (feasible, score min) ===\n")
        f.write("  b_reserve=%.2f  h_pre=%d  |  score=%.3f%% (gain %+.3f vs nu)  LPSP_nom=%.3f%%\n"
                % (best[1], best[2], best[0], ref_score - best[0], best[3]))
        f.write("  detail : " + "  ".join("%s=%.3f" % (k, best[5][k]) for k in scen_keys) + "\n")

    # Heatmap score (b_reserve x h_pre) ---------------------------------------
    Z = np.full((len(GRID_BRES), len(GRID_HPRE)), np.nan)
    for score, br, h, nom_m, feas, per in rows:
        ib = int(np.argmin(np.abs(GRID_BRES - br)))
        ih = int(np.argmin(np.abs(GRID_HPRE - h)))
        Z[ib, ih] = score
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[0, len(GRID_HPRE), 0, len(GRID_BRES)])
    fig.colorbar(im, ax=ax, label="LPSP-panne moy. (4 scenarios) [%]")
    ax.set_xticks(np.arange(len(GRID_HPRE)) + 0.5); ax.set_xticklabels(GRID_HPRE)
    ax.set_yticks(np.arange(len(GRID_BRES)) + 0.5); ax.set_yticklabels(GRID_BRES)
    ib_best = int(np.argmin(np.abs(GRID_BRES - best[1])))
    ih_best = int(np.argmin(np.abs(GRID_HPRE - best[2])))
    ax.scatter([ih_best + 0.5], [ib_best + 0.5], c="crimson", marker="*", s=240,
               edgecolors="black", label="optimum omniscient", zorder=5)
    ax.set_xlabel("h_pre  [h]"); ax.set_ylabel("b_reserve")
    ax.set_title("RB1(Pred) levier RESERVE -- optimisation OMNISCIENTE\n"
                 "(b_reserve=0.75 = levier OFF = RB1-opt nu)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # Console ------------------------------------------------------------------
    print("\n=== MEILLEUR (omniscient, feasible) ===")
    print("  >>> b_reserve=%.2f  h_pre=%d" % (best[1], best[2]))
    print("      score=%.3f%%  gain %+.3f vs nu (%.3f%%)  |  LPSP_nom=%.3f%% (nu %.3f%%)"
          % (best[0], ref_score - best[0], ref_score, best[3], ref_nom))
    print("      detail : " + "  ".join("%s=%.3f" % (k, best[5][k]) for k in scen_keys))
    print("\nTexte  -> %s\nFigure -> %s" % (OUT_TXT, OUT_PDF))


if __name__ == "__main__":
    main()
