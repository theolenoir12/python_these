"""
sweep_rb1.py -- ETAPE 1 : optimisation des seuils de RB1 pour la ROBUSTESSE
sous defaillance (composants H2).
=======================================================================
SOURCE 100% ASCII (cf. robustesse_common.py) : evite tout mojibake.

IDEE
----
RB1 (Vieillissement8/RB1/get_optimal_action_RB.py) est parametree implicitement
par DEUX seuils de SoC qui definissent une bande de melange lineaire batterie/H2 :
    - SoC_low  = 0.20 : en-dessous, la FC couvre TOUT le deficit (batterie protegee) ;
    - SoC_high = 0.60 : au-dessus, la batterie couvre TOUT le deficit (FC coupee),
                        ET l'ELY commence a absorber le surplus.
    - entre les deux, la fraction batterie monte lineairement (SoC-low)/(high-low).
Les pentes "5/2" du code d'origine sont exactement 1/(high-low) et 1/(1-high).

On BALAIE ce couple (SoC_low, SoC_high) et on evalue chaque variante A TRAVERS LE
MEME HARNESS Monte-Carlo de defaillance (robustesse_common) : memes 200 tirages
d'instants de panne, meme baseline RB2 figee, memes 4 scenarios (FC/ELY x total/50).
La variante candidate ne fait que REAGIR pendant la fenetre de panne, exactement
comme les strategies du nuage de Pareto -> comparaison appariee et coherente avec
results/robustesse_summary.txt.

SCORE (valide avec l'auteur, juillet 2026)
------------------------------------------
    score = MOYENNE, sur les 4 scenarios, de la LPSP-panne moyenne (sur 200 tirages)
avec un GARDE-FOU : on ne retient que les variantes dont la LPSP NOMINALE (marche
normale, meme fenetres) ne se degrade pas au-dela de RB1 ACTUELLE (0.20, 0.60),
a une tolerance TOL_NOM pres. Le point (0.20, 0.60) est INCLUS dans la grille : il
sert de reference (RB1-nu) et permet de lire directement le gain.

SORTIES
-------
    sweep_rb1.txt                 : tableau trie par score (feasible/best marques).
    results/sweep_rb1_heatmap.pdf : carte du score sur la grille (a, b).

LANCER (env conda simu_env, depuis ce dossier) :
    python sweep_rb1.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import robustesse_common as rc

# ======================= CONFIGURATION =======================
N_DRAWS   = 200                                    # tirages MC (== etude principale)
SEED      = 0                                       # memes tirages que run_robustesse
GRID_LOW  = np.round(np.linspace(0.10, 0.45, 8), 3)   # SoC_low  candidats
GRID_HIGH = np.round(np.linspace(0.50, 0.90, 9), 3)   # SoC_high candidats
MIN_GAP   = 0.15                                    # contrainte b - a >= MIN_GAP
TOL_NOM   = 0.02                                    # garde-fou : LPSP_nom <= ref + TOL (points de %)
RB1_DEFAULT = (0.20, 0.60)                          # RB1 actuelle (reference)

OUT_TXT   = "sweep_rb1.txt"
OUT_PDF   = os.path.join(rc.RESULTS_DIR, "sweep_rb1_heatmap.pdf")
N_WORKERS = rc.N_WORKERS
# =============================================================


# =============================================================================
# RB1 PARAMETREE : factory (cf. batch_optimize_rb2.make_rb2)
# =============================================================================
def make_rb1(soc_low, soc_high):
    """Renvoie une fonction d'action RB1 IDENTIQUE a l'originale mais parametree
    par (soc_low, soc_high). Signature 15 args (compatible harness)."""
    a = float(soc_low)
    b = float(soc_high)
    get_lol = rc.get_lol

    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if P_tot_ref_t > 0:                         # --- DEFICIT (decharge) ---
            if SoC_t <= a:
                frac = 0.0                          # FC couvre tout
            elif SoC_t >= b:
                frac = 1.0                          # batterie couvre tout
            else:
                frac = (SoC_t - a) / (b - a)        # melange lineaire
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_fc_t  = P_tot_ref_t - P_dc_bat_t
            P_dc_ely_t = 0.0
        else:                                        # --- SURPLUS (charge) ---
            if SoC_t <= b:
                frac = 1.0                          # batterie absorbe tout, ELY off
            elif SoC_t >= 1.0:
                frac = 0.0
            else:
                frac = (1.0 - SoC_t) / (1.0 - b)    # au-dela de b : ELY prend le reste
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_ely_t = P_tot_ref_t - P_dc_bat_t
            P_dc_fc_t  = 0.0

        # Clause defaillance interne (INERTE dans le harness : defaillances=[]).
        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t

        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol

    return get_optimal_action_RB


# =============================================================================
# WORKER : une tache = un couple (a, b) -> stats sur tous les tirages
# =============================================================================
def evaluate(args):
    """task = (soc_low, soc_high). Charge baseline+tirages (cache), simule les
    200 semaines NOMINALES et les 200 x 4 semaines de PANNE, renvoie les arrays
    LPSP (nominale + une par scenario)."""
    a, b = args
    rc._ensure_loaded()
    strat = make_rb1(a, b)
    bl, times = rc._BL, rc._TIMES
    n = len(times)

    lpsp_nom = np.empty(n)
    for i, t0 in enumerate(times):
        lpsp_nom[i], _ = rc.simulate_nominal_week(strat, bl, int(t0))

    per_scen = {}
    for sk, (comp, sev) in rc.SCENARIOS.items():
        lp = np.empty(n)
        for i, t0 in enumerate(times):
            lp[i], _ = rc.simulate_failure_week(strat, bl, int(t0), comp, sev)
        per_scen[sk] = lp

    return a, b, lpsp_nom, per_scen


# =============================================================================
# PILOTE
# =============================================================================
def build_grid():
    combos = [(a, b) for a in GRID_LOW for b in GRID_HIGH if b - a >= MIN_GAP - 1e-9]
    # Garantit la presence du point de reference RB1 (0.20, 0.60).
    if RB1_DEFAULT not in combos:
        combos.append(RB1_DEFAULT)
    return combos


def main():
    # 1. Baseline RB2 + tirages MC (memes que l'etude principale) --------------
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)
    times = rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED)
    np.savez_compressed(rc.MC_SETUP_CACHE, t=times)

    combos = build_grid()
    print("--- Sweep RB1 : %d couples (a,b) x %d tirages x (%d scenarios + nominal) "
          "(%d workers) ---" % (len(combos), N_DRAWS, len(rc.SCENARIOS), N_WORKERS),
          flush=True)
    print("    grille SoC_low=%s" % GRID_LOW, flush=True)
    print("    grille SoC_high=%s  |  contrainte b-a>=%.2f\n" % (GRID_HIGH, MIN_GAP), flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, combos), 1):
            a, b, nom, per = r
            score = float(np.mean([per[sk].mean() for sk in rc.SCENARIOS]))
            results.append((a, b, nom, per, score))
            print("  [%2d/%d] a=%.3f b=%.3f -> LPSP_panne_moy=%.3f%%  LPSP_nom=%.3f%%"
                  % (i, len(combos), a, b, score, nom.mean()), flush=True)
    print("--- %d sims en %.0fs ---\n" % (len(combos), time.time() - t0), flush=True)

    # 2. Reference RB1(0.20, 0.60) --------------------------------------------
    ref = next(r for r in results if (r[0], r[1]) == RB1_DEFAULT)
    ref_nom   = ref[2].mean()
    ref_score = ref[4]
    print("Reference RB1(0.20, 0.60) : score(panne)=%.3f%%  LPSP_nom=%.3f%%"
          % (ref_score, ref_nom), flush=True)

    # 3. Garde-fou nominal + classement ---------------------------------------
    rows = []
    for a, b, nom, per, score in results:
        nom_m = nom.mean()
        feasible = nom_m <= ref_nom + TOL_NOM
        rows.append((score, a, b, nom_m, feasible,
                     {sk: per[sk].mean() for sk in rc.SCENARIOS}))
    rows.sort(key=lambda x: x[0])

    feasibles = [r for r in rows if r[4]]
    best = feasibles[0] if feasibles else rows[0]

    # 4. Sortie texte ----------------------------------------------------------
    scen_keys = list(rc.SCENARIOS.keys())
    with open(OUT_TXT, "w") as f:
        f.write("Sweep RB1 (SoC_low, SoC_high) -- metrique DEFAILLANCE\n")
        f.write("N_DRAWS=%d  SEED=%d  EVAL_HOURS=%d  MIN_GAP=%.2f  TOL_NOM=%.2f\n"
                % (N_DRAWS, SEED, rc.EVAL_HOURS, MIN_GAP, TOL_NOM))
        f.write("score = moyenne des LPSP-panne moyennes sur les 4 scenarios [%]\n")
        f.write("feasible = LPSP_nom <= LPSP_nom(RB1 0.20/0.60)=%.3f%% + %.2f\n"
                % (ref_nom, TOL_NOM))
        f.write("reference RB1(0.20,0.60) : score=%.3f  LPSP_nom=%.3f\n\n"
                % (ref_score, ref_nom))
        header = "  a      b      score    LPSP_nom  feas  " + "  ".join("%-9s" % k for k in scen_keys)
        f.write(header + "\n")
        for score, a, b, nom_m, feas, per in rows:
            line = "  %.3f  %.3f  %7.3f  %7.3f   %s   " % (
                a, b, score, nom_m, " ok " if feas else "  - ")
            line += "  ".join("%9.3f" % per[k] for k in scen_keys)
            f.write(line + "\n")
        f.write("\n=== MEILLEUR (feasible, score min) ===\n")
        f.write("  SoC_low=%.3f  SoC_high=%.3f  |  score=%.3f%%  LPSP_nom=%.3f%%\n"
                % (best[1], best[2], best[0], best[3]))
        f.write("  gain score vs RB1(0.20,0.60) : %+.3f pts (%.1f%%)\n"
                % (best[0] - ref_score, 100.0 * (best[0] - ref_score) / ref_score))

    # 5. Heatmap du score ------------------------------------------------------
    Z = np.full((len(GRID_LOW), len(GRID_HIGH)), np.nan)
    for score, a, b, nom_m, feas, per in rows:
        ia = np.argmin(np.abs(GRID_LOW - a))
        ib = np.argmin(np.abs(GRID_HIGH - b))
        Z[ia, ib] = score
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[GRID_HIGH[0], GRID_HIGH[-1], GRID_LOW[0], GRID_LOW[-1]])
    fig.colorbar(im, ax=ax, label="LPSP-panne moyenne (4 scenarios) [%]")
    ax.scatter([RB1_DEFAULT[1]], [RB1_DEFAULT[0]], c="white", marker="o", s=90,
               edgecolors="black", label="RB1 actuelle (0.20/0.60)", zorder=5)
    ax.scatter([best[2]], [best[1]], c="crimson", marker="*", s=220,
               edgecolors="black", label="RB1-opt (%.2f/%.2f)" % (best[1], best[2]), zorder=6)
    ax.set_xlabel("SoC_high"); ax.set_ylabel("SoC_low")
    ax.set_title("Sweep RB1 -- robustesse sous defaillance\n(score = LPSP-panne moy. 4 scenarios)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # 6. Resume console --------------------------------------------------------
    print("\n=== MEILLEUR RB1 (feasible) ===")
    print("  >>> SoC_low=%.3f  SoC_high=%.3f" % (best[1], best[2]))
    print("      score(panne)=%.3f%% (ref %.3f%%, %+.3f)  |  LPSP_nom=%.3f%% (ref %.3f%%)"
          % (best[0], ref_score, best[0] - ref_score, best[3], ref_nom))
    print("      detail par scenario : " +
          "  ".join("%s=%.3f" % (k, best[5][k]) for k in scen_keys))
    print("\nTexte  -> %s" % OUT_TXT)
    print("Figure -> %s" % OUT_PDF)


if __name__ == "__main__":
    main()
