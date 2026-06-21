"""
run_robustesse.py -- pilote de l'etude de robustesse (defaillances H2).
=======================================================================
Enchaine :
  1. construction (ou lecture cache) de la trajectoire RB2 -> regime permanent ;
  2. tirage Monte-Carlo des instants de panne (memes pour toutes les strategies) ;
  3. simulation de la semaine de panne pour chaque (scenario x strategie x tirage) ;
  4. sauvegarde des resultats (results/robustesse_results.npz) + tableau resume.

Lancer :  python run_robustesse.py
Tracer :  python plot_robustesse.py  (lit le .npz produit ici)

Parametres modifiables ci-dessous (N_DRAWS, SEED, STRATEGIES, SCENARIOS).
"""
import os
import numpy as np

import robustesse_common as rc

# --- Parametres de l'experience ----------------------------------------------
N_DRAWS    = 200          # tirages Monte-Carlo (instants de panne)
SEED       = 0            # reproductibilite
STRATEGIES = rc.DEFAULT_STRATEGIES
SCENARIOS  = rc.SCENARIOS
OUT_NPZ    = os.path.join(rc.RESULTS_DIR, "robustesse_results.npz")


def main():
    # 1. Regime permanent (RB2) -----------------------------------------------
    baseline = rc.run_baseline_rb2(years=rc.YEARS_BASELINE, cache=True)
    n_steps = len(baseline["temps"])
    # On (re)ecrit le cache baseline a l'emplacement attendu par les workers.
    np.savez_compressed(rc.BASELINE_CACHE, **baseline)

    # 2. Instants de panne (memes tirages pour TOUT le monde) -----------------
    times = rc.sample_failure_times(n_steps, N_DRAWS, seed=SEED)
    np.savez_compressed(rc.MC_SETUP_CACHE, t=times)
    yr = rc.I.LOAD["Ts"] / 3600.0 / 24.0 / 365.0
    print("Trajectoire RB2 : %d h (%.2f ans). Instants de panne dans [%d h, %d h] "
          "soit mois %.1f -> %.1f. Panne=%d h, mesure LPSP sur %d h."
          % (n_steps, n_steps * yr, rc.SETTLE_HOURS, n_steps - rc.EVAL_HOURS,
             rc.SETTLE_HOURS / 730.0, (n_steps - rc.EVAL_HOURS) / 730.0,
             rc.WEEK_HOURS, rc.EVAL_HOURS))
    print("%d tirages x %d scenarios x %d strategies = %d semaines simulees.\n"
          % (N_DRAWS, len(SCENARIOS), len(STRATEGIES),
             N_DRAWS * len(SCENARIOS) * len(STRATEGIES)))

    # 3a. Contrefactuel SANS panne (meme semaines) : 1 passe par strategie -----
    nom_res = rc.run_pool(rc.evaluate_nominal, STRATEGIES,
                          "Contrefactuel marche normale",
                          line_fmt=lambda r: "%-7s : LPSP moy=%.3f%%" % (r[0], r[1].mean()))
    lpsp_nom = {st: lp for st, lp, _ in nom_res}

    # 3b. Monte-Carlo AVEC panne ----------------------------------------------
    tasks = []
    for sc_key, (comp, sev) in SCENARIOS.items():
        for strat in STRATEGIES:
            tasks.append((sc_key, comp, sev, strat))
    results = rc.run_pool(
        rc.evaluate, tasks, "Monte-Carlo defaillances",
        line_fmt=lambda r: "%-10s %-7s : LPSP moy=%.3f%%  p90=%.3f%%  max=%.3f%%"
        % (r[0], r[1], r[2].mean(), np.percentile(r[2], 90), r[2].max()))

    # 4. Mise en forme : tableaux [scenario][strategie] -> (n_draws,) ----------
    lpsp = {sk: {} for sk in SCENARIOS}
    ens  = {sk: {} for sk in SCENARIOS}
    for sc_key, strat, lp, en in results:
        lpsp[sc_key][strat] = lp
        ens[sc_key][strat]  = en

    # Sauvegarde npz (cles "scenario|strategie")
    save = {"times": times, "n_steps": np.array([n_steps]),
            "strategies": np.array(STRATEGIES),
            "scenarios": np.array(list(SCENARIOS.keys()))}
    for st in STRATEGIES:
        save["lpsp_nom|%s" % st] = lpsp_nom[st]
    for sk in SCENARIOS:
        for st in STRATEGIES:
            save["lpsp|%s|%s" % (sk, st)] = lpsp[sk][st]
            save["ens|%s|%s"  % (sk, st)] = ens[sk][st]
    np.savez_compressed(OUT_NPZ, **save)
    print("\nResultats -> %s" % OUT_NPZ)

    # 5. Resume texte ---------------------------------------------------------
    print_summary(lpsp, ens, lpsp_nom, STRATEGIES, SCENARIOS)
    write_summary_txt(lpsp, ens, lpsp_nom, STRATEGIES, SCENARIOS)


def _stats(a):
    return (a.mean(), np.median(a), np.percentile(a, 90), a.max(),
            100.0 * np.mean(a > 1e-9))   # % de semaines avec LPSP > 0


def print_summary(lpsp, ens, lpsp_nom, strategies, scenarios):
    print("\n" + "=" * 86)
    print("RESUME -- LPSP sur la semaine de panne [%]")
    print("  colonnes : LPSP_panne (moy/med/p90/max) | LPSP_normale_meme_sem | surcout(moy)")
    print("=" * 86)
    for sk in scenarios:
        print("\n### Scenario : %s (%s)" % (rc.SCENARIO_LABELS.get(sk, sk), sk))
        rows = []
        for st in strategies:
            m, med, p90, mx, _ = _stats(lpsp[sk][st])
            nom = lpsp_nom[st].mean()
            delta = (lpsp[sk][st] - lpsp_nom[st]).mean()   # surcout de robustesse
            rows.append((m, st, med, p90, mx, nom, delta))
        rows.sort()   # tri par LPSP_panne moyenne croissante -> meilleure en tete
        for rank, (m, st, med, p90, mx, nom, delta) in enumerate(rows, 1):
            star = "  <-- MEILLEURE" if rank == 1 else ""
            print("  %2d. %-7s  %6.3f / %6.3f / %6.3f / %6.3f   | norm %6.3f | +%6.3f%s"
                  % (rank, rc.STRATEGY_LABELS.get(st, st), m, med, p90, mx, nom, delta, star))

    # Strategie "parmi les meilleures" par tirage (ex aequo PARTAGES) ----------
    # On compte une strategie gagnante sur un tirage si sa LPSP est a EPS pres du
    # minimum de ce tirage. Les semaines calmes (toutes ~0 %) sont donc partagees
    # entre toutes, au lieu d'etre attribuees a la 1re de la liste (artefact argmin).
    EPS = 0.05   # points de %
    print("\n" + "=" * 86)
    print("STRATEGIE PARMI LES MEILLEURES PAR TIRAGE (LPSP a %.2f pt du min ; ex aequo partages)"
          % EPS)
    print("=" * 86)
    for sk in scenarios:
        M = np.vstack([lpsp[sk][st] for st in strategies])   # (n_strat, n_draws)
        mins = M.min(axis=0)
        is_best = M <= (mins + EPS)                           # (n_strat, n_draws) bool
        frac = 100.0 * is_best.mean(axis=1)                   # % de tirages "parmi les meilleures"
        order = np.argsort(-frac)
        txt = ", ".join("%s:%.0f%%" % (rc.STRATEGY_LABELS.get(strategies[i], strategies[i]),
                                       frac[i]) for i in order)
        print("  %-14s -> %s" % (rc.SCENARIO_LABELS.get(sk, sk), txt))


def write_summary_txt(lpsp, ens, lpsp_nom, strategies, scenarios):
    path = os.path.join(rc.RESULTS_DIR, "robustesse_summary.txt")
    with open(path, "w") as f:
        f.write("Etude de robustesse -- LPSP sur la semaine de panne\n")
        f.write("N_DRAWS=%d  SEED=%d  YEARS=%g\n" % (N_DRAWS, SEED, rc.YEARS_BASELINE))
        f.write("surcout = LPSP_panne - LPSP_normale (meme semaine, meme strategie) >= 0\n\n")
        for sk in scenarios:
            f.write("### %s (%s)\n" % (rc.SCENARIO_LABELS.get(sk, sk), sk))
            f.write("  strategie  LPSP_moy  LPSP_med  LPSP_p90  LPSP_max  LPSP_norm  surcout  %>0  ENS_moy[kWh]\n")
            rows = []
            for st in strategies:
                m, med, p90, mx, pos = _stats(lpsp[sk][st])
                nom = lpsp_nom[st].mean()
                delta = (lpsp[sk][st] - lpsp_nom[st]).mean()
                rows.append((m, st, med, p90, mx, nom, delta, pos, ens[sk][st].mean()))
            rows.sort()
            for m, st, med, p90, mx, nom, delta, pos, e in rows:
                f.write("  %-9s %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %7.3f  %5.1f  %10.2f\n"
                        % (st, m, med, p90, mx, nom, delta, pos, e))
            f.write("\n")
    print("Resume texte -> %s" % path)


if __name__ == "__main__":
    main()
