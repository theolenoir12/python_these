"""
bench_maintenance.py -- VALEUR DE LA RUL POUR LA MAINTENANCE INSULAIRE
                        (fenetres de visite + cout fixe d'intervention).
=============================================================================
SOURCE 100% ASCII (convention mesocentre).

MOTIVATION (proposition P3 de ../ANALYSE_CRITIQUE_integration_vieillissement.txt)
----------------------------------------------------------------------------------
Le readme RB2(RUL) a etabli que la RUL est dominee par le SoH comme signal de
derating temps reel. Sa place naturelle est la PLANIFICATION : sur un site
insulaire, les remplacements n'ont lieu qu'aux visites de maintenance
periodiques et chaque intervention a un cout fixe. Un composant qui meurt
entre deux visites reste HORS SERVICE jusqu'a la suivante (LPSP degradee).
C'est la que le pronostic a une valeur mecanique : anticiper le remplacement
A LA BONNE VISITE. Hypotheses (a defaut de donnees O&M) : periode de visite
3-12 mois, cout fixe 0.5-3 kEUR/intervention -- balayees.

PROTOCOLE
---------
Strategie de conduite FIXE (RB2(SoH)) pour toutes les politiques : l'ecart
mesure est attribuable a la POLITIQUE DE REMPLACEMENT seule. Quatre politiques
(boucle Common/main_init_and_loop_maintenance.py) :
  instant     remplacement immediat a l'EoL (reference "continentale",
              identique a la boucle de base -- test nul integre) ;
  corrective  remplacement a la 1re visite APRES la mort (outage entre-temps) ;
  calendar    + preventif si l'age depassera l'age de reference avant la
              visite suivante. Ages de reference = durees de vie NOMINALES
              (mesurees sur le run nominal instant) x CAL_FRAC : c'est la
              politique preventive SANS pronostic, calee sur le modele ;
  rul         + preventif si la RUL ESTIMEE en ligne (extrapolation lineaire
              du SoH, batterie incluse) < intervalle jusqu'a la visite
              suivante x RUL_MARGIN : politique preventive AVEC pronostic.
En monde NOMINAL, calendar ~ rul (les durees de vie sont previsibles) : la
valeur du pronostic doit apparaitre SOUS INCERTITUDE du vieillissement --
memes 200 mondes perturbes que bench_valeur_info (CRN, meme graine) : les
resultats des deux bancs sont croisables tirage a tirage.

COMPTABILITE
------------
  cout unifie de base  uni0 = deg + VoLL*EENS                      [kEUR]
  cout maintenance     = C_visite * n_interventions + gaspillage   [kEUR]
      n_interventions  = visites avec au moins un remplacement (groupage :
                         2 composants remplaces a la meme visite = 1 visite) ;
      gaspillage       = vie residuelle jetee au remplacement preventif
                         (SoH-EoL)/(1-EoL) * cout composant ;
  cout total           = uni0 + cout maintenance.
  C_visite est un POST-TRAITEMENT (grille C_VISIT_GRID) : il n'influence pas
  la simulation -> 1 run par (politique, tirage).

MARGE RUL (constat de validation, a garder en tete pour la lecture) : la RUL
par extrapolation LINEAIRE est systematiquement OPTIMISTE pour la batterie,
dont la degradation ACCELERE en fin de vie (C-rate qui monte quand la capacite
baisse : a 6 mois de la mort, RUL_lin ~ 280 j pour 182 j reels sur le cas
teste). Avec marge 1.0 la politique 'rul' peut donc rater la derniere fenetre
sure ; la marge compense le biais de l'estimateur. Balayer --margin
{1.0, 1.5, 2.0} (jobs separes ; une marge != 1 est incluse dans le tag du
fichier -> chaque marge a son propre txt/cache, rien n'est ecrase). Le
compromis marge trop basse (morts) / trop haute (gaspillage) est un RESULTAT,
pas un reglage a cacher.

REUTILISATION : comme bench_valeur_info, le banc relit son txt et ne relance
que les couples (politique, tirage) manquants. --fresh pour tout re-payer.

SORTIES (a cote de ce script ; tag = <Ny>y_T<mois>m)
----------------------------------------------------
  maintenance_<tag>.txt       nominaux + table par tirage + stats par C_visite
  maintenance_<tag>.pdf/.png  (1) cout total moyen vs C_visite par politique
                              (2) histogramme apparie rul - calendar
                              (3) gain du pronostic vs severite du monde

LANCER
------
  local (fumee)        : python bench_maintenance.py --quick
  mesocentre (nominal) : sbatch run_meso_maintenance.slurm
                         (~800 runs 25 ans, ~8-10 h / 32 coeurs ; autres
                          periodes : sbatch run_meso_maintenance.slurm
                          --tvisit 3 / --tvisit 12, jobs separes)
Options : --tvisit MOIS | --margin X | --nmc N | --years N | --seed S |
          --lo x --hi x | --workers N | --fresh
"""
import os
import re
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reutilise l'infrastructure du banc P1 : mondes CRN, apply_world, metriques,
# chargement de strategie, ellipses. UNE seule source pour tout ca.
import bench_valeur_info as VI                                        # noqa: E402
from Common.main_init_and_loop import init_and_run_loop              # noqa: E402
from Common.main_init_and_loop_maintenance import init_and_run_loop_maintenance  # noqa: E402

# ============================ CONFIGURATION ============================
N_MC      = 200
MC_SEED   = 2026          # MEME graine que bench_valeur_info -> mondes identiques
MC_LO     = 0.5
MC_HI     = 2.0
N_YEARS   = 25
VOLL      = 3.0           # doit = VI.VOLL (verifie a l'import plus bas)

T_VISIT_M   = 6.0         # periode de visite [mois] (--tvisit)
RUL_MARGIN  = 1.0         # preventif si RUL_est < intervalle*marge (--margin)
CAL_FRAC    = 1.0         # age calendaire de ref = vie nominale x CAL_FRAC
C_VISIT_GRID = [0.5, 1.5, 3.0]   # cout fixe par intervention [kEUR] (post-proc)

STRATEGY  = "RB2(SoH)"    # strategie de conduite FIXE pour toutes les politiques
POLICIES  = ["instant", "corrective", "calendar", "rul"]

assert abs(VOLL - VI.VOLL) < 1e-12, "VoLL incoherent avec bench_valeur_info"
# ======================================================================


def evaluate(task):
    """task = dict(policy, world, draw, years, tvisit_m, margin, cal_ages).
    Installe le monde, simule avec la politique de maintenance, mesure."""
    try:
        VI.apply_world(task['world'])
        strat = VI.load_strategy(STRATEGY)
        if task['policy'] == 'instant':
            data = init_and_run_loop_maintenance(strat, n_years=task['years'],
                                                 policy='instant')
        else:
            data = init_and_run_loop_maintenance(
                strat, n_years=task['years'],
                visit_period_months=task['tvisit_m'], policy=task['policy'],
                rul_margin=task['margin'], calendar_ages_y=task['cal_ages'])
        lpsp, deg, eens, uni0 = VI.metrics(data)
        m = data['maintenance']
        ok = True
        out = dict(lpsp=lpsp, deg=deg, eens=eens, uni0=uni0,
                   nint=m['n_interventions'],
                   nrep=sum(m['n_repl'].values()),
                   nprev=sum(m['n_prev'].values()),
                   waste=m['waste_eur'] / 1000.0,
                   outfc=m['outage_h']['fc'], outely=m['outage_h']['ely'],
                   repl_log=m['repl_log'])
    except Exception as e:
        ok = False
        out = dict(lpsp=None, deg=None, eens=None, uni0=None, nint=None,
                   nrep=None, nprev=None, waste=None, outfc=None, outely=None,
                   repl_log=[])
        print("  [FAIL] %-10s draw=%s : %s" % (task['policy'], task['draw'], e), flush=True)
    out.update(policy=task['policy'], draw=task['draw'], world=task['world'], ok=ok)
    return out


def _fmt(r):
    if not r['ok']:
        return "%-10s draw=%-3s FAIL" % (r['policy'], r['draw'])
    return ("%-10s draw=%-3s LPSP %7.4f%%  uni0 %8.3f  nint %2d  waste %6.3f  outage %5.0f h"
            % (r['policy'], r['draw'], r['lpsp'], r['uni0'], r['nint'],
               r['waste'], r['outfc'] + r['outely']))


# --------------------- reutilisation d'un resultat precedent ---------------------
COLS = ["lpsp", "deg", "uni0", "nint", "nprev", "waste", "outfc", "outely"]


def load_previous(out_txt, seed, lo, hi, tvisit_m, margin):
    """{(policy, draw): result} depuis un txt precedent (header verifie)."""
    done, factors = {}, {}
    if not os.path.isfile(out_txt):
        return done, factors
    with open(out_txt, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    head = " ".join(lines[:7])
    m = re.search(r"mult U_log\[([0-9.]+),([0-9.]+)\] \| seed=(\d+) \| VoLL=([0-9.]+)"
                  r" \| Tvisite=([0-9.]+)m \| marge=([0-9.]+)", head)
    if not m:
        print("  (reuse) header illisible dans %s -> ignore" % out_txt, flush=True)
        return {}, {}
    vals = [float(m.group(i)) for i in range(1, 7)]
    if (abs(vals[0] - lo) > 1e-9 or abs(vals[1] - hi) > 1e-9 or int(vals[2]) != seed
            or abs(vals[3] - VOLL) > 1e-9 or abs(vals[4] - tvisit_m) > 1e-9
            or abs(vals[5] - margin) > 1e-9):
        print("  (reuse) header de %s different -> ignore" % out_txt, flush=True)
        return {}, {}
    section = None
    pol_order = []
    for l in lines:
        if l.startswith("## Points nominaux"):
            section = "nom"; continue
        if l.startswith("## Tirages"):
            section = "draws"; continue
        if l.startswith("##"):
            section = None; continue
        if not l or l.startswith("#"):
            continue
        parts = l.split(";")
        if section == "nom":
            if parts[0] == "policy" or "FAIL" in l:
                continue
            r = dict(policy=parts[0], draw=-1, world=dict(VI.NOMINAL_WORLD), ok=True, repl_log=[])
            for k, c in enumerate(COLS):
                r[c] = float(parts[1 + k])
            r['eens'] = (r['uni0'] - r['deg']) / VOLL * 1000.0
            r['nrep'] = None
            done[(parts[0], -1)] = r
        elif section == "draws":
            if parts[0] == "draw":
                pol_order = [c[:-5] for c in parts[8:] if c.endswith("_lpsp")]
                continue
            if not pol_order:
                continue
            d = int(parts[0])
            factors[d] = [float(v) for v in parts[1:8]]
            for k, pol in enumerate(pol_order):
                base = 8 + len(COLS) * k
                r = dict(policy=pol, draw=d, world=None, ok=True, repl_log=[])
                for kk, c in enumerate(COLS):
                    r[c] = float(parts[base + kk])
                r['eens'] = (r['uni0'] - r['deg']) / VOLL * 1000.0
                r['nrep'] = None
                done[(pol, d)] = r
    return done, factors


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Valeur de la RUL pour la maintenance insulaire")
    ap.add_argument("--quick", action="store_true", help="fumee locale : 2 ans, N_MC=4")
    ap.add_argument("--tvisit", type=float, default=T_VISIT_M, help="periode de visite [mois]")
    ap.add_argument("--margin", type=float, default=RUL_MARGIN)
    ap.add_argument("--nmc", type=int, default=None)
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--seed", type=int, default=MC_SEED)
    ap.add_argument("--lo", type=float, default=MC_LO)
    ap.add_argument("--hi", type=float, default=MC_HI)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    years = args.years or (2 if args.quick else N_YEARS)
    n_mc  = args.nmc if args.nmc is not None else (4 if args.quick else N_MC)
    workers = args.workers or VI._detect_workers()

    tag = "%dy_T%gm" % (years, args.tvisit)
    if abs(args.margin - 1.0) > 1e-12:
        tag += "_m%g" % args.margin       # une marge != 1 a son propre txt/cache
    out_txt = os.path.join(HERE, "maintenance_%s.txt" % tag)
    out_fig = os.path.join(HERE, "maintenance_%s" % tag)

    print("=== VALEUR DE LA RUL -- maintenance insulaire (fenetres + cout fixe) ===", flush=True)
    print("    horizon=%d ans | N_MC=%d | Tvisite=%gm | marge RUL=%g | mult U_log[%.2f, %.2f] | seed=%d"
          % (years, n_mc, args.tvisit, args.margin, args.lo, args.hi, args.seed), flush=True)

    # --- 1) nominal instant : ages calendaires de reference + TEST NUL vs boucle de base ---
    VI.apply_world(VI.NOMINAL_WORLD)
    strat = VI.load_strategy(STRATEGY)
    print("\n--- Run nominal 'instant' (ages calendaires + test nul vs boucle de base)...", flush=True)
    t0 = time.time()
    data_m = init_and_run_loop_maintenance(strat, n_years=years, policy='instant')
    data_b = init_and_run_loop(strat, n_years=years)
    lpsp_m, deg_m, _, uni_m = VI.metrics(data_m)
    VI.apply_world(VI.NOMINAL_WORLD)   # metrics ne change pas le monde, par surete
    lpsp_b, deg_b, _, uni_b = VI.metrics(data_b)
    gap = abs(uni_m - uni_b)
    print("    TEST NUL instant==base : |unifie %.3f - %.3f| = %.3e kEUR -> %s  (%.0fs)"
          % (uni_m, uni_b, gap, "OK" if gap < 1e-6 else "ECHEC", time.time() - t0), flush=True)
    lives = VI.lifetimes(data_m)                 # [bat, fc, ely], None si aucun
    cal_ages = {c: (l * CAL_FRAC if l is not None else None)
                for c, l in zip(('bat', 'fc', 'ely'), lives)}
    print("    vies nominales (1er remplacement) : bat=%s fc=%s ely=%s -> ages calendaires %s"
          % (lives[0], lives[1], lives[2],
             {k: (round(v, 2) if v else None) for k, v in cal_ages.items()}), flush=True)

    # --- 2) mondes CRN (memes tirages que bench_valeur_info) ---
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(args.lo), np.log(args.hi)
    worlds = [{k: float(np.exp(rng.uniform(lo, hi))) for k in VI.FACTOR_KEYS}
              for _ in range(n_mc)]

    done, prev_factors = ({}, {}) if args.fresh else load_previous(
        out_txt, args.seed, args.lo, args.hi, args.tvisit, args.margin)
    if done:
        bad = [d for d, fac in prev_factors.items() if d < n_mc and any(
            abs(fac[i] - worlds[d][k]) > 5e-4 for i, k in enumerate(VI.FACTOR_KEYS))]
        if bad:
            print("  (reuse) facteurs incoherents avec la graine -> reuse IGNORE", flush=True)
            done = {}
        else:
            done = {k: v for k, v in done.items() if k[1] < n_mc}
            for (p, d), r in done.items():
                r['world'] = dict(VI.NOMINAL_WORLD) if d == -1 else worlds[d]
            print("  (reuse) %d resultats repris de %s" % (len(done), out_txt), flush=True)

    # le run nominal 'instant' (test nul + ages calendaires) compte comme resultat
    if ('instant', -1) not in done:
        m0 = data_m['maintenance']
        done[('instant', -1)] = dict(
            policy='instant', draw=-1, world=dict(VI.NOMINAL_WORLD), ok=True,
            lpsp=lpsp_m, deg=deg_m, eens=(uni_m - deg_m) / VOLL * 1000.0, uni0=uni_m,
            nint=m0['n_interventions'], nrep=sum(m0['n_repl'].values()),
            nprev=sum(m0['n_prev'].values()), waste=m0['waste_eur'] / 1000.0,
            outfc=m0['outage_h']['fc'], outely=m0['outage_h']['ely'],
            repl_log=m0['repl_log'])

    def mk(policy, world, draw):
        return dict(policy=policy, world=world, draw=draw, years=years,
                    tvisit_m=args.tvisit, margin=args.margin, cal_ages=cal_ages)

    tasks = []
    for p in POLICIES:
        if (p, -1) not in done:
            tasks.append(mk(p, dict(VI.NOMINAL_WORLD), -1))
    for d, w in enumerate(worlds):
        for p in POLICIES:
            if (p, d) not in done:
                tasks.append(mk(p, w, d))

    print("\n--- %d runs a lancer (%d repris, %d workers) ---" % (len(tasks), len(done), workers), flush=True)
    t0 = time.time(); res = list(done.values())
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(evaluate, tasks), 1):
                res.append(r)
                print("  [%3d/%d] %s" % (i, len(tasks), _fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)

    # --- 3) tri + stats ---
    nom = {r['policy']: r for r in res if r['draw'] == -1 and r['ok']}
    by  = {p: {r['draw']: r for r in res if r['policy'] == p and r['draw'] >= 0 and r['ok']}
           for p in POLICIES}
    draws_ok = sorted(set.intersection(*[set(by[p].keys()) for p in POLICIES])) if all(by[p] for p in POLICIES) else []

    def col(p, key):
        return np.array([by[p][d][key] for d in draws_ok], dtype=float)

    def total(p, cv):
        """Cout total [kEUR] pour un cout de visite cv [kEUR/intervention]."""
        return col(p, 'uni0') + cv * col(p, 'nint') + col(p, 'waste')

    def cvar_hi(x, q=0.9):
        if len(x) == 0:
            return float('nan')
        thr = np.quantile(x, q)
        tail = x[x >= thr]
        return float(tail.mean()) if len(tail) else float('nan')

    if draws_ok:
        sev = np.array([np.exp(np.mean([np.log(worlds[d][k]) for k in VI.FACTOR_KEYS]))
                        for d in draws_ok])

    # --- 4) txt ---
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Valeur de la RUL -- maintenance insulaire (fenetres de visite + cout fixe)\n")
        f.write("# horizon=%d ans | N_MC=%d | mult U_log[%.2f,%.2f] | seed=%d | VoLL=%.1f | Tvisite=%gm | marge=%g\n"
                % (years, n_mc, args.lo, args.hi, args.seed, VOLL, args.tvisit, args.margin))
        f.write("# strategie de conduite FIXE : %s ; politiques : %s\n" % (STRATEGY, POLICIES))
        f.write("# ages calendaires de ref (nominal x %.2f) : %s\n" % (CAL_FRAC,
                {k: (round(v, 2) if v else None) for k, v in cal_ages.items()}))
        f.write("# cout total = uni0 + C_visite*n_interventions + gaspillage ; C_visite en post-proc\n")
        f.write("# TEST NUL instant==base : %s (ecart %.3e kEUR)\n" % ("OK" if gap < 1e-6 else "ECHEC", gap))
        f.write("\n## Points nominaux (multiplicateurs = 1)\n")
        f.write("policy;" + ";".join(COLS) + "\n")
        for p in POLICIES:
            r = nom.get(p)
            if r is None:
                f.write("%s;NOMINAL_FAIL\n" % p); continue
            f.write(p + ";" + ";".join("%.4f" % r[c] if c == "lpsp" else "%.3f" % r[c] for c in COLS) + "\n")
        f.write("\n## Tirages (mondes perturbes, %d complets)\n" % len(draws_ok))
        f.write("draw;" + ";".join(VI.FACTOR_KEYS)
                + ";" + ";".join(";".join("%s_%s" % (p, c) for c in COLS) for p in POLICIES) + "\n")
        for d in draws_ok:
            w = worlds[d]
            row = [str(d)] + ["%.4f" % w[k] for k in VI.FACTOR_KEYS]
            for p in POLICIES:
                r = by[p][d]
                row += ["%.4f" % r['lpsp']] + ["%.3f" % r[c] for c in COLS[1:]]
            f.write(";".join(row) + "\n")

        if draws_ok:
            for cv in C_VISIT_GRID:
                f.write("\n## Cout total (kEUR) a C_visite = %.1f kEUR/intervention\n" % cv)
                f.write("policy;mean;std;P5;P50;P95;CVaR90;LPSP_mean;nint_mean;waste_mean;outage_h_mean\n")
                for p in POLICIES:
                    u = total(p, cv)
                    f.write("%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.4f;%.2f;%.3f;%.0f\n"
                            % (p, u.mean(), u.std(), np.quantile(u, 0.05), np.quantile(u, 0.50),
                               np.quantile(u, 0.95), cvar_hi(u), col(p, 'lpsp').mean(),
                               col(p, 'nint').mean(), col(p, 'waste').mean(),
                               (col(p, 'outfc') + col(p, 'outely')).mean()))
                f.write("# differences appariees CRN (negatif = 1er gagne)\n")
                f.write("paire;mean;std;P5;P95;pct_gagne\n")
                for a, b in [("rul", "corrective"), ("rul", "calendar"),
                             ("calendar", "corrective"), ("rul", "instant")]:
                    dif = total(a, cv) - total(b, cv)
                    f.write("%s - %s;%.3f;%.3f;%.3f;%.3f;%.1f%%\n"
                            % (a, b, dif.mean(), dif.std(), np.quantile(dif, 0.05),
                               np.quantile(dif, 0.95), 100.0 * (dif < 0).mean()))

    # --- 5) figure ---
    if draws_ok:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
        colors = {"instant": "tab:gray", "corrective": "tab:red",
                  "calendar": "tab:blue", "rul": "tab:green"}
        cv_axis = np.linspace(0.0, max(C_VISIT_GRID), 25)
        ax = axes[0]
        for p in POLICIES:
            means = [total(p, cv).mean() for cv in cv_axis]
            ax.plot(cv_axis, means, color=colors[p], lw=1.8, label=p)
        for cv in C_VISIT_GRID:
            ax.axvline(cv, color='k', lw=0.6, ls=':', alpha=0.5)
        ax.set_xlabel("Cout fixe par intervention C_visite [kEUR]")
        ax.set_ylabel("Cout total moyen [kEUR]")
        ax.set_title("Cout total vs cout d'intervention (Tvisite=%gm)" % args.tvisit)
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

        ax = axes[1]
        cv0 = C_VISIT_GRID[len(C_VISIT_GRID) // 2]
        d1 = total("rul", cv0) - total("calendar", cv0)
        d2 = total("rul", cv0) - total("corrective", cv0)
        bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=25)
        ax.hist(d2, bins=bins, alpha=0.55, color="tab:red",  label="rul - corrective")
        ax.hist(d1, bins=bins, alpha=0.55, color="tab:blue", label="rul - calendar")
        ax.axvline(0.0, color='k', lw=0.9)
        ax.set_xlabel("Difference appariee de cout total [kEUR]  (<0 = la RUL gagne)")
        ax.set_ylabel("Tirages")
        ax.set_title("Valeur du pronostic par tirage (C_visite=%.1f kEUR)" % cv0)
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

        ax = axes[2]
        gain = -(total("rul", cv0) - total("calendar", cv0))
        ax.scatter(sev, gain, s=16, color="tab:green", alpha=0.45, zorder=3)
        if len(sev) > 2:
            a1, a0 = np.polyfit(sev, gain, 1)
            xs = np.linspace(sev.min(), sev.max(), 50)
            corr = float(np.corrcoef(gain, sev)[0, 1])
            ax.plot(xs, a1 * xs + a0, color='k', lw=1.2, zorder=4,
                    label="tendance (corr=%.2f)" % corr)
            ax.legend(fontsize=9)
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.axvline(1.0, color='k', lw=0.8, ls=':', alpha=0.6)
        ax.set_xlabel("Severite du monde (moy. geometrique des multiplicateurs)")
        ax.set_ylabel("Gain rul vs calendar [kEUR]  (>0 = gain)")
        ax.set_title("Valeur du pronostic vs ecart au modele")
        ax.grid(True, ls='--', alpha=0.5)

        fig.suptitle("Valeur de la RUL pour la maintenance insulaire (%d ans, N=%d, Tvisite=%gm)"
                     % (years, len(draws_ok), args.tvisit), fontsize=12)
        fig.tight_layout()
        fig.savefig(out_fig + ".pdf", bbox_inches="tight")
        fig.savefig(out_fig + ".png", dpi=160, bbox_inches="tight")
        plt.close()

    # --- 6) resume console ---
    print("\n" + "=" * 78)
    if draws_ok:
        cv0 = C_VISIT_GRID[len(C_VISIT_GRID) // 2]
        print("COUT TOTAL [kEUR] a C_visite=%.1f (N=%d, Tvisite=%gm)" % (cv0, len(draws_ok), args.tvisit))
        print("%-11s | %8s | %8s | %8s | %6s | %7s | %8s" %
              ("policy", "mean", "P95", "CVaR90", "nint", "waste", "outage h"))
        for p in POLICIES:
            u = total(p, cv0)
            print("%-11s | %8.3f | %8.3f | %8.3f | %6.2f | %7.3f | %8.0f"
                  % (p, u.mean(), np.quantile(u, 0.95), cvar_hi(u),
                     col(p, 'nint').mean(), col(p, 'waste').mean(),
                     (col(p, 'outfc') + col(p, 'outely')).mean()))
        print("-" * 78)
        for a, b in [("rul", "corrective"), ("rul", "calendar"), ("calendar", "corrective")]:
            dif = total(a, cv0) - total(b, cv0)
            print("  %-22s : %+.3f +/- %.3f kEUR ; gagne %.0f%%"
                  % ("%s - %s" % (a, b), dif.mean(), dif.std(), 100.0 * (dif < 0).mean()))
    print("=" * 78)
    print("Resultats : %s" % out_txt)
    print("Figure    : %s.pdf" % out_fig)


if __name__ == "__main__":
    main()
