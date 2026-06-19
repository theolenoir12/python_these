"""
sens_cweights.py -- ETAPE 3 : sensibilite aux POIDS DE COUT ("C-weights").
=============================================================================
Reviewers APEN : R3-major3 ("sensitivity scan over ... C-weights ...", avec
bandes d'incertitude sur le plan de Pareto) ; R1-6 (arbitrage economique : les
pertes de vieillissement vs le cout de remplacement des composants).

Qu'est-ce qu'un "C-weight" ici ?
--------------------------------
L'axe Y du Pareto est un cout de degradation [kEUR] = somme de trois
contributions, chacune = (degradation normalisee a l'EoL) * cout_de_remplacement
du composant (cf. cost_fcn_total2.get_cost_total = cost_bat + cost_fc + cost_ely).
Les POIDS sont donc les couts de remplacement, derives des CAPEX (Init) :
    BAT['cost']  ~ CAPEX batterie (150 EUR/kWh)
    FC['cost']   ~ CAPEX_stack FC (0.3*2500 = 750 EUR/kW)
    ELY['cost']  ~ CAPEX_stack ELY (563 EUR/kW)
Ce sont des hypotheses technico-economiques incertaines -> on les fait varier.

PROPRIETE CLE (verifiee dans le code)
-------------------------------------
La trajectoire SoH, les REMPLACEMENTS et la LPSP sont INVARIANTS aux poids :
dans get_soh / la boucle, SoH = 1 - (deg_cumulee_en_EUR)/['cost']*(1-EoL), et la
deg_cumulee est elle-meme proportionnelle a ['cost'] -> le facteur ['cost'] se
SIMPLIFIE. Seul le cout total (axe Y) change, et il est LINEAIRE en chaque poids :
    cout(m_bat, m_fc, m_ely) = m_bat*cost_bat + m_fc*cost_fc + m_ely*cost_ely
ou (cost_bat, cost_fc, cost_ely) sont les composantes au poids NOMINAL.

Consequences :
  (1) UNE seule simulation par strategie suffit (10 simus) ; le Monte-Carlo sur
      les poids est du POST-TRAITEMENT analytique (N_MC=5000 ~ gratuit).
  (2) La bande d'incertitude par point est PUREMENT VERTICALE (la LPSP ne bouge
      pas) -> barres d'erreur verticales, pas d'ellipses 2D. C'est un RESULTAT :
      le classement en LPSP est insensible aux hypotheses de cout ; seule
      l'amplitude de l'axe degradation se dilate/contracte.
  NB : multiplier les 3 poids par un MEME facteur ne fait que redimensionner
  l'axe (ne change pas le classement) ; c'est la variation RELATIVE entre
  composants (tirages independants) qui est informative.

SOURCE 100% ASCII (volontaire ; cf. sens_soh_estimation.py).
NE MODIFIE RIEN dans Vieillissement8 (import lecture seule via sens_common).

Sorties (dans ./results/) : sens_cweights_pareto.pdf (front + barres verticales),
sens_cweights_breakdown.pdf (composition du cout par strategie), sens_cweights.txt.
Lancer :  python sens_cweights.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics_components,
                         lps_cost_keur, run_pool, RESULTS_DIR)

# ============================ CONFIGURATION ============================
# --- Front de Pareto : les 10 EMS (memes dossiers que sens_eol / batch_pareto) ---
SCENARIOS = [
    ("0-100",    "0-100"),
    ("25-75",    "25-75"),
    ("50-50",    "50-50"),
    ("75-25",    "75-25"),
    ("100-0",    "100-0"),
    ("RB2",      "RB2"),
    ("RB2(SoH)", "RB2(SoH)"),
    ("RB1",      "RB1"),
    ("SoC1",     "SoC1"),
    ("SoC06",    "SoC06"),
]

# --- Monte Carlo sur les poids (multiplicateurs centres sur 1, tirage uniforme) ---
# Incertitude relative de chaque cout de remplacement. +/-30% par defaut ;
# l'hydrogene (FC/ELY) est en general plus incertain que la batterie -> ajustable.
DELTA   = dict(bat=0.30, fc=0.30, ely=0.30)
# MC a 200 tirages (homogene avec sens_eol / sens_hthresholds / sens_sizing).
# Rappel : pour les poids de cout le MC est ANALYTIQUE (1 simu/strategie suffit,
# cf. propriete de linearite ci-dessus) -> 200 tirages sont gratuits ; la LPSP
# etant INVARIANTE aux poids, la bande par point reste PUREMENT VERTICALE (barres
# d'erreur, pas d'ellipse 2D possible).
N_MC    = 200
MC_SEED = 7
CI      = 95            # intervalle de confiance affiche (percentiles), en %

# --- OAT (figure 2 = decomposition) : strategie de reference pour le focus ---
REF_LABEL = "RB2(SoH)"

OUT_TXT = os.path.join(RESULTS_DIR, "sens_cweights.txt")
# ======================================================================


def evaluate(params):
    """Worker picklable. params = dict(folder, label). Lance la strategie UNE
    fois au poids NOMINAL et renvoie LPSP + les 3 composantes de cout [kEUR].
    Aucune mutation de poids : la simu est invariante aux C-weights."""
    try:
        strat = load_strategy(params['folder'])
        data = init_and_run_loop(strat)
        lpsp, cb, cf, ce = metrics_components(data)
        clps = lps_cost_keur(data)   # cout LPS pas-a-pas (invariant aux poids de cout)
        ok = True
    except Exception as e:
        lpsp = cb = cf = ce = clps = None
        ok = False
        print("  [FAIL] %-9s : %s" % (params['label'], e), flush=True)
    return dict(label=params['label'], folder=params['folder'],
                lpsp=lpsp, cb=cb, cf=cf, ce=ce, clps=clps, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "%-9s -> FAIL" % r['label']
    return ("%-9s -> LPSP %6.4f%%  cout %7.2f kEUR  (bat %.1f / fc %.1f / ely %.1f)"
            % (r['label'], r['lpsp'], r['cb'] + r['cf'] + r['ce'], r['cb'], r['cf'], r['ce']))


def main():
    print("=== ETAPE 3 -- Sensibilite aux poids de cout (C-weights) (10 EMS, 25 ans) ===", flush=True)
    print("    DELTA=%s | N_MC=%d (post-traitement) | CI=%d%%" % (DELTA, N_MC, CI), flush=True)
    nominal_w = dict(bat=I.BAT['cost'], fc=I.FC['cost'], ely=I.ELY['cost'])
    print("    couts de remplacement nominaux [EUR] : bat=%.0f fc=%.0f ely=%.0f"
          % (nominal_w['bat'], nominal_w['fc'], nominal_w['ely']), flush=True)

    # -------- 1 simulation par strategie (la seule partie couteuse) --------
    tasks = [dict(folder=f, label=l) for f, l in SCENARIOS]
    res = {r['label']: r for r in run_pool(evaluate, tasks,
           "C-weights -- 1 simu/strategie", _fmt) if r['ok']}

    # -------- Monte-Carlo POST-TRAITEMENT (memes tirages pour toutes) --------
    rng = np.random.default_rng(MC_SEED)
    M = np.empty((N_MC, 3))
    for j, comp in enumerate(('bat', 'fc', 'ely')):
        M[:, j] = rng.uniform(1.0 - DELTA[comp], 1.0 + DELTA[comp], N_MC)
    plo, phi = (100 - CI) / 2.0, 100 - (100 - CI) / 2.0

    stats = {}
    for _, label in SCENARIOS:
        r = res.get(label)
        if r is None:
            continue
        samples = r['cb'] * M[:, 0] + r['cf'] * M[:, 1] + r['ce'] * M[:, 2]
        stats[label] = dict(
            lpsp=r['lpsp'], nominal=r['cb'] + r['cf'] + r['ce'],
            cb=r['cb'], cf=r['cf'], ce=r['ce'], clps=r['clps'],
            mean=float(samples.mean()), std=float(samples.std()),
            lo=float(np.percentile(samples, plo)),
            hi=float(np.percentile(samples, phi)))

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite C-weights -- 10 EMS 25 ans | DELTA=%s | N_MC=%d | CI=%d%%\n"
                % (DELTA, N_MC, CI))
        f.write("# couts de remplacement nominaux [EUR]: bat=%.0f fc=%.0f ely=%.0f\n"
                % (nominal_w['bat'], nominal_w['fc'], nominal_w['ely']))
        f.write("# LPSP INVARIANT aux poids ; cout lineaire -> bande verticale\n")
        f.write("# clps = cout LPS pas-a-pas [kEUR] (invariant aux poids de cout)\n\n")
        f.write("strat;LPSP_%%;cout_nominal;cost_bat;cost_fc;cost_ely;"
                "cout_mean;cout_std;cout_lo%d;cout_hi%d;clps\n" % (CI, CI))
        for _, label in SCENARIOS:
            s = stats.get(label)
            if s is None:
                f.write("%s;FAIL\n" % label); continue
            f.write("%s;%.4f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f\n"
                    % (label, s['lpsp'], s['nominal'], s['cb'], s['cf'], s['ce'],
                       s['mean'], s['std'], s['lo'], s['hi'], s['clps']))

    # ===================== FIGURE 1 : FRONT + BARRES VERTICALES (CI) =====================
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_off = {
        '0-100': (6, 4), '25-75': (6, 4), '50-50': (6, 4), '75-25': (6, 4),
        '100-0': (6, 4), 'RB2': (6, -2), 'RB2(SoH)': (6, -12), 'RB1': (6, 4),
        'SoC1': (6, 4), 'SoC06': (6, 4),
    }
    for (folder, label), col in zip(SCENARIOS, colors):
        s = stats.get(label)
        if s is None:
            continue
        yerr = [[s['nominal'] - s['lo']], [s['hi'] - s['nominal']]]
        ax.errorbar([s['lpsp']], [s['nominal']], yerr=yerr, fmt='o', color=col,
                    ecolor=col, elinewidth=1.6, capsize=5, capthick=1.6,
                    markersize=8, markeredgecolor='k', markeredgewidth=0.6, zorder=5)
        dx, dy = label_off.get(label, (6, 4))
        ax.annotate(label, (s['lpsp'], s['nominal']), textcoords="offset points",
                    xytext=(dx, dy), fontsize=11, color=col, weight='bold', zorder=7)
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Robustesse du front de Pareto aux poids de cout", fontsize=12)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_cweights_pareto.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== FIGURE 2 : COMPOSITION DU COUT PAR STRATEGIE =====================
    labels_ok = [l for _, l in SCENARIOS if l in stats]
    cb = np.array([stats[l]['cb'] for l in labels_ok])
    cf = np.array([stats[l]['cf'] for l in labels_ok])
    ce = np.array([stats[l]['ce'] for l in labels_ok])
    x = np.arange(len(labels_ok))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, cb, label='Batterie', color='tab:blue')
    ax.bar(x, cf, bottom=cb, label='PEMFC', color='tab:red')
    ax.bar(x, ce, bottom=cb + cf, label='PEMWE', color='tab:green')
    ax.set_xticks(x); ax.set_xticklabels(labels_ok, rotation=35, ha='right')
    ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Composition du cout de degradation par strategie (poids nominaux)", fontsize=12)
    ax.legend(loc='best'); ax.grid(True, axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_cweights_breakdown.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 78)
    print("C-WEIGHTS : LPSP fixe ; cout nominal et bande %d%% (poids +/-%s)" % (CI, DELTA))
    print("-" * 78)
    print("%-9s | %-8s | %-9s | %-22s | composantes bat/fc/ely" % (
        "strat", "LPSP", "cout_nom", "bande CI%d [kEUR]" % CI))
    for _, label in SCENARIOS:
        s = stats.get(label)
        if s is None:
            print("%-9s | FAIL" % label); continue
        print("%-9s | %6.4f%% | %8.2f | [%7.2f .. %7.2f] | %.1f / %.1f / %.1f"
              % (label, s['lpsp'], s['nominal'], s['lo'], s['hi'],
                 s['cb'], s['cf'], s['ce']))
    print("=" * 78)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_cweights_pareto.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_cweights_breakdown.pdf"))


if __name__ == "__main__":
    main()
