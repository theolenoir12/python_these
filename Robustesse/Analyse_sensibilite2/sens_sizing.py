"""
sens_sizing.py -- ETAPE 5 : sensibilite au DIMENSIONNEMENT (Monte-Carlo).
=============================================================================
Reviewer APEN R3-major5 : le dimensionnement (FC = 1.5 kW < pic 4.2 kW ; PV/ELY
sur-dimensionnes) "force structurellement" certaines strategies ; la robustesse
des resultats aux variations de taille doit etre discutee.

APPROCHE MONTE-CARLO (unifiee avec sens_eol / sens_hthresholds)
--------------------------------------------------------------
On traite la taille des composants comme une INCERTITUDE : chaque composant
(BAT, FC, ELY) est multiplie par un facteur tire UNIFORMEMENT dans [0.8, 1.2]
(+/-20%), les 3 facteurs etant independants. On reproduit le FRONT DE PARETO
COMPLET des 10 EMS (un point = une strategie au dimensionnement nominal) ou
CHAQUE point porte son nuage Monte-Carlo + une ELLIPSE DE CONFIANCE 1-sigma/
2-sigma issue de la variation conjointe des tailles. C'est la "bande
d'incertitude par strategie" demandee par R3, exactement comme pour l'EoL.
Les MEMES N_MC triplets sont appliques a toutes les strategies (common random
numbers) -> comparaison des ellipses non polluee par le bruit MC.

Leviers (choix utilisateur : PV NON touche)
-------------------------------------------
  - Batterie : capacite via BAT['series_num'] (nb de cellules en serie).
  - PEMFC    : puissance via FC['n_series']  (nb de cellules).
  - PEMWE    : puissance via ELY['n_series'] (nb de cellules).

Pourquoi n_series (et pas n_parallel) -> AUCUNE edition du code de base
----------------------------------------------------------------------
n_series est un FACTEUR LINEAIRE de la puissance (P_max ~ n_series) qui :
  (i)  se SIMPLIFIE dans le ratio du brentq (voltage = n_series*(...) ->
       voltage(alpha)/V_bol independant de n_series) => calibration SoH->alpha
       INCHANGEE ;
  (ii) est COHERENT entre la boucle (recompute P_*_max_t ~ n_series) et Init.
On mute donc juste les tailles dans le worker et on met a l'echelle (x facteur)
les rares grandeurs derivees FIGEES a l'import : BAT['cost'], FC['P_fc_max'],
FC['cost'], ELY['P_ely_max'], ELY['cost']. Le reste (dynamique SoC, P_*_max_t,
couts de degradation) est relu EN DIRECT.

SOURCE 100% ASCII (volontaire). NE MODIFIE RIEN dans Vieillissement8.

Sorties (dans ./results/) : sens_sizing_pareto.pdf (front + ellipses),
sens_sizing_oat.pdf (figure d'appui OAT), sens_sizing.txt.
Lancer :  python sens_sizing.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics,
                         lps_cost_keur, lifetimes, run_pool, confidence_ellipse,
                         RESULTS_DIR)

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
    ("RB1_hist_020_060", "RB1 historique"),
    ("SoC1",     "SoC1"),
    ("SoC06",    "SoC06"),
]

# --- Monte Carlo : facteurs de taille des 3 composants, echantillonnes
#     CONJOINTEMENT et UNIFORMEMENT dans [1-DELTA, 1+DELTA]. Memes triplets pour
#     toutes les strategies (common random numbers). 1.0 = dimensionnement nominal.
DELTA     = 0.20      # +/-20% par composant
MC_RANGES = dict(bat=(1 - DELTA, 1 + DELTA),
                 fc=(1 - DELTA, 1 + DELTA),
                 ely=(1 - DELTA, 1 + DELTA))
N_MC    = 200         # tirages par strategie (run mesocentre : 200 pour ellipses lisses)
MC_SEED = 2024

# --- OAT (figure d'appui) : un composant a la fois, sur UNE strategie de reference ---
REF_FOLDER, REF_LABEL = "RB2(SoH)", "RB2(SoH)"
BASE_FAC = dict(bat=1.0, fc=1.0, ely=1.0)
BAT_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]
FC_GRID  = [0.8, 0.9, 1.0, 1.1, 1.2]
ELY_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]

# Cout total ~ 10 nominaux + 10*N_MC + OAT. N_MC=200 -> 10 + 2000 + ~12 = ~2022 sims.
OUT_TXT = os.path.join(RESULTS_DIR, "sens_sizing.txt")
# ======================================================================

# Valeurs NOMINALES des grandeurs impactees (capturees a l'import, AVANT toute
# mutation -> reference stable, pas de cumul entre taches d'un worker reutilise).
BASE = dict(
    bat_series=I.BAT['series_num'], bat_cost=I.BAT['cost'],
    fc_nseries=I.FC['n_series'],  fc_pmax=I.FC['P_fc_max'],   fc_cost=I.FC['cost'],
    ely_nseries=I.ELY['n_series'], ely_pmax=I.ELY['P_ely_max'], ely_cost=I.ELY['cost'],
)


def _apply_sizing(fbat, ffc, fely):
    """Met a l'echelle tailles + grandeurs derivees figees (depuis BASE)."""
    I.BAT['series_num'] = BASE['bat_series'] * fbat
    I.BAT['cost']       = BASE['bat_cost']   * fbat
    I.FC['n_series']    = BASE['fc_nseries'] * ffc
    I.FC['P_fc_max']    = BASE['fc_pmax']    * ffc
    I.FC['cost']        = BASE['fc_cost']    * ffc
    I.ELY['n_series']   = BASE['ely_nseries'] * fely
    I.ELY['P_ely_max']  = BASE['ely_pmax']   * fely
    I.ELY['cost']       = BASE['ely_cost']   * fely


def evaluate(params):
    """Worker picklable. params = dict(folder, label, kind, bat, fc, ely)
    (bat/fc/ely = FACTEURS de taille). Applique le dimensionnement, charge la
    strategie, lance la simu."""
    try:
        _apply_sizing(params['bat'], params['fc'], params['ely'])
        strat = load_strategy(params['folder'])
        data = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        clps = lps_cost_keur(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:
        lpsp = cost = clps = lb = lf = le = None
        ok = False
        print("  [FAIL] %-9s size=(%.3f,%.3f,%.3f) : %s"
              % (params['label'], params['bat'], params['fc'], params['ely'], e), flush=True)
    return dict(params=params, label=params['label'], kind=params['kind'],
                bat=params['bat'], fc=params['fc'], ely=params['ely'],
                lpsp=lpsp, cost=cost, clps=clps,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def _fmt(r):
    if not r['ok']:
        return "%-9s [%s] size=(%.2f,%.2f,%.2f) -> FAIL" % (
            r['label'], r['kind'], r['bat'], r['fc'], r['ely'])
    return ("%-9s [%-7s] size=(%.2f,%.2f,%.2f) -> LPSP %6.4f%%  deg %7.2f kEUR  vie B %s"
            % (r['label'], r['kind'], r['bat'], r['fc'], r['ely'],
               r['lpsp'], r['cost'], _yr(r['life_bat'])))


def build_tasks(size_samples):
    """Construit la liste complete des taches : nominaux + MC (toutes strategies)
    + OAT (strategie de reference). Une seule pool -> coeurs satures en continu."""
    tasks = []
    # 1) point NOMINAL de chaque strategie (facteurs 1,1,1) -> le point du front
    for folder, label in SCENARIOS:
        tasks.append(dict(folder=folder, label=label, kind='nom', **BASE_FAC))
    # 2) nuage MONTE-CARLO de chaque strategie (memes triplets pour toutes)
    for folder, label in SCENARIOS:
        for (b, f, e) in size_samples:
            tasks.append(dict(folder=folder, label=label, kind='mc', bat=b, fc=f, ely=e))
    # 3) OAT sur la strategie de reference (figure d'appui)
    for v in BAT_GRID:
        if v != BASE_FAC['bat']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_bat',
                              bat=v, fc=BASE_FAC['fc'], ely=BASE_FAC['ely']))
    for v in FC_GRID:
        if v != BASE_FAC['fc']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_fc',
                              bat=BASE_FAC['bat'], fc=v, ely=BASE_FAC['ely']))
    for v in ELY_GRID:
        if v != BASE_FAC['ely']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_ely',
                              bat=BASE_FAC['bat'], fc=BASE_FAC['fc'], ely=v))
    return tasks


def main():
    print("=== ETAPE 5 -- Sensibilite au dimensionnement : front de Pareto + bandes (25 ans) ===", flush=True)
    print("    %d strategies | N_MC=%d/strat | +/-%.0f%% uniforme par composant | nominal: bat_series=%g fc_nseries=%g ely_nseries=%g"
          % (len(SCENARIOS), N_MC, DELTA * 100, BASE['bat_series'], BASE['fc_nseries'], BASE['ely_nseries']), flush=True)

    # --- echantillons de taille communs a toutes les strategies (common random numbers) ---
    rng = np.random.default_rng(MC_SEED)
    size_samples = [(float(rng.uniform(*MC_RANGES['bat'])),
                     float(rng.uniform(*MC_RANGES['fc'])),
                     float(rng.uniform(*MC_RANGES['ely']))) for _ in range(N_MC)]

    tasks = build_tasks(size_samples)
    res = run_pool(evaluate, tasks, "Dimensionnement -- nominaux + Monte-Carlo + OAT", _fmt)

    # --- tri des resultats ---
    nom = {r['label']: r for r in res if r['kind'] == 'nom' and r['ok']}
    mc_by = {label: [r for r in res
                     if r['kind'] == 'mc' and r['label'] == label and r['ok']]
             for _, label in SCENARIOS}
    oat = {k: sorted([r for r in res if r['kind'] == k and r['ok']],
                     key=lambda r: r[{'oat_bat': 'bat', 'oat_fc': 'fc', 'oat_ely': 'ely'}[k]])
           for k in ('oat_bat', 'oat_fc', 'oat_ely')}
    # la reference au nominal appartient a chaque grille OAT
    ref_nom = nom.get(REF_LABEL)
    for k, comp in (('oat_bat', 'bat'), ('oat_fc', 'fc'), ('oat_ely', 'ely')):
        if ref_nom is not None:
            oat[k] = sorted(oat[k] + [ref_nom], key=lambda r: r[comp])

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite dimensionnement -- front de Pareto + bandes\n")
        f.write("# leviers: BAT['series_num'], FC['n_series'], ELY['n_series'] (PV non touche)\n")
        f.write("# MC: N=%d/strat, +/-%.0f%% uniforme/composant, seed=%d (memes triplets pour toutes)\n\n"
                % (N_MC, DELTA * 100, MC_SEED))
        f.write("## Front de Pareto : point nominal + dispersion MC par strategie\n")
        f.write("strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;deg_min;deg_max;N_ok;clps_nom;clps_mean;clps_std\n")
        for _, label in SCENARIOS:
            rn = nom.get(label)
            sub = mc_by.get(label, [])
            if rn is None:
                f.write("%s;NOMINAL_FAIL\n" % label); continue
            lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
            cl = np.array([r['clps'] for r in sub])
            if len(sub):
                f.write("%s;%.4f;%.3f;%.4f;%.4f;%.3f;%.3f;%.3f;%.3f;%d;%.3f;%.3f;%.3f\n"
                        % (label, rn['lpsp'], rn['cost'], lp.mean(), lp.std(),
                           dg.mean(), dg.std(), dg.min(), dg.max(), len(sub),
                           rn['clps'], cl.mean(), cl.std()))
            else:
                f.write("%s;%.4f;%.3f;-;-;-;-;-;-;0;%.3f;-;-\n"
                        % (label, rn['lpsp'], rn['cost'], rn['clps']))
        f.write("\n## OAT (strategie %s) : un composant varie, les deux autres au nominal\n" % REF_LABEL)
        for k, comp, lifekey in (('oat_bat', 'bat', 'life_bat'),
                                 ('oat_fc', 'fc', 'life_fc'),
                                 ('oat_ely', 'ely', 'life_ely')):
            f.write("# %s\nfacteur_taille;LPSP_%%;deg_kEUR;vie_composant_ans\n" % comp.upper())
            for r in oat[k]:
                f.write("%.3f;%.4f;%.3f;%s\n" % (r[comp], r['lpsp'], r['cost'], _yr(r[lifekey])))
            f.write("\n")

    # ===================== FIGURE 1 : FRONT DE PARETO + ELLIPSES PAR POINT =====================
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_off = {
        '0-100': (4, 4), '25-75': (4, 4), '50-50': (4, 4), '75-25': (4, 4),
        '100-0': (4, 4), 'RB2': (6, -2), 'RB2(SoH)': (6, -10), 'RB1': (4, 4),
        'SoC1': (4, 4), 'SoC06': (4, 4),
    }
    for (folder, label), col in zip(SCENARIOS, colors):
        rn = nom.get(label)
        sub = mc_by.get(label, [])
        if rn is None:
            continue
        if len(sub) >= 3:
            x = np.array([r['lpsp'] for r in sub]); y = np.array([r['cost'] for r in sub])
            ax.scatter(x, y, s=12, color=col, alpha=0.25, zorder=2)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor='none',
                               lw=1.6, zorder=4)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor='none',
                               lw=0.9, ls='--', alpha=0.6, zorder=4)
        # point NOMINAL (taille de base) = le point "officiel" du front
        ax.scatter([rn['lpsp']], [rn['cost']], marker='o', s=70, color=col,
                   edgecolor='k', linewidth=0.6, zorder=6)
        dx, dy = label_off.get(label, (4, 4))
        ax.annotate(label, (rn['lpsp'], rn['cost']), textcoords="offset points",
                    xytext=(dx, dy), fontsize=11, color=col, weight='bold', zorder=7)

    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Robustesse du front de Pareto au dimensionnement (+/-20%)", fontsize=12)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_sizing_pareto.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== FIGURE 2 : OAT (1x3) sur la strategie de reference =====================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [('oat_bat', 'bat', 'life_bat', 'Batterie'),
             ('oat_fc',  'fc',  'life_fc',  'PEMFC'),
             ('oat_ely', 'ely', 'life_ely', 'PEMWE')]
    for ax, (k, comp, lifekey, title) in zip(axes, specs):
        rows = oat[k]
        if not rows:
            ax.set_title("%s (aucun run OK)" % title); continue
        x = np.array([r[comp] for r in rows])
        dgc = np.array([r['cost'] for r in rows])
        life = np.array([r[lifekey] if r[lifekey] is not None else np.nan for r in rows])
        ax.plot(x, dgc, 'o-', color='tab:red')
        ax.set_title(title, fontsize=11); ax.set_xlabel("facteur de taille")
        ax.grid(True, ls='--', alpha=0.4)
        ax.axvline(BASE_FAC[comp], color='k', ls=':', lw=0.8, alpha=0.6)
        axb = ax.twinx()
        axb.plot(x, life, 's--', color='tab:blue', alpha=0.8)
        if comp == 'bat':
            ax.set_ylabel("Cout degradation total [kEUR]", color='tab:red')
        if comp == 'ely':
            axb.set_ylabel("Vie du composant [ans]", color='tab:blue')
        axb.tick_params(axis='y', labelcolor='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle("Sensibilite OAT au dimensionnement (%s)" % REF_LABEL, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_sizing_oat.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 78)
    print("FRONT DE PARETO (point nominal) + dispersion MC sur le dimensionnement")
    print("-" * 78)
    print("%-9s | %-18s | %-26s" % ("strat", "nominal (LPSP/deg)", "MC deg mean+/-std [min..max]"))
    for _, label in SCENARIOS:
        rn = nom.get(label); sub = mc_by.get(label, [])
        if rn is None:
            print("%-9s | NOMINAL FAIL" % label); continue
        if sub:
            dg = np.array([r['cost'] for r in sub]); lp = np.array([r['lpsp'] for r in sub])
            print("%-9s | %6.4f%% / %7.2f | %7.2f +/- %.2f  [%.2f..%.2f]  (LPSP %6.4f+/-%.4f, N=%d)"
                  % (label, rn['lpsp'], rn['cost'], dg.mean(), dg.std(), dg.min(), dg.max(),
                     lp.mean(), lp.std(), len(sub)))
        else:
            print("%-9s | %6.4f%% / %7.2f | (aucun MC OK)" % (label, rn['lpsp'], rn['cost']))
    print("=" * 78)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_sizing_pareto.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_sizing_oat.pdf"))


if __name__ == "__main__":
    main()
