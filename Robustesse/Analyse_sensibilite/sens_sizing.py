"""
sens_sizing.py -- ETAPE 5 : robustesse du CLASSEMENT des EMS au DIMENSIONNEMENT.
=============================================================================
Reviewer APEN R3-major5 : le dimensionnement (FC = 1.5 kW < pic 4.2 kW ; PV/ELY
sur-dimensionnes) "force structurellement" les strategies battery-heavy a battre
les strategies hydrogen-heavy ; la robustesse du CLASSEMENT des EMS aux
variations de taille doit etre discutee, sinon l'ordre du front (Fig. 6) ne
peut pas etre generalise.

Approche : SCENARIOS DISCRETS de dimensionnement (un dimensionnement = un CHOIX
de conception, pas une incertitude aleatoire -> scenarios, pas Monte-Carlo). Pour
CHAQUE scenario on recalcule le FRONT DE PARETO complet des 10 EMS, puis on
regarde si l'ORDRE (classement par cout) change.

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
(NB : n_parallel, lui, est incoherent boucle/Init et touche la densite de
courant -> on l'evite.) On mute donc juste les tailles dans le worker et on met
a l'echelle (x facteur) les rares grandeurs derivees FIGEES a l'import :
BAT['cost'], FC['P_fc_max'], FC['cost'], ELY['P_ely_max'], ELY['cost'].
Le reste (dynamique SoC, P_*_max_t, couts de degradation) est relu EN DIRECT.

SOURCE 100% ASCII (volontaire). NE MODIFIE RIEN dans Vieillissement8.

Sorties (dans ./results/) : sens_sizing_fronts.pdf (fronts par scenario),
sens_sizing_ranking.pdf (heatmap du classement), sens_sizing.txt.
Lancer :  python sens_sizing.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics,
                         lifetimes, run_pool, RESULTS_DIR)

# ============================ CONFIGURATION ============================
# --- Les 10 EMS (memes dossiers que sens_eol / batch_pareto) ---
EMS_LIST = [
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

# --- Scenarios de dimensionnement : (label, facteurs bat / fc / ely) ---
# Cibles R3-5 : FC sous-dimensionnee -> l'agrandir ; ELY sur-dimensionnee ->
# la reduire ; batterie -> les deux sens. 1.0 = nominal.
SIZINGS = [
    ("nominal",        dict(bat=1.0, fc=1.0, ely=1.0)),
    ("FC x2",          dict(bat=1.0, fc=2.0, ely=1.0)),
    ("FC x3",          dict(bat=1.0, fc=3.0, ely=1.0)),
    ("ELY x0.5",       dict(bat=1.0, fc=1.0, ely=0.5)),
    ("BAT x0.5",       dict(bat=0.5, fc=1.0, ely=1.0)),
    ("BAT x2",         dict(bat=2.0, fc=1.0, ely=1.0)),
    ("FC x2 / ELY x0.5", dict(bat=1.0, fc=2.0, ely=0.5)),
]

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
    """Worker picklable. params = dict(folder, ems, sizing, bat, fc, ely)."""
    try:
        _apply_sizing(params['bat'], params['fc'], params['ely'])
        strat = load_strategy(params['folder'])
        data = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:
        lpsp = cost = lb = lf = le = None
        ok = False
        print("  [FAIL] %-9s @ %-12s : %s" % (params['ems'], params['sizing'], e), flush=True)
    return dict(ems=params['ems'], sizing=params['sizing'],
                bat=params['bat'], fc=params['fc'], ely=params['ely'],
                lpsp=lpsp, cost=cost, life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "%-9s @ %-12s -> FAIL" % (r['ems'], r['sizing'])
    return ("%-9s @ %-12s -> LPSP %6.4f%%  deg %7.2f kEUR"
            % (r['ems'], r['sizing'], r['lpsp'], r['cost']))


def _ranks_and_dominance(rows):
    """rows = liste de dicts ok pour UN scenario. Renvoie {ems: (rank_cout, nondom)}.
    rank_cout : 1 = cout le plus bas. nondom : True si non-domine en 2D (LPSP, cout)."""
    out = {}
    order = sorted(rows, key=lambda r: r['cost'])
    for rk, r in enumerate(order, 1):
        nondom = not any((o['lpsp'] <= r['lpsp'] and o['cost'] <= r['cost']
                          and (o['lpsp'] < r['lpsp'] or o['cost'] < r['cost']))
                         for o in rows)
        out[r['ems']] = (rk, nondom)
    return out


def main():
    print("=== ETAPE 5 -- Robustesse du classement des EMS au dimensionnement (25 ans) ===", flush=True)
    print("    %d EMS x %d scenarios = %d simus | leviers: BAT.series, FC.n_series, ELY.n_series"
          % (len(EMS_LIST), len(SIZINGS), len(EMS_LIST) * len(SIZINGS)), flush=True)
    print("    nominal: bat_series=%g fc_nseries=%g ely_nseries=%g"
          % (BASE['bat_series'], BASE['fc_nseries'], BASE['ely_nseries']), flush=True)

    tasks = []
    for slabel, fac in SIZINGS:
        for folder, ems in EMS_LIST:
            tasks.append(dict(folder=folder, ems=ems, sizing=slabel, **fac))
    res = run_pool(evaluate, tasks, "Dimensionnement -- fronts par scenario", _fmt)

    # res[scenario][ems] = run
    by = {s: {} for s, _ in SIZINGS}
    for r in res:
        if r['ok']:
            by[r['sizing']][r['ems']] = r
    ranks = {s: _ranks_and_dominance(list(by[s].values())) for s, _ in SIZINGS}

    ems_labels = [e for _, e in EMS_LIST]

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Robustesse du classement des EMS au dimensionnement -- 25 ans\n")
        f.write("# leviers: BAT['series_num'], FC['n_series'], ELY['n_series'] (PV non touche)\n")
        f.write("# nominal: bat_series=%g fc_nseries=%g ely_nseries=%g\n\n"
                % (BASE['bat_series'], BASE['fc_nseries'], BASE['ely_nseries']))
        for slabel, fac in SIZINGS:
            f.write("## Scenario '%s' (bat x%g, fc x%g, ely x%g)\n"
                    % (slabel, fac['bat'], fac['fc'], fac['ely']))
            f.write("ems;LPSP_%;deg_kEUR;rank_cout;non_domine\n")
            rk = ranks[slabel]
            for ems in ems_labels:
                r = by[slabel].get(ems)
                if r is None:
                    f.write("%s;FAIL\n" % ems); continue
                rank, nd = rk[ems]
                f.write("%s;%.4f;%.3f;%d;%s\n"
                        % (ems, r['lpsp'], r['cost'], rank, "oui" if nd else "non"))
            f.write("\n")

    # ===================== FIGURE 1 : FRONTS PAR SCENARIO (petits multiples) =====================
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    col_of = {e: colors[i] for i, e in enumerate(ems_labels)}
    n = len(SIZINGS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    # auto-echelle PAR panneau (la LPSP peut exploser dans certains scenarios ->
    # des axes partages ecraseraient les autres ; le classement inter-scenarios
    # est porte par la heatmap figure 2).
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    for idx, (slabel, fac) in enumerate(SIZINGS):
        ax = axes[idx // ncol][idx % ncol]
        for ems in ems_labels:
            r = by[slabel].get(ems)
            if r is None:
                continue
            nd = ranks[slabel][ems][1]
            ax.scatter([r['lpsp']], [r['cost']], color=col_of[ems], s=55,
                       edgecolor=('k' if nd else 'none'),
                       linewidth=(1.1 if nd else 0), zorder=3)
        ax.set_title(slabel, fontsize=10)
        ax.grid(True, ls='--', alpha=0.4)
        if idx % ncol == 0:
            ax.set_ylabel("deg [kEUR]")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("LPSP [%]")
    # cases vides
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis('off')
    # legende unique (EMS -> couleur) + note non-domine
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=col_of[e],
                          markersize=8, label=e) for e in ems_labels]
    handles.append(plt.Line2D([0], [0], marker='o', ls='', mfc='none', mec='k',
                              markersize=8, label='non-domine (bord noir)'))
    fig.legend(handles=handles, loc='center right', fontsize=8, framealpha=0.9)
    fig.suptitle("Fronts de Pareto des 10 EMS selon le dimensionnement", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    fig.savefig(os.path.join(RESULTS_DIR, "sens_sizing_fronts.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== FIGURE 2 : HEATMAP DU CLASSEMENT (rank par cout) =====================
    R = np.full((len(ems_labels), len(SIZINGS)), np.nan)
    for jc, (slabel, _) in enumerate(SIZINGS):
        for ir, ems in enumerate(ems_labels):
            if ems in ranks[slabel]:
                R[ir, jc] = ranks[slabel][ems][0]
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(SIZINGS), 0.55 * len(ems_labels) + 1.5))
    im = ax.imshow(R, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(ems_labels))
    ax.set_xticks(range(len(SIZINGS))); ax.set_xticklabels([s for s, _ in SIZINGS],
                                                           rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(ems_labels))); ax.set_yticklabels(ems_labels, fontsize=9)
    for ir in range(len(ems_labels)):
        for jc in range(len(SIZINGS)):
            if not np.isnan(R[ir, jc]):
                nd = ranks[SIZINGS[jc][0]][ems_labels[ir]][1]
                ax.text(jc, ir, "%d%s" % (int(R[ir, jc]), "*" if nd else ""),
                        ha='center', va='center', fontsize=9,
                        weight=('bold' if nd else 'normal'))
    cb = plt.colorbar(im, ax=ax, pad=0.02); cb.set_label("rang par cout (1 = moins cher)")
    ax.set_title("Classement des EMS par cout de degradation selon le dimensionnement\n"
                 "(* = non-domine en 2D LPSP/cout)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_sizing_ranking.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 78)
    print("CLASSEMENT PAR COUT (1 = moins cher ; * = non-domine 2D)")
    hdr = "%-9s" % "EMS" + "".join(" %10s" % s[:10] for s, _ in SIZINGS)
    print(hdr); print("-" * len(hdr))
    for ems in ems_labels:
        line = "%-9s" % ems
        for slabel, _ in SIZINGS:
            if ems in ranks[slabel]:
                rk, nd = ranks[slabel][ems]
                line += " %9s%s" % (rk, "*" if nd else " ")
            else:
                line += " %10s" % "FAIL"
        print(line)
    print("=" * 78)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_sizing_fronts.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_sizing_ranking.pdf"))


if __name__ == "__main__":
    main()
