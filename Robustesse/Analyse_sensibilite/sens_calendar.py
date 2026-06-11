"""
sens_calendar.py -- ETAPE 6 : impact du VIEILLISSEMENT CALENDAIRE de la batterie.
=============================================================================
Reviewer APEN R3-major3-ii : le vieillissement calendaire batterie est exclu au
motif que l'indicateur cible des mecanismes "directement influences par l'EMS" ;
or le TEMPS DE RESIDENCE en SoC est justement une variable controlee par l'EMS,
donc l'exclusion n'est pas justifiee. On AJOUTE donc un terme calendaire et on
mesure son impact (et son incertitude) sur le front de Pareto.

Modele ajoute (dans cost_fcn_total2.get_cost_bat, OFF par defaut)
----------------------------------------------------------------
Perte de capacite calendaire = somme_t k_cal(SoC_t) * dt, avec
  k_cal(SoC) = k_cal(1) * g(SoC),  g(SoC) = SoC          (forme LINEAIRE)
  k_cal(1) calibre pour qu'a SoC=100% constant la batterie atteigne l'EoL (perte
  1-SoH_EoL) en T_cal annees.
Terme ADDITIF par pas -> telescope correctement dans l'accumulation incrementale
de la boucle -> agit a la fois sur le SoH/les remplacements ET sur le cout.
Desactive si BAT_CAL_TCAL_Y est None (defaut) -> resultats de base inchanges.

Structure (comme l'EoL)
-----------------------
Par EMS : point BASELINE (calendaire OFF) + nuage MONTE-CARLO (calendaire ON,
T_cal ~ U[10,20] ans) + ellipse. Le DECALAGE baseline -> nuage = l'impact
calendaire. Bonus : on correle le surcout calendaire au SoC MOYEN de chaque EMS
(montre que l'EMS *pilote* le calendaire via la residence en SoC -> coeur de
l'argument R3).

SOURCE 100% ASCII (volontaire). NE MODIFIE RIEN d'autre dans Vieillissement8.

Sorties (dans ./results/) : sens_calendar_pareto.pdf, sens_calendar_insight.pdf,
sens_calendar.txt.
Lancer :  python sens_calendar.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics, lifetimes,
                         run_pool, confidence_ellipse, RESULTS_DIR)
from Common import cost_fcn_total2 as CF   # pour activer le calendaire (global module)

# ============================ CONFIGURATION ============================
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

# --- Monte Carlo : vie calendaire a SoC=100% (annees), uniforme. Memes tirages
# pour toutes les strategies (common random numbers). ---
TCAL_RANGE = (10.0, 20.0)
N_MC    = 15
MC_SEED = 99

# --- OAT (figure d'appui) : grille de T_cal sur une strategie de reference ---
REF_FOLDER, REF_LABEL = "RB2(SoH)", "RB2(SoH)"
TCAL_GRID = [10.0, 12.5, 15.0, 17.5, 20.0]

OUT_TXT = os.path.join(RESULTS_DIR, "sens_calendar.txt")
# ======================================================================


def evaluate(params):
    """Worker picklable. params = dict(folder, ems, kind, tcal). tcal=None ->
    calendaire DESACTIVE (baseline)."""
    try:
        CF.BAT_CAL_TCAL_Y = params['tcal']    # None -> terme nul
        strat = load_strategy(params['folder'])
        data = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        lb, lf, le = lifetimes(data)
        mean_soc = float(np.mean(data['SoC']))
        ok = True
    except Exception as e:
        lpsp = cost = lb = lf = le = mean_soc = None
        ok = False
        print("  [FAIL] %-9s tcal=%s : %s" % (params['ems'], params['tcal'], e), flush=True)
    finally:
        CF.BAT_CAL_TCAL_Y = None              # on remet OFF (proprete inter-taches)
    return dict(ems=params['ems'], kind=params['kind'], tcal=params['tcal'],
                lpsp=lpsp, cost=cost, life_bat=lb, life_fc=lf, life_ely=le,
                mean_soc=mean_soc, ok=ok)


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def _fmt(r):
    if not r['ok']:
        return "%-9s [%s] tcal=%s -> FAIL" % (r['ems'], r['kind'], r['tcal'])
    tc = "OFF" if r['tcal'] is None else "%.1fy" % r['tcal']
    return ("%-9s [%-3s] cal=%-5s -> LPSP %6.4f%%  deg %7.2f kEUR  vie_bat %s  SoC_moy %.3f"
            % (r['ems'], r['kind'], tc, r['lpsp'], r['cost'], _yr(r['life_bat']), r['mean_soc']))


def main():
    print("=== ETAPE 6 -- Vieillissement calendaire batterie (10 EMS, 25 ans) ===", flush=True)
    print("    T_cal ~ U%s ans | N_MC=%d/strat | g(SoC)=SoC (lineaire)" % (TCAL_RANGE, N_MC), flush=True)

    rng = np.random.default_rng(MC_SEED)
    tcal_samples = [float(rng.uniform(*TCAL_RANGE)) for _ in range(N_MC)]

    tasks = []
    for folder, ems in EMS_LIST:
        tasks.append(dict(folder=folder, ems=ems, kind='nom', tcal=None))   # baseline (OFF)
    for folder, ems in EMS_LIST:
        for tc in tcal_samples:
            tasks.append(dict(folder=folder, ems=ems, kind='mc', tcal=tc))  # calendaire ON
    for tc in TCAL_GRID:
        tasks.append(dict(folder=REF_FOLDER, ems=REF_LABEL, kind='oat', tcal=tc))
    res = run_pool(evaluate, tasks, "Calendaire -- baseline + Monte-Carlo + OAT", _fmt)

    nom = {r['ems']: r for r in res if r['kind'] == 'nom' and r['ok']}
    mc_by = {ems: [r for r in res if r['kind'] == 'mc' and r['ems'] == ems and r['ok']]
             for _, ems in EMS_LIST}
    oat = sorted([r for r in res if r['kind'] == 'oat' and r['ok']], key=lambda r: r['tcal'])

    ems_labels = [e for _, e in EMS_LIST]

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Vieillissement calendaire batterie -- 10 EMS 25 ans\n")
        f.write("# T_cal ~ U%s ans (vie calendaire a SoC=100%%), g(SoC)=SoC, N_MC=%d, seed=%d\n\n"
                % (TCAL_RANGE, N_MC, MC_SEED))
        f.write("## Front : baseline (calendaire OFF) -> avec calendaire (MC)\n")
        f.write("ems;SoC_moy;LPSP_off;deg_off;LPSP_cal_mean;deg_cal_mean;deg_cal_std;"
                "ddeg_pct;vie_bat_off;vie_bat_cal_mean\n")
        for _, ems in EMS_LIST:
            rn = nom.get(ems); sub = mc_by.get(ems, [])
            if rn is None:
                f.write("%s;NOMINAL_FAIL\n" % ems); continue
            if sub:
                lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
                vb = [r['life_bat'] for r in sub if r['life_bat'] is not None]
                vb_mean = np.mean(vb) if vb else None
                ddeg = (dg.mean() - rn['cost']) / rn['cost'] * 100 if rn['cost'] else 0.0
                f.write("%s;%.4f;%.4f;%.3f;%.4f;%.3f;%.3f;%+.1f;%s;%s\n"
                        % (ems, rn['mean_soc'], rn['lpsp'], rn['cost'], lp.mean(),
                           dg.mean(), dg.std(), ddeg, _yr(rn['life_bat']),
                           _yr(vb_mean)))
            else:
                f.write("%s;%.4f;%.4f;%.3f;-;-;-;-;%s;-\n"
                        % (ems, rn['mean_soc'], rn['lpsp'], rn['cost'], _yr(rn['life_bat'])))
        f.write("\n## OAT (strategie %s) : deg & vie_bat vs T_cal\n" % REF_LABEL)
        f.write("T_cal_ans;LPSP_%;deg_kEUR;vie_bat_ans\n")
        for r in oat:
            f.write("%.1f;%.4f;%.3f;%s\n" % (r['tcal'], r['lpsp'], r['cost'], _yr(r['life_bat'])))

    # ===================== FIGURE 1 : FRONT baseline -> calendaire + ellipses =====================
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_off = {
        '0-100': (6, 4), '25-75': (6, 4), '50-50': (6, 4), '75-25': (6, 4),
        '100-0': (6, 4), 'RB2': (6, -2), 'RB2(SoH)': (6, -12), 'RB1': (6, 4),
        'SoC1': (6, 4), 'SoC06': (6, 4),
    }
    for (folder, ems), col in zip(EMS_LIST, colors):
        rn = nom.get(ems); sub = mc_by.get(ems, [])
        if rn is None:
            continue
        if len(sub) >= 3:
            x = np.array([r['lpsp'] for r in sub]); y = np.array([r['cost'] for r in sub])
            ax.scatter(x, y, s=12, color=col, alpha=0.25, zorder=2)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor='none',
                               lw=1.6, zorder=4)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor='none',
                               lw=0.9, ls='--', alpha=0.6, zorder=4)
            xm, ym = x.mean(), y.mean()
            ax.plot([rn['lpsp'], xm], [rn['cost'], ym], '-', color=col, lw=0.8,
                    alpha=0.5, zorder=3)   # decalage baseline -> calendaire
            ax.scatter([xm], [ym], marker='D', s=45, color=col, edgecolor='k',
                       linewidth=0.5, zorder=6)
        # point BASELINE (calendaire OFF) : marqueur ouvert
        ax.scatter([rn['lpsp']], [rn['cost']], marker='o', s=70, facecolor='white',
                   edgecolor=col, linewidth=1.6, zorder=5)
        dx, dy = label_off.get(ems, (6, 4))
        ax.annotate(ems, (rn['lpsp'], rn['cost']), textcoords="offset points",
                    xytext=(dx, dy), fontsize=11, color=col, weight='bold', zorder=7)
    ax.scatter([], [], marker='o', facecolor='white', edgecolor='k',
               label='baseline (calendaire OFF)')
    ax.scatter([], [], marker='D', color='0.4', edgecolor='k', label='moyenne MC (calendaire ON)')
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Impact du vieillissement calendaire batterie sur le front", fontsize=12)
    ax.grid(True, ls='--', alpha=0.5); ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_calendar_pareto.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== FIGURE 2 : OAT + insight (SoC moyen vs surcout calendaire) =====================
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
    # (a) OAT sur la reference
    if oat:
        x = np.array([r['tcal'] for r in oat]); dgc = np.array([r['cost'] for r in oat])
        life = np.array([r['life_bat'] if r['life_bat'] is not None else np.nan for r in oat])
        axa.plot(x, dgc, 'o-', color='tab:red')
        axa.set_xlabel("T_cal [ans] (vie calendaire a SoC=100%)")
        axa.set_ylabel("Cout degradation total [kEUR]", color='tab:red')
        axa.tick_params(axis='y', labelcolor='tab:red'); axa.grid(True, ls='--', alpha=0.4)
        axc = axa.twinx(); axc.plot(x, life, 's--', color='tab:blue', alpha=0.8)
        axc.set_ylabel("Vie batterie [ans]", color='tab:blue')
        axc.tick_params(axis='y', labelcolor='tab:blue')
        axa.set_title("OAT calendaire (%s)" % REF_LABEL, fontsize=11)
    # (b) surcout calendaire vs SoC moyen (l'EMS pilote le calendaire)
    sx, sy, scol = [], [], []
    for (folder, ems), col in zip(EMS_LIST, colors):
        rn = nom.get(ems); sub = mc_by.get(ems, [])
        if rn is None or not sub:
            continue
        dcost = np.mean([r['cost'] for r in sub]) - rn['cost']
        sx.append(rn['mean_soc']); sy.append(dcost); scol.append(col)
        axb.annotate(ems, (rn['mean_soc'], dcost), textcoords="offset points",
                     xytext=(4, 3), fontsize=8)
    axb.scatter(sx, sy, c=scol, s=60, edgecolor='k', linewidth=0.4, zorder=3)
    axb.set_xlabel("SoC moyen de la batterie"); axb.set_ylabel("Surcout calendaire moyen [kEUR]")
    axb.set_title("Le calendaire suit la residence en SoC (pilotee par l'EMS)", fontsize=11)
    axb.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_calendar_insight.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 80)
    print("IMPACT CALENDAIRE : baseline (OFF) -> calendaire ON (MC T_cal ~ U%s)" % (TCAL_RANGE,))
    print("-" * 80)
    print("%-9s | SoC_moy | deg OFF -> deg ON (moy)  | d_deg%% | vie_bat OFF->ON" % "EMS")
    for _, ems in EMS_LIST:
        rn = nom.get(ems); sub = mc_by.get(ems, [])
        if rn is None:
            print("%-9s | FAIL" % ems); continue
        if sub:
            dg = np.array([r['cost'] for r in sub])
            vb = [r['life_bat'] for r in sub if r['life_bat'] is not None]
            vbm = np.mean(vb) if vb else None
            ddeg = (dg.mean() - rn['cost']) / rn['cost'] * 100 if rn['cost'] else 0.0
            print("%-9s |  %.3f  | %7.2f -> %7.2f         | %+5.1f | %s -> %s"
                  % (ems, rn['mean_soc'], rn['cost'], dg.mean(), ddeg,
                     _yr(rn['life_bat']), _yr(vbm)))
        else:
            print("%-9s |  %.3f  | %7.2f -> (aucun MC)" % (ems, rn['mean_soc'], rn['cost']))
    print("=" * 80)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_calendar_pareto.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_calendar_insight.pdf"))


if __name__ == "__main__":
    main()
