"""
sens_hthresholds.py -- ETAPE 4 : sensibilite aux SEUILS DE DEGRADATION des
composants HYDROGENE (PEMFC / PEMWE).
=============================================================================
Reviewers APEN : R3-major3-i ("PEMFC/PEMWE degradation is a linear superposition
of four mechanisms with seemingly arbitrary thresholds (1%, 80%, 60%)"). R3
demande aussi des bandes d'incertitude sur le plan de Pareto.

Les "4 seuils de regime" (cf. cost_fcn_total2)
----------------------------------------------
  PEMFC : FC_FHIGH = 0.80  -> seuil haute puissance ("80%")
          FC_FLOW  = 0.01  -> seuil idling / basse puissance ("1%")
  PEMWE : ELY_F30  = 0.30  -> debut de degradation (1 A/cm2 = 30% Pmax)
          ELY_F60  = 0.60  -> saturation au "rated" (2 A/cm2 = 60% Pmax, "60%")

Difference avec les axes precedents
-----------------------------------
Ces seuils pilotent la DEGRADATION cumulee -> SoH -> REMPLACEMENTS -> LPSP. Ils
agissent donc sur les DEUX axes (LPSP ET cout) -> la bande par point est 2D
(ELLIPSES, comme l'EoL), et il FAUT RE-SIMULER (contrairement aux C-weights).

Override PROPRE (sans dupliquer le modele)
------------------------------------------
Les 4 seuils sont des CONSTANTES DE MODULE de cost_fcn_total2 (FC_FHIGH/FC_FLOW
promus depuis des litteraux ; ELY_F30/ELY_F60 deja exposes). La boucle ET le
calcul de cout appellent les MEMES fonctions (get_cost_fc, _ely_advance ->
_ely_rates) qui relisent ces globals a chaque appel. On les mute donc dans
CHAQUE worker (process separes -> aucune contamination), SANS toucher au reste de
Vieillissement8. Override coherent boucle+cout garanti.

SOURCE 100% ASCII (volontaire ; cf. sens_soh_estimation.py).

Sorties (dans ./results/) : sens_hthresholds_pareto.pdf, sens_hthresholds_oat.pdf,
sens_hthresholds.txt.
Lancer :  python sens_hthresholds.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics,
                         lps_cost_keur, lifetimes, run_pool, confidence_ellipse,
                         RESULTS_DIR)
from Common import cost_fcn_total2 as CF   # pour muter les seuils (globals module)

# ============================ CONFIGURATION ============================
# Seuils nominaux (== valeurs par defaut de cost_fcn_total2).
NOMINAL = dict(fhigh=0.80, flow=0.01, f30=0.30, f60=0.60)

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

# --- Monte Carlo : 4 seuils echantillonnes CONJOINTEMENT (uniforme). CHAQUE
# seuil varie de +/-20% autour de son nominal (taille uniforme -> plus clair a
# expliquer). Memes N_MC quadruplets pour toutes les strategies (common random
# numbers). Les contraintes physiques restent satisfaites sur tout le domaine :
# f30 < f60 (max f30=0.36 < min f60=0.48) et flow << fhigh ; fhigh max=0.96 < 1.
MC_RANGES = dict(fhigh=(0.64, 0.96), flow=(0.008, 0.012),
                 f30=(0.24, 0.36), f60=(0.48, 0.72))
N_MC    = 200         # tirages par strategie (run mesocentre : 200 pour ellipses lisses)
MC_SEED = 4242

# --- OAT (figure d'appui) : un seuil a la fois, sur UNE strategie de reference ---
# Grilles a +/-20% (nominal +/-{20%,10%,0}) pour rester coherent avec le MC.
REF_FOLDER, REF_LABEL = "RB2(SoH)", "RB2(SoH)"
FHIGH_GRID = [0.64, 0.72, 0.80, 0.88, 0.96]
FLOW_GRID  = [0.008, 0.009, 0.010, 0.011, 0.012]
F30_GRID   = [0.24, 0.27, 0.30, 0.33, 0.36]
F60_GRID   = [0.48, 0.54, 0.60, 0.66, 0.72]

# Cout total ~ 10 nominaux + 10*N_MC + OAT. N_MC=15 -> 10 + 150 + ~16 = ~176 sims.
OUT_TXT = os.path.join(RESULTS_DIR, "sens_hthresholds.txt")
# ======================================================================


def evaluate(params):
    """Worker picklable. params = dict(folder, label, kind, fhigh, flow, f30, f60).
    Mute les 4 seuils (globals de cost_fcn_total2) puis lance la strategie."""
    try:
        CF.FC_FHIGH = params['fhigh']
        CF.FC_FLOW  = params['flow']
        CF.ELY_F30  = params['f30']
        CF.ELY_F60  = params['f60']
        strat = load_strategy(params['folder'])
        data = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        clps = lps_cost_keur(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:
        lpsp = cost = clps = lb = lf = le = None
        ok = False
        print("  [FAIL] %-9s seuils=(%.3f,%.4f,%.3f,%.3f) : %s"
              % (params['label'], params['fhigh'], params['flow'],
                 params['f30'], params['f60'], e), flush=True)
    return dict(params=params, label=params['label'], kind=params['kind'],
                fhigh=params['fhigh'], flow=params['flow'],
                f30=params['f30'], f60=params['f60'],
                lpsp=lpsp, cost=cost, clps=clps,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def _fmt(r):
    if not r['ok']:
        return "%-9s [%s] -> FAIL" % (r['label'], r['kind'])
    return ("%-9s [%-8s] (fh%.2f fl%.3f f30%.2f f60%.2f) -> LPSP %6.4f%%  deg %7.2f kEUR"
            % (r['label'], r['kind'], r['fhigh'], r['flow'], r['f30'], r['f60'],
               r['lpsp'], r['cost']))


def build_tasks(mc_samples):
    """nominaux (toutes strategies) + MC (toutes strategies) + OAT (reference)."""
    tasks = []
    for folder, label in SCENARIOS:
        tasks.append(dict(folder=folder, label=label, kind='nom', **NOMINAL))
    for folder, label in SCENARIOS:
        for s in mc_samples:
            tasks.append(dict(folder=folder, label=label, kind='mc', **s))
    # OAT : un seuil varie, les 3 autres au nominal
    for key, grid in (('fhigh', FHIGH_GRID), ('flow', FLOW_GRID),
                      ('f30', F30_GRID), ('f60', F60_GRID)):
        for v in grid:
            if v == NOMINAL[key]:
                continue
            p = dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_' + key, **NOMINAL)
            p[key] = v
            tasks.append(p)
    return tasks


def main():
    print("=== ETAPE 4 -- Sensibilite aux seuils de degradation H2 (PEMFC/PEMWE) (25 ans) ===", flush=True)
    print("    %d strategies | N_MC=%d/strat | ranges=%s | nominal=%s"
          % (len(SCENARIOS), N_MC, MC_RANGES, NOMINAL), flush=True)

    rng = np.random.default_rng(MC_SEED)
    mc_samples = [dict(fhigh=float(rng.uniform(*MC_RANGES['fhigh'])),
                       flow=float(rng.uniform(*MC_RANGES['flow'])),
                       f30=float(rng.uniform(*MC_RANGES['f30'])),
                       f60=float(rng.uniform(*MC_RANGES['f60']))) for _ in range(N_MC)]

    tasks = build_tasks(mc_samples)
    res = run_pool(evaluate, tasks, "Seuils H2 -- nominaux + Monte-Carlo + OAT", _fmt)

    nom = {r['label']: r for r in res if r['kind'] == 'nom' and r['ok']}
    mc_by = {label: [r for r in res
                     if r['kind'] == 'mc' and r['label'] == label and r['ok']]
             for _, label in SCENARIOS}
    oat = {k: [r for r in res if r['kind'] == k and r['ok']]
           for k in ('oat_fhigh', 'oat_flow', 'oat_f30', 'oat_f60')}
    ref_nom = nom.get(REF_LABEL)
    for k, key in (('oat_fhigh', 'fhigh'), ('oat_flow', 'flow'),
                   ('oat_f30', 'f30'), ('oat_f60', 'f60')):
        if ref_nom is not None:
            oat[k] = sorted(oat[k] + [ref_nom], key=lambda r: r[key])

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite seuils degradation H2 -- 10 EMS 25 ans | nominal=%s\n" % NOMINAL)
        f.write("# MC: N=%d/strat, ranges=%s, seed=%d (memes quadruplets pour toutes)\n\n"
                % (N_MC, MC_RANGES, MC_SEED))
        f.write("## Front de Pareto : nominal + dispersion MC par strategie\n")
        f.write("strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;deg_min;deg_max;N_ok;clps_nom;clps_mean;clps_std\n")
        for _, label in SCENARIOS:
            rn = nom.get(label); sub = mc_by.get(label, [])
            if rn is None:
                f.write("%s;NOMINAL_FAIL\n" % label); continue
            if sub:
                lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
                cl = np.array([r['clps'] for r in sub])
                f.write("%s;%.4f;%.3f;%.4f;%.4f;%.3f;%.3f;%.3f;%.3f;%d;%.3f;%.3f;%.3f\n"
                        % (label, rn['lpsp'], rn['cost'], lp.mean(), lp.std(),
                           dg.mean(), dg.std(), dg.min(), dg.max(), len(sub),
                           rn['clps'], cl.mean(), cl.std()))
            else:
                f.write("%s;%.4f;%.3f;-;-;-;-;-;-;0;%.3f;-;-\n"
                        % (label, rn['lpsp'], rn['cost'], rn['clps']))
        f.write("\n## OAT (strategie %s) : un seuil varie, les 3 autres au nominal\n" % REF_LABEL)
        for k, key, lifekey in (('oat_fhigh', 'fhigh', 'life_fc'),
                                ('oat_flow', 'flow', 'life_fc'),
                                ('oat_f30', 'f30', 'life_ely'),
                                ('oat_f60', 'f60', 'life_ely')):
            f.write("# %s (vie %s)\nseuil;LPSP_%%;deg_kEUR;vie_ans\n" % (key, lifekey[5:]))
            for r in oat[k]:
                f.write("%.4f;%.4f;%.3f;%s\n" % (r[key], r['lpsp'], r['cost'], _yr(r[lifekey])))
            f.write("\n")

    # ===================== FIGURE 1 : FRONT + ELLIPSES PAR POINT =====================
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_off = {
        '0-100': (6, 4), '25-75': (6, 4), '50-50': (6, 4), '75-25': (6, 4),
        '100-0': (6, 4), 'RB2': (6, -2), 'RB2(SoH)': (6, -12), 'RB1': (6, 4),
        'SoC1': (6, 4), 'SoC06': (6, 4),
    }
    for (folder, label), col in zip(SCENARIOS, colors):
        rn = nom.get(label); sub = mc_by.get(label, [])
        if rn is None:
            continue
        if len(sub) >= 3:
            x = np.array([r['lpsp'] for r in sub]); y = np.array([r['cost'] for r in sub])
            ax.scatter(x, y, s=12, color=col, alpha=0.25, zorder=2)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor='none',
                               lw=1.6, zorder=4)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor='none',
                               lw=0.9, ls='--', alpha=0.6, zorder=4)
        ax.scatter([rn['lpsp']], [rn['cost']], marker='o', s=70, color=col,
                   edgecolor='k', linewidth=0.6, zorder=6)
        dx, dy = label_off.get(label, (6, 4))
        ax.annotate(label, (rn['lpsp'], rn['cost']), textcoords="offset points",
                    xytext=(dx, dy), fontsize=11, color=col, weight='bold', zorder=7)
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Robustesse du front de Pareto aux seuils de degradation H2", fontsize=12)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_hthresholds_pareto.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== FIGURE 2 : OAT (1x4) sur la strategie de reference =====================
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    specs = [('oat_fhigh', 'fhigh', 'life_fc',  'FC : seuil haut (80%)'),
             ('oat_flow',  'flow',  'life_fc',  'FC : seuil idling (1%)'),
             ('oat_f30',   'f30',   'life_ely', 'ELY : debut (30%)'),
             ('oat_f60',   'f60',   'life_ely', 'ELY : saturation (60%)')]
    for ax, (k, key, lifekey, title) in zip(axes, specs):
        rows = oat[k]
        if not rows:
            ax.set_title("%s (aucun run OK)" % title); continue
        x = np.array([r[key] for r in rows])
        dgc = np.array([r['cost'] for r in rows])
        life = np.array([r[lifekey] if r[lifekey] is not None else np.nan for r in rows])
        ax.plot(x, dgc, 'o-', color='tab:red')
        ax.set_title(title, fontsize=10); ax.set_xlabel("seuil")
        ax.grid(True, ls='--', alpha=0.4)
        ax.axvline(NOMINAL[key], color='k', ls=':', lw=0.8, alpha=0.6)
        axb = ax.twinx()
        axb.plot(x, life, 's--', color='tab:blue', alpha=0.8)
        if k == 'oat_fhigh':
            ax.set_ylabel("Cout degradation total [kEUR]", color='tab:red')
        if k == 'oat_f60':
            axb.set_ylabel("Vie du composant [ans]", color='tab:blue')
        axb.tick_params(axis='y', labelcolor='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle("Sensibilite OAT aux seuils de degradation H2 (%s)" % REF_LABEL, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_hthresholds_oat.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 78)
    print("FRONT (nominal) + dispersion MC sur les 4 seuils de degradation H2")
    print("-" * 78)
    for _, label in SCENARIOS:
        rn = nom.get(label); sub = mc_by.get(label, [])
        if rn is None:
            print("%-9s | NOMINAL FAIL" % label); continue
        if sub:
            dg = np.array([r['cost'] for r in sub]); lp = np.array([r['lpsp'] for r in sub])
            print("%-9s | nom %6.4f%%/%7.2f | MC deg %7.2f+/-%.2f [%.2f..%.2f] LPSP %6.4f+/-%.4f (N=%d)"
                  % (label, rn['lpsp'], rn['cost'], dg.mean(), dg.std(), dg.min(), dg.max(),
                     lp.mean(), lp.std(), len(sub)))
        else:
            print("%-9s | nom %6.4f%%/%7.2f | (aucun MC OK)" % (label, rn['lpsp'], rn['cost']))
    print("=" * 78)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_hthresholds_pareto.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_hthresholds_oat.pdf"))


if __name__ == "__main__":
    main()
