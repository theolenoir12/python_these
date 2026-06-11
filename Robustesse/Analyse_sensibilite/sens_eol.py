"""
sens_eol.py -- ETAPE 2 : sensibilite aux SEUILS DE FIN DE VIE (EoL).
=============================================================================
Reviewers APEN : R3-major3-iii (EoL batterie 40% de perte = SoH 0.60 juge eleve,
tester 20-30% = SoH 0.70-0.80), R4-9/R4-10 (vie batterie courte, justifier le
seuil), R1-6 (arbitrage remplacement vs vieillissement). R3 demande surtout des
BANDES D'INCERTITUDE sur le PLAN DE PARETO.

Difference avec l'etape 1 (estimation du SoH)
---------------------------------------------
L'erreur d'estimation du SoH ne concernait QUE RB2(SoH) (seule strategie qui
utilise le SoH dans ses setpoints). Le seuil EoL, lui, est un parametre GLOBAL
qui agit sur TOUTES les strategies, via trois canaux (verifies dans le code de
base) :
  (i)   declenchement des remplacements dans la boucle
        (main_init_and_loop : "if SoH_*_tp1 < *['SoH_EoL']") ;
  (ii)  normalisation du cout de degradation
        (cost_fcn_total2 : cout = indicateur / (1 - SoH_EoL) * cout_remplacement) ;
  (iii) conversion indicateur->SoH (get_soh) et bornes alpha (brentq).

=> On reproduit donc le FRONT DE PARETO COMPLET de batch_pareto.py (un point =
une strategie), mais ou CHAQUE point porte son propre nuage Monte-Carlo + une
ELLIPSE DE CONFIANCE issue de la variation conjointe des seuils EoL. C'est la
"bande d'incertitude par strategie" reclamee par R3.

SOURCE 100% ASCII (volontaire ; cf. sens_soh_estimation.py).

Override PROPRE : on mute I.BAT/FC/ELY['SoH_EoL'] DANS chaque worker (process
separes -> aucune contamination), SANS TOUCHER a Vieillissement8.

Limite du modele (FC/ELY)
-------------------------
Les bornes alpha sont resolues par brentq sur [0, ~0.222] (FC) et [0, ~0.226]
(ELY) dans Common/main_init_and_loop.py (lignes ~78-84). La borne haute est
JUSTE SOUS une singularite du modele de tension (le terme log(1 - i/(...(1-a)))
diverge), et le modele de degradation n'est calibre que jusqu'a ~10-20% de perte
de tension. On NE descend donc PAS l'EoL FC/ELY sous 0.90 par defaut (sinon
extrapolation hors calibration). Pour elargir : modifier les brackets brentq
dans Vieillissement8 PUIS etendre MC_RANGES['fc'/'ely'] ci-dessous. Tout
echantillon qui ferait echouer le brentq est ABANDONNE proprement (try/except).
La batterie (pas de alpha) est libre : on balaie [0.60, 0.80].

Sorties (dans ./results/) : sens_eol_pareto.pdf (figure principale),
sens_eol_oat.pdf (figure d'appui), sens_eol.txt.
Lancer :  python sens_eol.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics,
                         lifetimes, run_pool, confidence_ellipse, RESULTS_DIR)

# ============================ CONFIGURATION ============================
# Seuils EoL du fichier de base (Init_EMR_MG_v16_python : BAT 0.70, FC/ELY 0.90).
BASE_EOL = dict(bat=0.70, fc=0.90, ely=0.90)

# --- Front de Pareto : les 10 EMS de batch_pareto.py (dossiers de Vieillissement8) ---
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

# --- Monte Carlo : seuils EoL des 3 composants echantillonnes CONJOINTEMENT ---
# Tirage UNIFORME dans chaque plage. Les MEMES N_MC triplets sont appliques a
# toutes les strategies (common random numbers) -> comparaison des ellipses non
# polluee par le bruit MC.
#   bat : libre, 0.60 (= 40% de perte, valeur article) .. 0.80 (= 20%, R3).
#   fc/ely : >= 0.90 (limite calibration ; voir docstring). Elargir = editer les
#            brackets brentq dans Vieillissement8 puis ces plages.
MC_RANGES = dict(bat=(0.60, 0.80), fc=(0.85, 0.95), ely=(0.85, 0.95))
N_MC    = 15          # tirages par strategie (auto-scalable : monter sur grosse machine)
MC_SEED = 2024

# --- OAT (figure d'appui) : un seuil a la fois, sur UNE strategie de reference ---
REF_FOLDER, REF_LABEL = "RB2(SoH)", "RB2(SoH)"
BAT_EOL_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]    # batterie : libre
FC_EOL_GRID  = [0.85, 0.875, 0.90, 0.925, 0.95]  # FC  : >= 0.90 (limite modele)
ELY_EOL_GRID = [0.85, 0.875, 0.90, 0.925, 0.95]  # ELY : >= 0.90 (limite modele)

# Cout total ~ 10 nominaux + 10*N_MC + OAT. N_MC=15 -> 10 + 150 + ~11 = ~170 sims.
OUT_TXT = os.path.join(RESULTS_DIR, "sens_eol.txt")
# ======================================================================


def evaluate(params):
    """Worker picklable. params = dict(folder, label, kind, bat, fc, ely).
    Charge la strategie du dossier, mute les seuils EoL, lance la simu."""
    try:
        strat = load_strategy(params['folder'])
        I.BAT['SoH_EoL'] = params['bat']
        I.FC['SoH_EoL']  = params['fc']
        I.ELY['SoH_EoL'] = params['ely']
        data = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:                 # ex. brentq hors bracket si EoL trop bas
        lpsp = cost = lb = lf = le = None
        ok = False
        print("  [FAIL] %-9s EoL=(%.3f,%.3f,%.3f) : %s"
              % (params['label'], params['bat'], params['fc'], params['ely'], e), flush=True)
    return dict(params=params, label=params['label'], kind=params['kind'],
                bat=params['bat'], fc=params['fc'], ely=params['ely'],
                lpsp=lpsp, cost=cost, life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def _fmt(r):
    if not r['ok']:
        return "%-9s [%s] EoL=(%.2f,%.2f,%.2f) -> FAIL" % (
            r['label'], r['kind'], r['bat'], r['fc'], r['ely'])
    return ("%-9s [%-7s] EoL=(%.2f,%.2f,%.2f) -> LPSP %6.4f%%  deg %7.2f kEUR  vie B %s"
            % (r['label'], r['kind'], r['bat'], r['fc'], r['ely'],
               r['lpsp'], r['cost'], _yr(r['life_bat'])))


def build_tasks(eol_samples):
    """Construit la liste complete des taches : nominaux + MC (toutes strategies)
    + OAT (strategie de reference). Une seule pool -> coeurs satures en continu."""
    tasks = []
    # 1) point NOMINAL de chaque strategie (seuils EoL de base) -> le point du front
    for folder, label in SCENARIOS:
        tasks.append(dict(folder=folder, label=label, kind='nom', **BASE_EOL))
    # 2) nuage MONTE-CARLO de chaque strategie (memes triplets EoL pour toutes)
    for folder, label in SCENARIOS:
        for (b, f, e) in eol_samples:
            tasks.append(dict(folder=folder, label=label, kind='mc', bat=b, fc=f, ely=e))
    # 3) OAT sur la strategie de reference (figure d'appui)
    for v in BAT_EOL_GRID:
        if v != BASE_EOL['bat']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_bat',
                              bat=v, fc=BASE_EOL['fc'], ely=BASE_EOL['ely']))
    for v in FC_EOL_GRID:
        if v != BASE_EOL['fc']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_fc',
                              bat=BASE_EOL['bat'], fc=v, ely=BASE_EOL['ely']))
    for v in ELY_EOL_GRID:
        if v != BASE_EOL['ely']:
            tasks.append(dict(folder=REF_FOLDER, label=REF_LABEL, kind='oat_ely',
                              bat=BASE_EOL['bat'], fc=BASE_EOL['fc'], ely=v))
    return tasks


def main():
    print("=== ETAPE 2 -- Sensibilite aux seuils EoL : front de Pareto + bandes (25 ans) ===", flush=True)
    print("    %d strategies | N_MC=%d/strat | ranges=%s | base EoL=%s"
          % (len(SCENARIOS), N_MC, MC_RANGES, BASE_EOL), flush=True)

    # --- echantillons EoL communs a toutes les strategies (common random numbers) ---
    rng = np.random.default_rng(MC_SEED)
    eol_samples = [(float(rng.uniform(*MC_RANGES['bat'])),
                    float(rng.uniform(*MC_RANGES['fc'])),
                    float(rng.uniform(*MC_RANGES['ely']))) for _ in range(N_MC)]

    tasks = build_tasks(eol_samples)
    res = run_pool(evaluate, tasks, "EoL -- nominaux + Monte-Carlo + OAT", _fmt)

    # --- tri des resultats ---
    nom = {r['label']: r for r in res if r['kind'] == 'nom' and r['ok']}
    mc_by = {label: [r for r in res
                     if r['kind'] == 'mc' and r['label'] == label and r['ok']]
             for _, label in SCENARIOS}
    oat = {k: sorted([r for r in res if r['kind'] == k and r['ok']],
                     key=lambda r: r[{'oat_bat': 'bat', 'oat_fc': 'fc', 'oat_ely': 'ely'}[k]])
           for k in ('oat_bat', 'oat_fc', 'oat_ely')}
    # la reference au baseline appartient a chaque grille OAT
    ref_nom = nom.get(REF_LABEL)
    for k, comp in (('oat_bat', 'bat'), ('oat_fc', 'fc'), ('oat_ely', 'ely')):
        if ref_nom is not None and all(abs(ref_nom[c] - BASE_EOL[c]) < 1e-12
                                       for c in ('bat', 'fc', 'ely')):
            oat[k] = sorted(oat[k] + [ref_nom], key=lambda r: r[comp])

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite seuils EoL -- front de Pareto + bandes | base EoL=%s\n" % BASE_EOL)
        f.write("# MC: N=%d/strat, ranges=%s, seed=%d (memes triplets pour toutes les strategies)\n\n"
                % (N_MC, MC_RANGES, MC_SEED))
        f.write("## Front de Pareto : point nominal + dispersion MC par strategie\n")
        f.write("strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;deg_min;deg_max;N_ok\n")
        for _, label in SCENARIOS:
            rn = nom.get(label)
            sub = mc_by.get(label, [])
            if rn is None:
                f.write("%s;NOMINAL_FAIL\n" % label); continue
            lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
            if len(sub):
                f.write("%s;%.4f;%.3f;%.4f;%.4f;%.3f;%.3f;%.3f;%.3f;%d\n"
                        % (label, rn['lpsp'], rn['cost'], lp.mean(), lp.std(),
                           dg.mean(), dg.std(), dg.min(), dg.max(), len(sub)))
            else:
                f.write("%s;%.4f;%.3f;-;-;-;-;-;-;0\n" % (label, rn['lpsp'], rn['cost']))
        f.write("\n## OAT (strategie %s) : un seuil varie, les deux autres au baseline\n" % REF_LABEL)
        for k, comp, lifekey in (('oat_bat', 'bat', 'life_bat'),
                                 ('oat_fc', 'fc', 'life_fc'),
                                 ('oat_ely', 'ely', 'life_ely')):
            f.write("# %s\nSoH_EoL;LPSP_%%;deg_kEUR;vie_composant_ans\n" % comp.upper())
            for r in oat[k]:
                f.write("%.3f;%.4f;%.3f;%s\n" % (r[comp], r['lpsp'], r['cost'], _yr(r[lifekey])))
            f.write("\n")

    # ===================== FIGURE 1 : FRONT DE PARETO + ELLIPSES PAR POINT =====================
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    # offsets de label (dx, dy) par strategie pour limiter les chevauchements ; a ajuster.
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
        # point NOMINAL (seuils EoL de base) = le point "officiel" du front
        ax.scatter([rn['lpsp']], [rn['cost']], marker='o', s=70, color=col,
                   edgecolor='k', linewidth=0.6, zorder=6)
        dx, dy = label_off.get(label, (4, 4))
        ax.annotate(label, (rn['lpsp'], rn['cost']), textcoords="offset points",
                    xytext=(dx, dy), fontsize=11, color=col, weight='bold', zorder=7)

    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Robustesse du front de Pareto aux seuils de fin de vie (EoL)\n"
                 "point = seuils nominaux ; ellipses 1$\\sigma$/2$\\sigma$ = incertitude EoL "
                 "(bat %.2f-%.2f, FC/ELY %.2f-%.2f)"
                 % (MC_RANGES['bat'][0], MC_RANGES['bat'][1],
                    MC_RANGES['fc'][0], MC_RANGES['fc'][1]), fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_eol_pareto.pdf"), bbox_inches="tight")
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
        ax.set_title(title, fontsize=11); ax.set_xlabel("SoH_EoL")
        ax.grid(True, ls='--', alpha=0.4)
        ax.axvline(BASE_EOL[comp], color='k', ls=':', lw=0.8, alpha=0.6)
        axb = ax.twinx()
        axb.plot(x, life, 's--', color='tab:blue', alpha=0.8)
        if comp == 'bat':
            ax.set_ylabel("Cout degradation total [kEUR]", color='tab:red')
        if comp == 'ely':
            axb.set_ylabel("Vie du composant [ans]", color='tab:blue')
        axb.tick_params(axis='y', labelcolor='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle("Sensibilite aux seuils EoL (OAT, strategie %s)" % REF_LABEL, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sens_eol_oat.pdf"), bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 78)
    print("FRONT DE PARETO (point nominal) + dispersion MC sur les seuils EoL")
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
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_eol_pareto.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_eol_oat.pdf"))


if __name__ == "__main__":
    main()
