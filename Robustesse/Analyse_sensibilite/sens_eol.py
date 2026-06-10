"""
sens_eol.py -- ETAPE 2 : sensibilite de RB2(SoH) aux SEUILS DE FIN DE VIE (EoL).
=============================================================================
Reviewers APEN : R3-major3-iii (EoL batterie 40% de perte = SoH 0.7 juge eleve,
tester 20-30% = SoH 0.75-0.8), R4-9/R4-10 (vie batterie courte, justifier le
seuil), R1-6 (arbitrage remplacement vs vieillissement). R3 demande des BANDES
D'INCERTITUDE sur le plan de Pareto.

SOURCE 100% ASCII (volontaire ; cf. sens_soh_estimation.py).

Principe
--------
On fait varier les seuils de fin de vie SoH_EoL de chaque composant et on
observe l'effet sur (LPSP, cout de degradation, duree de vie). Le seuil EoL
controle (i) la frequence des remplacements dans la boucle et (ii) la
normalisation du cout (cout = indicateur / (1-SoH_EoL) * cout_remplacement) :
les deux se combinent en "cout cycle de vie ~ nb_remplacements * cout_unitaire".

Override PROPRE : on mute I.BAT/FC/ELY['SoH_EoL'] dans chaque worker (process
separes -> pas de contamination), SANS TOUCHER a Vieillissement8.

Limite du modele : les bornes alpha (brentq) du code de base correspondent
EXACTEMENT a SoH=0.90 pour FC et ELY (modele de degradation calibre jusqu'a
~10% de perte de tension). On ne peut donc PAS descendre l'EoL FC/ELY sous 0.90
sans recalibrer le modele -> on teste FC/ELY dans [0.90, 0.96] (EoL plus
stricte). La batterie (sans alpha) est libre : [0.60, 0.80].

Sorties (dans ./results/) : sens_eol_oat.pdf, sens_eol_pareto.pdf, sens_eol.txt
Lancer :  python sens_eol.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, BASE_STRAT, metrics, lifetimes,
                         run_pool, confidence_ellipse, RESULTS_DIR)

# ============================ CONFIGURATION ============================
BASE_EOL = dict(bat=0.70, fc=0.90, ely=0.90)     # seuils EoL du fichier de base

# --- OAT : un composant a la fois, les autres au baseline ---
BAT_EOL_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]    # batterie : libre
FC_EOL_GRID  = [0.90, 0.92, 0.94, 0.96]          # FC  : >= 0.90 (limite modele)
ELY_EOL_GRID = [0.90, 0.92, 0.94, 0.96]          # ELY : >= 0.90 (limite modele)

# --- Monte Carlo : EoL des 3 composants echantillonnes conjointement ---
MC_RANGES = dict(bat=(0.60, 0.80), fc=(0.90, 0.96), ely=(0.90, 0.96))
N_MC = 35
MC_SEED = 2024

# Cout total = 1(base) + (5-1)+(4-1)+(4-1) OAT + N_MC = 1 + 10 + 35 = 46 sims.
OUT_TXT = os.path.join(RESULTS_DIR, "sens_eol.txt")
# ======================================================================


def evaluate(params):
    """Worker picklable. params = dict(bat, fc, ely, group). Mute les seuils
    EoL puis lance RB2(SoH) de base (strategie inchangee)."""
    try:
        I.BAT['SoH_EoL'] = params['bat']
        I.FC['SoH_EoL']  = params['fc']
        I.ELY['SoH_EoL'] = params['ely']
        data = init_and_run_loop(BASE_STRAT)
        lpsp, cost = metrics(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:                 # ex. brentq hors bracket
        lpsp = cost = lb = lf = le = None
        ok = False
        print("  [FAIL] EoL=(%.3f,%.3f,%.3f) : %s" % (params['bat'], params['fc'],
              params['ely'], e), flush=True)
    return dict(**params, lpsp=lpsp, cost=cost,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "EoL=(%.2f,%.2f,%.2f) -> FAIL" % (r['bat'], r['fc'], r['ely'])
    return ("EoL=(%.2f,%.2f,%.2f) -> LPSP %6.4f%%  deg %7.2f kEUR  vie B/F/E %s/%s/%s"
            % (r['bat'], r['fc'], r['ely'], r['lpsp'], r['cost'],
               _yr(r['life_bat']), _yr(r['life_fc']), _yr(r['life_ely'])))


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def main():
    print("=== ETAPE 2 -- Sensibilite aux seuils de fin de vie (EoL) (RB2(SoH), 25 ans) ===", flush=True)
    print("    base EoL = %s | FC/ELY bornes >= 0.90 (limite modele)" % BASE_EOL, flush=True)

    base = evaluate(dict(group='base', **BASE_EOL))
    print("\nBASELINE : %s" % _fmt(base), flush=True)

    # -------- OAT : un composant a la fois --------
    p_oat = []
    for v in BAT_EOL_GRID:
        p_oat.append(dict(group='oat_bat', bat=v, fc=BASE_EOL['fc'], ely=BASE_EOL['ely']))
    for v in FC_EOL_GRID:
        p_oat.append(dict(group='oat_fc', bat=BASE_EOL['bat'], fc=v, ely=BASE_EOL['ely']))
    for v in ELY_EOL_GRID:
        p_oat.append(dict(group='oat_ely', bat=BASE_EOL['bat'], fc=BASE_EOL['fc'], ely=v))
    # retire les doublons exacts du baseline (deja calcule) pour economiser
    p_oat = [p for p in p_oat if (p['bat'], p['fc'], p['ely'])
             != (BASE_EOL['bat'], BASE_EOL['fc'], BASE_EOL['ely'])]
    r_oat = run_pool(evaluate, p_oat, "OAT -- un composant a la fois", _fmt)
    r_oat_all = [base] + r_oat   # baseline appartient a chaque grille

    # -------- Monte Carlo conjoint --------
    rng = np.random.default_rng(MC_SEED)
    p_mc = []
    for _ in range(N_MC):
        p_mc.append(dict(group='mc',
                         bat=float(rng.uniform(*MC_RANGES['bat'])),
                         fc=float(rng.uniform(*MC_RANGES['fc'])),
                         ely=float(rng.uniform(*MC_RANGES['ely']))))
    r_mc = [r for r in run_pool(evaluate, p_mc, "Monte Carlo -- EoL conjoints", _fmt) if r['ok']]

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    def _g(group):
        return [r for r in r_oat_all if r['group'] in (group, 'base')]
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite seuils EoL -- RB2(SoH) 25 ans | base EoL=%s\n" % BASE_EOL)
        f.write("BASELINE; LPSP=%.4f%%; deg=%.3fkEUR; vie_bat=%s; vie_fc=%s; vie_ely=%s\n\n"
                % (base['lpsp'], base['cost'], base['life_bat'], base['life_fc'], base['life_ely']))
        for comp, grid, key in (('bat', BAT_EOL_GRID, 'life_bat'),
                                ('fc', FC_EOL_GRID, 'life_fc'),
                                ('ely', ELY_EOL_GRID, 'life_ely')):
            f.write("## OAT %s : SoH_EoL varie, autres au baseline\n" % comp)
            f.write("SoH_EoL;LPSP_%;deg_kEUR;vie_composant_ans\n")
            # on ne garde que les runs ou SEUL `comp` differe du baseline
            rows = sorted([r for r in r_oat_all
                           if all(r[c] == BASE_EOL[c] for c in ('bat', 'fc', 'ely') if c != comp)],
                          key=lambda r: r[comp])
            for r in rows:
                f.write("%.3f;%.4f;%.3f;%s\n" % (r[comp], r['lpsp'], r['cost'], _yr(r[key])))
            f.write("\n")
        lp = np.array([r['lpsp'] for r in r_mc]); dg = np.array([r['cost'] for r in r_mc])
        f.write("## Monte Carlo conjoint (N=%d) ranges=%s\n" % (len(r_mc), MC_RANGES))
        f.write("LPSP_mean=%.4f LPSP_std=%.4f (min %.4f max %.4f)\n"
                % (lp.mean(), lp.std(), lp.min(), lp.max()))
        f.write("deg_mean=%.3f deg_std=%.3f (min %.3f max %.3f)\n"
                % (dg.mean(), dg.std(), dg.min(), dg.max()))

    # ===================== FIGURE 1 : OAT (1x3) =====================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [('bat', BAT_EOL_GRID, 'life_bat', 'Batterie'),
             ('fc',  FC_EOL_GRID,  'life_fc',  'PEMFC'),
             ('ely', ELY_EOL_GRID, 'life_ely', 'PEMWE')]
    for ax, (comp, grid, lifekey, title) in zip(axes, specs):
        rows = sorted([r for r in r_oat_all
                       if all(r[c] == BASE_EOL[c] for c in ('bat', 'fc', 'ely') if c != comp)],
                      key=lambda r: r[comp])
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
    fig.suptitle("Sensibilite aux seuils de fin de vie (OAT)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "sens_eol_oat.pdf"),
                                    bbox_inches="tight"); plt.close()

    # ===================== FIGURE 2 : Pareto + bande de confiance =====================
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.array([r['lpsp'] for r in r_mc]); y = np.array([r['cost'] for r in r_mc])
    sc = ax.scatter(x, y, c=[r['bat'] for r in r_mc], cmap='viridis', s=30, alpha=0.8, zorder=2)
    cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label("SoH_EoL batterie")
    confidence_ellipse(x, y, ax, n_std=1.0, edgecolor='0.2', facecolor='none', lw=1.8, zorder=4)
    confidence_ellipse(x, y, ax, n_std=2.0, edgecolor='0.2', facecolor='none', lw=1.0,
                       ls='--', alpha=0.7, zorder=4)
    ax.scatter([x.mean()], [y.mean()], marker='D', s=60, color='0.2', zorder=5, label='moyenne MC')
    ax.scatter([base['lpsp']], [base['cost']], marker='*', s=320, color='red',
               edgecolor='k', linewidth=0.6, zorder=6, label='baseline (0.70/0.90/0.90)')
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Sensibilite aux seuils de fin de vie (EoL)", fontsize=11)
    ax.grid(True, ls='--', alpha=0.5); ax.legend(loc='best', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "sens_eol_pareto.pdf"),
                                    bbox_inches="tight"); plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 72)
    print("BASELINE : LPSP %.4f%%  deg %.2f kEUR  vie B/F/E %s/%s/%s"
          % (base['lpsp'], base['cost'], _yr(base['life_bat']), _yr(base['life_fc']),
             _yr(base['life_ely'])))
    print("-" * 72)
    for comp, grid, lifekey, title in specs:
        rows = sorted([r for r in r_oat_all
                       if all(r[c] == BASE_EOL[c] for c in ('bat', 'fc', 'ely') if c != comp)],
                      key=lambda r: r[comp])
        print("OAT %s :" % title)
        for r in rows:
            print("   SoH_EoL=%.2f : LPSP %6.4f%%  deg %7.2f kEUR  vie %s ans"
                  % (r[comp], r['lpsp'], r['cost'], _yr(r[lifekey])))
    lp = np.array([r['lpsp'] for r in r_mc]); dg = np.array([r['cost'] for r in r_mc])
    print("-" * 72)
    print("Monte Carlo conjoint (N=%d): LPSP %.4f+/-%.4f  deg %.2f+/-%.2f kEUR  (deg %.1f..%.1f)"
          % (len(r_mc), lp.mean(), lp.std(), dg.mean(), dg.std(), dg.min(), dg.max()))
    print("=" * 72)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_eol_oat.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_eol_pareto.pdf"))


if __name__ == "__main__":
    main()
