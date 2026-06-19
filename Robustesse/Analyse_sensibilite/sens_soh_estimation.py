"""
sens_soh_estimation.py -- ETAPE 1 : robustesse de RB2(SoH) a l'ERREUR
D'ESTIMATION DU SoH  (reviewers APEN : R2-6 "+/-10%", R3-min2 "+/-2% gaussien").
=============================================================================
SOURCE 100% ASCII (volontaire) : certains interpreteurs decodent le .py en
latin-1 et mojibakent accents / euro / sigma. En ASCII pur, console, .txt et
figures restent propres quel que soit l'encodage de l'interpreteur.

L'EMS RB2(SoH) calcule ses setpoints a partir du SoH ESTIME :
    P_fc_set  = coef_fc  * P_fc_max_nom  * SoH_fc_est
    P_ely_set = coef_ely * P_ely_max_nom * SoH_ely_est
On remplace SoH_*_est par une version BRUITEE du VRAI SoH :
    SoH_est = clip( SoH_vrai * (1 + e) , clip_lo, clip_hi )
avec e = biais (constant) + bruit gaussien (sigma). L'estimation est
PIECEWISE-CONSTANTE, rafraichie tous les `refresh_steps` pas (defaut 1 semaine),
conforme a "SoH estime au moins 1x/semaine" (article Sec. 2.4.1) : un bruit
horaire blanc se moyennerait sur 168 pas et sous-estimerait l'impact.

Seul le SoH *vu par le controleur* est bruite. Le VRAI SoH se degrade
normalement ; LPSP et cout sont calcules sur les VRAIES trajectoires -> on
mesure l'effet de l'erreur d'estimation sur la performance REELLE.

NE MODIFIE RIEN dans Vieillissement8 (import lecture seule via sens_common).

Sorties (dans ./results/) : sens_soh_bias.pdf, sens_soh_pareto.pdf, sens_soh.txt
Lancer :  python sens_soh_estimation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, BASE_STRAT, metrics, lps_cost_keur,
                         lifetimes, run_pool, confidence_ellipse, RESULTS_DIR)

# ============================ CONFIGURATION ============================
REFRESH_STEPS = int(24 * 7 * 3600 // I.LOAD['Ts'])   # rafraichissement estimation = 1 semaine
CLIP = (0.2, 1.10)                                   # bornes physiques du SoH estime
COMPONENTS = ('fc', 'ely')                           # composants dont le SoH est bruite

# --- Regime 1 : BIAIS systematique (deterministe) ---
BIAS_GRID = [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]

# --- Regime 2 : BRUIT gaussien (Monte Carlo) ---
# sigma=2% = valeur reviewer R3 (estimateur realiste) ; sigma=10% = stress.
# N_MC tirages par sigma. Cout total = 1(base) + len(BIAS_GRID) + len(SIGMA_LIST)*N_MC.
#   N_MC=20 -> ~50 sims (rapide) ; N_MC=50 -> ~110 sims (nuages/ellipses plus lisses).
SIGMA_LIST = [0.02, 0.10]
N_MC = 200            # run mesocentre : 200 pour nuages/ellipses lisses
MC_SEED0 = 1000

OUT_TXT = os.path.join(RESULTS_DIR, "sens_soh.txt")
# ======================================================================


def make_noisy_soh(bias_rel, sigma_rel, seed):
    """Enveloppe RB2(SoH) : injecte une erreur (biais + bruit) sur le SoH vu
    par le controleur. Estimation piecewise-constante. Construite DANS le
    worker (closure non picklable)."""
    rng = np.random.default_rng(seed)
    st = {'k': 0, 'fac_fc': 1.0, 'fac_ely': 1.0}

    def strat(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
              SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
              RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if st['k'] % REFRESH_STEPS == 0:
            n_fc  = rng.normal(0.0, sigma_rel) if sigma_rel > 0 else 0.0
            n_ely = rng.normal(0.0, sigma_rel) if sigma_rel > 0 else 0.0
            st['fac_fc']  = 1.0 + bias_rel + n_fc
            st['fac_ely'] = 1.0 + bias_rel + n_ely
        st['k'] += 1
        sfc  = min(max(SoH_fc_t  * st['fac_fc'],  CLIP[0]), CLIP[1]) if 'fc'  in COMPONENTS else SoH_fc_t
        sely = min(max(SoH_ely_t * st['fac_ely'], CLIP[0]), CLIP[1]) if 'ely' in COMPONENTS else SoH_ely_t
        return BASE_STRAT(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                          alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                          P_ely_max_t, RUL_fc_t, RUL_ely_t, sfc, sely)
    return strat


def evaluate(params):
    """Worker picklable. params = dict(bias, sigma, seed, tag)."""
    strat = make_noisy_soh(params['bias'], params['sigma'], params['seed'])
    data = init_and_run_loop(strat)
    lpsp, cost = metrics(data)
    clps = lps_cost_keur(data)
    lb, lf, le = lifetimes(data)
    return dict(params=params, bias=params['bias'], sigma=params['sigma'],
                lpsp=lpsp, cost=cost, clps=clps, life_bat=lb, life_fc=lf, life_ely=le)


def _fmt(r):
    return ("bias=%+.3f sigma=%.3f -> LPSP %7.4f%%  deg %7.2f kEUR"
            % (r['bias'], r['sigma'], r['lpsp'], r['cost']))


def main():
    print("=== ETAPE 1 -- Sensibilite a l'erreur d'estimation du SoH (RB2(SoH), 25 ans) ===", flush=True)
    print("    refresh=%d pas (~%.0f j) | clip=%s | composants=%s"
          % (REFRESH_STEPS, REFRESH_STEPS * I.LOAD['Ts'] / 3600 / 24, CLIP, COMPONENTS), flush=True)

    base = evaluate(dict(bias=0.0, sigma=0.0, seed=0, tag='base'))
    print("\nBASELINE (SoH exact) : LPSP %.4f%%  deg %.2f kEUR  | vie B/F/E = %s/%s/%s"
          % (base['lpsp'], base['cost'], base['life_bat'], base['life_fc'], base['life_ely']), flush=True)

    p_bias = [dict(bias=b, sigma=0.0, seed=0, tag='bias') for b in BIAS_GRID]
    r_bias = run_pool(evaluate, p_bias, "REGIME 1 -- biais systematique", _fmt)

    p_mc = [dict(bias=0.0, sigma=sg, seed=MC_SEED0 + k, tag='mc%.3f' % sg)
            for sg in SIGMA_LIST for k in range(N_MC)]
    r_mc = run_pool(evaluate, p_mc, "REGIME 2 -- bruit gaussien (Monte Carlo)", _fmt)

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite estimation SoH -- RB2(SoH) 25 ans | refresh=%dpas clip=%s comp=%s\n"
                % (REFRESH_STEPS, CLIP, COMPONENTS))
        f.write("BASELINE; LPSP=%.4f%%; deg=%.3fkEUR; clps=%.3fkEUR; vie_bat=%s; vie_fc=%s; vie_ely=%s\n\n"
                % (base['lpsp'], base['cost'], base['clps'],
                   base['life_bat'], base['life_fc'], base['life_ely']))
        f.write("## Regime 1 : biais systematique\n")
        f.write("bias;LPSP_%;deg_kEUR;dLPSP_pts;ddeg_%;clps_kEUR\n")
        for r in r_bias:
            f.write("%+.4f;%.4f;%.3f;%+.4f;%+.2f;%.3f\n"
                    % (r['bias'], r['lpsp'], r['cost'], r['lpsp'] - base['lpsp'],
                       (r['cost'] - base['cost']) / base['cost'] * 100, r['clps']))
        f.write("\n## Regime 2 : bruit gaussien (Monte Carlo)\n")
        f.write("sigma;N;LPSP_mean;LPSP_std;LPSP_min;LPSP_max;deg_mean;deg_std;deg_min;deg_max;clps_mean\n")
        for sg in SIGMA_LIST:
            sub = [r for r in r_mc if abs(r['sigma'] - sg) < 1e-12]
            lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
            cl = np.array([r['clps'] for r in sub])
            f.write("%.3f;%d;%.4f;%.4f;%.4f;%.4f;%.3f;%.3f;%.3f;%.3f;%.3f\n"
                    % (sg, len(sub), lp.mean(), lp.std(), lp.min(), lp.max(),
                       dg.mean(), dg.std(), dg.min(), dg.max(), cl.mean()))

    # ===================== FIGURE 1 : biais (double axe) =====================
    b = np.array([r['bias'] for r in r_bias]) * 100
    lp = np.array([r['lpsp'] for r in r_bias]); dg = np.array([r['cost'] for r in r_bias])
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    c1, c2 = 'tab:blue', 'tab:red'
    ax1.plot(b, lp, 'o-', color=c1, label='LPSP')
    ax1.axhline(base['lpsp'], color=c1, ls=':', alpha=0.6)
    ax1.set_xlabel("Biais d'estimation du SoH [%]"); ax1.set_ylabel("LPSP [%]", color=c1)
    ax1.tick_params(axis='y', labelcolor=c1); ax1.grid(True, ls='--', alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(b, dg, 's-', color=c2, label='Cout degradation')
    ax2.axhline(base['cost'], color=c2, ls=':', alpha=0.6)
    ax2.set_ylabel("Cout de degradation [kEUR]", color=c2); ax2.tick_params(axis='y', labelcolor=c2)
    ax1.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax1.set_title("Biais d'estimation du SoH", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "sens_soh_bias.pdf"),
                                    bbox_inches="tight"); plt.close()

    # ===================== FIGURE 2 : plan de Pareto + intervalles de confiance =====================
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lpb = np.array([r['lpsp'] for r in r_bias]); dgb = np.array([r['cost'] for r in r_bias])
    ax.plot(lpb, dgb, '-', color='0.6', lw=1.2, zorder=1)
    sc = ax.scatter(lpb, dgb, c=np.array([r['bias'] for r in r_bias]) * 100,
                    cmap='coolwarm', s=70, zorder=3, edgecolor='k', linewidth=0.4)
    cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Biais d'estimation du SoH [%]")
    for r in (r_bias[0], r_bias[-1]):
        ax.annotate("%+.0f%%" % (r['bias'] * 100), (r['lpsp'], r['cost']),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    mc_colors = ['tab:green', 'tab:orange', 'tab:purple']
    for sg, col in zip(SIGMA_LIST, mc_colors):
        sub = [r for r in r_mc if abs(r['sigma'] - sg) < 1e-12]
        x = np.array([r['lpsp'] for r in sub]); y = np.array([r['cost'] for r in sub])
        ax.scatter(x, y, s=14, color=col, alpha=0.35, zorder=2)
        confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor='none', lw=1.8, zorder=4)
        confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor='none', lw=1.0,
                           ls='--', alpha=0.7, zorder=4)
        ax.scatter([x.mean()], [y.mean()], marker='D', s=55, color=col,
                   edgecolor='k', linewidth=0.5, zorder=5,
                   label="bruit sigma=%.0f%% (N=%d)" % (sg * 100, len(sub)))

    ax.scatter([base['lpsp']], [base['cost']], marker='*', s=340, color='red',
               edgecolor='k', linewidth=0.6, zorder=6, label='baseline (SoH exact)')

    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Sensibilite a l'estimation du SoH", fontsize=11)
    ax.grid(True, ls='--', alpha=0.5); ax.legend(loc='best', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "sens_soh_pareto.pdf"),
                                    bbox_inches="tight"); plt.close()

    # ===================== RESUME CONSOLE (ASCII) =====================
    print("\n" + "=" * 72)
    print("BASELINE : LPSP %.4f%%  deg %.2f kEUR" % (base['lpsp'], base['cost']))
    print("-" * 72)
    print("Biais  ->  LPSP        deg")
    for r in r_bias:
        print("  %+.3f : %8.4f%%  %8.2f kEUR  (dLPSP %+.4f pts, ddeg %+.2f%%)"
              % (r['bias'], r['lpsp'], r['cost'], r['lpsp'] - base['lpsp'],
                 (r['cost'] - base['cost']) / base['cost'] * 100))
    for sg in SIGMA_LIST:
        sub = [r for r in r_mc if abs(r['sigma'] - sg) < 1e-12]
        lp = np.array([r['lpsp'] for r in sub]); dg = np.array([r['cost'] for r in sub])
        print("  MC sigma=%.3f (N=%d): LPSP %.4f+/-%.4f  deg %.2f+/-%.2f kEUR"
              % (sg, len(sub), lp.mean(), lp.std(), dg.mean(), dg.std()))
    print("=" * 72)
    print("Resultats : %s" % OUT_TXT)
    print("Figures   : %s" % os.path.join(RESULTS_DIR, "sens_soh_bias.pdf"))
    print("            %s" % os.path.join(RESULTS_DIR, "sens_soh_pareto.pdf"))


if __name__ == "__main__":
    main()
