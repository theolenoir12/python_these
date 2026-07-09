# -*- coding: utf-8 -*-
"""make_mpc_pareto.py -- figures d'ESPACE DES OBJECTIFS avec les points MPC(LP),
au STYLE EXACT de make_chap3_pareto.py (palette tab10, front PD magenta #c2185b,
iso-couts pointilles pente 8.2014, figsize (10,6), encart zoom).
=================================================================================
N'ECRASE AUCUNE figure existante : ecrit des fichiers NEUFS dans figures_chap3/ :
    pareto_strategies_mpc      synthese complete (toutes RB + MPC + MPC omni + PD)
    pareto_mpc_ceiling         focus "deployable vs potentiel" (RB2, ultime RB,
                               MPC deployable, MPC omni, front PD)

Reutilise le style et les points RB de make_chap3_pareto (import, sans lancer son
main). Les points MPC sont LUS des resultats reels de ../MPC3/ (re-import 2026-07-09) :
    MPC (deployable)  = 'sw x3' du sweep_mpc_robust.txt (defaut retenu, plan C)
    MPC omni          = 'MPC omni (H=48)' de bench_mpc.txt (plafond informationnel)
Coordonnees = (LPSP %, cout de degradation kEUR), meme repere que les RB.

Aucune simulation : executable directement (F5 / python make_mpc_pareto.py).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style + points RB + helpers, importes tels quels (make_chap3 ne lance rien a
# l'import ; son main() est protege). rcParams/palette sont appliques a l'import.
from make_chap3_pareto import (
    BASE, EXTRA, L_SOH, L_PRED, L_SHP, color_of, dp_front, draw_isocost,
    DP_COLOR, LABEL_STROKE, FIGSIZE, HERE, OUTDIR,
)

# --- Couleurs dediees MPC (distinctes de la palette RB et du magenta PD) ------
MPC_COLORS = {
    'MPC':      '#cc3311',   # rouge brique : MPC deployable (sw x3, bruite)
    'MPC omni': '#7b3294',   # violet       : MPC a prevision parfaite (plafond)
}


def _parse_bench(path):
    """{label: (LPSP%, deg kEUR)} depuis un tableau bench_mpc/sweep (';' separe)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("label"):
                continue
            p = [x.strip() for x in line.split(";")]
            if len(p) < 6:
                continue
            out[p[0]] = (float(p[2]), float(p[4]))    # LPSP%, deg kEUR
    return out


def mpc_points():
    """Points MPC reels lus de ../MPC/ ; fallback sur les valeurs consignees."""
    mpcdir = os.path.abspath(os.path.join(HERE, "..", "MPC3"))
    pts = {}
    try:
        rob = _parse_bench(os.path.join(mpcdir, "sweep_mpc_robust.txt"))
        cand = [v for k, v in rob.items() if k.startswith("sw x3")]
        if cand:
            pts['MPC'] = cand[0]
    except OSError:
        pass
    try:
        ben = _parse_bench(os.path.join(mpcdir, "bench_mpc.txt"))
        if "MPC omni (H=48)" in ben:
            pts['MPC omni'] = ben["MPC omni (H=48)"]
    except OSError:
        pass
    pts.setdefault('MPC',      (1.41, 66.29))   # sw x3   (MPC3/sweep_mpc_robust)
    pts.setdefault('MPC omni', (0.46, 55.65))   # omni H48 (MPC3/bench_mpc.txt)
    return pts


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name + ".pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), format="png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved ->", os.path.relpath(p, HERE))


def _scatter_label(ax, x, y, lab, col, dx=0.5, dy=0.5, ha="left", va="bottom",
                   s=60, fs=14):
    ax.scatter(x, y, color=col, s=s, alpha=0.95, zorder=4,
               edgecolors="white", linewidths=0.6)
    ax.text(x + dx, y + dy, lab, fontsize=fs, color=col, weight="bold",
            path_effects=LABEL_STROKE, ha=ha, va=va, zorder=5)


def build_strategies_mpc():
    """Synthese complete (comme pareto_strategies_v2) + points MPC deployable et
    omni. Zoom ELARGI a gauche pour capter les MPC (LPSP bas) et l'amas RB2."""
    pts = dict(BASE)
    for lab in (L_SOH, L_PRED, L_SHP):
        pts[lab] = EXTRA[lab]
    mpc = mpc_points()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    fl, fd = dp_front()
    ax.plot(fl, fd, "-", color=DP_COLOR, lw=1.6, zorder=2)
    ax.scatter(fl, fd, color=DP_COLOR, s=18, zorder=2)

    for lab, (x, y) in pts.items():
        ax.scatter(x, y, color=color_of(lab), s=60, alpha=0.9, zorder=3)
    for lab, (x, y) in pts.items():
        if lab in ('RB2', '75-25') or lab in EXTRA:
            continue                       # etiquetes dans le zoom (amas dense)
        col = color_of(lab)
        if lab == 'SoC06':
            ax.text(x - 2, y - 3, lab, fontsize=14, color=col, weight='bold',
                    path_effects=LABEL_STROKE, va='top')
        else:
            ax.text(x + 0.5, y + 0.5, lab, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)
    # MPC sur le plan principal (etiquetes aussi dans le zoom -> ici sans texte)
    for lab in ('MPC', 'MPC omni'):
        ax.scatter(*mpc[lab], color=MPC_COLORS[lab], s=95, marker='*',
                   zorder=4, edgecolors='white', linewidths=0.7)

    # --- encart zoom ELARGI : capte MPC (LPSP 0.5) + amas RB2 ---
    zx, zy = (0.2, 4.3), (50, 70)
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    axins.plot(fl, fd, "-", color=DP_COLOR, lw=1.6, zorder=2)
    axins.scatter(fl, fd, color=DP_COLOR, s=16, zorder=2)
    zoom_rb = [l for l in (['75-25', '100-0', 'RB2', L_SOH, L_PRED, L_SHP])
               if zx[0] < pts[l][0] < zx[1] and zy[0] < pts[l][1] < zy[1]]
    off = {  # (dx, dy, ha, va)
        '100-0': (0.06, 0.8, 'left', 'bottom'), 'RB2': (0.08, -0.4, 'left', 'top'),
        '75-25': (0.08, 0.0, 'left', 'center'), L_SOH: (0.08, 0.0, 'left', 'center'),
        L_PRED: (0.0, 0.6, 'center', 'bottom'), L_SHP: (-0.10, -0.3, 'right', 'top'),
    }
    for lab in zoom_rb:
        axins.scatter(*pts[lab], color=color_of(lab), s=70, alpha=0.9, zorder=3)
        dx, dy, ha, va = off[lab]
        axins.text(pts[lab][0] + dx, pts[lab][1] + dy, lab, fontsize=11,
                   color=color_of(lab), weight='bold', path_effects=LABEL_STROKE,
                   ha=ha, va=va)
    mpc_off = {'MPC': (0.08, 0.4, 'left', 'bottom'),
               'MPC omni': (0.08, -0.5, 'left', 'top')}
    for lab in ('MPC', 'MPC omni'):
        axins.scatter(*mpc[lab], color=MPC_COLORS[lab], s=130, marker='*',
                      zorder=4, edgecolors='white', linewidths=0.7)
        dx, dy, ha, va = mpc_off[lab]
        axins.text(mpc[lab][0] + dx, mpc[lab][1] + dy, lab, fontsize=11,
                   color=MPC_COLORS[lab], weight='bold', path_effects=LABEL_STROKE,
                   ha=ha, va=va)
    axins.set_xlim(*zx); axins.set_ylim(*zy)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)
    draw_isocost(ax, list(range(70, 141, 10)), ax.get_xlim(), ax.get_ylim())
    draw_isocost(axins, [55, 60, 65, 70, 75], zx, zy)

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    _save(fig, "pareto_strategies_mpc")


def build_mpc_ceiling():
    """Figure focalisee : deployable (MPC) vs potentiel (MPC omni), face aux
    reperes RB2 socle / ultime RB et au front PD. Sans amas complet."""
    mpc = mpc_points()
    ref = {'RB2': BASE['RB2'], L_SHP: EXTRA[L_SHP]}
    fig, ax = plt.subplots(figsize=FIGSIZE)

    fl, fd = dp_front()
    ax.plot(fl, fd, "-", color=DP_COLOR, lw=1.6, zorder=2, label="Front PD (borne offline)")
    ax.scatter(fl, fd, color=DP_COLOR, s=20, zorder=2)

    _scatter_label(ax, *ref['RB2'], 'RB2', color_of('RB2'), dx=0.06, dy=1.0)
    _scatter_label(ax, *ref[L_SHP], L_SHP, color_of(L_SHP), dx=0.06, dy=-1.4, va='top')
    _scatter_label(ax, *mpc['MPC'], 'MPC (déployable)', MPC_COLORS['MPC'],
                   dx=0.08, dy=0.8, s=150, fs=15)
    _scatter_label(ax, *mpc['MPC omni'], 'MPC omni (potentiel)', MPC_COLORS['MPC omni'],
                   dx=0.08, dy=-1.6, va='top', s=150, fs=15)
    # fleche potentiel -> deployable (l'ecart = fragilite a la prevision)
    ax.annotate("", xy=mpc['MPC'], xytext=mpc['MPC omni'],
                arrowprops=dict(arrowstyle="->", color='0.4', lw=1.4, ls='--'), zorder=3)
    xm = 0.5 * (mpc['MPC'][0] + mpc['MPC omni'][0])
    ym = 0.5 * (mpc['MPC'][1] + mpc['MPC omni'][1])
    ax.text(xm - 0.1, ym, "fragilité\nprévision", fontsize=11, color='0.35',
            ha='right', va='center', path_effects=LABEL_STROKE)

    draw_isocost(ax, list(range(45, 86, 5)), (0.0, 3.5), (44, 72))
    ax.set_xlim(0.0, 3.5); ax.set_ylim(44, 72)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    _save(fig, "pareto_mpc_ceiling")


def main():
    build_strategies_mpc()
    build_mpc_ceiling()


if __name__ == "__main__":
    main()
