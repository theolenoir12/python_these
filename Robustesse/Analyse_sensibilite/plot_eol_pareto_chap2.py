"""plot_eol_pareto_chap2.py -- front de Pareto sous echantillonnage EoL, CHAPITRE 2.
Derive de plot_eol_pareto_en.py : relit results_meso/sens_eol.txt (aucune
simulation), EXCLUT RB2(SoH) (reportee au chapitre suivant), labels FR.
Ellipses alignees sur les axes (le .txt ne stocke que moyenne/ecart-type).
Couleurs identiques a la figure Pareto principale du chapitre. Sortie -> manuscrit.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys
from matplotlib.patches import Ellipse


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(HERE, "results_meso", "sens_eol.txt")
OUT = ("/home/theo/Documents/Doctorat/GENIAL/LaTeX/Manuscrit_post_chap1_v1/"
       "Chapitre 2/Sensibilite/Figures/sens_eol_pareto.pdf")

# Mapping couleur sur les 10 strategies (RB2(SoH) incluse) -> couleurs stables,
# identiques a la figure Pareto principale ; mais on ne TRACE que les 9 statiques.
COLOR_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(COLOR_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
STRAT_PLOT = [s for s in COLOR_ORDER if s != 'RB2(SoH)']
IDEAL_COLOR = '0.3'

LABEL_PLACEMENT = {
    'RB2':   dict(dx=-2.7, dy=+4.0, ha='left', va='top'),
    'Ideal': dict(dx=+0.5, dy=+0.5, ha='left', va='bottom'),
}
DEFAULT_PLACEMENT = dict(dx=+0.5, dy=+0.5, ha='left', va='bottom')


def parse_front(path):
    rows, in_block = {}, False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Front"):
                in_block = True
                continue
            if in_block:
                if s.startswith("## OAT"):
                    break
                if not s or s.startswith("strat;") or s.startswith("#"):
                    continue
                p = s.split(";")
                if len(p) < 7 or p[0] not in COLOR_ORDER:
                    continue
                rows[p[0]] = dict(lpsp_nom=float(p[1]), deg_nom=float(p[2]),
                                  lpsp_mean=float(p[3]), lpsp_std=float(p[4]),
                                  deg_mean=float(p[5]), deg_std=float(p[6]))
    return rows


def draw(ax, rows, strats, zoom=False):
    rng = np.random.default_rng(0)
    for strat in strats:
        if strat not in rows:
            continue
        r = rows[strat]
        col = darken(COLORS[strat], 0.7)
        if r['lpsp_std'] > 0 or r['deg_std'] > 0:
            cx = rng.normal(r['lpsp_mean'], r['lpsp_std'], 40)
            cy = rng.normal(r['deg_mean'], r['deg_std'], 40)
            ax.scatter(cx, cy, s=6, color=col, alpha=0.4, edgecolor='none', zorder=2)
        for n_std, lw, ls, al in ((1.0, 1.6, '-', 0.9), (2.0, 0.9, '--', 0.5)):
            ax.add_patch(Ellipse((r['lpsp_mean'], r['deg_mean']),
                                 2 * n_std * r['lpsp_std'], 2 * n_std * r['deg_std'],
                                 edgecolor=col, facecolor='none', lw=lw, ls=ls,
                                 alpha=al, zorder=4))
        ax.scatter([r['lpsp_nom']], [r['deg_nom']], s=(70 if zoom else 60),
                   color=col, alpha=0.9, zorder=6)


def main():
    rows = parse_front(TXT)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw(ax, rows, STRAT_PLOT)
    for strat in STRAT_PLOT:
        if strat not in rows:
            continue
        r = rows[strat]
        col = darken(COLORS[strat], 0.7)
        p = LABEL_PLACEMENT.get(strat, DEFAULT_PLACEMENT)
        ax.text(r['lpsp_nom'] + p['dx'], r['deg_nom'] + p['dy'], strat, fontsize=14,
                color=col, weight='bold', path_effects=LABEL_STROKE,
                ha=p['ha'], va=p['va'], zorder=7)
    ic = darken(IDEAL_COLOR, 0.7)
    ax.scatter([0], [0], s=60, color=ic, alpha=0.9, zorder=6)
    ax.text(0.5, 0.5, 'Ideal', fontsize=14, color=ic, weight='bold',
            path_effects=LABEL_STROKE, ha='left', va='bottom', zorder=7)

    # encart zoom bas-gauche
    zoom_strats = [s for s in ['75-25', '100-0', 'RB2', 'RB1'] if s in rows]
    axins = ax.inset_axes([0.42, 0.06, 0.5, 0.42])
    draw(axins, rows, zoom_strats, zoom=True)
    zoff = {'RB1': (0.10, 0.0, 'left', 'center'),
            '100-0': (0.10, 1.3, 'left', 'bottom'),
            'RB2': (-0.10, -0.8, 'right', 'top'),
            '75-25': (0.10, 0.0, 'left', 'center')}
    xs, ys = [], []
    for strat in zoom_strats:
        r = rows[strat]
        col = darken(COLORS[strat], 0.7)
        dx, dy, ha, va = zoff[strat]
        axins.text(r['lpsp_nom'] + dx, r['deg_nom'] + dy, strat, fontsize=11,
                   color=col, weight='bold', path_effects=LABEL_STROKE,
                   ha=ha, va=va, zorder=7)
        xs += [r['lpsp_nom'], r['lpsp_mean'] - 2 * r['lpsp_std'], r['lpsp_mean'] + 2 * r['lpsp_std']]
        ys += [r['deg_nom'], r['deg_mean'] - 2 * r['deg_std'], r['deg_mean'] + 2 * r['deg_std']]
    mx, my = 0.10 * (max(xs) - min(xs) or 1), 0.10 * (max(ys) - min(ys) or 1)
    axins.set_xlim(min(xs) - mx, max(xs) + mx)
    axins.set_ylim(min(ys) - my, max(ys) + my)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close()
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
