"""Pareto 25 ans -- PREDICTIONS -- EMS de base + RB2(Pred) (SANS vieillissement).

RB2(Pred) = RB2 (socle cost-min 0.440/0.310) + pre-charge batterie previsionnelle,
RE-OPTIMISEE sur ce socle (reopt_pred.py). Point = moyenne Monte-Carlo de la
version bruitee+hysteresis (seule legitime, cf README). Comparaison honnete :
la baseline est RB2 a son socle cost-min (80.11 k€), donc le gain previsionnel
affiche est celui d'une pre-charge sur un RB2 DEJA optimise.

Figure SANS les strategies integrant le vieillissement (RB2(SoH)/RB2(RUL)) --
volontaire (1er temps). Deux versions :
  - pareto_ems_pred.pdf/.png          : le Pareto (LPSP, deg).
  - pareto_ems_pred_isocost.pdf/.png  : + lignes d'ISO-COUT unifie.
Ne remplace aucun fichier existant."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Analyse_sensibilite")))
try:
    import voll_common as V
    ISO_SLOPE = V.cost_lpsp_keur(1.0)
except Exception:
    ISO_SLOPE = 8.2014

# --- Point RB2(Pred) (rempli depuis reopt_pred.py, moyenne MC bruite+hyst) ---
RB2PRED = (2.5301, 58.9666)   # (LPSP %, deg k€)


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

points = np.array([
    [10.3855, 124.1937],  # 0-100
    [20.2667, 110.7658],  # 25-75
    [8.0744, 109.0235],   # 50-50
    [3.8032, 59.6765],    # 75-25
    [2.4851, 66.4122],    # 100-0
    [2.5920, 58.8499],    # RB2  (cost-min 0.440/0.310)
    [2.5301, 58.9666],        # RB2(Pred)  (re-optimise, moyenne MC)
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000]      # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(Pred)',
          'RB1', 'SoC1', 'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
EXTRA_COLORS = {'RB2(Pred)': '#117733'}     # vert dedie, distinct du cluster
IDEAL_COLOR = '0.3'


def color_of(label):
    if label in EXTRA_COLORS:
        return darken(EXTRA_COLORS[label], 0.9)
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def draw_isocost(axis, C_levels, xlim, ylim, label_y=None, inset=False):
    axis.set_xlim(xlim); axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 100)
    for C in C_levels:
        axis.plot(xs, C - ISO_SLOPE * xs, ls=':', color='0.6', lw=0.9, zorder=0)
        if label_y is not None:
            xl = (C - label_y) / ISO_SLOPE
            if xlim[0] < xl < xlim[1]:
                axis.text(xl, label_y, f"{C:.0f}", fontsize=7.5 if not inset else 6.5,
                          color='0.4', ha='center', va='center', zorder=1,
                          path_effects=LABEL_STROKE)


def build_figure(iso_cost):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)
    for i, label in enumerate(labels):
        col = color_of(label)
        if label == 'RB2':
            ax.text(points[i, 0] + 0.3, points[i, 1] + 0.6, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, va='bottom', ha='left')
        elif label == 'SoC06':
            ax.text(points[i, 0] - 2, points[i, 1] - 3, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='top')
        elif label == 'RB2(Pred)':
            ax.text(points[i, 0] - 0.3, points[i, 1] - 1.0, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, va='top', ha='right')
        else:
            ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)

    zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(Pred)', 'RB1']
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    for i, label in enumerate(labels):
        if label in zoom_labels:
            axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=70, alpha=0.9)
    zoom_offsets = {
        'RB1':       (0.10, 0.0,  'left',   'center'),
        '100-0':     (0.10, 1.0,  'left',   'bottom'),
        'RB2':       (0.12, 0.2,  'left',   'bottom'),
        'RB2(Pred)': (-0.10, -0.5, 'right', 'top'),
        '75-25':     (0.10, 0.0,  'left',   'center'),
    }
    for i, label in enumerate(labels):
        if label in zoom_labels:
            dx, dy, ha, va = zoom_offsets[label]
            axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                       color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                       horizontalalignment=ha, verticalalignment=va)
    axins.set_xlim(1.0, 4.5)
    axins.set_ylim(55, 84)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)

    if iso_cost:
        main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
        draw_isocost(ax, list(range(70, 141, 10)), main_xlim, main_ylim, label_y=13)
        draw_isocost(axins, [75, 80, 85, 90, 95, 100], (1.0, 4.5), (55, 84))

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    stem = "pareto_ems_pred_isocost" if iso_cost else "pareto_ems_pred"
    plt.savefig(stem + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(stem + ".png", format='png', dpi=150, bbox_inches='tight')
    print("saved ->", stem + ".pdf / .png")


build_figure(iso_cost=False)
build_figure(iso_cost=True)
