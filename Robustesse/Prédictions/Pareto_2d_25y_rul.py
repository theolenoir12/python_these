"""Pareto 25 ans -- PREDICTIONS -- donnees ACTUALISEES + RB2(RUL).

Reprend le Pareto actualise (RB2 et RB2(SoH) a leurs setpoints qui MINIMISENT le
cout unifie, comparaison honnete best-vs-best) et ajoute JUSTE le RB2(RUL), lui
aussi a son point cost-min (sweep_rul_attribution.py, meme socle/metrique).
    RB2      : c_fc=0.440, c_ely=0.310 (constant)             -> (2.5920, 58.8499)  unif 80.11
    RB2(SoH) : 0.440*SoH_fc^1 / 0.310*SoH_ely^2               -> (2.9089, 54.9115)  unif 78.77
    RB2(RUL) : c_ely*min(RUL/1000,1)^exp (cost-min du sweep)   -> voir RB2RUL ci-dessous
(RB2(Pred)/RB2(SoH+Pred) volontairement EXCLUS : "juste le RUL en plus".)

Deux figures :
  - pareto_ems_rul.pdf/.png          : le Pareto (LPSP, deg).
  - pareto_ems_rul_isocost.pdf/.png  : le MEME + lignes d'ISO-COUT unifie
    (deg = C - m*LPSP, m = cout d'un point de LPSP via voll_common, VoLL=3).

Ne remplace PAS les fichiers existants (Pareto_2d_25y.py / pareto_ems.pdf)."""
import os
import sys
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

# --- Point RB2(RUL) cost-min (sweep_rul_attribution.py) ---
# Resultat : l'optimum de RB2(RUL) est EXP_ELY=0 -> aucune modulation RUL -> le
# point COINCIDE avec RB2 (0.440/0.310). Le RUL n'apporte rien une fois le socle
# optimise (toute modulation exp>0 degrade). On le represente par un ANNEAU sur RB2.
RB2RUL = (2.5920, 58.8499)     # (LPSP %, deg k€)  == RB2 (best-vs-best honnete)


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
    [2.5920, 58.8499],    # RB2       (cost-min 0.440/0.310)
    [2.9089, 54.9115],    # RB2(SoH)  (0.440*SoH_fc^1 / 0.310*SoH_ely^2)
    [2.5920, 58.8499],    # RB2(RUL)  (cost-min sweep_rul_attribution)
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000]      # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)',
          'RB2(RUL)', 'RB1', 'SoC1', 'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
EXTRA_COLORS = {'RB2(RUL)': '#0072B2', 'RB2': '#4AE81A'}  # RB2 ambre vif + RB2(RUL) bleu
IDEAL_COLOR = '0.3'


def color_of(label):
    if label in EXTRA_COLORS:
        return darken(EXTRA_COLORS[label], 0.95)
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
        if label == 'RB2(RUL)':   # coincide avec RB2 -> anneau creux
            ax.scatter(points[i, 0], points[i, 1], facecolors='none',
                       edgecolors=color_of(label), s=140, linewidths=2.3, zorder=4)
        else:
            ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)

    for i, label in enumerate(labels):
        col = color_of(label)
        if label == 'RB2':
            ax.text(points[i, 0] - 2.7, points[i, 1] + 0, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='top')
        elif label == 'SoC06':
            ax.text(points[i, 0] - 2, points[i, 1] - 3, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='top')
        elif label == 'RB2(SoH)':
            ax.text(points[i, 0] + 0.35, points[i, 1] - 1.0, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='top',
                    horizontalalignment='left')
        elif label == 'RB2(RUL)':
            ax.text(points[i, 0] - 0.35, points[i, 1] + 0.8, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='bottom',
                    horizontalalignment='right')
        else:
            ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)

    zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB2(RUL)', 'RB1']
    axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])
    for i, label in enumerate(labels):
        if label in zoom_labels:
            if label == 'RB2(RUL)':   # coincide avec RB2 -> anneau creux
                axins.scatter(points[i, 0]+0.01, points[i, 1]-0.1, facecolors='none',
                              edgecolors=color_of(label), s=140, linewidths=2.4, zorder=4)
            else:
                axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.95)
    zoom_offsets = {
        'RB1':      (0.10, 0.0,  'left',   'center'),
        '100-0':    (0.10, 1.3,  'left',   'bottom'),
        'RB2':      (-0.15, -1.1, 'right', 'top'),
        'RB2(SoH)': (0.12, 0.6, 'left',  'top'),
        'RB2(RUL)': (-0.15, 1.1, 'right', 'bottom'),
        '75-25':    (0.10, 0.0,  'left',   'center'),
    }
    for i, label in enumerate(labels):
        if label in zoom_labels:
            dx, dy, ha, va = zoom_offsets[label]
            axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                       color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                       horizontalalignment=ha, verticalalignment=va)
    axins.set_xlim(1.0, 4.5)
    axins.set_ylim(53, 84)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)

    if iso_cost:
        main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
        draw_isocost(ax, list(range(70, 141, 10)), main_xlim, main_ylim, label_y=13)
        draw_isocost(axins, [75, 80, 85, 90, 95, 100], (1.0, 4.5), (53, 84))

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    stem = "pareto_ems_rul_isocost" if iso_cost else "pareto_ems_rul"
    plt.savefig(stem + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(stem + ".png", format='png', dpi=150, bbox_inches='tight')
    print("saved ->", stem + ".pdf / .png")


build_figure(iso_cost=False)
build_figure(iso_cost=True)
plt.show()
