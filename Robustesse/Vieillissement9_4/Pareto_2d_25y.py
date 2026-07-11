"""Pareto 25 ans -- VERSION 2 avec le NOUVEAU RB2(SoH) issu du balayage
d'attribution exhaustif (sweep_soh_attribution.py).

Nouveau RB2(SoH) = meilleure augmentation 100% attribuable au vieillissement :
    P_fc_set  = 0.440 * Pmax_fc  * SoH_fc  ^ 1.0
    P_ely_set = 0.310 * Pmax_ely * SoH_ely ^ 2.0
-> (LPSP 2.9089 %, deg 54.9115 k€), cout unifie 78.77 k€ (VoLL=3).
   Gain pur attribuable au SoH = +1.34 k€ (+1.67 %) vs la meilleure constante.

Comparaison HONNETE (best-vs-best) : RB2 ET RB2(SoH) sont pris a leurs setpoints
qui MINIMISENT le cout unifie (pas le RB2 nominal non optimise) :
    RB2      : c_fc=0.440, c_ely=0.310 (constant, sans SoH)
               -> (LPSP 2.5920 %, deg 58.8499 k€), unifie 80.11 k€
    RB2(SoH) : 0.440*SoH_fc^1 / 0.310*SoH_ely^2
               -> (LPSP 2.9089 %, deg 54.9115 k€), unifie 78.77 k€
Le gain PUR attribuable au SoH est donc l'ecart 80.11 -> 78.77 = 1.34 k€ (1.67 %).

Ce script produit DEUX figures :
  - pareto_ems_soh2.pdf/.png          : le Pareto (LPSP, deg).
  - pareto_ems_soh2_isocost.pdf/.png  : le MEME, + les lignes d'ISO-COUT unifie.
Les iso-couts sont des droites  deg = C - m*LPSP  avec m = cout d'un point de
LPSP = VoLL * E_REF/1e5 (voll_common.py, VoLL=3) ; tout point sur une meme droite
a le meme cout total unifie. Plus bas-gauche = moins cher.

NE REMPLACE PAS les fichiers existants (Pareto_2d_25y.py / pareto_ems.pdf). Les
autres EMS gardent leurs valeurs de reference (metrique identique au sweep : RB2
plain concorde a 1e-3). Seul le point RB2(SoH) est mis a jour."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

# --- Pente d'iso-cout : cout d'un point de LPSP [k€], via voll_common (VoLL=3) ---
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Analyse_sensibilite")))
try:
    import voll_common as V
    ISO_SLOPE = V.cost_lpsp_keur(1.0)          # ~8.2014 k€ par point de LPSP
except Exception:
    ISO_SLOPE = 8.2014                          # repli si voll_common indisponible


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})

# Points (LPSP %, deg k€). RB2(SoH) = NOUVELLE version attribuable (cf entete).
points = np.array([
    [10.3855, 124.1937],  # 0-100
    [20.2667, 110.7658],  # 25-75
    [8.0744, 109.0235],   # 50-50
    [3.8032, 59.6765],    # 75-25
    [2.4851, 66.4122],    # 100-0
    [2.5920, 58.8499],    # RB2  (setpoints qui MINIMISENT le cout : 0.440/0.310)
    [2.9089, 54.9115],    # RB2(SoH) v2  (0.440*SoH_fc^1 / 0.310*SoH_ely^2)
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000]      # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)',
          'RB1', 'SoC1', 'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'


def color_of(label):
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def draw_isocost(axis, C_levels, xlim, ylim, label_y=None, inset=False):
    """Trace les droites deg = C - ISO_SLOPE*LPSP dans les limites donnees.
    label_y : si non None, ecrit la valeur C la ou la droite passe a cette
    ordonnee (place dans une bande vide)."""
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 100)
    for C in C_levels:
        axis.plot(xs, C - ISO_SLOPE * xs, ls=':', color='0.6',
                  lw=0.9, zorder=0)
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
            ax.text(points[i, 0] - 2.7, points[i, 1] + 4, label,
                    fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                    verticalalignment='top')
        elif label == 'SoC06':
            ax.text(points[i, 0] - 2, points[i, 1] - 3, label,
                    fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                    verticalalignment='top')
        elif label == 'RB2(SoH)':
            ax.text(points[i, 0] + 0.35, points[i, 1] - 1.0, label,
                    fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                    verticalalignment='top', horizontalalignment='left')
        else:
            ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label,
                    fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE)

    # --- Encadre zoom sur le cluster bas-gauche ---
    zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB1']
    axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])
    for i, label in enumerate(labels):
        if label in zoom_labels:
            axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=70, alpha=0.9)

    zoom_offsets = {
        'RB1':      (0.10, 0.0,  'left',   'center'),
        '100-0':    (0.10, 1.3,  'left',   'bottom'),
        'RB2':      (-0.10, -0.8, 'right', 'top'),
        'RB2(SoH)': (0.12, 1.4,  'left',   'bottom'),
        '75-25':    (0.10, 0.0,  'left',   'center'),
    }
    for i, label in enumerate(labels):
        if label in zoom_labels:
            dx, dy, ha, va = zoom_offsets[label]
            axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                       color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                       horizontalalignment=ha, verticalalignment=va)

    axins.set_xlim(1.0, 4.3)
    axins.set_ylim(53, 84)          # abaisse pour inclure RB2(SoH) v2 a 54.9
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)

    # --- Lignes d'iso-cout unifie (2e figure seulement) ---
    if iso_cost:
        main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
        draw_isocost(ax, list(range(70, 141, 10)), main_xlim, main_ylim, label_y=13)
        # dans l'encadre : plus fin, sans labels (place restreinte)
        draw_isocost(axins, [75, 80, 85, 90, 95, 100], (1.0, 4.3), (53, 84))

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout()
    stem = "pareto_ems_soh2_isocost" if iso_cost else "pareto_ems_soh2"
    plt.savefig(stem + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(stem + ".png", format='png', dpi=150, bbox_inches='tight')
    print("saved ->", stem + ".pdf / .png")
    return fig


build_figure(iso_cost=False)   # figure 1 : identique a la v2
build_figure(iso_cost=True)    # figure 2 : + lignes d'iso-cout
plt.show()
