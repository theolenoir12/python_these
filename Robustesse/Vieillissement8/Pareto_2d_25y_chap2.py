"""Pareto 25 ans pour le CHAPITRE 2 (manuscrit) : 9 EMS statiques, SANS RB2(SoH)
(qui est introduite au chapitre 3). Derive de Pareto_2d_25y.py (valeurs codees
en dur, identiques a batch_results_summary_25y.txt). Sauvegarde directement dans
le dossier figures du chapitre 2 du manuscrit."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys


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

# 9 EMS statiques + Ideal (RB2(SoH) volontairement exclue -> chapitre 3)
points = np.array([
    [10.3855, 124.1937],  # 0-100
    [20.2667, 110.7658],  # 25-75
    [8.0744, 109.0235],   # 50-50
    [3.8032, 59.6765],    # 75-25
    [2.4851, 66.4122],    # 100-0
    [2.4540, 65.4218],    # RB2
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000],     # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2',
          'RB1', 'SoC1', 'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'


def color_of(label):
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


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
    else:
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE)

# Zoom cluster bas-gauche
zoom_labels = ['75-25', '100-0', 'RB2', 'RB1']
axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])
for i, label in enumerate(labels):
    if label in zoom_labels:
        axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=70, alpha=0.9)
zoom_offsets = {
    'RB1':   (0.10, 0.0,  'left',   'center'),
    '100-0': (0.10, 1.3,  'left',   'bottom'),
    'RB2':   (-0.10, -0.8, 'right', 'top'),
    '75-25': (0.10, 0.0,  'left',   'center'),
}
for i, label in enumerate(labels):
    if label in zoom_labels:
        dx, dy, ha, va = zoom_offsets[label]
        axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                   color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                   horizontalalignment=ha, verticalalignment=va)
axins.set_xlim(1.0, 4.3)
axins.set_ylim(55, 84)
axins.grid(True, linestyle='--', alpha=0.5)
axins.tick_params(labelsize=10)
ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
OUT = ("/home/theo/Documents/Doctorat/GENIAL/LaTeX/Manuscrit_post_chap1_v1/"
       "Chapitre 2/EMS/Figures/pareto_ems.pdf")
plt.savefig(OUT, format='pdf', bbox_inches='tight')
print("saved ->", OUT)
