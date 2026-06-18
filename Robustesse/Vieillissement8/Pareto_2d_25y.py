import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys


def darken(color, factor=0.7):
    """Assombrit une couleur en multipliant sa luminosite (HLS) par factor."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


# Halo blanc fin autour du texte (lisibilite sur fond + grille)
LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]

plt.rcParams.update({
    "text.usetex": False,          # Pas besoin de LaTeX externe
    "mathtext.fontset": "cm",      # Computer Modern (style LaTeX)
    "font.family": "serif",        # Police générale
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6
})

# Données bleues (points d'origine)
points = np.array([
    [10.3855, 124.1937],  # 0-100
    [20.2667, 110.7658],  # 25-75
    [8.0744, 109.0235],   # 50-50
    [3.8032, 59.6765],    # 75-25
    [2.4851, 66.4122],    # 100-0
    [2.7127, 65.1767],    # RB2
    [2.6496, 58.5350],    # RB2(SoH)
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000]      # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)',
          'RB1', 'SoC1', 'SoC06', 'Ideal']

# --- Code couleur par strategie, identique a la figure a ellipses (tab10) ---
STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'  # gris fonce pour 'Ideal' (hors STRAT_ORDER)

def color_of(label):
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))

# Affichage des points, colores par strategie
for i, label in enumerate(labels):
    ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)

# Ajout des labels (meme couleur que le point, halo blanc) avec conditions specifiques
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
        ax.text(points[i, 0] + 0.3, points[i, 1] - 4, label,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                verticalalignment='top', horizontalalignment='center')

    else:
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE)

# Style de l'axe
ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Degradation cost [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("pareto_ems.pdf", format='pdf', bbox_inches='tight')
plt.show()