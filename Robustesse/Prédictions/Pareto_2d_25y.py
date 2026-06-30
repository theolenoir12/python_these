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
    [2.4540, 65.4218],    # RB2
    [2.5475, 59.3644],    # RB2(SoH)
    [2.3297, 65.0030],    # RB2(Pred)        reel (bruite+hyst) ; omni etait (2.2280, 65.2184)
    [2.5763, 59.9217],    # RB2(RUL)
    [2.4580, 59.4033],    # RB2(SoH+Pred)    reel (bruite+hyst) ; omni etait (2.3449, 59.5982)
    [1.2597, 80.1562],    # RB1
    [1.3389, 140.6745],   # SoC1
    [29.4642, 109.2535],  # SoC06
    [0.0000, 0.0000]      # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB2(Pred)',
          'RB2(RUL)', 'RB2(SoH+Pred)', 'RB1', 'SoC1', 'SoC06', 'Ideal']

# --- Code couleur par strategie, identique a la figure a ellipses (tab10) ---
STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'  # gris fonce pour 'Ideal' (hors STRAT_ORDER)

# RB2(Pred)/RB2(RUL) absents de STRAT_ORDER -> couleurs dediees (sinon confondus)
EXTRA_COLORS = {'RB2(Pred)': '#000000', 'RB2(RUL)': '#117733',
                'RB2(SoH+Pred)': '#d95f02'}

def color_of(label):
    if label in EXTRA_COLORS:
        return EXTRA_COLORS[label]
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))

# --- Encadre zoom defini tot : les points du cluster ne sont labelises QUE dans le zoom ---
zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB1', 'RB2(Pred)', 'RB2(RUL)',
               'RB2(SoH+Pred)']

# Affichage des points, colores par strategie
for i, label in enumerate(labels):
    ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)

# Labels du plot principal : uniquement les strategies hors cluster (le cluster est dans le zoom)
for i, label in enumerate(labels):
    if label in zoom_labels:
        continue
    col = color_of(label)

    if label == 'SoC06':
        ax.text(points[i, 0] - 2, points[i, 1] - 3, label,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                verticalalignment='top')

    else:
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE)

# --- Encadre zoom sur le cluster bas-gauche, place dans le coin bas-droit (vide) ---
axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])  # [x0, y0, w, h] en fraction des axes

for i, label in enumerate(labels):
    if label in zoom_labels:
        axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=70, alpha=0.9)

# Decalages de labels adaptes a l'echelle agrandie : (dx, dy, ha, va)
zoom_offsets = {
    'RB1':      (0.08, 0.0,  'left',   'center'),
    '100-0':    (0.06, 1.1,  'left',   'bottom'),
    'RB2':      (0.13, 0.1,  'left',   'center'),
    'RB2(Pred)':(-0.12, 0.0, 'right',  'center'),
    'RB2(SoH)': (0.0, -1.1,  'center', 'top'),
    'RB2(RUL)': (0.0,  1.0,  'center', 'bottom'),
    'RB2(SoH+Pred)': (-0.12, 0.0, 'right', 'center'),
    '75-25':    (0.0,  1.1,  'center', 'bottom'),
}
for i, label in enumerate(labels):
    if label in zoom_labels:
        dx, dy, ha, va = zoom_offsets[label]
        axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                   color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                   horizontalalignment=ha, verticalalignment=va, zorder=6)

axins.set_xlim(0.9, 4.5)
axins.set_ylim(55, 84)
axins.grid(True, linestyle='--', alpha=0.5)
axins.tick_params(labelsize=10)
ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

# Style de l'axe
ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Degradation cost [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("pareto_ems.pdf", format='pdf', bbox_inches='tight')
plt.show()