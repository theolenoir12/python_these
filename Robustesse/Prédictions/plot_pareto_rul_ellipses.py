# -*- coding: utf-8 -*-
"""
plot_pareto_rul_ellipses.py -- Pareto 2D 25 ans de TOUTES les EMS (identique a
Pareto_2d_25y.py) avec, DANS L'ENCADRE ZOOM EXISTANT, les ellipses d'incertitude
du pronostic 1σ/2σ autour de RB2(RUL) -- pour juger l'AMPLITUDE de l'ellipse au
milieu des positions des autres strategies.

NB : PAS d'encart supplementaire, PAS d'ellipse RB2(SoH).

Entree  : mc_rul_uncertainty_cloud.csv (produit par mc_rul_uncertainty.py, N=200).
Sorties : pareto_ems_rul_ellipses.pdf / .png
"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import colorsys

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD_CSV = os.path.join(HERE, "mc_rul_uncertainty_cloud.csv")
RUL_COLOR = '#117733'   # meme vert que RB2(RUL) dans Pareto_2d_25y.py


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


def confidence_ellipse(x, y, ax, n_std=1.0, **kw):
    x = np.asarray(x); y = np.asarray(y)
    cov = np.cov(x, y)
    pear = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0.0
    rx, ry = np.sqrt(1 + pear), np.sqrt(1 - pear)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry, **kw)
    sx, sy = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    tr = (transforms.Affine2D().rotate_deg(45).scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    return ax.add_patch(ell)


def load_rul_cloud():
    cloud = []
    with open(CLOUD_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] == "RB2(RUL)" and r["variant"] == "cloud":
                cloud.append((float(r["lpsp"]), float(r["deg"])))
    return np.array(cloud)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

points = np.array([
    [10.3855, 124.1937], [20.2667, 110.7658], [8.0744, 109.0235], [3.8032, 59.6765],
    [2.4851, 66.4122], [2.4540, 65.4218], [2.5475, 59.3644], [2.3297, 65.0030],
    [2.5763, 59.9217], [2.4580, 59.4033], [1.2597, 80.1562], [1.3389, 140.6745],
    [29.4642, 109.2535], [0.0000, 0.0000],
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB2(Pred)',
          'RB2(RUL)', 'RB2(SoH+Pred)', 'RB1', 'SoC1', 'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'
EXTRA_COLORS = {'RB2(Pred)': '#000000', 'RB2(RUL)': RUL_COLOR, 'RB2(SoH+Pred)': '#d95f02'}


def color_of(label):
    if label in EXTRA_COLORS:
        return EXTRA_COLORS[label]
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


cloud = load_rul_cloud()

fig, ax = plt.subplots(figsize=(8, 6))
zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB1', 'RB2(Pred)', 'RB2(RUL)',
               'RB2(SoH+Pred)']

for i, label in enumerate(labels):
    ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)

for i, label in enumerate(labels):
    if label in zoom_labels:
        continue
    col = color_of(label)
    if label == 'SoC06':
        ax.text(points[i, 0] - 2, points[i, 1] - 3, label, fontsize=14, color=col,
                weight='bold', path_effects=LABEL_STROKE, verticalalignment='top')
    else:
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, fontsize=14, color=col,
                weight='bold', path_effects=LABEL_STROKE)

# --- Encadre zoom RESSERRE : uniquement RB2(RUL) (+ ellipses 1σ/2σ) et le point
#     unique de RB2(SoH). Bornes serrees -> l'ellipse de RB2(RUL) devient visible
#     et se juge directement contre l'ecart a RB2(SoH). Pas d'ellipse RB2(SoH).
i_rul = labels.index('RB2(RUL)')
i_soh = labels.index('RB2(SoH)')
axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])

# ellipses d'incertitude du pronostic autour de RB2(RUL), RECENTREES exactement
# sur le point nominal trace (le nuage bruite decale la moyenne de ~0.7 sigma).
cloud_c = cloud + (points[i_rul] - cloud.mean(axis=0))
for ns, lw, ls in ((1.0, 1.9, '-'), (2.0, 1.3, '--')):
    confidence_ellipse(cloud_c[:, 0], cloud_c[:, 1], axins, n_std=ns, edgecolor=RUL_COLOR,
                       facecolor='none', lw=lw, ls=ls, alpha=0.95, zorder=5)
# points : RB2(RUL) et RB2(SoH)
axins.scatter(points[i_rul, 0], points[i_rul, 1], color=color_of('RB2(RUL)'),
              s=80, alpha=0.95, zorder=6)
axins.scatter(points[i_soh, 0], points[i_soh, 1], color=color_of('RB2(SoH)'),
              s=80, alpha=0.95, zorder=6)
axins.text(points[i_rul, 0], points[i_rul, 1] + 0.05, 'RB2(RUL)', fontsize=12,
           color=color_of('RB2(RUL)'), weight='bold', path_effects=LABEL_STROKE,
           ha='center', va='bottom', zorder=7)
axins.text(points[i_soh, 0], points[i_soh, 1] - 0.05, 'RB2(SoH)', fontsize=12,
           color=color_of('RB2(SoH)'), weight='bold', path_effects=LABEL_STROKE,
           ha='center', va='top', zorder=7)

axins.set_xlim(2.520, 2.610); axins.set_ylim(59.15, 60.10)
axins.grid(True, linestyle='--', alpha=0.5); axins.tick_params(labelsize=10)
ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Degradation cost [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "pareto_ems_rul_ellipses.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(HERE, "pareto_ems_rul_ellipses.png"), dpi=150, bbox_inches='tight')
print("Figure -> pareto_ems_rul_ellipses.pdf / .png")
print("Amplitude ellipse RB2(RUL) : sigma_LPSP=%.4f pt, sigma_deg=%.4f kEUR ; 2sigma ~ %.4f pt x %.4f kEUR"
      % (cloud[:, 0].std(), cloud[:, 1].std(), 4 * cloud[:, 0].std(), 4 * cloud[:, 1].std()))
