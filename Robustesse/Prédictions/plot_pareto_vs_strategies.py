"""
plot_pareto_vs_strategies.py -- figure "front de Pareto propre" (PREDICTIONS).
Superpose le FRONT DE PARETO de la PD (degradation <-> fiabilite, eps variable)
au nuage des strategies, INCLUANT toutes les variantes RB2 modifiees du chapitre
Predictions (RB2(SoH), RB2(Pred), RB2(RUL), RB2(SoH+Pred)) en plus des bases.
Meme plan que Pareto_2d_25y.py : encart en bas a droite, sobre, sans etoile ni
legende.
    x = LPSP [%]   ,   y = cout de degradation [k EUR / 25 ans]

Le front PD provient du run mesocentre de Vieillissement8/DP :
    ../Vieillissement8/DP/results_meso/dp_pareto_25y_51x51.npz  (sinon results/)

A lancer EN LOCAL :
    python plot_pareto_vs_strategies.py [chemin/dp_pareto_25y_51x51.npz]

Sortie -> pareto_strategies_vs_DP.{pdf,png}   (dans Predictions/)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

_THIS = os.path.dirname(os.path.abspath(__file__))
# dossier des resultats DP (front de Pareto), dans Vieillissement8/DP
_DP_DIR = os.path.join(_THIS, '..', 'Vieillissement8', 'DP')


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]

plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 13,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

# --- Tous les points connus (LPSP %, deg kEUR/25ans), == Pareto_2d_25y.py -----
# RB2(Pred) et RB2(SoH+Pred) : points REALISTES (prevision bruitee + hysteresis
# anti-clignotement M_SIGMA=1.0/MIN_DWELL=12), 25 ans, Monte-Carlo N=8/16 graines.
# Ce ne sont PLUS les bornes omniscientes (prevision parfaite) :
#     RB2(Pred)     omni (2.2280, 65.2188) -> reel (2.3297, 65.0030)
#     RB2(SoH+Pred) omni (2.3449, 59.5982) -> reel (2.4580, 59.4033)
# cf. robustesse_bruit_prevision.txt et RB2(SoH)/readme.txt.
PT = {
    '0-100': (10.3855, 124.1937), '25-75': (20.2667, 110.7658),
    '50-50': (8.0744, 109.0235),  '75-25': (3.8032, 59.6765),
    '100-0': (2.4851, 66.4122),   'RB2': (2.4540, 65.4218),
    'RB2(SoH)': (2.5475, 59.3644), 'RB2(Pred)': (2.3297, 65.0030),
    'RB2(RUL)': (2.5763, 59.9217), 'RB2(SoH+Pred)': (2.4580, 59.4033),
    'RB1': (1.2597, 80.1562),     'SoC1': (1.3389, 140.6745),
    'SoC06': (29.4642, 109.2535), 'Ideal': (0.0, 0.0),
}

# --- Selection PREDICTIONS : bases + toutes les variantes RB2 modifiees -------
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)',
          'RB2(Pred)', 'RB2(RUL)', 'RB2(SoH+Pred)', 'RB1', 'SoC1', 'SoC06', 'Ideal']
# points labelises UNIQUEMENT dans l'encart (cluster bas-gauche)
zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB1', 'RB2(Pred)',
               'RB2(RUL)', 'RB2(SoH+Pred)']
# decalages de label dans l'encart : (dx, dy, ha, va)
# Trois variantes (SoH, RUL, SoH+Pred) sont quasi coincidentes vers (2.5, 59.5)
# -> on etale leurs labels (gauche / bas / haut) ; 75-25 (x=3.8) est isole a droite.
zoom_offsets = {
    'RB1':           (0.05, 0.0,  'left',   'center'),
    '100-0':         (0.05, 0.8,  'left',   'bottom'),
    'RB2':           (0.12, 0.0,  'left',   'center'),
    'RB2(Pred)':     (-0.10, 0.0, 'right',  'center'),
    'RB2(SoH+Pred)': (-0.10, 0.0, 'right',  'center'),
    'RB2(SoH)':      (0.0, -1.2,  'center', 'top'),
    'RB2(RUL)':      (0.0,  1.2,  'center', 'bottom'),
    '75-25':         (0.10, 0.0,  'left',   'center'),
}

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'
EXTRA_COLORS = {'RB2(Pred)': '#000000', 'RB2(RUL)': '#117733',
                'RB2(SoH+Pred)': '#d95f02'}
DP_COLOR = '#c2185b'   # magenta soutenu pour le front PD


def color_of(label):
    if label in EXTRA_COLORS:
        return EXTRA_COLORS[label]
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def find_npz():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for sub in ("results_meso", "results"):
        p = os.path.join(_DP_DIR, sub, "dp_pareto_25y_51x51.npz")
        if os.path.exists(p):
            return p
    sys.exit("dp_pareto_25y_51x51.npz introuvable (Vieillissement8/DP/results_meso/ ou results/).")


def main():
    d = np.load(find_npz())
    eps, lpsp, deg = d['eps'], d['lpsp'], d['deg_keur']
    order = np.argsort(eps)
    eps, lpsp, deg = eps[order], lpsp[order], deg[order]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- nuage des strategies (labels hors cluster sur le plan principal) -----
    for label in labels:
        x, y = PT[label]
        ax.scatter(x, y, color=color_of(label), s=60, alpha=0.9, zorder=4)
    for label in labels:
        if label in zoom_labels:
            continue
        x, y = PT[label]
        col = color_of(label)
        if label == 'SoC06':
            ax.text(x - 2, y - 3, label, fontsize=14, color=col, weight='bold',
                    path_effects=LABEL_STROKE, va='top')
        else:
            ax.text(x + 0.5, y + 0.5, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)

    # --- FRONT DE PARETO PD (sobre : ligne + petits points, pas d'etoile) -----
    ax.plot(lpsp, deg, '-', color=DP_COLOR, lw=1.6, zorder=5)
    ax.scatter(lpsp, deg, color=DP_COLOR, s=22, zorder=6)


    # --- encart : zoom du cluster bas-gauche, dans le coin BAS-DROITE (vide,    -
    #     a droite de la queue du front pour ne pas la recouvrir). Plus grand    -
    #     que la figure Vieillissement8 : cluster RB2 plus dense ici. -----------
    axins = ax.inset_axes([0.57, 0.06, 0.41, 0.52])
    for label in zoom_labels:
        x, y = PT[label]
        axins.scatter(x, y, color=color_of(label), s=70, alpha=0.9, zorder=4)
    for label in zoom_labels:
        x, y = PT[label]
        dx, dy, ha, va = zoom_offsets[label]
        axins.text(x + dx, y + dy, label, fontsize=10, color=color_of(label),
                   weight='bold', path_effects=LABEL_STROKE, ha=ha, va=va, zorder=8)
    axins.plot(lpsp, deg, '-', color=DP_COLOR, lw=1.6, zorder=5)
    axins.scatter(lpsp, deg, color=DP_COLOR, s=20, zorder=6)
    for e, off in ((0.2, (6, -10)), (0.5, (6, 5))):
        k = int(np.argmin(np.abs(eps - e)))

    axins.set_xlim(-0.1, 4.5)
    axins.set_ylim(48, 84)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(_THIS, "pareto_strategies_vs_DP.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(_THIS, "pareto_strategies_vs_DP.png"), dpi=130, bbox_inches='tight')
    print("Figure ecrite ->", os.path.join(_THIS, "pareto_strategies_vs_DP.{pdf,png}"))


if __name__ == "__main__":
    main()
