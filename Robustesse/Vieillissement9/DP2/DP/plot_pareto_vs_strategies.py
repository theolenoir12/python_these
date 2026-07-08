"""
plot_pareto_vs_strategies.py -- figure "front de Pareto propre" (VIEILLISSEMENT8).
Superpose le FRONT DE PARETO de la PD (degradation <-> fiabilite, eps variable)
au nuage des strategies de BASE du chapitre (EMS statiques, RB1, RB2, SoC, Ideal
-- SANS les variantes RB2 modifiees, qui sont traitees dans le chapitre
Predictions). Meme plan que Pareto_2d_25y.py : encart en bas a droite, sobre,
sans etoile ni legende.
    x = LPSP [%]   ,   y = cout de degradation [k EUR / 25 ans]

A lancer EN LOCAL :
    python plot_pareto_vs_strategies.py [chemin/dp_pareto_25y_51x51.npz]
(par defaut : results_meso/ puis results/)

Sortie -> figures_pareto_meso/pareto_strategies_vs_DP.{pdf,png}
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
PT = {
    '0-100': (10.3855, 124.1937), '25-75': (20.2667, 110.7658),
    '50-50': (8.0744, 109.0235),  '75-25': (3.8032, 59.6765),
    '100-0': (2.4851, 66.4122),   'RB2': (2.4540, 65.4218),
    'RB2(SoH)': (2.5475, 59.3644), 'RB2(Pred)': (2.2280, 65.2184),
    'RB2(RUL)': (2.5763, 59.9217), 'RB2(SoH+Pred)': (2.3167, 59.6867),
    'RB1': (1.2597, 80.1562),     'SoC1': (1.3389, 140.6745),
    'SoC06': (29.4642, 109.2535), 'Ideal': (0.0, 0.0),
}

# --- Selection VIEILLISSEMENT8 : strategies de base, SANS les RB2 modifies ----
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2',
          'RB1', 'SoC1', 'SoC06', 'Ideal']
# points labelises UNIQUEMENT dans l'encart (cluster bas-gauche)
zoom_labels = ['75-25', '100-0', 'RB2', 'RB1']
# decalages de label dans l'encart : (dx, dy, ha, va)
zoom_offsets = {
    'RB1':   (0.08, 0.0, 'left',   'center'),
    '100-0': (0.06, 1.1, 'left',   'bottom'),
    'RB2':   (0.13, 0.1, 'left',   'center'),
    '75-25': (0.0,  1.1, 'center', 'bottom'),
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
    # priorite au front V2 (projection vieillissement + rollout), sinon legacy
    for name in ("dp_pareto_25y_51x51_v2.npz", "dp_pareto_25y_51x51.npz"):
        for sub in ("results_meso", "results"):
            p = os.path.join(_THIS, sub, name)
            if os.path.exists(p):
                return p
    sys.exit("dp_pareto_25y_51x51[_v2].npz introuvable (results_meso/ ou results/).")


def main():
    src = find_npz()
    print("front PD :", src)
    d = np.load(src)
    eps, lpsp, deg = d['eps'], d['lpsp'], d['deg_keur']
    nd = d['nondominated'].astype(bool) if 'nondominated' in d.files \
        else np.ones(len(eps), dtype=bool)
    order = np.argsort(eps)
    eps, lpsp, deg, nd = eps[order], lpsp[order], deg[order], nd[order]

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
    # la ligne ne relie que les points NON-DOMINES ; les points domines (des
    # politiques PD valides mais hors front) restent en marqueurs creux.
    o = np.argsort(lpsp[nd])
    ax.plot(lpsp[nd][o], deg[nd][o], '-', color=DP_COLOR, lw=1.6, zorder=5)
    ax.scatter(lpsp[nd], deg[nd], color=DP_COLOR, s=22, zorder=6)
    if (~nd).any():
        ax.scatter(lpsp[~nd], deg[~nd], facecolors='none', edgecolors=DP_COLOR,
                   s=22, lw=0.9, zorder=6)


    # --- encart : zoom du cluster bas-gauche, dans le coin BAS-DROITE (vide,    -
    #     a droite de la queue du front pour ne pas la recouvrir) ---------------
    axins = ax.inset_axes([0.58, 0.07, 0.40, 0.45])
    for label in zoom_labels:
        x, y = PT[label]
        axins.scatter(x, y, color=color_of(label), s=70, alpha=0.9, zorder=4)
    for label in zoom_labels:
        x, y = PT[label]
        dx, dy, ha, va = zoom_offsets[label]
        axins.text(x + dx, y + dy, label, fontsize=11, color=color_of(label),
                   weight='bold', path_effects=LABEL_STROKE, ha=ha, va=va, zorder=8)
    axins.plot(lpsp[nd][o], deg[nd][o], '-', color=DP_COLOR, lw=1.6, zorder=5)
    axins.scatter(lpsp[nd], deg[nd], color=DP_COLOR, s=20, zorder=6)
    if (~nd).any():
        axins.scatter(lpsp[~nd], deg[~nd], facecolors='none', edgecolors=DP_COLOR,
                      s=20, lw=0.9, zorder=6)

    axins.set_xlim(-0.1, 4.5)
    axins.set_ylim(48, 84)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    out_dir = os.path.join(_THIS, "figures_pareto_meso")
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_strategies_vs_DP.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "pareto_strategies_vs_DP.png"), dpi=130, bbox_inches='tight')
    print("Figure ecrite ->", os.path.join(out_dir, "pareto_strategies_vs_DP.{pdf,png}"))


if __name__ == "__main__":
    main()
