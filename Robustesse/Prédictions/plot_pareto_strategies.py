# -*- coding: utf-8 -*-
"""
plot_pareto_strategies.py -- PLAN DE PARETO : TOUTES LES STRATEGIES + FRONT PD (NETTOYE).
=========================================================================================
Figure "vue d'ensemble" pour situer les positions RELATIVES des strategies. Memes
points / couleurs / encart que Pareto_2d_25y.py, avec en plus le FRONT DE PARETO PD
(programmation dynamique), restreint a ses points NON DOMINES (front nettoye, cf.
../Vieillissement8/DP/.../dp_pareto_points_25y.txt). PAS d'ellipses ici : les
incertitudes de prevision (minuscules) sont montrees separement par
plot_pred_uncertainty.py.

RB2(Pred) et RB2(SoH+Pred) sont places a leur MOYENNE Monte-Carlo (variante
hysteresis, robuste), lue dans sens_pred_noise_N200_meso.txt -- pas a la borne
omnisciente.

Entrees :
    sens_pred_noise_N200_meso.txt        (moyennes hyst des 2 strategies pred)
    ../Vieillissement8/DP/.../dp_pareto_25y_51x51.npz   (front PD ; filtre non-domine)
Sortie  -> pareto_strategies.{pdf,png}  (dans Predictions/)

Lancer en LOCAL :
    python plot_pareto_strategies.py [chemin/sens_pred_noise.txt] [chemin/dp.npz]
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
from matplotlib.lines import Line2D

_THIS = os.path.dirname(os.path.abspath(__file__))
_DP_DIR = os.path.join(_THIS, "..", "Vieillissement8", "DP")
STATS_TXT_DEFAULT = os.path.join(_THIS, "sens_pred_noise_N200_meso.txt")


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground="white")]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 12,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "grid.alpha": 0.7, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})

PT = {
    "0-100": (10.3855, 124.1937), "25-75": (20.2667, 110.7658),
    "50-50": (8.0744, 109.0235),  "75-25": (3.8032, 59.6765),
    "100-0": (2.4851, 66.4122),   "RB2": (2.4540, 65.4218),
    "RB2(SoH)": (2.5475, 59.3644), "RB2(Pred)": (2.3642, 65.0248),
    "RB2(RUL)": (2.5763, 59.9217), "RB2(SoH+Pred)": (2.4796, 59.3898),
    "RB1": (1.2597, 80.1562),     "SoC1": (1.3389, 140.6745),
    "SoC06": (29.4642, 109.2535), "Ideal": (0.0, 0.0),
}
LABELS = list(PT)
PRED_STRATS = ["RB2(Pred)", "RB2(SoH+Pred)"]

STRAT_ORDER = ["0-100", "25-75", "50-50", "75-25", "100-0",
               "RB2", "RB2(SoH)", "RB1", "SoC1", "SoC06"]
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = "0.3"
EXTRA_COLORS = {"RB2(Pred)": "#000000", "RB2(RUL)": "#117733",
                "RB2(SoH+Pred)": "#d95f02"}
FRONT_COL = "red"


def color_of(label):
    if label in EXTRA_COLORS:
        return EXTRA_COLORS[label]
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def parse_stats(path):
    out = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("strat;"):
                continue
            p = s.split(";")
            if len(p) >= 7:
                out[(p[0], p[1])] = (float(p[2]), float(p[3]), float(p[4]),
                                     float(p[5]), int(p[6]))
    return out


def find_npz():
    if len(sys.argv) > 2:
        return sys.argv[2]
    for sub in ("results_meso2/results", "results_meso", "results"):
        p = os.path.join(_DP_DIR, *sub.split("/"), "dp_pareto_25y_51x51.npz")
        if os.path.exists(p):
            return p
    sys.exit("dp_pareto_25y_51x51.npz introuvable sous Vieillissement8/DP/.")


def nondominated(lpsp, deg):
    """Masque des points NON domines (minimisation de lpsp ET deg)."""
    n = len(lpsp)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and lpsp[j] <= lpsp[i] and deg[j] <= deg[i] and (
                    lpsp[j] < lpsp[i] or deg[j] < deg[i]):
                keep[i] = False
                break
    return keep


def main():
    stats = parse_stats(sys.argv[1] if len(sys.argv) > 1 else STATS_TXT_DEFAULT)
    for s in PRED_STRATS:                           # moyenne MC hyst (robuste)
        mx, _, my, _, _ = stats[(s, "hyst")]
        PT[s] = (mx, my)

    d = np.load(find_npz())
    lpsp, deg = d["lpsp"], d["deg_keur"]
    keep = nondominated(lpsp, deg)                  # front NETTOYE
    o = np.argsort(lpsp[keep])
    fl, fd = lpsp[keep][o], deg[keep][o]

    fig, ax = plt.subplots(figsize=(8, 6))

    # front PD nettoye (reference legere)
    ax.plot(fl, fd, "-", color=FRONT_COL, lw=1.4, zorder=2)
    ax.scatter(fl, fd, color=FRONT_COL, s=10, zorder=2,marker="D")

    # nuage de TOUTES les strategies
    for label in LABELS:
        x, y = PT[label]
        ax.scatter(x, y, color=color_of(label), s=60, alpha=0.9, zorder=4)

    zoom_labels = ["75-25", "100-0", "RB2", "RB2(SoH)", "RB1", "RB2(Pred)",
                   "RB2(RUL)", "RB2(SoH+Pred)"]
    for label in LABELS:
        if label in zoom_labels:
            continue
        x, y = PT[label]
        col = color_of(label)
        if label == "SoC06":
            ax.text(x - 2, y - 3, label, fontsize=14, color=col, weight="bold",
                    path_effects=LABEL_STROKE, va="top")
        else:
            ax.text(x + 0.5, y + 0.5, label, fontsize=14, color=col,
                    weight="bold", path_effects=LABEL_STROKE)

    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- encart : cluster bas-gauche, coin bas-droit (identique a Pareto_2d_25y) -
    axins = ax.inset_axes([0.45, 0.08, 0.52, 0.46])
    axins.plot(fl, fd, "-", color=FRONT_COL, lw=1.2, zorder=2)
    axins.scatter(fl, fd, color=FRONT_COL, s=14, zorder=2)
    for label in zoom_labels:
        x, y = PT[label]
        axins.scatter(x, y, color=color_of(label), s=70, alpha=0.9, zorder=4)
    zoom_offsets = {
        "RB1":           (0.08, 0.0,  "left",   "center"),
        "100-0":         (0.06, 1.1,  "left",   "bottom"),
        "RB2":           (0.13, 0.1,  "left",   "center"),
        "RB2(Pred)":     (-0.12, 0.0, "right",  "center"),
        "RB2(SoH)":      (0.0, -1.1,  "center", "top"),
        "RB2(RUL)":      (0.0,  1.0,  "center", "bottom"),
        "RB2(SoH+Pred)": (-0.12, 0.0, "right",  "center"),
        "75-25":         (0.0,  1.1,  "center", "bottom"),
    }
    for label in zoom_labels:
        dx, dy, ha, va = zoom_offsets[label]
        x, y = PT[label]
        axins.text(x + dx, y + dy, label, fontsize=11, color=color_of(label),
                   weight="bold", path_effects=LABEL_STROKE, ha=ha, va=va, zorder=6)
    axins.set_xlim(0.9, 4.5)
    axins.set_ylim(55, 84)
    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(labelsize=10)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)

    # legende minimale (juste l'identite du front), coin haut-droit vide
    ax.legend(handles=[Line2D([0], [0], color=FRONT_COL, lw=1.4, marker="D",
                              markersize=4, label="front de Pareto (PD)")],
              loc="upper right", framealpha=0.95, fontsize=12, borderpad=0.6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_THIS, "pareto_strategies." + ext),
                    dpi=130, bbox_inches="tight")
    print("Figure ecrite -> pareto_strategies.{pdf,png}")


if __name__ == "__main__":
    main()
