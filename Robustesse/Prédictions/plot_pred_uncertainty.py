# -*- coding: utf-8 -*-
"""
plot_pred_uncertainty.py -- ZOOM SUR L'INCERTITUDE DE PREVISION (ellipses).
===========================================================================
Les ellipses d'incertitude (bruit de prevision, Monte-Carlo N=200) sont MINUSCULES
a l'echelle du plan de Pareto complet (cf. plot_pareto_strategies.py). On les montre
donc dans une figure DEDIEE, zoomee sur les seules strategies qui incorporent la
prevision, avec LEUR BASELINE respective :

    RB2(Pred)      (version hysteresis, robuste)  vs  RB2        (sans prevision)
    RB2(SoH+Pred)  (version hysteresis, robuste)  vs  RB2(SoH)   (sans prevision)

Ellipses ALIGNEES SUR LES AXES (demi-axes = n_std * sigma_LPSP, n_std * sigma_deg),
1 sigma (trait plein) / 2 sigma (tiretes), centrees sur la moyenne Monte-Carlo.
Rien d'autre sur la figure (ni front, ni autres strategies). Legende HORS du nuage.

Entree : sens_pred_noise_N200_meso.txt   (means/stds Monte-Carlo N=200, ici-meme)
Sortie -> pred_uncertainty_zoom.{pdf,png}  (dans Predictions/)

Lancer en LOCAL :
    python plot_pred_uncertainty.py [chemin/sens_pred_noise.txt]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

_THIS = os.path.dirname(os.path.abspath(__file__))
STATS_TXT_DEFAULT = os.path.join(_THIS, "sens_pred_noise_N200_meso.txt")

LABEL_STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.labelsize": 17, "axes.titlesize": 16, "legend.fontsize": 11,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "pdf.fonttype": 42,
})

# couleurs identiques a la figure d'ensemble (Pareto_2d_25y)
COL = {"RB2(Pred)": "#000000", "RB2(SoH+Pred)": "#d95f02"}
BASE_OF = {"RB2(Pred)": "RB2", "RB2(SoH+Pred)": "RB2(SoH)"}
BASE_COL = "0.45"


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


def draw_ellipses(ax, mx, my, sx, sy, col):
    for n_std, lw, ls, al in ((1.0, 2.0, "-", 0.95), (2.0, 1.1, "--", 0.55)):
        ax.add_patch(Ellipse((mx, my), 2 * n_std * sx, 2 * n_std * sy,
                             edgecolor=col, facecolor="none", lw=lw, ls=ls,
                             alpha=al, zorder=5))


def main():
    stats = parse_stats(sys.argv[1] if len(sys.argv) > 1 else STATS_TXT_DEFAULT)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # baselines (references sans prevision) : carres gris
    for base in ("RB2", "RB2(SoH)"):
        bx, _, by, _, _ = stats[(base, "baseline")]
        ax.scatter([bx], [by], s=140, color=BASE_COL, marker="s",
                   edgecolor="k", linewidth=0.7, zorder=6)

    # strategies previsionnelles robustes (hysteresis) : losange + ellipses
    pts = {}
    for s, col in COL.items():
        mx, sx, my, sy, _ = stats[(s, "hyst")]
        pts[s] = (mx, my, sx, sy)
        draw_ellipses(ax, mx, my, sx, sy, col)
        ax.scatter([mx], [my], s=110, color=col, marker="D",
                   edgecolor="k", linewidth=0.9, zorder=7)

    # etiquettes (hors legende)
    bx_rb2, _, by_rb2, _, _ = stats[("RB2", "baseline")]
    bx_soh, _, by_soh, _, _ = stats[("RB2(SoH)", "baseline")]
    px, py, _, sy_p = pts["RB2(Pred)"]
    qx, qy, _, sy_q = pts["RB2(SoH+Pred)"]
    txt = [
        (px, py - 2 * sy_p - 0.10, "RB2(Pred)",     COL["RB2(Pred)"],     "center", "top"),
        (bx_rb2 + 0.006, by_rb2 + 0.10, "RB2",        BASE_COL,             "left",   "bottom"),
        (qx, qy - 2 * sy_q - 0.10, "RB2(SoH+Pred)",  COL["RB2(SoH+Pred)"], "center", "top"),
        (bx_soh, by_soh + 0.22, "RB2(SoH)",          BASE_COL,             "center", "bottom"),
    ]
    for tx, ty, name, c, ha, va in txt:
        ax.text(tx, ty, name, fontsize=12, color=c, weight="bold",
                path_effects=LABEL_STROKE, ha=ha, va=va, zorder=9)

    ax.set_xlim(2.27, 2.60)
    ax.set_ylim(58.7, 66.1)
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)

    leg = [
        Line2D([0], [0], color=BASE_COL, marker="s", ls="none", markeredgecolor="k",
               markersize=11, label="référence sans prévision"),
        Line2D([0], [0], color="0.2", marker="D", ls="none", markeredgecolor="k",
               markersize=10, label="règle à hystérésis (robuste)"),
        Line2D([0], [0], color="0.2", lw=2.0, ls="-",
               label="incertitude de prévision  1σ (plein) / 2σ (tirets)"),
    ]
    # ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.14),
    #           ncol=1, framealpha=0.95, fontsize=11, handlelength=2.0, borderpad=0.7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_THIS, "pred_uncertainty_zoom." + ext),
                    dpi=130, bbox_inches="tight")
    print("Figure ecrite -> pred_uncertainty_zoom.{pdf,png}")


if __name__ == "__main__":
    main()
