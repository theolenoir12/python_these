# -*- coding: utf-8 -*-
"""
make_pred_zoom_fig.py -- ZOOM SUR L'INCERTITUDE DE PREVISION (ellipses), v2.
============================================================================
Remplace pred_uncertainty_zoom.pdf (ancien socle) par une version a trois
panneaux, un par paire (base sans prevision -> strategie previsionnelle),
sur le socle optimise et avec la nomenclature du chapitre :

    RB2                  ->  RB2(Pred)                (bench_fable, N=200)
    RB2(SoH_H2)          ->  RB2(SoH_H2+Pred)         (bench_ultime, N=200)
    RB2(SoH_all)         ->  RB2(SoH_all+Pred)        (bench_ultime, N=200)

Ellipses alignees sur les axes (demi-axes = n_std * sigma), 1 sigma (plein) /
2 sigma (tirets), centrees sur la moyenne Monte-Carlo. Les ellipses sont
minuscules a l'echelle du plan complet, d'ou le zoom par paire.

Sortie -> Synthese/Figures/pred_uncertainty_zoom_v2.{pdf,png}

Lancer en LOCAL :
    python make_pred_zoom_fig.py
"""
import os
import colorsys
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse

_THIS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_THIS, "Synthese", "Figures", "pred_uncertainty_zoom_v2")

LABEL_STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.labelsize": 15, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42,
})
BASE_COL = "0.45"


def darken(color, factor=0.9):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


# (label base, (x,y) base, label strat, (x,y) strat, (sx,sy) 1sigma, couleur)
PAIRS = [
    ("RB2", (2.5921, 58.843),
     "RB2(Pred)", (2.5312, 58.962), (0.0136, 0.014), darken('#7d3fbf')),
    ("RB2(SoH$_{\\mathrm{H_2}}$)", (2.9089, 54.9115),
     "RB2(SoH$_{\\mathrm{H_2}}$+Pred)", (2.7575, 55.075), (0.0126, 0.016),
     darken('#b8860b')),
    ("RB2(SoH$_{\\mathrm{all}}$)", (3.1051, 52.870),
     "RB2(SoH$_{\\mathrm{all}}$+Pred)", (2.9421, 53.095), (0.0138, 0.017),
     darken('#c02020')),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, (lab_b, (bx, by), lab_s, (mx, my), (sx, sy), col) in zip(axes, PAIRS):
        # base sans prevision : carre gris
        ax.scatter([bx], [by], s=130, color=BASE_COL, marker="s",
                   edgecolor="k", linewidth=0.7, zorder=6)
        # strategie previsionnelle : losange + ellipses 1/2 sigma
        for n_std, lw, ls, al in ((1.0, 2.0, "-", 0.95), (2.0, 1.1, "--", 0.55)):
            ax.add_patch(Ellipse((mx, my), 2 * n_std * sx, 2 * n_std * sy,
                                 edgecolor=col, facecolor="none", lw=lw, ls=ls,
                                 alpha=al, zorder=5))
        ax.scatter([mx], [my], s=100, color=col, marker="D",
                   edgecolor="k", linewidth=0.9, zorder=7)
        # etiquettes
        ax.text(bx, by + 0.028, lab_b, fontsize=11.5, color=BASE_COL,
                weight="bold", path_effects=LABEL_STROKE,
                ha="center", va="bottom", zorder=9)
        ax.text(mx, my - 2 * sy - 0.022, lab_s, fontsize=11.5, color=col,
                weight="bold", path_effects=LABEL_STROKE,
                ha="center", va="top", zorder=9)
        # cadrage par paire
        xmin, xmax = min(bx, mx), max(bx, mx)
        ymin, ymax = min(by, my), max(by, my)
        ax.set_xlim(xmin - 0.062, xmax + 0.062)
        ax.set_ylim(ymin - 0.115, ymax + 0.115)
        ax.set_xlabel("LPSP [%]")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Coût de dégradation [k€]")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT + "." + ext, dpi=130, bbox_inches="tight")
    print("Figure ecrite ->", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
