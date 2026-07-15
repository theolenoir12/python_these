# -*- coding: utf-8 -*-
"""generate_ellipses.py -- plans de Pareto AVEC ELLIPSES d'incertitude (sensibilité).
=================================================================================
RE-TRACE (aucune simulation) les plans de Pareto à ellipses des analyses de
sensibilité, en LISANT les résultats Monte-Carlo DÉJÀ stockés dans
../Analyse_sensibilite/results_meso/sens_<axe>.txt. Pour chaque stratégie : le
point NOMINAL + des ellipses 1σ/2σ (alignées sur les axes : les .txt ne stockent
que moyenne/écart-type marginaux) autour du nuage MC. Même identité visuelle que
generate_pareto.py (couleurs tab10, encart zoom du cluster bas-gauche).

Modèle : ../Analyse_sensibilite/plot_eol_pareto_chap2.py, généralisé à tous les
axes multi-stratégies.

Axes régénérés (5) :
    eol         seuils de fin de vie          sens_eol.txt         (section ## Front)
    hthresholds seuils fonctions dég. H2      sens_hthresholds.txt (section ## Front)
    sizing      dimensionnement BAT/FC/ELY    sens_sizing.txt      (section ## Front)
    cweights    poids = coûts de remplacement sens_cweights.txt    (LPSP invariant ->
                                                                    ellipse verticale)
    calendar    vieillissement calendaire     sens_calendar.txt    (nominal=OFF,
                                                                    ellipse=calendaire)
NON couvert : soh (sens_soh.txt) est MONO-STRATÉGIE (RB2(SoH) : biais + bruit) et
ne rentre pas dans ce plotter multi-stratégies -> figure sens_soh_pareto.pdf laissée
STATIQUE dans figures_ellipses/. Les figures d'ellipses de prévision/RUL ont leurs
propres scripts (cf. figures_ellipses/prediction/ et README).

Sorties -> figures_ellipses/sens_<axe>_pareto.{pdf,png}

Lancer (env conda habituel : numpy + matplotlib) :
    python generate_ellipses.py                 # les 5 axes
    python generate_ellipses.py eol sizing      # certains axes
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import colorsys

THIS = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS, "..", "Analyse_sensibilite", "results_meso")
FIG_DIR = os.path.join(THIS, "figures_ellipses")

# --- style partagé avec generate_pareto.py -------------------------------- #
STRAT_ORDER = ["0-100", "25-75", "50-50", "75-25", "100-0",
               "RB2", "RB2(SoH)", "RB1", "SoC1", "SoC06"]
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = "0.3"
LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground="white")]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

# cluster dense étiqueté UNIQUEMENT dans l'encart
CLUSTER = ["75-25", "100-0", "RB2", "RB2(SoH)", "RB1"]
ZOOM_OFFSETS = {
    "RB1":      (0.10,  0.0, "left",  "center"),
    "100-0":    (0.06,  1.3, "left",  "bottom"),
    "RB2":      (-0.10, -0.8, "right", "top"),
    "RB2(SoH)": (0.10,  0.4, "left",  "bottom"),
    "75-25":    (0.10,  0.0, "left",  "center"),
}


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


def color_of(strat):
    return darken(COLORS.get(strat, IDEAL_COLOR), 0.7)


# --------------------------------------------------------------------------- #
#  PARSERS : chacun renvoie {strat: (lpsp_nom, deg_nom, lpsp_mean, lpsp_std,   #
#  deg_mean, deg_std)}. Structure UNIFIÉE -> un seul traceur.                  #
# --------------------------------------------------------------------------- #
def parse_front(path):
    """sens_eol / sens_hthresholds / sens_sizing : section '## Front ...',
    cols strat;LPSP_nom;deg_nom;LPSP_mean;LPSP_std;deg_mean;deg_std;..."""
    rows, in_block = {}, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Front"):
                in_block = True
                continue
            if in_block:
                if s.startswith("## ") or s.startswith("## OAT"):
                    break
                if not s or s.startswith("strat;") or s.startswith("#"):
                    continue
                p = s.split(";")
                if len(p) < 7 or p[0] not in STRAT_ORDER:
                    continue
                rows[p[0]] = (float(p[1]), float(p[2]), float(p[3]),
                              float(p[4]), float(p[5]), float(p[6]))
    return rows


def parse_cweights(path):
    """sens_cweights : LPSP invariante ; deg = cout. cols
    strat;LPSP_%;cout_nominal;cost_bat;cost_fc;cost_ely;cout_mean;cout_std;..."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("strat;"):
                continue
            p = s.split(";")
            if len(p) < 8 or p[0] not in STRAT_ORDER:
                continue
            lpsp, deg_nom = float(p[1]), float(p[2])
            deg_mean, deg_std = float(p[6]), float(p[7])
            rows[p[0]] = (lpsp, deg_nom, lpsp, 0.0, deg_mean, deg_std)
    return rows


def parse_calendar(path):
    """sens_calendar : nominal = calendaire OFF ; ellipse = calendaire ON (MC).
    cols ems;SoC_moy;LPSP_off;deg_off;LPSP_cal_mean;deg_cal_mean;deg_cal_std;..."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("ems;"):
                continue
            p = s.split(";")
            if len(p) < 7 or p[0] not in STRAT_ORDER:
                continue
            lpsp_off, deg_off = float(p[2]), float(p[3])
            lpsp_cal, deg_cal_mean, deg_cal_std = float(p[4]), float(p[5]), float(p[6])
            rows[p[0]] = (lpsp_off, deg_off, lpsp_cal, 0.0, deg_cal_mean, deg_cal_std)
    return rows


AXES = {
    "eol":         ("sens_eol.txt", parse_front),
    "hthresholds": ("sens_hthresholds.txt", parse_front),
    "sizing":      ("sens_sizing.txt", parse_front),
    "cweights":    ("sens_cweights.txt", parse_cweights),
    "calendar":    ("sens_calendar.txt", parse_calendar),
}


# --------------------------------------------------------------------------- #
#  TRACÉ                                                                       #
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(0)


def draw(ax, rows, strats, zoom=False):
    """Pour chaque stratégie : nuage MC léger + ellipses 1σ/2σ + point nominal."""
    for strat in strats:
        if strat not in rows:
            continue
        lpsp_nom, deg_nom, mx, sx, my, sy = rows[strat]
        col = color_of(strat)
        if sx > 0 or sy > 0:
            cx = _RNG.normal(mx, sx, 40)
            cy = _RNG.normal(my, sy, 40)
            ax.scatter(cx, cy, s=6, color=col, alpha=0.4, edgecolor="none", zorder=2)
            for n_std, lw, ls, al in ((1.0, 1.6, "-", 0.9), (2.0, 0.9, "--", 0.5)):
                ax.add_patch(Ellipse((mx, my), 2 * n_std * sx, 2 * n_std * sy,
                                     edgecolor=col, facecolor="none", lw=lw, ls=ls,
                                     alpha=al, zorder=4))
        ax.scatter([lpsp_nom], [deg_nom], s=(70 if zoom else 60),
                   color=col, alpha=0.9, zorder=6)


def build_axis_figure(name, rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = [s for s in STRAT_ORDER if s in rows]

    # --- nuage principal + étiquettes hors cluster ---------------------------
    draw(ax, rows, plotted)
    for strat in plotted:
        if strat in CLUSTER:
            continue
        lpsp_nom, deg_nom = rows[strat][0], rows[strat][1]
        col = color_of(strat)
        if strat == "SoC06":
            ax.text(lpsp_nom - 2, deg_nom - 3, strat, fontsize=14, color=col,
                    weight="bold", path_effects=LABEL_STROKE, va="top")
        else:
            ax.text(lpsp_nom + 0.5, deg_nom + 0.5, strat, fontsize=14, color=col,
                    weight="bold", path_effects=LABEL_STROKE)
    # point Idéal (0,0), comme sur les plans de Pareto
    ic = color_of("Ideal")
    ax.scatter([0], [0], s=60, color=ic, alpha=0.9, zorder=6)
    ax.text(0.5, 0.5, "Ideal", fontsize=14, color=ic, weight="bold",
            path_effects=LABEL_STROKE)

    # --- encart : zoom du cluster bas-gauche (bornes auto sur ±2σ) ------------
    zoom_strats = [s for s in CLUSTER if s in rows]
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    draw(axins, rows, zoom_strats, zoom=True)
    xs, ys = [], []
    for strat in zoom_strats:
        lpsp_nom, deg_nom, mx, sx, my, sy = rows[strat]
        dx, dy, ha, va = ZOOM_OFFSETS[strat]
        axins.text(lpsp_nom + dx, deg_nom + dy, strat, fontsize=11, color=color_of(strat),
                   weight="bold", path_effects=LABEL_STROKE, ha=ha, va=va, zorder=7)
        xs += [lpsp_nom, mx - 2 * sx, mx + 2 * sx]
        ys += [deg_nom, my - 2 * sy, my + 2 * sy]
    if xs:
        mxg = 0.10 * (max(xs) - min(xs) or 1.0)
        myg = 0.10 * (max(ys) - min(ys) or 1.0)
        axins.set_xlim(min(xs) - mxg, max(xs) + mxg)
        axins.set_ylim(min(ys) - myg, max(ys) + myg)
    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(labelsize=10)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    stem = "sens_" + name + "_pareto"
    fig.savefig(os.path.join(FIG_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved ->", stem + ".{pdf,png}")


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    axes = wanted or list(AXES)
    for name in axes:
        if name not in AXES:
            print("axe inconnu :", name, "(", ", ".join(AXES), ")")
            continue
        fname, parser = AXES[name]
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print("[", name, "] données introuvables ->", path)
            continue
        print("[", name, "]", fname)
        rows = parser(path)
        if not rows:
            print("  aucune ligne exploitable -> ignoré")
            continue
        build_axis_figure(name, rows)


if __name__ == "__main__":
    main()
