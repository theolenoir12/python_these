# -*- coding: utf-8 -*-
"""generate_pareto.py -- generateur UNIFIE des plans de Pareto (LPSP, degradation).
=================================================================================
Regroupe en UN seul script tous les plans de Pareto du chapitre robustesse. Les
CHIFFRES sont ceux du chapitre PREDICTIONS (socle cost-min, VoLL=3, horizon 25 ans,
moyennes Monte-Carlo N=200 pour les strategies previsionnelles). Ce jeu est le seul
COHERENT contenant a la fois RB2(RUL), l'unique levier RB2(SoH), RB2(Pred) et
RB2(SoH+Pred). Points identiques a ../Prédictions/plot_pareto_strategies.py.

Trois FAMILLES de figures :
    base      : les stratégies de base seules
                (0-100, 25-75, 50-50, 75-25, 100-0, RB2, RB1, SoC1, SoC06, Ideal)
    base_soh  : les stratégies de base + RB2(SoH)   (unique levier état de santé)
    pred      : intégration de la prévision, TOUTES les variantes
                (+ RB2(SoH), RB2(Pred), RB2(RUL), RB2(SoH+Pred))

Chaque famille est déclinée sur DEUX axes indépendants :
    iso_cost = {False, True}   -> lignes d'iso-coût unifié (VoLL=3), SANS étiquette
                                  de valeur (les petits encadrés ont été retirés).
    dp       = {False, True}   -> superposition du front de Pareto de la PD
                                  (référence optimale globale, data/dp_*_v2.npz).

Sorties -> figures/<famille>[_dp][_isocost].{pdf,png}

Lancer (dans l'environnement conda habituel avec numpy + matplotlib) :
    python generate_pareto.py            # génère tout (12 figures)
    python generate_pareto.py base pred  # seulement ces familles

NB : ISO_SLOPE = cost_lpsp_keur(1.0) = 3 * (E_REF/100)/1000 = 8.2014 kEUR/pt
(cf. ../Analyse_sensibilite/voll_common.py). Valeur figée en dur ici pour rendre
le script autonome ; le fallback historique donnait déjà la même valeur.
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

THIS = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(THIS, "figures")
DP_NPZ = os.path.join(THIS, "data", "dp_pareto_25y_51x51_v2.npz")

ISO_SLOPE = 8.2014      # kEUR de coût LPSP par point de LPSP (VoLL=3)
DP_COLOR = "#c2185b"    # magenta soutenu pour le front PD

# --------------------------------------------------------------------------- #
#  POINTS canoniques (LPSP [%], coût de dégradation [kEUR / 25 ans])           #
#  Chapitre PREDICTIONS ; strategies previsionnelles = moyennes MC hyst N=200  #
#  (sens_pred_noise_N200_meso.txt), RB2(RUL) déterministe.                     #
# --------------------------------------------------------------------------- #
POINTS = {
    "0-100": (10.3855, 124.1937),
    "25-75": (20.2667, 110.7658),
    "50-50": (8.0744, 109.0235),
    "75-25": (3.8032, 59.6765),
    "100-0": (2.4851, 66.4122),
    "RB2":   (2.4540, 65.4218),   # socle cost-min
    "RB1":   (1.2597, 80.1562),
    "SoC1":  (1.3389, 140.6745),
    "SoC06": (29.4642, 109.2535),
    "Ideal": (0.0, 0.0),
    # levier état de santé (UNIQUE : setpoints H2 x SoH, noté simplement "SoH")
    "RB2(SoH)":      (2.5475, 59.3644),
    # intégration prévision
    "RB2(Pred)":     (2.3642, 65.0248),   # pré-charge +-1sigma (moyenne MC hyst)
    "RB2(RUL)":      (2.5763, 59.9217),   # augmentation par pronostic RUL
    "RB2(SoH+Pred)": (2.4796, 59.3898),   # prévision appliquée à RB2(SoH), MC hyst
}

BASE = ["0-100", "25-75", "50-50", "75-25", "100-0", "RB2",
        "RB1", "SoC1", "SoC06", "Ideal"]

# --------------------------------------------------------------------------- #
#  Familles : liste des labels + réglages de l'encart (zoom du cluster dense)  #
# --------------------------------------------------------------------------- #
FAMILIES = {
    "base": dict(
        labels=BASE,
        zoom=["75-25", "100-0", "RB2", "RB1"],
        offsets={
            "RB1":   (0.08, 0.0, "left",   "center"),
            "100-0": (0.06, 1.1, "left",   "bottom"),
            "RB2":   (0.13, 0.1, "left",   "center"),
            "75-25": (0.0,  1.1, "center", "bottom"),
        },
        ins_xlim=(0.9, 4.5), ins_ylo=55,
    ),
    "base_soh": dict(
        labels=BASE + ["RB2(SoH)"],
        zoom=["75-25", "100-0", "RB2", "RB2(SoH)", "RB1"],
        offsets={
            "RB1":      (0.08,  0.0, "left",   "center"),
            "100-0":    (0.06,  1.1, "left",   "bottom"),
            "RB2":      (0.13,  0.1, "left",   "center"),
            "RB2(SoH)": (0.0,  -1.1, "center", "top"),
            "75-25":    (0.0,   1.1, "center", "bottom"),
        },
        ins_xlim=(0.9, 4.5), ins_ylo=55,
    ),
    "pred": dict(
        labels=BASE + ["RB2(SoH)", "RB2(Pred)", "RB2(RUL)", "RB2(SoH+Pred)"],
        zoom=["75-25", "100-0", "RB2", "RB2(SoH)", "RB1", "RB2(Pred)",
              "RB2(RUL)", "RB2(SoH+Pred)"],
        offsets={
            "RB1":           (0.08,  0.0, "left",   "center"),
            "100-0":         (0.06,  1.1, "left",   "bottom"),
            "RB2":           (0.13,  0.1, "left",   "center"),
            "RB2(Pred)":     (-0.12, 0.0, "right",  "center"),
            "RB2(SoH)":      (0.0,  -1.1, "center", "top"),
            "RB2(RUL)":      (0.0,   1.0, "center", "bottom"),
            "RB2(SoH+Pred)": (-0.12, 0.0, "right",  "center"),
            "75-25":         (0.0,   1.1, "center", "bottom"),
        },
        ins_xlim=(0.9, 4.5), ins_ylo=55,
    ),
}

# --------------------------------------------------------------------------- #
#  Couleurs (tab10 partagé avec sens_*.py ; teintes dédiées pour les extras)   #
# --------------------------------------------------------------------------- #
STRAT_ORDER = ["0-100", "25-75", "50-50", "75-25", "100-0",
               "RB2", "RB2(SoH)", "RB1", "SoC1", "SoC06"]
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
EXTRA_COLORS = {"RB2(Pred)": "#000000", "RB2(RUL)": "#117733",
                "RB2(SoH+Pred)": "#d95f02"}
IDEAL_COLOR = "0.3"

LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground="white")]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


def color_of(label):
    if label in EXTRA_COLORS:
        return EXTRA_COLORS[label]           # teinte dédiée (non éclaircie)
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def load_dp_front():
    """Front de Pareto PD (points NON dominés triés par LPSP), ou None si absent."""
    if not os.path.exists(DP_NPZ):
        print("  front PD introuvable ->", DP_NPZ, "(variantes _dp ignorées)")
        return None
    d = np.load(DP_NPZ)
    nd = d["nondominated"].astype(bool) if "nondominated" in d.files \
        else np.ones(len(d["lpsp"]), dtype=bool)
    o = np.argsort(d["lpsp"][nd])
    return d["lpsp"][nd][o], d["deg_keur"][nd][o]


def draw_isocost(axis, levels, xlim, ylim):
    """Lignes d'iso-coût coût_total = deg + ISO_SLOPE*LPSP. SANS étiquette de
    valeur (les petits encadrés de texte ont été retirés à la demande)."""
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 100)
    for C in levels:
        axis.plot(xs, C - ISO_SLOPE * xs, ls=":", color="0.6", lw=0.9, zorder=0)


def build_figure(family, iso_cost, dp_front):
    cfg = FAMILIES[family]
    labels, zoom = cfg["labels"], cfg["zoom"]
    ins_ylo = 45 if dp_front is not None else cfg["ins_ylo"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- front PD (sous le nuage : il domine les stratégies) ------------------
    if dp_front is not None:
        ax.plot(dp_front[0], dp_front[1], "-", color=DP_COLOR, lw=1.6, zorder=3)
        ax.scatter(dp_front[0], dp_front[1], color=DP_COLOR, s=18, zorder=3)

    # --- nuage principal + étiquettes hors cluster ---------------------------
    for label in labels:
        x, y = POINTS[label]
        ax.scatter(x, y, color=color_of(label), s=60, alpha=0.9, zorder=4)
    for label in labels:
        if label in zoom:
            continue                       # étiqueté uniquement dans l'encart
        x, y = POINTS[label]
        col = color_of(label)
        if label == "SoC06":
            ax.text(x - 2, y - 3, label, fontsize=14, color=col, weight="bold",
                    path_effects=LABEL_STROKE, va="top")
        else:
            ax.text(x + 0.5, y + 0.5, label, fontsize=14, color=col,
                    weight="bold", path_effects=LABEL_STROKE)

    # --- encart : zoom du cluster bas-gauche ---------------------------------
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    if dp_front is not None:
        axins.plot(dp_front[0], dp_front[1], "-", color=DP_COLOR, lw=1.6, zorder=3)
        axins.scatter(dp_front[0], dp_front[1], color=DP_COLOR, s=16, zorder=3)
    for label in zoom:
        x, y = POINTS[label]
        axins.scatter(x, y, color=color_of(label), s=70, alpha=0.9, zorder=4)
    for label in zoom:
        dx, dy, ha, va = cfg["offsets"][label]
        x, y = POINTS[label]
        axins.text(x + dx, y + dy, label, fontsize=11, color=color_of(label),
                   weight="bold", path_effects=LABEL_STROKE, ha=ha, va=va, zorder=6)
    axins.set_xlim(cfg["ins_xlim"])
    axins.set_ylim(ins_ylo, 84)
    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(labelsize=10)

    # --- iso-coûts (optionnels, sans étiquette de valeur) --------------------
    if iso_cost:
        draw_isocost(ax, list(range(70, 141, 10)), ax.get_xlim(), ax.get_ylim())
        draw_isocost(axins, [70, 75, 80, 85, 90, 95, 100],
                     cfg["ins_xlim"], (ins_ylo, 84))

    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    stem = family + ("_dp" if dp_front is not None else "") \
        + ("_isocost" if iso_cost else "")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved ->", stem + ".{pdf,png}")


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    families = wanted or list(FAMILIES)
    dp_front = load_dp_front()
    for family in families:
        if family not in FAMILIES:
            print("famille inconnue :", family, "(", ", ".join(FAMILIES), ")")
            continue
        print("[", family, "]")
        for iso_cost in (False, True):
            build_figure(family, iso_cost, None)          # sans front DP
            if dp_front is not None:
                build_figure(family, iso_cost, dp_front)  # avec front DP


if __name__ == "__main__":
    main()
