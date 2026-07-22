"""Figures de synthèse Pareto V11-p=2 : toutes les familles EMS et leurs
augmentations, horizon 1 an, avec le front PD en repère.

Plans tracés (dégradation en ordonnée, kEUR/an) :
  - LPSP [%]  vs dégradation ;
  - EENS [kWh/an] vs dégradation (composante fiabilité de J3 = deg + 3·EENS).

Entrées :
  - points des stratégies : ``pareto_points_v11.tsv`` (cette campagne, même moteur
    et même comptabilité) ;
  - front PD (repère) : le ``points.tsv`` d'un export DP 1 an
    (``DP/results/mpc_reference_1y_*``). Le front est l'export canonique de la PD ;
    un point PD recalculé au même harnais est ajouté par le runner de synthèse
    pour quantifier tout écart de comptabilité (cf. note §13).

Usage (local ou mésocentre) :
  python -m FuzzyRules.plot_pareto_families_v11 \
      [--points pareto_points_v11.tsv] [--dp-front <points.tsv>] [--outdir <dir>]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
V11 = HERE.parent
DEFAULT_POINTS = HERE / "pareto_points_v11.tsv"
DEFAULT_DP_FRONT = (V11 / "DP" / "results" / "mpc_reference_1y_1b54f384caa8"
                    / "points.tsv")

# Style sobre, cohérent avec les figures Pareto du dossier DP.
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 10,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.markersize": 8,
    "grid.alpha": 0.6, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})

# Famille -> (couleur, marqueur, libellé de légende).
FAMILY_STYLE = {
    "reference": ("#111111", "s", "Références (RB1, RB2)"),
    "flc": ("#3b6fb0", "D", "FLC experte (+ prévision)"),
    "learned_tree": ("#e06c1f", "o", "Règles apprises — arbre (+ augmentations)"),
    "learned_anfis": ("#2f8f4e", "^", "ANFIS Takagi-Sugeno"),
    "pd": ("#c0392b", "*", "PD (harnais, eps=3)"),
}
_DEFAULT_STYLE = ("#666666", "x", "autre")


def load_points(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n"))
    reader = csv.DictReader(rows, delimiter="\t")
    out = []
    for r in reader:
        out.append({
            "label": r["label"], "famille": r["famille"],
            "augmentation": r["augmentation"], "info": r["info"],
            "lpsp": float(r["lpsp_pct"]), "eens": float(r["eens_kwh"]),
            "deg": float(r["deg_keur"]), "j3": float(r["j3_eur"]),
        })
    return out


def load_dp_front(path):
    if not Path(path).exists():
        return None
    eps, lpsp, deg, eens = [], [], [], []
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for r in reader:
            eps.append(float(r["eps"]))
            lpsp.append(float(r["lpsp_pct"]))
            deg.append(float(r["degradation_keur"]))
            eens.append(float(r["eens_kwh"]))
    order = np.argsort(lpsp)
    return {"eps": np.array(eps)[order], "lpsp": np.array(lpsp)[order],
            "deg": np.array(deg)[order], "eens": np.array(eens)[order]}


# Étiquettes courtes (None = point non étiqueté, ex. itérations DAgger
# intermédiaires ou variantes gaussiennes adjacentes) et décalage (dx,dy) en
# points pour éviter les collisions.
TAGS = {
    "RB1": ("RB1", (6, -13)),
    "RB2": ("RB2", (6, 5)),
    "FLC-IF (oracle)": ("FLC-IF", (-10, 8)),
    "FLC-IF (gaussian)": (None, None),
    "arbre I0 d4": ("arbre I0", (2, -13)),
    "arbre+precharge (oracle)": ("arbre+préch.", (-10, -14)),
    "arbre+precharge (gaussian)": (None, None),
    "arbre+DAgger iter1": (None, None),
    "arbre+DAgger iter2 (best)": ("DAgger*", (5, 6)),
    "arbre+DAgger iter3": (None, None),
    "arbre+DAgger iter4": (None, None),
    "arbre+DAgger iter5": (None, None),
    "ANFIS I0 (mf2)": ("ANFIS mf2", (6, 2)),
    "ANFIS I0 (mf3)": ("ANFIS mf3", (-22, 8)),
    "ANFIS IS (mf3)": ("ANFIS-IS", (-52, 2)),
    "FLC-I0 réglée": ("FLC-I0", (5, -11)),
    "PD (eps=3, harnais)": ("PD", (6, 2)),
}


def _dagger_path(points):
    """Trajectoire DAgger (arbre I0 -> iter1..5) pour la relier en pointillé."""
    start = [p for p in points if p["label"] == "arbre I0 d4"]
    iters = sorted([p for p in points if p["augmentation"] == "dagger"],
                   key=lambda p: p["label"])
    return start + iters


def xkey_map(xkey):
    return {"lpsp": "lpsp", "eens": "eens"}[xkey]


def _eens_per_lpsp(points):
    """Facteur EENS[kWh] par point de LPSP[%] : LPSP = EENS/energie_charge,
    donc le rapport est ~constant (meme profil). Mediane robuste."""
    ratios = [p["eens"] / p["lpsp"] for p in points if p["lpsp"] > 1e-9]
    return float(np.median(ratios)) if ratios else 1.0


def _draw_isocost(ax, xkey, eens_per_x, levels):
    """Droites iso-J3 (J3 = C_deg[EUR] + 3*EENS[kWh]) : y[kEUR] = (J3-3*eens)/1000.

    Dans le plan EENS, x = EENS (eens_per_x=1) ; dans le plan LPSP, EENS = k*x."""
    x0, x1 = ax.get_xlim()
    xs = np.array([max(x0, 0.0), x1])
    for j3 in levels:
        ys = (j3 - 3.0 * eens_per_x * xs) / 1000.0
        ax.plot(xs, ys, "-", color="#cfcfcf", lw=0.8, zorder=0)
        # étiquette près du bord droit, au-dessus de la droite.
        ax.annotate(f"J3={j3:.0f}", (xs[1], ys[1]), fontsize=7,
                    color="#9a9a9a", ha="right", va="bottom",
                    xytext=(-2, 1), textcoords="offset points", zorder=0)


def _plot_plane(points, dp_front, xkey, xlabel, outpath, xmax=None, zoom=False,
                iso_levels=None):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    xk = xkey_map(xkey)

    # Limites d'abord (les isocoûts en dépendent).
    if zoom:
        degs = [p["deg"] for p in points]
        ax.set_ylim(min(degs) - 0.02, max(degs) + 0.03)
    if xmax is not None:
        ax.set_xlim(left=-0.03 * xmax, right=xmax)

    # Droites iso-J3 en arrière-plan (J3 = deg + 3*EENS ; LPSP proportionnel a EENS).
    if iso_levels:
        eens_per_x = 1.0 if xkey == "eens" else _eens_per_lpsp(points)
        _draw_isocost(ax, xkey, eens_per_x, iso_levels)

    if dp_front is not None and not zoom:
        ax.plot(dp_front[xk], dp_front["deg"], "-", color="#999999",
                lw=1.6, zorder=1, label="Front PD (export canonique, 1 an)")
        ax.plot(dp_front[xk], dp_front["deg"], ".", color="#999999",
                ms=6, zorder=1)

    path = _dagger_path(points)
    if len(path) > 1:
        ax.plot([p[xk] for p in path], [p["deg"] for p in path], ":",
                color=FAMILY_STYLE["learned_tree"][0], lw=1.1, alpha=0.7,
                zorder=2)

    seen = set()
    for p in points:
        color, marker, legend = FAMILY_STYLE.get(p["famille"], _DEFAULT_STYLE)
        lab = legend if p["famille"] not in seen else None
        seen.add(p["famille"])
        x = p[xk]
        ax.scatter(x, p["deg"], s=75, marker=marker, color=color,
                   edgecolor="white", linewidth=0.7, zorder=4, label=lab)
        tag, off = TAGS.get(p["label"], (p["label"], (4, 4)))
        if tag:
            ax.annotate(tag, (x, p["deg"]), fontsize=9, xytext=off,
                        textcoords="offset points", zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Coût de dégradation [kEUR / an]")
    suffix = " — zoom stratégies" if zoom else " vs front PD"
    ax.set_title(f"Familles EMS V11-p=2 (1 an){suffix}")
    ax.grid(True)
    loc = "lower left" if zoom else "upper right"
    ax.legend(loc=loc, framealpha=0.95)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath}.{ext}", dpi=160)
    plt.close(fig)
    return f"{outpath}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default=str(DEFAULT_POINTS))
    parser.add_argument("--dp-front", default=str(DEFAULT_DP_FRONT))
    parser.add_argument("--outdir", default=str(HERE / "figures_synthese"))
    args = parser.parse_args()

    points = load_points(args.points)
    dp_front = load_dp_front(args.dp_front)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    iso = [3000, 3500, 4000, 4500, 5000]   # niveaux J3 [EUR/an] des isocoûts
    produced = [
        _plot_plane(points, dp_front, "lpsp", r"LPSP [%]",
                    str(outdir / "pareto_lpsp_deg"), xmax=4.5, iso_levels=iso),
        _plot_plane(points, dp_front, "lpsp", r"LPSP [%]",
                    str(outdir / "pareto_lpsp_deg_zoom"), xmax=4.5, zoom=True,
                    iso_levels=iso),
        _plot_plane(points, dp_front, "eens", r"EENS [kWh / an]",
                    str(outdir / "pareto_eens_deg"), xmax=900.0, iso_levels=iso),
        _plot_plane(points, dp_front, "eens", r"EENS [kWh / an]",
                    str(outdir / "pareto_eens_deg_zoom"), xmax=900.0, zoom=True,
                    iso_levels=iso),
    ]

    # Classement J3 (lecture directe du "qui est dominé").
    order = sorted(points, key=lambda p: p["j3"])
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ys = np.arange(len(order))
    colors = [FAMILY_STYLE[p["famille"]][0] for p in order]
    ax.barh(ys, [p["j3"] for p in order], color=colors, edgecolor="white")
    ax.set_yticks(ys)
    ax.set_yticklabels([p["label"] for p in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(order[0]["j3"], color="#888888", ls="--", lw=1.0)
    ax.set_xlabel(r"$J_3 = C_{deg} + 3\,EENS$ [EUR / an]")
    ax.set_title("Classement J3 (1 an) — plus bas = meilleur")
    ax.grid(True, axis="x")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(outdir / f"classement_j3.{ext}"), dpi=160)
    plt.close(fig)

    produced.append(str(outdir / "classement_j3.png"))
    print("Figures écrites dans", outdir)
    for f in produced:
        print("  ", f)


if __name__ == "__main__":
    main()
