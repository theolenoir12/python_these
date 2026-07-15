# -*- coding: utf-8 -*-
"""Figures de Pareto compatibles avec le modèle Vieillissement10.

Les données proviennent directement des sorties 25 ans de Vieillissement10.
RUL, prévisions et programmation dynamique restent hors périmètre pour l'instant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
V10 = HERE.parent / "Vieillissement10"
BASE_RESULTS = V10 / "rank_base_strategies_25y.txt"
SOH_RESULTS = (
    V10 / "Optimization_results_validated" / "optimization_soh_validated25.csv",
    V10 / "Optimization_results_validated" / "optimization_soh_validated_shapes25.csv",
)
FIG_DIR = HERE / "figures"
VOLL_EUR_PER_KWH = 3.0
ISOCOST_TOLERANCE_KEUR = 1e-6

BASE = [
    "0-100", "25-75", "50-50", "75-25", "100-0", "RB2", "RB1",
    "SoC1", "SoC06", "Ideal",
]
STRAT_ORDER = [
    "0-100", "25-75", "50-50", "75-25", "100-0", "RB2",
    "RB2(SoH)", "RB1", "SoC1", "SoC06",
]
COLORS = {
    label: color
    for label, color in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))
}
IDEAL_COLOR = "0.3"
SOH_FRONT_COLOR = "#3366aa"
LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground="white")]

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def darken(color, factor=0.7):
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def color_of(label):
    return darken(COLORS.get(label, IDEAL_COLOR))


def load_base_points(path=BASE_RESULTS):
    """Lit LPSP, EENS et dégradation depuis le classement canonique V10."""
    points = {}
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("rang;"):
                continue
            fields = line.split(";")
            points[fields[1]] = {
                "lpsp_pct": float(fields[3]),
                "eens_kwh": float(fields[2]),
                "degradation_keur": float(fields[7]),
                "unified_keur": float(fields[8]),
            }
    points["Ideal"] = {
        "lpsp_pct": 0.0,
        "eens_kwh": 0.0,
        "degradation_keur": 0.0,
        "unified_keur": 0.0,
    }
    missing = sorted(set(BASE) - set(points))
    if missing:
        raise ValueError("Stratégies V10 absentes : %s" % ", ".join(missing))
    return points


def load_soh_rows(paths=SOH_RESULTS):
    rows = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for raw in csv.DictReader(stream):
                row = {
                    "lpsp_pct": float(raw["lpsp_pct"]),
                    "eens_kwh": float(raw["eens_kwh"]),
                    "degradation_keur": float(raw["degradation_keur"]),
                    "unified_keur": float(raw["unified_keur"]),
                    "parameters": json.loads(raw["parameters_json"]),
                }
                key = (
                    round(row["lpsp_pct"], 9),
                    round(row["degradation_keur"], 9),
                    json.dumps(row["parameters"], sort_keys=True),
                )
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def pareto_front(rows):
    front = []
    best_degradation = float("inf")
    for row in sorted(rows, key=lambda r: (r["lpsp_pct"], r["degradation_keur"])):
        if row["degradation_keur"] < best_degradation - 1e-9:
            front.append(row)
            best_degradation = row["degradation_keur"]
    return front


def select_soh_point(front, rb2_unified_keur):
    """Point de dégradation minimale sans dépasser le coût unifié de RB2."""
    feasible = [
        row for row in front
        if row["unified_keur"] <= rb2_unified_keur + ISOCOST_TOLERANCE_KEUR
    ]
    if not feasible:
        raise ValueError("Aucun point RB2(SoH) à coût unifié inférieur ou égal à RB2")
    return min(feasible, key=lambda row: row["degradation_keur"])


def iso_slope_keur_per_lpsp_point(points):
    """Coût EENS d'un point de LPSP, déduit des données V10 et du VoLL."""
    rb2 = points["RB2"]
    return VOLL_EUR_PER_KWH * rb2["eens_kwh"] / 1000.0 / rb2["lpsp_pct"]


def draw_isocost(axis, slope, levels, xlim, ylim):
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 150)
    for level in levels:
        axis.plot(xs, level - slope * xs, ls=":", color="0.6", lw=0.9, zorder=0)


def draw_soh_front(axis, front, small=False):
    xs = [row["lpsp_pct"] for row in front]
    ys = [row["degradation_keur"] for row in front]
    axis.plot(xs, ys, "-", color=SOH_FRONT_COLOR, lw=1.4, alpha=0.85, zorder=2)
    axis.scatter(
        xs, ys, color=SOH_FRONT_COLOR, s=12 if small else 18,
        alpha=0.55, zorder=2,
    )


def build_figure(family, iso_cost, points, front, slope):
    labels = BASE if family == "base" else BASE + ["RB2(SoH)"]
    show_front = family == "base_soh_front"
    zoom = ["75-25", "100-0", "RB2", "RB1"]
    if family != "base":
        zoom.insert(3, "RB2(SoH)")
    offsets = {
        "RB2": (-0.03, 0.8, "right", "bottom"),
        "RB2(SoH)": (0.05, -0.7, "left", "top"),
        "RB1": (0.05, 0.7, "left", "bottom"),
        "100-0": (0.05, 0.8, "left", "bottom"),
        "75-25": (0.0, 0.9, "center", "bottom"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    if show_front:
        draw_soh_front(ax, front)
    for label in labels:
        row = points[label]
        ax.scatter(
            row["lpsp_pct"], row["degradation_keur"], color=color_of(label),
            s=65, alpha=0.95, zorder=4,
        )
    for label in labels:
        if label in zoom:
            continue
        row = points[label]
        dx, dy = (-1.2, -4.0) if label == "SoC06" else (0.35, 0.5)
        ax.text(
            row["lpsp_pct"] + dx, row["degradation_keur"] + dy, label,
            fontsize=13, color=color_of(label), weight="bold",
            path_effects=LABEL_STROKE,
        )

    axins = ax.inset_axes([0.43, 0.10, 0.54, 0.46])
    if show_front:
        draw_soh_front(axins, front, small=True)
    for label in zoom:
        row = points[label]
        axins.scatter(
            row["lpsp_pct"], row["degradation_keur"], color=color_of(label),
            s=70, alpha=0.95, zorder=4,
        )
        dx, dy, ha, va = offsets[label]
        axins.text(
            row["lpsp_pct"] + dx, row["degradation_keur"] + dy, label,
            fontsize=10.5, color=color_of(label), weight="bold", ha=ha, va=va,
            path_effects=LABEL_STROKE, zorder=6,
        )

    inset_xlim = (0.55, 2.05)
    inset_ylim = (18.0, 27.5)
    if iso_cost:
        draw_isocost(ax, slope, range(20, 241, 20), ax.get_xlim(), ax.get_ylim())
        draw_isocost(
            axins, slope, [30, 35, 40, 45, 50, 55, 60], inset_xlim, inset_ylim
        )
    else:
        axins.set_xlim(inset_xlim)
        axins.set_ylim(inset_ylim)

    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)
    ax.set_xlabel("LPSP sur la charge totale [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    stem = family + ("_isocost" if iso_cost else "")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / (stem + ".pdf"), bbox_inches="tight")
    png_path = FIG_DIR / (stem + ".png")
    fig.savefig(
        png_path, dpi=180, bbox_inches="tight", facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    # Certains lecteurs Windows interprètent mal les PNG RGBA complexes issus
    # d'Agg. Une conversion RGB explicite stabilise l'affichage sans changer la
    # figure ni le PDF vectoriel.
    from PIL import Image
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path)
    print("saved -> figures/%s.{pdf,png}" % stem)


def write_snapshot(points, selected, slope):
    path = HERE / "points_v10.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow([
            "Label", "LPSP_pct", "EENS_kWh", "degradation_kEUR",
            "unified_kEUR",
        ])
        for label in BASE:
            row = points[label]
            writer.writerow([
                label, "%.6f" % row["lpsp_pct"], "%.3f" % row["eens_kwh"],
                "%.6f" % row["degradation_keur"], "%.6f" % row["unified_keur"],
            ])
        writer.writerow([
            "RB2(SoH)", "%.6f" % selected["lpsp_pct"],
            "%.3f" % selected["eens_kwh"], "%.6f" % selected["degradation_keur"],
            "%.6f" % selected["unified_keur"],
        ])
    print("iso-slope V10 = %.6f kEUR/point de LPSP" % slope)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "families", nargs="*", choices=["base", "base_soh", "base_soh_front"],
    )
    args = parser.parse_args()
    families = args.families or ["base", "base_soh", "base_soh_front"]

    points = load_base_points()
    front = pareto_front(load_soh_rows())
    selected = select_soh_point(front, points["RB2"]["unified_keur"])
    points["RB2(SoH)"] = selected
    slope = iso_slope_keur_per_lpsp_point(points)
    write_snapshot(points, selected, slope)
    for family in families:
        for iso_cost in (False, True):
            build_figure(family, iso_cost, points, front, slope)


if __name__ == "__main__":
    main()
