#!/usr/bin/env python3
"""Figures de lecture des reruns corriges P1/P3/P4, sans simulation.

Le script ne lit que les caches pleine precision deja valides. Les plans
d'objectifs utilisent :
  x = EENS [MWh] ;
  y = cout hors defaillance [kEUR].
La pente des droites d'iso-cout vaut -VoLL = -3 kEUR/MWh.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
V9 = OUT.parent
VOLL = 3.0  # EUR/kWh = kEUR/MWh
C_VISIT = 1.5  # kEUR/intervention

P1_PATH = V9 / "runs/p1_value_info_248b9c6a69b3/results_raw.tsv"
P3_PATHS = {
    "T3": V9 / "runs/p3_maintenance_748b2c584716/results_raw.tsv",
    "T6": V9 / "runs/p3_maintenance_e9a82d234288/results_raw.tsv",
    "T12": V9 / "runs/p3_maintenance_f9746cd965ec/results_raw.tsv",
}
P4_PATH = V9 / "runs/p4_dwell_db5f2e82eff7/results_raw.tsv"
CONTRASTS_PATH = V9 / "runs/p1_p3_stats_dc427b9da2f4/contrasts.tsv"

COLORS = {
    "RB2": "#4C78A8",
    "RB2(Recale)": "#8064A2",
    "RB2(Sched)": "#F28E2B",
    "RB2(SoH)": "#2E8B57",
    "instant": "#777777",
    "corrective": "#D62728",
    "calendar": "#4C78A8",
    "rul": "#2E8B57",
    "base": "#111111",
    "minoff": "#7F7F7F",
    "noisy": "#F28E2B",
    "omni": "#2CA02C",
}


def read_table(path: Path, delimiter: str = ";") -> list[dict[str, str]]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) < 2:
        raise RuntimeError(f"Table vide ou incomplete : {path}")
    return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter))


def load_long(
    path: Path,
    entity_field: str,
    entities: tuple[str, ...],
    fields: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    rows = [row for row in read_table(path) if int(row["draw"]) >= 0]
    result: dict[str, dict[str, np.ndarray]] = {}
    reference_draws = None
    for entity in entities:
        selected = sorted(
            (row for row in rows if row[entity_field] == entity),
            key=lambda row: int(row["draw"]),
        )
        draws = np.array([int(row["draw"]) for row in selected], dtype=int)
        if len(draws) != 200 or not np.array_equal(draws, np.arange(200)):
            raise RuntimeError(f"{entity} : les 200 tirages contigus sont requis")
        if reference_draws is None:
            reference_draws = draws
        elif not np.array_equal(draws, reference_draws):
            raise RuntimeError(f"CRN rompus pour {entity}")
        result[entity] = {
            field: np.array([float(row[field]) for row in selected], dtype=float)
            for field in fields
        }
    return result


def validate_closures(p1, p3_by_t) -> None:
    for name, values in p1.items():
        expected = values["deg"] + VOLL * values["eens"] / 1000.0
        if not np.allclose(values["uni"], expected, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Fermeture P1 invalide : {name}")
    for tag, dataset in p3_by_t.items():
        for name, values in dataset.items():
            expected = values["deg"] + VOLL * values["eens"] / 1000.0
            if not np.allclose(values["uni0"], expected, rtol=0.0, atol=1e-10):
                raise RuntimeError(f"Fermeture P3 invalide : {tag}/{name}")
            waste = values["wbat"] + values["wfc"] + values["wely"]
            if not np.allclose(values["waste"], waste, rtol=0.0, atol=1e-10):
                raise RuntimeError(f"Fermeture gaspillage invalide : {tag}/{name}")


def p1_xy(values):
    return values["eens"] / 1000.0, values["deg"]


def p1_cost(values):
    x, y = p1_xy(values)
    return y + VOLL * x


def p3_xy(values):
    x = values["eens"] / 1000.0
    y = values["deg"] + C_VISIT * values["nint"] + values["waste"]
    return x, y


def p3_cost(values):
    x, y = p3_xy(values)
    return y + VOLL * x


def p4_xy(rows, family: str, n: int):
    selected = [
        row for row in rows
        if row["family"] == family and int(row["N"]) == n and row["ok"] == "True"
    ]
    if not selected:
        raise RuntimeError(f"P4 absent : {family}, N={n}")
    return (
        np.array([float(row["eens"]) / 1000.0 for row in selected]),
        np.array([float(row["deg"]) for row in selected]),
        np.array([float(row["uni"]) for row in selected]),
    )


def mean_point(x, y):
    return float(np.mean(x)), float(np.mean(y))


def setup_ax(ax, title: str, delta: bool = False, show_direction: bool = True):
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(True, color="0.88", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if delta:
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        ax.axvline(0, color="0.35", lw=0.8, zorder=1)
        ax.set_xlabel(r"$\Delta$ EENS [MWh]")
        ax.set_ylabel(r"$\Delta$ coût hors défaillance [k€]")
    else:
        ax.set_xlabel("Énergie non servie EENS [MWh]")
        ax.set_ylabel("Coût hors défaillance [k€]")
    if show_direction:
        ax.text(
            0.02, 0.03, "meilleur  ↙", transform=ax.transAxes,
            fontsize=9, color="0.35", weight="bold",
        )


def pad_limits(ax, frac: float = 0.08):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = max(xmax - xmin, 1e-6) * frac
    dy = max(ymax - ymin, 1e-6) * frac
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)


def iso_absolute(ax, ref_x: float, ref_y: float, label: str):
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 200)
    ys = ref_y + VOLL * (ref_x - xs)
    ax.plot(xs, ys, color="0.25", ls="--", lw=1.2, zorder=2,
            label=f"iso-coût de {label} (VoLL=3)")


def iso_delta(ax):
    pad_limits(ax, 0.05)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xs = np.linspace(xmin, xmax, 300)
    boundary = -VOLL * xs
    clipped = np.minimum(boundary, ymax)
    ax.fill_between(
        xs, ymin, clipped, where=clipped > ymin,
        color="#DDEFD9", alpha=0.65, zorder=-5,
    )
    ax.plot(xs, boundary, color="0.20", ls="--", lw=1.2,
            label=r"$\Delta C=0$ à VoLL=3")
    ax.text(
        0.03, 0.93, "gain économique", transform=ax.transAxes,
        fontsize=9, color="#2E6B34", weight="bold", va="top",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def arrow(ax, start, end, color, lw=2.0):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        shrinkA=7, shrinkB=7, mutation_scale=13),
        zorder=7,
    )


def plot_mean(ax, x, y, label, color, marker="o", size=95):
    point = mean_point(x, y)
    ax.scatter(*point, s=size, marker=marker, color=color,
               edgecolor="white", linewidth=1.2, zorder=8, label=label)
    return point


def annotate_delta_cost(ax, point, delta, offset=(7, 7)):
    ax.annotate(
        f"ΔC={delta:+.3f} k€", xy=point, xytext=offset,
        textcoords="offset points", fontsize=8.5, color="0.15",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
        zorder=10,
    )


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_p1(p1):
    labels = {
        "RB2": "RB2",
        "RB2(Recale)": "RB2 recalé",
        "RB2(Sched)": "Horloge globale",
        "RB2(SoH)": "SoH exact",
    }
    order = tuple(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax = axes[0]
    setup_ax(ax, "Niveaux et trajectoire moyenne dans l’espace des objectifs")
    means = {}
    for name in order:
        x, y = p1_xy(p1[name])
        ax.scatter(x, y, s=10, color=COLORS[name], alpha=0.10, zorder=2)
        means[name] = plot_mean(ax, x, y, labels[name], COLORS[name])
    for a, b in zip(order[:-1], order[1:]):
        arrow(ax, means[a], means[b], COLORS[b])
    pad_limits(ax)
    iso_absolute(ax, *means["RB2"], "RB2 moyen")
    ax.legend(fontsize=8.5, loc="best")

    ax = axes[1]
    setup_ax(ax, "Déplacements appariés : provenance du gain", delta=True)
    comparisons = [
        ("Recalage − RB2", "RB2(Recale)", "RB2", COLORS["RB2(Recale)"]),
        ("Horloge − recalage", "RB2(Sched)", "RB2(Recale)", COLORS["RB2(Sched)"]),
        ("SoH − horloge", "RB2(SoH)", "RB2(Sched)", COLORS["RB2(SoH)"]),
        ("SoH − RB2 (total)", "RB2(SoH)", "RB2", "#006D2C"),
    ]
    for label, a, b, color in comparisons:
        xa, ya = p1_xy(p1[a]); xb, yb = p1_xy(p1[b])
        dx, dy = xa - xb, ya - yb
        ax.scatter(dx, dy, s=13, color=color, alpha=0.22, zorder=3)
        point = plot_mean(ax, dx, dy, label, color, marker="D", size=80)
        offset = (8, -20) if name == "calendar" else (-82, 8)
        annotate_delta_cost(ax, point, float(np.mean(dy + VOLL * dx)), offset=offset)
    iso_delta(ax)
    ax.legend(fontsize=8.2, loc="best")

    fig.suptitle(
        "P1 — valeur informationnelle du SoH sous incertitude de vieillissement\n"
        "25 ans, 200 mondes CRN ; SoH exact ; horloge globale non recalée aux remplacements",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    save(fig, "01_P1_SoH_espace_objectifs")


def load_contrasts():
    with CONTRASTS_PATH.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def p3_interval_records(contrasts):
    result = []
    for months in (3, 6, 12):
        scenario = f"T{months}m_marge1_corrige"
        matches = [
            row for row in contrasts
            if row["analysis"] == "P3"
            and row["scenario"] == scenario
            and float(row["voll"]) == 3.0
            and float(row["c_visit"]) == 1.5
            and row["contrast"] == "rul-corrective"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Contraste P3 absent/duplique : {scenario}")
        row = matches[0]
        result.append((
            months, float(row["mean"]),
            float(row["mean_ci95_low"]), float(row["mean_ci95_high"]),
            int(row["wins"]), int(row["ties"]), int(row["losses"]),
        ))
    return result


def figure_p3(p3, contrasts):
    labels = {
        "instant": "Remplacement instantané",
        "corrective": "Correctif",
        "calendar": "Calendaire",
        "rul": "RUL parfaite",
    }
    central = p3["T6"]
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.6))

    ax = axes[0]
    setup_ax(ax, "Scénario central : niveaux moyens")
    means = {}
    for name in labels:
        x, y = p3_xy(central[name])
        ax.scatter(x, y, s=10, color=COLORS[name], alpha=0.10, zorder=2)
        means[name] = plot_mean(ax, x, y, labels[name], COLORS[name])
    arrow(ax, means["corrective"], means["calendar"], COLORS["calendar"])
    arrow(ax, means["corrective"], means["rul"], COLORS["rul"], lw=2.5)
    pad_limits(ax)
    iso_absolute(ax, *means["corrective"], "correctif moyen")
    ax.legend(fontsize=7.8, loc="best")

    ax = axes[1]
    setup_ax(ax, "Déplacements appariés depuis le correctif", delta=True)
    for name in ("calendar", "rul"):
        xa, ya = p3_xy(central[name]); xb, yb = p3_xy(central["corrective"])
        dx, dy = xa - xb, ya - yb
        ax.scatter(dx, dy, s=15, color=COLORS[name], alpha=0.24, zorder=3)
        point = plot_mean(ax, dx, dy, f"{labels[name]} − correctif",
                          COLORS[name], marker="D", size=85)
        annotate_delta_cost(ax, point, float(np.mean(dy + VOLL * dx)))
    iso_delta(ax)
    ax.legend(fontsize=8.3, loc="best")

    ax = axes[2]
    setup_ax(ax, "Valeur de la RUL selon l’intervalle de visite", show_direction=False)
    records = p3_interval_records(contrasts)
    months = np.array([row[0] for row in records])
    means_cost = np.array([row[1] for row in records])
    low = np.array([row[2] for row in records])
    high = np.array([row[3] for row in records])
    ax.axhline(0, color="0.25", lw=1.0)
    ax.fill_between(months, low, high, color=COLORS["rul"], alpha=0.18)
    ax.plot(months, means_cost, "-o", color=COLORS["rul"], lw=2.2, ms=7)
    for month, mean, _, _, wins, ties, losses in records:
        ax.annotate(
            f"{mean:+.3f} k€\n{wins}/{ties}/{losses}",
            (month, mean), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8.5,
        )
    ax.set_xticks(months, [f"{month} mois" for month in months])
    ax.set_xlabel("Période entre visites")
    ax.set_ylabel("RUL − correctif [k€]  (<0 : gain RUL)")
    ax.text(0.03, 0.04, "ruban : IC95 de la moyenne appariée",
            transform=ax.transAxes, fontsize=8.5, color="0.35")

    fig.suptitle(
        "P3 — valeur de la RUL pour la maintenance insulaire\n"
        "25 ans, 200 mondes CRN, VoLL=3, intervention=1,5 k€ ; RUL exacte (borne structurelle)",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    save(fig, "02_P3_RUL_espace_objectifs")


def p4_series(rows, family: str, ns=(2, 4, 6, 8, 12)):
    result = []
    for n in ns:
        x, y, u = p4_xy(rows, family, n)
        result.append((n, x, y, u))
    return result


def figure_p4(rows):
    base_x, base_y, base_u = p4_xy(rows, "base", 0)
    minoff = p4_series(rows, "minoff")
    noisy = p4_series(rows, "noisy")
    omni = p4_series(rows, "omni")
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.6))

    ax = axes[0]
    setup_ax(ax, "Zone utile dans l’espace des objectifs (N≤6 h)")
    base_point = plot_mean(ax, base_x, base_y, "Base", COLORS["base"], marker="*", size=180)
    for family, series, label in (
        ("minoff", minoff, "Temps minimal sans prévision"),
        ("noisy", noisy, "Prévision bruitée (32 graines)"),
        ("omni", omni, "Prévision clairvoyante"),
    ):
        use = [item for item in series if item[0] <= 6]
        mx = [float(np.mean(item[1])) for item in use]
        my = [float(np.mean(item[2])) for item in use]
        if family == "noisy":
            for _, x, y, _ in use:
                ax.scatter(x, y, s=11, color=COLORS[family], alpha=0.18, zorder=2)
        ax.plot(mx, my, "-o", color=COLORS[family], lw=2.0, ms=6,
                label=label, zorder=5)
        for item, x, y in zip(use, mx, my):
            ax.annotate(f"N={item[0]}", (x, y), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=COLORS[family])
    pad_limits(ax)
    iso_absolute(ax, *base_point, "base")
    ax.legend(fontsize=7.8, loc="best")

    ax = axes[1]
    setup_ax(ax, "Prévision bruitée N=4 : déplacements", delta=True)
    noisy4_x, noisy4_y, noisy4_u = p4_xy(rows, "noisy", 4)
    min8_x, min8_y, min8_u = p4_xy(rows, "minoff", 8)
    for label, rx, ry, ru, color in (
        ("N=4 bruité − base", base_x, base_y, base_u, COLORS["noisy"]),
        ("N=4 bruité − meilleur min-off", min8_x, min8_y, min8_u, "#A65628"),
    ):
        dx, dy = noisy4_x - rx[0], noisy4_y - ry[0]
        ax.scatter(dx, dy, s=24, color=color, alpha=0.50, zorder=3)
        point = plot_mean(ax, dx, dy, label, color, marker="D", size=90)
        annotate_delta_cost(ax, point, float(np.mean(noisy4_u - ru[0])))
    iso_delta(ax)
    ax.legend(fontsize=8.2, loc="best")

    ax = axes[2]
    setup_ax(ax, "Gain économique et falaise au-delà de N=6 h", show_direction=False)
    base_cost = float(base_u[0])
    for family, series, label in (
        ("minoff", minoff, "Temps minimal"),
        ("noisy", noisy, "Prévision bruitée"),
        ("omni", omni, "Prévision clairvoyante"),
    ):
        ns = np.array([item[0] for item in series])
        means = np.array([np.mean(item[3]) - base_cost for item in series])
        ax.plot(ns, means, "-o", color=COLORS[family], lw=2.0, ms=6, label=label)
        if family == "noisy":
            std = np.array([np.std(item[3], ddof=1) for item in series])
            ax.fill_between(ns, means - std, means + std,
                            color=COLORS[family], alpha=0.18)
    ax.axhline(0, color="0.25", lw=1.0)
    ax.axvspan(1.7, 4.3, color="#DDEFD9", alpha=0.45, zorder=-3)
    ax.annotate("optimum bruité\nN=4 : −1,470 k€", (4, np.mean(noisy4_u) - base_cost),
                xytext=(12, 18), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=COLORS["noisy"]),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))
    ax.set_xticks([2, 4, 6, 8, 12])
    ax.set_xlabel("Persistance N [h]")
    ax.set_ylabel("Coût − base [k€]  (<0 : gain)")
    ax.legend(fontsize=8.2, loc="best")

    fig.suptitle(
        "P4 — filtre de persistance au démarrage de l’électrolyseur\n"
        "25 ans, vieillissement nominal ; prévision bruitée AR(1), 32 graines",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    save(fig, "03_P4_prevision_espace_objectifs")


def figure_summary(p1, p3, p4_rows):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.3))

    ax = axes[0]
    setup_ax(ax, "P1 — augmentation par le SoH")
    order = ("RB2", "RB2(Recale)", "RB2(Sched)", "RB2(SoH)")
    labels = ("RB2", "Recalé", "Horloge", "SoH exact")
    pts = []
    for name, label in zip(order, labels):
        x, y = p1_xy(p1[name])
        pts.append(plot_mean(ax, x, y, label, COLORS[name]))
    for i in range(len(pts) - 1):
        arrow(ax, pts[i], pts[i + 1], COLORS[order[i + 1]])
    pad_limits(ax, 0.18)
    iso_absolute(ax, *pts[0], "RB2")
    ax.legend(fontsize=7.5, loc="best")
    ax.text(0.98, 0.04, "SoH−RB2 = −2,305 k€\nSoH−horloge = −0,716 k€",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    ax = axes[1]
    setup_ax(ax, "P3 — maintenance par RUL")
    central = p3["T6"]
    order = ("corrective", "calendar", "rul")
    labels = ("Correctif", "Calendaire", "RUL parfaite")
    pts = {}
    for name, label in zip(order, labels):
        x, y = p3_xy(central[name])
        pts[name] = plot_mean(ax, x, y, label, COLORS[name])
    arrow(ax, pts["corrective"], pts["calendar"], COLORS["calendar"])
    arrow(ax, pts["corrective"], pts["rul"], COLORS["rul"], lw=2.5)
    pad_limits(ax, 0.16)
    iso_absolute(ax, *pts["corrective"], "correctif")
    ax.legend(fontsize=7.5, loc="best")
    ax.text(0.98, 0.04, "RUL−correctif = −2,625 k€\n178/200 mondes gagnants",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    ax = axes[2]
    setup_ax(ax, "P4 — persistance ELY")
    points = []
    for family, n, label, marker, size in (
        ("base", 0, "Base", "*", 180),
        ("minoff", 8, "Meilleur min-off", "o", 95),
        ("noisy", 4, "Prévision bruitée N=4", "o", 95),
        ("omni", 4, "Clairvoyante N=4", "o", 95),
    ):
        x, y, _ = p4_xy(p4_rows, family, n)
        points.append((family, plot_mean(ax, x, y, label, COLORS[family], marker, size)))
    for (fa, pa), (fb, pb) in zip(points[:-1], points[1:]):
        arrow(ax, pa, pb, COLORS[fb])
    pad_limits(ax, 0.14)
    iso_absolute(ax, *points[0][1], "base")
    ax.legend(fontsize=7.5, loc="best")
    ax.text(0.98, 0.04, "bruitée−base = −1,470 k€\nau-delà du min-off = −0,387 k€",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    fig.suptitle(
        "Résultats corrigés dans les espaces d’objectifs — chaque panneau a son propre protocole\n"
        "bas-gauche = moins d’énergie non servie et moins de coût hors défaillance",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    save(fig, "00_SYNTHESE_espaces_objectifs")


def write_data_summary(p1, p3, p4_rows):
    lines = [
        "figure;strategie;EENS_mean_MWh;cout_hors_defaillance_mean_kEUR;cout_total_mean_kEUR",
    ]
    for name, values in p1.items():
        x, y = p1_xy(values)
        lines.append(f"P1;{name};{np.mean(x):.9f};{np.mean(y):.9f};{np.mean(y + VOLL*x):.9f}")
    for name, values in p3["T6"].items():
        x, y = p3_xy(values)
        lines.append(f"P3_T6_C1.5;{name};{np.mean(x):.9f};{np.mean(y):.9f};{np.mean(y + VOLL*x):.9f}")
    for family, n in (("base", 0), ("minoff", 8), ("noisy", 4), ("omni", 4)):
        x, y, u = p4_xy(p4_rows, family, n)
        lines.append(f"P4;{family}_N{n};{np.mean(x):.9f};{np.mean(y):.9f};{np.mean(u):.9f}")
    (OUT / "DONNEES_MOYENNES_FIGURES.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    p1 = load_long(
        P1_PATH, "strat", ("RB2", "RB2(Recale)", "RB2(Sched)", "RB2(SoH)"),
        ("lpsp", "deg", "eens", "uni"),
    )
    p3 = {
        tag: load_long(
            path, "policy", ("instant", "corrective", "calendar", "rul"),
            ("lpsp", "deg", "eens", "uni0", "nint", "waste", "wbat", "wfc", "wely"),
        )
        for tag, path in P3_PATHS.items()
    }
    validate_closures(p1, p3)
    p4_rows = read_table(P4_PATH)
    if sum(row["family"] == "noisy" and row["N"] == "4" and row["ok"] == "True"
           for row in p4_rows) != 32:
        raise RuntimeError("P4 noisy N=4 doit contenir exactement 32 graines")

    contrasts = load_contrasts()
    figure_summary(p1, p3, p4_rows)
    figure_p1(p1)
    figure_p3(p3, contrasts)
    figure_p4(p4_rows)
    write_data_summary(p1, p3, p4_rows)
    print(f"Figures generees dans {OUT}")


if __name__ == "__main__":
    main()
