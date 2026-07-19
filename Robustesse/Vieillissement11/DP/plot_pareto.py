"""Trace le front PD V11-p=2 apres rapatriement du calcul 25 ans.

Usage :

    python plot_pareto.py [runs/dp_pareto_v11_p2_25y_51x51_rollout.npz]

Le script refuse les anciens caches sans metadata V11-p=2. Il ne relance
aucune simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "runs" / "dp_pareto_v11_p2_25y_51x51_rollout.npz"


def main():
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT
    if not source.exists():
        raise SystemExit(
            f"Front 25 ans absent : {source}\n"
            "Les caches 1y_7x7 sont des smokes et ne doivent pas etre traces."
        )
    data = np.load(source, allow_pickle=False)
    if "model_id" not in data.files or "ely_stress_exponent" not in data.files:
        raise ValueError("cache non attribuable : metadata V11 absente")
    if not np.isclose(float(data["ely_stress_exponent"]), 2.0):
        raise ValueError("ce traceur canonique attend le nominal p=2")

    eps = np.asarray(data["eps"], dtype=float)
    deg = np.asarray(data["deg_keur"], dtype=float)
    eens = np.asarray(data["eens_kwh"], dtype=float)
    lpsp = np.asarray(data["lpsp"], dtype=float)
    unif3 = np.asarray(data["unif3_keur"], dtype=float)
    nd = (np.asarray(data["nondominated"], dtype=bool)
          if "nondominated" in data.files else np.ones(len(eps), dtype=bool))

    rb1 = {
        "lpsp": float(data["RB1_lpsp"]),
        "deg": float(data["RB1_deg_keur"]),
        "eens": float(data["RB1_eens_kwh"]),
        "unif": float(data["RB1_unif3_keur"]),
    }
    rb2 = {
        "lpsp": float(data["RB2_lpsp"]),
        "deg": float(data["RB2_deg_keur"]),
        "eens": float(data["RB2_eens_kwh"]),
        "unif": float(data["RB2_unif3_keur"]),
    }
    demand_kwh = float(data["demand_kwh"])

    order = np.argsort(eps)
    eps, deg, eens, lpsp, unif3, nd = (
        array[order] for array in (eps, deg, eens, lpsp, unif3, nd)
    )
    output = source.parent
    best = int(np.argmin(unif3))

    # Front degradation--EENS : seuls les points non domines sont relies.
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    front_order = np.argsort(eens[nd])
    ax.plot(eens[nd][front_order], deg[nd][front_order], "-", color="#7a5195",
            lw=1.7, zorder=1, label="Front PD non domine")
    colors = np.log10(eps)
    scatter = ax.scatter(eens[nd], deg[nd], c=colors[nd], cmap="viridis",
                         s=58, edgecolor="k", linewidth=0.4, zorder=3)
    if (~nd).any():
        ax.scatter(eens[~nd], deg[~nd], facecolor="none", edgecolor="0.5",
                   s=48, linewidth=0.9, zorder=2, label="Points PD domines")
    for x, y, value, keep in zip(eens, deg, eps, nd):
        if keep:
            ax.annotate(f"{value:g}", (x, y), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color="0.3")
    ax.scatter(rb1["eens"], rb1["deg"], marker="s", s=95, facecolor="white",
               edgecolor="#4c78a8", linewidth=1.8, label="RB1 optimisee")
    ax.scatter(rb2["eens"], rb2["deg"], marker="D", s=90, facecolor="white",
               edgecolor="#f58518", linewidth=1.8, label="RB2 optimisee")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("log10(epsilon), epsilon en EUR/kWh")
    ax.set_xlabel("Energie non servie EENS [kWh / 25 ans]")
    ax.set_ylabel("Cout de degradation [kEUR / 25 ans]")
    ax.set_title("Front PD V11-p=2 : degradation--fiabilite")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pareto_deg_eens_v11_p2.pdf")
    fig.savefig(output / "pareto_deg_eens_v11_p2.png", dpi=180)
    plt.close(fig)

    # Plan canonique du manuscrit : LPSP--degradation. EENS et LPSP sont ici
    # strictement proportionnelles, car tous les points partagent le profil de
    # charge et la meme definition de la fiabilite.
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    front_order = np.argsort(lpsp[nd])
    ax.plot(lpsp[nd][front_order], deg[nd][front_order], "-", color="#7a5195",
            lw=1.7, zorder=1, label="Front PD non domine")
    scatter = ax.scatter(lpsp[nd], deg[nd], c=colors[nd], cmap="viridis",
                         s=58, edgecolor="k", linewidth=0.4, zorder=3)
    if (~nd).any():
        ax.scatter(lpsp[~nd], deg[~nd], facecolor="none", edgecolor="0.5",
                   s=48, linewidth=0.9, zorder=2, label="Points PD domines")
    shown = {0.05, 0.1, 0.12, 0.2, 0.5, 1.0, 3.0, 10.0, 20.0, 50.0}
    for x, y, value, keep in zip(lpsp, deg, eps, nd):
        if keep and float(value) in shown:
            ax.annotate(f"{value:g}", (x, y), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color="0.3")
    ax.scatter(rb1["lpsp"], rb1["deg"], marker="s", s=95, facecolor="white",
               edgecolor="#4c78a8", linewidth=1.8, label="RB1 optimisee")
    ax.scatter(rb2["lpsp"], rb2["deg"], marker="D", s=90, facecolor="white",
               edgecolor="#f58518", linewidth=1.8, label="RB2 optimisee")
    ax.scatter(lpsp[best], deg[best], marker="*", s=190, facecolor="#54a24b",
               edgecolor="k", linewidth=0.6, zorder=5,
               label=f"Minimum realise a VoLL=3 (epsilon={eps[best]:g})")
    slope_j3 = 3.0 * demand_kwh / 100_000.0
    xline = np.linspace(0.0, max(float(lpsp.max()), rb1["lpsp"], rb2["lpsp"]), 300)
    ax.plot(xline, unif3[best] - slope_j3 * xline, ":", color="0.4", lw=1.2,
            label="Iso-cout tangent, VoLL=3")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("log10(epsilon), epsilon en EUR/kWh")
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Cout de degradation [kEUR / 25 ans]")
    ax.set_title("Front PD V11-p=2 : LPSP--degradation")
    ax.set_ylim(float(deg.min()) - 0.8,
                max(float(deg.max()), rb1["deg"], rb2["deg"]) + 1.4)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pareto_lpsp_deg_v11_p2.pdf")
    fig.savefig(output / "pareto_lpsp_deg_v11_p2.png", dpi=180)
    plt.close(fig)

    # Zoom sur la partie decisionnelle du front. Les references RB sont hors de
    # l'ordonnee de ce zoom et restent visibles dans le plan complet ci-dessus.
    zoom = (lpsp <= 1.35) & nd
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    zoom_order = np.argsort(lpsp[zoom])
    ax.plot(lpsp[zoom][zoom_order], deg[zoom][zoom_order], "-", color="#7a5195",
            lw=1.7, zorder=1)
    sc_zoom = ax.scatter(lpsp[zoom], deg[zoom], c=colors[zoom], cmap="viridis",
                         s=64, edgecolor="k", linewidth=0.4, zorder=3)
    shown_zoom = {0.15, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0, 20.0, 50.0}
    offsets = {3.0: (6, -13), 10.0: (9, 7), 20.0: (-2, 15), 50.0: (-16, 7)}
    for x, y, value in zip(lpsp[zoom], deg[zoom], eps[zoom]):
        if float(value) in shown_zoom:
            ax.annotate(f"{value:g}", (x, y),
                        xytext=offsets.get(float(value), (5, 4)),
                        textcoords="offset points", fontsize=7, color="0.3")
    ax.scatter(lpsp[best], deg[best], marker="*", s=210, facecolor="#54a24b",
               edgecolor="k", linewidth=0.6, zorder=5,
               label=f"Minimum J@3 observe : epsilon={eps[best]:g}")
    i_eps3 = int(np.argmin(np.abs(eps - 3.0)))
    ax.scatter(lpsp[i_eps3], deg[i_eps3], marker="X", s=105,
               facecolor="#e45756", edgecolor="k", linewidth=0.5, zorder=5,
               label="Politique resolue avec epsilon=3")
    xline = np.linspace(max(0.0, float(lpsp[zoom].min()) - 0.03),
                        float(lpsp[zoom].max()) + 0.06, 250)
    ax.plot(xline, unif3[best] - slope_j3 * xline, ":", color="0.4", lw=1.2,
            label="Iso-cout tangent, VoLL=3")
    cb = fig.colorbar(sc_zoom, ax=ax)
    cb.set_label("log10(epsilon), epsilon en EUR/kWh")
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Cout de degradation [kEUR / 25 ans]")
    ax.set_title("Front PD V11-p=2 : zone de compromis")
    ax.set_ylim(float(deg[zoom].min()) - 0.45, float(deg[zoom].max()) + 0.55)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pareto_lpsp_deg_zoom_v11_p2.pdf")
    fig.savefig(output / "pareto_lpsp_deg_zoom_v11_p2.png", dpi=180)
    plt.close(fig)

    # Cout de reporting a VoLL=3 selon le poids epsilon de la resolution.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogx(eps, unif3, "o-", color="#4c78a8", label="PD rollout")
    ax.axhline(rb1["unif"], ls="--", color="#4c78a8",
               label=f"RB1 = {rb1['unif']:.2f} kEUR")
    ax.axhline(rb2["unif"], ls=":", color="#f58518",
               label=f"RB2 = {rb2['unif']:.2f} kEUR")
    ax.scatter(eps[best], unif3[best], s=110, color="#54a24b", zorder=3,
               label=f"Minimum epsilon={eps[best]:g}: {unif3[best]:.2f} kEUR")
    ax.set_xlabel("Poids epsilon dans la resolution PD [EUR/kWh]")
    ax.set_ylabel("Cout unifie evalue a VoLL=3 [kEUR / 25 ans]")
    ax.set_title("Selection du compromis PD V11-p=2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pareto_unif_vs_eps_v11_p2.pdf")
    fig.savefig(output / "pareto_unif_vs_eps_v11_p2.png", dpi=180)
    plt.close(fig)

    print(f"Source : {source}")
    print(f"Points : {len(eps)}, non domines : {int(nd.sum())}")
    print(f"Minimum J@VoLL3 : epsilon={eps[best]:g}, {unif3[best]:.3f} kEUR")
    print(f"Figures -> {output}/pareto_*_v11_p2.{{pdf,png}}")


if __name__ == "__main__":
    main()
