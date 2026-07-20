"""Analyse reproductible du screening annuel MPC V11-p=2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DISPLAY = {
    "rb1_v11_p2_020_040": "RB1",
    "rb2_v11_p2_0574_0465": "RB2",
    "mpc_no_soh_h6": "MPC H6 sans SoH",
    "mpc_no_soh_h24": "MPC H24 sans SoH",
    "mpc_soh_fc1_h6": "MPC H6 SoH-FC",
    "mpc_soh_ely1_h6": "MPC H6 SoH-ELY",
    "mpc_soh_both1_h6": "MPC H6 SoH-FC+ELY",
    "mpc_soh_both1_h24": "MPC H24 SoH-FC+ELY",
}


def _nondominated(points: list[dict]) -> list[bool]:
    mask = []
    for i, point in enumerate(points):
        dominated = any(
            j != i
            and other["lpsp_pct"] <= point["lpsp_pct"] + 1e-12
            and other["degradation_keur"] <= point["degradation_keur"] + 1e-12
            and (
                other["lpsp_pct"] < point["lpsp_pct"] - 1e-12
                or other["degradation_keur"] < point["degradation_keur"] - 1e-12
            )
            for j, other in enumerate(points)
        )
        mask.append(not dominated)
    return mask


def _latest_run() -> Path:
    candidates = sorted(
        (HERE / "runs").glob("screen_1y_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("Aucun runs/screen_1y_* trouve ; fournir --run.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    run = args.run or _latest_run()
    protocol = json.loads((run / "protocol.json").read_text())
    summaries = json.loads((run / "summary.json").read_text())
    order = [config["label"] for config in protocol["configs"]]
    points = [
        {
            "label": label,
            "display": DISPLAY.get(label, label),
            **summaries[label],
        }
        for label in order if label in summaries
    ]
    if len(points) != len(order):
        raise SystemExit("Screening incomplet : summary.json ne contient pas tous les points.")
    mask = _nondominated(points)
    for point, keep in zip(points, mask):
        point["nondominated"] = keep

    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        "label\tdisplay\tlpsp_pct\tdegradation_keur\teens_kwh\t"
        "j_voll3_keur\tnondominated\twall_seconds"
    ]
    for point in points:
        rows.append(
            f"{point['label']}\t{point['display']}\t{point['lpsp_pct']:.10g}\t"
            f"{point['degradation_keur']:.10g}\t{point['eens_kwh']:.10g}\t"
            f"{point['j_voll3_keur']:.10g}\t{int(point['nondominated'])}\t"
            f"{point['wall_seconds']:.10g}"
        )
    (args.output / "screen_1y_pareto_points.tsv").write_text(
        "\n".join(rows) + "\n")

    front = sorted(
        [point for point in points if point["nondominated"]],
        key=lambda point: point["lpsp_pct"])
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(
        [point["lpsp_pct"] for point in front],
        [point["degradation_keur"] for point in front],
        color="#264653", lw=1.5, zorder=1, label="Front non domine")
    for point in points:
        is_rb = point["label"].startswith("rb")
        ax.scatter(
            point["lpsp_pct"], point["degradation_keur"],
            marker="s" if is_rb else "o", s=58,
            color="#6C757D" if is_rb else "#D1495B",
            edgecolor="white", linewidth=0.6, zorder=2)
        ax.annotate(
            point["display"],
            (point["lpsp_pct"], point["degradation_keur"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("LPSP (%) ? minimiser")
    ax.set_ylabel("Cout de degradation (kEUR) ? minimiser")
    ax.set_title("Screening MPC V11-p=2 ? un an, prevision parfaite")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "screen_1y_pareto.png", dpi=180)
    fig.savefig(args.output / "screen_1y_pareto.pdf")
    plt.close(fig)

    by_label = {point["label"]: point for point in points}
    h6 = by_label["mpc_no_soh_h6"]
    h24 = by_label["mpc_no_soh_h24"]
    soh24 = by_label["mpc_soh_both1_h24"]
    delta_j_h = 100.0 * (h24["j_voll3_keur"] / h6["j_voll3_keur"] - 1.0)
    delta_j_soh = 100.0 * (soh24["j_voll3_keur"] / h24["j_voll3_keur"] - 1.0)
    names = "\n".join(
        f"- {point['display']}" for point in points if point["nondominated"])
    report = f"""# Analyse du screening MPC V11-p=2

Source : {run}.

## Validite

Les {len(points)} points utilisent le meme profil sur {protocol['years']:g} an(s),
le modele {protocol['model_id']} et une prevision parfaite. Cette analyse ne
doit pas etre superposee a un front DP de 25 ans.

## Resultats

- Passage H6 vers H24 sans SoH : variation de J3 = {delta_j_h:+.3f} %.
- Effet SoH FC+ELY a H24 : variation de J3 = {delta_j_soh:+.3f} %.

## Front non domine du screening

{names}

La comparaison scientifique avec le DP doit etre effectuee avec
`compare_mpc_dp_v11.py` sur un horizon strictement identique.
"""
    (args.output / "ANALYSE_SCREEN_MPC_V11_P2.md").write_text(report)
    print(f"OK -> {args.output}")


if __name__ == "__main__":
    main()
