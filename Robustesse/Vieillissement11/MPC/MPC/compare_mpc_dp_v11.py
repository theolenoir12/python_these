"""Compare un front MPC et le front DP V11 sans melanger les horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def _years_from_npz(path: Path, data: np.lib.npyio.NpzFile) -> float | None:
    if "years" in data.files:
        return float(np.asarray(data["years"]).item())
    for text in (path.name, path.parent.name):
        match = re.search(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)y(?:_|\b)", text)
        if match:
            return float(match.group(1))
    return None


def _nondominated(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominates_i = (
            (x <= x[i] + 1e-12) & (y <= y[i] + 1e-12)
            & ((x < x[i] - 1e-12) | (y < y[i] - 1e-12))
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            mask[i] = False
    return mask


def _dominates(ax: float, ay: float, bx: float, by: float) -> bool:
    return (
        ax <= bx + 1e-12 and ay <= by + 1e-12
        and (ax < bx - 1e-12 or ay < by - 1e-12)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpc-run", type=Path, required=True,
                        help="dossier contenant protocol.json et summary.json")
    parser.add_argument("--dp", type=Path, required=True, help="front DP .npz")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()

    protocol = json.loads((args.mpc_run / "protocol.json").read_text())
    summaries = json.loads((args.mpc_run / "summary.json").read_text())
    mpc_years = float(protocol["years"])

    with np.load(args.dp, allow_pickle=False) as dp:
        dp_years = _years_from_npz(args.dp, dp)
        if dp_years is None:
            raise SystemExit(
                "Horizon DP introuvable : utiliser le nouveau lanceur ou un nom contenant '<N>y'.")
        if not np.isclose(dp_years, mpc_years):
            raise SystemExit(
                f"Comparaison refusee : MPC={mpc_years:g} ans, DP={dp_years:g} ans.")
        dp_x = np.asarray(dp["lpsp"], dtype=float)
        dp_y = np.asarray(dp["deg_keur"], dtype=float)
        dp_eps = np.asarray(dp["eps"], dtype=float)
        dp_model = str(np.asarray(dp["model_id"]).item()) if "model_id" in dp.files else None

    mpc_model = protocol.get("model_id")
    if dp_model is not None and mpc_model is not None and dp_model != mpc_model:
        raise SystemExit(f"model_id incompatible : MPC={mpc_model}, DP={dp_model}")

    configs = {item["label"]: item for item in protocol.get("configs", [])}
    points = []
    for label, result in summaries.items():
        config = configs.get(label, {})
        if config.get("kind", "mpc") != "mpc":
            continue
        points.append({
            "label": label,
            "lpsp_pct": float(result["lpsp_pct"]),
            "degradation_keur": float(result["degradation_keur"]),
            "forecast_mode": config.get(
                "forecast_mode", protocol.get("forecast_mode", "unknown")),
        })
    if not points:
        raise SystemExit("Aucun point MPC dans summary.json")

    dp_nd = _nondominated(dp_x, dp_y)
    front_x, front_y, front_eps = dp_x[dp_nd], dp_y[dp_nd], dp_eps[dp_nd]
    order = np.argsort(front_x)
    front_x, front_y, front_eps = front_x[order], front_y[order], front_eps[order]

    for point in points:
        x, y = point["lpsp_pct"], point["degradation_keur"]
        dominated_by = [
            float(eps) for eps, dx, dy in zip(front_eps, front_x, front_y)
            if _dominates(float(dx), float(dy), x, y)
        ]
        dominates = [
            float(eps) for eps, dx, dy in zip(front_eps, front_x, front_y)
            if _dominates(x, y, float(dx), float(dy))
        ]
        point["dominated_by_dp"] = bool(dominated_by)
        point["dominating_dp_eps"] = dominated_by
        point["n_dp_front_points_dominated"] = len(dominates)
        point["dominates_entire_dp_front"] = len(dominates) == len(front_x)

    args.output.mkdir(parents=True, exist_ok=True)
    tag = f"{mpc_years:g}y"
    tsv = [
        "label\tforecast_mode\tlpsp_pct\tdegradation_keur\tdominated_by_dp\t"
        "n_dp_front_points_dominated\tdominates_entire_dp_front"
    ]
    for point in points:
        tsv.append(
            f"{point['label']}\t{point['forecast_mode']}\t{point['lpsp_pct']:.10g}\t"
            f"{point['degradation_keur']:.10g}\t{int(point['dominated_by_dp'])}\t"
            f"{point['n_dp_front_points_dominated']}\t"
            f"{int(point['dominates_entire_dp_front'])}"
        )
    (args.output / f"dp_mpc_comparison_{tag}.tsv").write_text("\n".join(tsv) + "\n")
    (args.output / f"dp_mpc_comparison_{tag}.json").write_text(
        json.dumps({
            "years": mpc_years, "dp_file": str(args.dp),
            "dp_front_points": int(len(front_x)), "mpc_points": points,
        }, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.plot(front_x, front_y, "-o", color="#243B6B", lw=1.8, ms=4,
            label="Front DP V11")
    colors = {"perfect": "#D1495B", "noisy": "#E69F00", "persistence": "#6A4C93"}
    for point in points:
        ax.scatter(
            point["lpsp_pct"], point["degradation_keur"], s=52,
            color=colors.get(point["forecast_mode"], "#2A9D8F"),
            edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(point["label"], (point["lpsp_pct"], point["degradation_keur"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("LPSP (%) ? minimiser")
    ax.set_ylabel("Cout de degradation (kEUR) ? minimiser")
    ax.set_title(f"Front DP et points MPC V11 ? horizon commun {mpc_years:g} an(s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / f"dp_mpc_pareto_{tag}.png", dpi=180)
    fig.savefig(args.output / f"dp_mpc_pareto_{tag}.pdf")
    plt.close(fig)

    global_dominators = [
        point["label"] for point in points if point["dominates_entire_dp_front"]]
    if global_dominators:
        raise RuntimeError(
            "Un point MPC domine tout le front DP : verifier formulations/grilles : "
            + ", ".join(global_dominators))
    print(f"OK -- aucun MPC ne domine globalement le front DP -> {args.output}")


if __name__ == "__main__":
    main()
