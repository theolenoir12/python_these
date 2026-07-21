"""Visualise et audite les surfaces de commande de la FLC experte I0.

L'audit ne suppose pas qu'une defuzzification Mamdani conserve strictement la
monotonie de la table linguistique. Il mesure donc les inversions locales au
lieu de les masquer et les sauvegarde avec l'empreinte de la specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .flc_policy_v11 import make_expert_flc_policy_v11


EXPECTED_DIRECTIONS = {
    "deficit": {"h2": 1, "soc": -1, "severity": 1},
    "surplus": {"h2": -1, "soc": 1, "severity": 1},
}
AXES = {"h2": 0, "soc": 1, "severity": 2}


def _evaluate_cube(policy, branch, grid):
    cube = np.empty((len(grid), len(grid), len(grid)), dtype=float)
    for h2_i, h2 in enumerate(grid):
        for soc_i, soc in enumerate(grid):
            for severity_i, severity in enumerate(grid):
                cube[h2_i, soc_i, severity_i] = policy.command_fraction(
                    branch, severity, soc, h2
                )
    return cube


def _monotonicity_diagnostics(cube, branch, tolerance):
    diagnostics = {}
    for variable, direction in EXPECTED_DIRECTIONS[branch].items():
        delta = np.diff(cube, axis=AXES[variable])
        expected_delta = direction * delta
        violations = expected_delta < -tolerance
        diagnostics[variable] = {
            "expected_direction": "nondecreasing" if direction > 0 else "nonincreasing",
            "comparisons": int(expected_delta.size),
            "violation_count": int(np.count_nonzero(violations)),
            "violation_fraction": float(np.mean(violations)),
            "largest_local_reversal": float(
                max(0.0, -float(np.min(expected_delta)))
            ),
            "largest_step_in_expected_direction": float(
                max(0.0, float(np.max(expected_delta)))
            ),
        }
    return diagnostics


def analyze(grid_points=31, tolerance=1e-12, output_dir=None):
    if int(grid_points) < 11:
        raise ValueError("grid_points doit etre >= 11")
    policy = make_expert_flc_policy_v11()
    spec_sha256 = policy.flc_metadata["spec_sha256"]
    grid = np.linspace(0.0, 1.0, int(grid_points), dtype=float)
    cubes = {
        branch: _evaluate_cube(policy, branch, grid)
        for branch in ("deficit", "surplus")
    }

    report = {
        "policy_id": policy.policy_id,
        "spec_sha256": spec_sha256,
        "grid_points_per_axis": int(grid_points),
        "grid_evaluations_per_branch": int(grid_points) ** 3,
        "deadband_fraction": policy.flc_parameters["output_deadband"],
        "tolerance": float(tolerance),
        "branches": {},
    }
    for branch, cube in cubes.items():
        report["branches"][branch] = {
            "command_fraction_min": float(np.min(cube)),
            "command_fraction_max": float(np.max(cube)),
            "zero_fraction": float(np.mean(cube == 0.0)),
            "monotonicity": _monotonicity_diagnostics(
                cube, branch, tolerance
            ),
        }

    output_dir = (
        Path(__file__).resolve().parent / "figures"
        if output_dir is None else Path(output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"flc_surfaces_i0_{spec_sha256[:12]}"

    h2_targets = (0.1, 0.5, 0.9)
    figure, axes = plt.subplots(
        2, len(h2_targets), figsize=(12.0, 6.8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for row, branch in enumerate(("deficit", "surplus")):
        for column, h2_target in enumerate(h2_targets):
            h2_i = int(np.argmin(np.abs(grid - h2_target)))
            axis = axes[row, column]
            mesh = axis.pcolormesh(
                grid, grid, cubes[branch][h2_i],
                shading="nearest", vmin=0.0, vmax=1.0, cmap="viridis",
            )
            axis.set_title(
                f"{branch} — remplissage H2={grid[h2_i]:.2f}"
            )
            axis.set_xlabel("Severite de Pnet normalisee")
            if column == 0:
                axis.set_ylabel("SoC normalise")
    figure.colorbar(
        mesh, ax=axes, label="Fraction de commande apres zone morte",
        shrink=0.88,
    )
    figure.suptitle(
        "FLC Mamdani experte I0 — surfaces de commande",
        fontsize=13,
    )
    figure_path = output_dir / f"{stem}.png"
    report_path = output_dir / f"{stem}_diagnostics.json"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report["figure"] = str(figure_path.resolve())
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=31)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report_path, report = analyze(
        grid_points=args.grid_points,
        tolerance=args.tolerance,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Diagnostic sauvegarde dans {report_path}")


if __name__ == "__main__":
    main()
