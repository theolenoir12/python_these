"""Audit independant de la promotion FLC-IS SoH sur 25 ans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_promoted_flc_25y_v11 import audit as audit_trajectories
from .run_promoted_flc_25y_v11 import ARRAY_KEYS


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    base_report = audit_trajectories(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = summary["manifest"]
    null_result = summary["null_control"]
    parent = summary["parent"]
    failures = list(base_report["failures"])

    null_path = run_dir / f"{null_result['candidate_id']}.npz"
    parent_path = Path(parent["trajectory"])
    exact_by_array = {}
    with np.load(parent_path, allow_pickle=False) as parent_cache, np.load(
        null_path, allow_pickle=False
    ) as null_cache:
        for key in ARRAY_KEYS:
            exact_by_array[key] = bool(
                np.array_equal(parent_cache[key], null_cache[key])
            )
    for key, exact in exact_by_array.items():
        if not exact:
            failures.append(f"test nul: trajectoire {key} differente du parent")

    parent_metrics = dict(parent["metrics"])
    null_metrics = dict(null_result["metrics"])
    parent_metrics.pop("runtime_s", None)
    null_metrics.pop("runtime_s", None)
    metrics_exact = parent_metrics == null_metrics
    ledger_exact = parent["ledger"] == null_result["ledger"]
    if not metrics_exact:
        failures.append("test nul: metriques differentes du parent")
    if not ledger_exact:
        failures.append("test nul: ledger different du parent")
    if manifest["null_control_candidate_id"] != null_result["candidate_id"]:
        failures.append("identifiant du test nul incoherent")

    report = {
        **base_report,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "null_control": {
            "status": "PASS" if (
                all(exact_by_array.values()) and metrics_exact and ledger_exact
            ) else "FAIL",
            "array_exact_by_key": exact_by_array,
            "metrics_exact_excluding_runtime": metrics_exact,
            "ledger_exact": ledger_exact,
        },
    }
    (run_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Audit FLC-IS SoH 25 ans — {report['status']}",
        "",
        f"Empreinte : `{report['fingerprint']}`.",
        "",
    ]
    if failures:
        lines.extend(["## Échecs", "", *[f"- {item}" for item in failures]])
    else:
        lines.extend([
            "Les contrôles de trajectoire, bilan, métriques et ledgers passent.",
            "Le contrôle nul reproduit bit-à-bit les quinze tableaux du parent I0,",
            "ainsi que ses métriques hors temps de calcul et son ledger.",
        ])
    (run_dir / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
