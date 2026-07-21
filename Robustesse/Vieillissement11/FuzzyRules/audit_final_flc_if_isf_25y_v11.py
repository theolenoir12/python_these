"""Audit independant de la clôture FLC-IF/ISF sur 25 ans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_promoted_flc_25y_v11 import audit as audit_trajectories
from .run_promoted_flc_25y_v11 import ARRAY_KEYS


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    base = audit_trajectories(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    failures = list(base["failures"])
    parent = summary["parent"]
    null = next(item for item in summary["results"] if item["kind"] == "null")
    exact_by_array = {}
    with np.load(parent["trajectory"], allow_pickle=False) as parent_cache, np.load(
        run_dir / f"{null['candidate_id']}.npz", allow_pickle=False
    ) as null_cache:
        for key in ARRAY_KEYS:
            exact_by_array[key] = bool(np.array_equal(parent_cache[key], null_cache[key]))
    if not all(exact_by_array.values()):
        failures.append("test nul: trajectoire differente du parent")

    parent_metrics = dict(parent["metrics"])
    null_metrics = dict(null["metrics"])
    parent_metrics.pop("runtime_s", None)
    null_metrics.pop("runtime_s", None)
    metrics_exact = parent_metrics == null_metrics
    ledger_exact = parent["ledger"] == null["ledger"]
    if not metrics_exact:
        failures.append("test nul: metriques differentes")
    if not ledger_exact:
        failures.append("test nul: ledger different")

    iid_if = {
        item["parameters"]["noise_seed"]: item
        for item in summary["results"] if item["kind"] == "iid_if"
    }
    iid_isf = {
        item["parameters"]["noise_seed"]: item
        for item in summary["results"] if item["kind"] == "iid_isf"
    }
    crn_ok = set(iid_if) == set(iid_isf)
    if not crn_ok:
        failures.append("graines IF/ISF non appariees")
    for seed in sorted(set(iid_if) & set(iid_isf)):
        diag_if = iid_if[seed]["forecast_diagnostics"]
        diag_isf = iid_isf[seed]["forecast_diagnostics"]
        if (
            diag_if["noise_draws"] != diag_isf["noise_draws"]
            or diag_if["eps"] != diag_isf["eps"]
        ):
            failures.append(f"graine {seed}: suite de bruit IF/ISF incoherente")

    report = {
        **base,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "null_control": {
            "array_exact_by_key": exact_by_array,
            "metrics_exact_excluding_runtime": metrics_exact,
            "ledger_exact": ledger_exact,
        },
        "crn_if_isf": {
            "status": "PASS" if crn_ok else "FAIL",
            "seeds": sorted(set(iid_if) & set(iid_isf)),
        },
    }
    (run_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Audit final FLC-IF/ISF 25 ans — {report['status']}",
        "",
        f"Empreinte : `{report['fingerprint']}`.",
        "",
    ]
    if failures:
        lines.extend(["## Échecs", "", *[f"- {item}" for item in failures]])
    else:
        lines.extend([
            "Les trajectoires, profils, bilans, métriques et ledgers passent.",
            "Le test nul est bit-à-bit exact et les huit paires IF/ISF",
            "emploient les mêmes graines et suites de bruit.",
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
