"""Audit independant des trajectoires FLC promues sur 25 ans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Common.reliability_metrics import compute_reliability_metrics


TOL = 1e-9


def _check(condition, message, failures):
    if not bool(condition):
        failures.append(message)


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = summary["manifest"]
    failures = []
    diagnostics = {}
    reference_path = Path(manifest["reference_sources"]["trajectories"])
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_load = np.asarray(reference["RB1_p2_tuned__P_dc_load"])
        reference_pv = np.asarray(reference["RB1_p2_tuned__P_dc_pv"])

    _check(manifest["years"] == 25.0, "horizon different de 25 ans", failures)
    _check(manifest["ely_stress_exponent"] == 2.0, "exposant PEMWE different de 2", failures)
    _check(manifest["replacement_accounting"] == "corrected", "ledger non corrige", failures)

    for result in summary["results"]:
        label = result["candidate_id"]
        with np.load(run_dir / f"{label}.npz", allow_pickle=False) as cache:
            arrays = {key: np.asarray(cache[key]) for key in cache.files}
        n = len(arrays["lol_tab"])
        local_failures = []
        _check(n == manifest["expected_steps"], f"{label}: nombre de pas", local_failures)
        for key, values in arrays.items():
            _check(np.all(np.isfinite(values)), f"{label}: {key} non fini", local_failures)
        _check(np.array_equal(arrays["P_dc_load"], reference_load),
               f"{label}: charge differente de la reference", local_failures)
        _check(np.array_equal(arrays["P_dc_pv"], reference_pv),
               f"{label}: PV different de la reference", local_failures)
        _check(np.min(arrays["SoC"]) >= 0.2 - TOL, f"{label}: SoC min", local_failures)
        _check(np.max(arrays["SoC"]) <= 0.995 + TOL, f"{label}: SoC max", local_failures)
        _check(np.min(arrays["E_h2"]) >= -TOL, f"{label}: H2 min", local_failures)
        _check(np.max(arrays["E_h2"]) <= 200.0 + TOL, f"{label}: H2 max", local_failures)
        _check(np.min(arrays["lol_tab"]) >= -TOL, f"{label}: LOL min", local_failures)
        _check(np.max(arrays["lol_tab"]) <= 1.0 + TOL, f"{label}: LOL max", local_failures)
        simultaneous = (
            (np.abs(arrays["P_dc_fc"]) > TOL)
            & (np.abs(arrays["P_dc_ely"]) > TOL)
        )
        _check(not np.any(simultaneous), f"{label}: FC/ELY simultanes", local_failures)

        p_net = arrays["P_dc_load"] - arrays["P_dc_pv"]
        dispatch = arrays["P_dc_bat"] + arrays["P_dc_fc"] + arrays["P_dc_ely"]
        direct_lol = np.zeros(n, dtype=float)
        deficit = p_net > TOL
        direct_lol[deficit] = np.clip(
            (p_net[deficit] - dispatch[deficit]) / p_net[deficit], 0.0, 1.0
        )
        balance_error = float(np.max(np.abs(direct_lol - arrays["lol_tab"])))
        _check(balance_error <= 1e-8, f"{label}: fermeture deficit", local_failures)

        reliability = compute_reliability_metrics(arrays)
        metrics = result["metrics"]
        _check(abs(reliability["eens_kwh"] - metrics["eens_kwh"]) <= 1e-8,
               f"{label}: EENS non reproductible", local_failures)
        _check(abs(reliability["lpsp_pct"] - metrics["lpsp_pct"]) <= 1e-10,
               f"{label}: LPSP non reproductible", local_failures)
        ledger = result["ledger"]
        for component in ("bat", "fc", "ely"):
            expected = ledger["retired_eur"][component] + ledger["current_eur"][component]
            _check(abs(ledger["total_eur"][component] - expected) <= TOL,
                   f"{label}: ledger {component}", local_failures)
        ledger_total = float(sum(ledger["total_eur"].values()))
        _check(abs(ledger_total - metrics["degradation_eur"]) <= TOL,
               f"{label}: degradation non reproductible", local_failures)

        diagnostics[label] = {
            "status": "PASS" if not local_failures else "FAIL",
            "failures": local_failures,
            "steps": n,
            "balance_lol_max_abs_error": balance_error,
            "simultaneous_fc_ely_steps": int(np.count_nonzero(simultaneous)),
            "soc_min": float(np.min(arrays["SoC"])),
            "soc_max": float(np.max(arrays["SoC"])),
            "h2_min_kwh": float(np.min(arrays["E_h2"])),
            "h2_max_kwh": float(np.max(arrays["E_h2"])),
        }
        failures.extend(local_failures)

    report = {
        "run_dir": str(run_dir),
        "fingerprint": summary["fingerprint"],
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "diagnostics": diagnostics,
    }
    (run_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Audit FLC promues 25 ans — {report['status']}",
        "",
        f"Empreinte : `{report['fingerprint']}`.",
        "",
    ]
    if failures:
        lines.extend(["## Échecs", "", *[f"- {item}" for item in failures]])
    else:
        lines.extend([
            "Les profils sont bit-à-bit identiques aux références V11-p=2.",
            "Les contrôles de finitude, bornes, bilan, exclusivité FC/ELY,",
            "recalcul EENS/LPSP et identités des ledgers passent.",
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
