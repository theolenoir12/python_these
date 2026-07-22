"""Audit mecanique d'un smoke FLC V11-p=2 deja calcule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TOL = 1e-9


def _check(condition, message, failures):
    if not bool(condition):
        failures.append(message)


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    ledgers = json.loads((run_dir / "ledgers.json").read_text(encoding="utf-8"))
    failures = []
    diagnostics = {}
    expected_steps = int(summary["manifest"]["expected_steps"])

    _check(summary["manifest"]["ely_stress_exponent"] == 2.0,
           "l'exposant PEMWE nominal n'est pas p=2", failures)
    _check(summary["manifest"]["replacement_accounting"] == "corrected",
           "le ledger corrige n'est pas actif", failures)

    for label, result in summary["results"].items():
        with np.load(run_dir / f"{label}.npz", allow_pickle=False) as cache:
            arrays = {key: np.asarray(cache[key]) for key in cache.files}
        n = len(arrays["lol_tab"])
        _check(n == expected_steps, f"{label}: nombre de pas inattendu", failures)
        _check(result["steps"] == n, f"{label}: steps incoherent", failures)
        for key, values in arrays.items():
            _check(np.all(np.isfinite(values)), f"{label}: {key} non fini", failures)

        _check(len(arrays["SoC"]) == n + 1, f"{label}: taille SoC", failures)
        _check(len(arrays["E_h2"]) == n + 1, f"{label}: taille E_h2", failures)
        _check(float(np.min(arrays["SoC"])) >= 0.2 - TOL,
               f"{label}: SoC sous la borne", failures)
        _check(float(np.max(arrays["SoC"])) <= 0.995 + TOL,
               f"{label}: SoC au-dessus de la borne", failures)
        _check(float(np.min(arrays["E_h2"])) >= -TOL,
               f"{label}: stock H2 negatif", failures)
        _check(float(np.max(arrays["E_h2"])) <= 200.0 + TOL,
               f"{label}: stock H2 au-dessus de la capacite", failures)
        _check(float(np.min(arrays["lol_tab"])) >= -TOL,
               f"{label}: LOL negative", failures)
        _check(float(np.max(arrays["lol_tab"])) <= 1.0 + TOL,
               f"{label}: LOL superieure a 1", failures)

        simultaneous = (
            (np.abs(arrays["P_dc_fc"]) > TOL)
            & (np.abs(arrays["P_dc_ely"]) > TOL)
        )
        _check(not np.any(simultaneous),
               f"{label}: PEMFC et PEMWE actifs simultanement", failures)

        p_net = arrays["P_dc_load"] - arrays["P_dc_pv"]
        p_dispatch = (
            arrays["P_dc_bat"] + arrays["P_dc_fc"] + arrays["P_dc_ely"]
        )
        deficit = p_net > TOL
        direct_lol = np.zeros(n, dtype=float)
        direct_lol[deficit] = np.clip(
            (p_net[deficit] - p_dispatch[deficit]) / p_net[deficit], 0.0, 1.0
        )
        max_lol_error = float(np.max(np.abs(direct_lol - arrays["lol_tab"])))
        _check(max_lol_error <= 1e-8,
               f"{label}: LOL ne ferme pas le bilan deficit", failures)

        ledger = ledgers[label]
        for component in ("bat", "fc", "ely"):
            expected_total = (
                ledger["retired_eur"][component]
                + ledger["current_eur"][component]
            )
            _check(
                abs(ledger["total_eur"][component] - expected_total) <= TOL,
                f"{label}: identite ledger {component}", failures,
            )
        ledger_total = float(sum(ledger["total_eur"].values()))
        _check(abs(ledger_total - result["degradation_eur"]) <= TOL,
               f"{label}: total degradation incoherent", failures)
        _check(ledger["end_step_exclusive"] == n,
               f"{label}: fin de ledger incoherente", failures)

        diagnostics[label] = {
            "steps": n,
            "soc_min": float(np.min(arrays["SoC"])),
            "soc_max": float(np.max(arrays["SoC"])),
            "h2_min_kwh": float(np.min(arrays["E_h2"])),
            "h2_max_kwh": float(np.max(arrays["E_h2"])),
            "lol_max": float(np.max(arrays["lol_tab"])),
            "balance_lol_max_abs_error": max_lol_error,
            "simultaneous_fc_ely_steps": int(np.count_nonzero(simultaneous)),
            "ledger_total_eur": ledger_total,
        }

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
        f"# Audit smoke FLC — {report['status']}",
        "",
        f"Empreinte : `{report['fingerprint']}`.",
        "",
    ]
    if failures:
        lines += ["## Échecs", ""] + [f"- {item}" for item in failures]
    else:
        lines += [
            "Tous les contrôles de finitude, bornes SoC/H2, exclusivité FC/ELY,",
            "fermeture du déficit et identités du ledger corrigé passent.",
        ]
    (run_dir / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    report = audit(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
