"""Smoke reproductible de la FLC experte face aux references V11-p=2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
V11 = HERE.parent
if str(V11) not in sys.path:
    sys.path.insert(0, str(V11))

from Common import Init_EMR_MG_v16_python as I
from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.main_init_and_loop import init_and_run_loop
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy
from Common.reliability_metrics import compute_reliability_metrics
from FuzzyRules.flc_policy_v11 import make_expert_flc_policy_v11


VOLL = 3.0
ARRAY_KEYS = (
    "temps", "SoC", "E_h2", "P_bat", "P_fc", "P_ely",
    "P_dc_load", "P_dc_pv", "P_dc_bat", "P_dc_fc", "P_dc_ely",
    "lol_tab", "SoH_bat", "SoH_fc", "SoH_ely",
)


def _starts(power):
    on = np.abs(np.asarray(power, dtype=float)) > 1e-9
    return int(np.count_nonzero(on[1:] & ~on[:-1]))


def _profile_hash(n_steps):
    digest = hashlib.sha256()
    digest.update(np.asarray(I.LOAD["P_ref"][:n_steps], dtype=np.float64).tobytes())
    digest.update(np.asarray(I.PV["P"][:n_steps], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _evaluate(policy, years):
    started = perf_counter()
    data = init_and_run_loop(
        policy, n_years=years, replacement_accounting="corrected"
    )
    reliability = compute_reliability_metrics(data)
    components = data["degradation_ledger"]["total_eur"]
    degradation = float(sum(components.values()))
    summary = {
        "steps": int(len(data["lol_tab"])),
        "eens_kwh": float(reliability["eens_kwh"]),
        "lpsp_pct": float(reliability["lpsp_pct"]),
        "load_energy_kwh": float(reliability["load_energy_kwh"]),
        "degradation_eur": degradation,
        "unified_voll3_eur": degradation + VOLL * reliability["eens_kwh"],
        "degradation_components_eur": {
            key: float(value) for key, value in components.items()
        },
        "fc_starts": _starts(data["P_fc"]),
        "ely_starts": _starts(data["P_ely"]),
        "runtime_s": perf_counter() - started,
    }
    arrays = {key: np.asarray(data[key]) for key in ARRAY_KEYS}
    return summary, arrays, data["degradation_ledger"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=7.0)
    args = parser.parse_args()
    if not 0.0 < args.days <= 365.0:
        raise ValueError("--days doit appartenir a ]0, 365]")
    years = args.days / 365.0

    flc = make_expert_flc_policy_v11()
    policies = {
        "flc_mamdani_expert_v11_p2_i0_v1": flc,
        "rb1_v11_p2_020_040": make_rb1_policy_v11(0.20, 0.40),
        "rb2_v11_p2_0574_0465": make_rb2_policy(0.574, 0.465),
    }
    expected_steps = max(0, int(round(args.days * 24.0)) - 1)
    manifest = {
        "protocol_id": "smoke-flc-v11-p2-i0-v1-2026-07-21",
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "voll_eur_per_kwh": VOLL,
        "days_requested": float(args.days),
        "years_argument": years,
        "expected_steps": expected_steps,
        "profile_sha256": _profile_hash(expected_steps),
        "flc_policy_id": flc.policy_id,
        "flc_spec_sha256": flc.flc_metadata["spec_sha256"],
        "flc_parameters": flc.flc_parameters,
        "references": {
            "rb1": {"soc_low": 0.20, "soc_high": 0.40},
            "rb2": {"fc_setpoint": 0.574, "ely_setpoint": 0.465},
        },
        "replacement_accounting": "corrected",
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    run_dir = HERE / "runs" / f"smoke_flc_i0_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    ledgers = {}
    for label, policy in policies.items():
        summary, arrays, ledger = _evaluate(policy, years)
        if summary["steps"] != expected_steps:
            raise AssertionError(
                f"{label}: {summary['steps']} pas, attendu {expected_steps}"
            )
        results[label] = summary
        ledgers[label] = ledger
        np.savez_compressed(run_dir / f"{label}.npz", **arrays)
        print(
            f"{label:36s} J3={summary['unified_voll3_eur']:10.4f} EUR "
            f"deg={summary['degradation_eur']:10.4f} "
            f"EENS={summary['eens_kwh']:8.4f} kWh "
            f"starts={summary['fc_starts']}/{summary['ely_starts']} "
            f"({summary['runtime_s']:.2f}s)",
            flush=True,
        )

    output = {"manifest": manifest, "fingerprint": fingerprint, "results": results}
    (run_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "ledgers.json").write_text(
        json.dumps(ledgers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
