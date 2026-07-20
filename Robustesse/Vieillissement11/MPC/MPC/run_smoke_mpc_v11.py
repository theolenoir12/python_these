"""Smoke local court du MPC V11-p=2 sur la vraie boucle physique.

Par defaut, le script simule sept jours pour H=6 et H=24, execute le test nul
complet ``soh, beta=0 == no_soh`` pour H=6, et sauvegarde les caches pleine
precision dans un dossier empreinte sous ``runs/``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Common.cost_fcn_total2 import get_cost_from_ledger  # noqa: E402
from Common.degradation_v11 import MODEL_ID  # noqa: E402
from Common.main_init_and_loop import init_and_run_loop  # noqa: E402
from Common.rb1_policy_v11 import make_rb1_policy_v11  # noqa: E402
from Common.rb2_policy import make_rb2_policy  # noqa: E402
from Common.reliability_metrics import compute_reliability_metrics  # noqa: E402
from MPC.mpc_v11 import MPCConfig, MPCPolicyV11  # noqa: E402


VOLL_REPORTING = 3.0
ARRAY_KEYS = (
    "temps", "SoC", "E_h2", "P_bat", "P_fc", "P_ely", "P_dc_load",
    "P_dc_pv", "P_dc_bat", "P_dc_fc", "P_dc_ely", "lol_tab",
    "SoH_bat", "SoH_fc", "SoH_ely", "alpha_fc", "alpha_ely",
)


def summarize(data: dict, wall_seconds: float, diagnostics: dict | None) -> dict:
    reliability = compute_reliability_metrics(data)
    degradation_eur = get_cost_from_ledger(data)
    p_ref = np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"])
    deficit = p_ref > 1e-12
    balance = np.zeros_like(p_ref, dtype=float)
    balance[deficit] = (
        np.asarray(data["P_dc_bat"])[deficit]
        + np.asarray(data["P_dc_fc"])[deficit]
        + np.asarray(data["P_dc_ely"])[deficit]
        + np.asarray(data["lol_tab"])[deficit] * p_ref[deficit]
        - p_ref[deficit]
    )
    return {
        "n_steps": int(data["n"]),
        "wall_seconds": float(wall_seconds),
        "degradation_keur": degradation_eur / 1000.0,
        "eens_kwh": reliability["eens_kwh"],
        "lpsp_pct": reliability["lpsp_pct"],
        "demand_kwh": reliability["load_energy_kwh"],
        "j_voll3_keur": degradation_eur / 1000.0
        + VOLL_REPORTING * reliability["eens_kwh"] / 1000.0,
        "max_deficit_balance_residual_w": float(np.max(np.abs(balance), initial=0.0)),
        "max_lol": float(np.max(data["lol_tab"], initial=0.0)),
        "ledger_events": len(data["degradation_ledger"]["events"]),
        "diagnostics": diagnostics,
    }


def save_run(output: Path, label: str, data: dict) -> None:
    np.savez_compressed(
        output / f"{label}.npz",
        model_id=np.array(MODEL_ID),
        ely_stress_exponent=np.array(2.0),
        **{key: np.asarray(data[key]) for key in ARRAY_KEYS},
    )
    (output / f"{label}_ledger.json").write_text(
        json.dumps(data["degradation_ledger"], indent=2) + "\n")


def assert_null_exact(a: dict, b: dict) -> None:
    for key in ARRAY_KEYS:
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
            raise AssertionError(f"test nul non exact pour {key}")
    if a["degradation_ledger"] != b["degradation_ledger"]:
        raise AssertionError("test nul non exact pour le ledger")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 24])
    args = parser.parse_args()
    if args.days <= 0.0 or any(h < 2 for h in args.horizons):
        raise SystemExit("jours > 0 et horizons >= 2 requis")

    protocol = {
        "model_id": MODEL_ID,
        "ely_stress_exponent": 2.0,
        "days": float(args.days),
        "horizons": list(dict.fromkeys(args.horizons)),
        "forecast_mode": "perfect",
        "voll_reporting": VOLL_REPORTING,
        "rb1_parameters": [0.20, 0.40],
        "rb2_parameters": [0.574, 0.465],
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = HERE / "runs" / f"smoke_{args.days:g}d_{fingerprint}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    years = float(args.days) / 365.0
    summaries: dict[str, dict] = {}

    for label, policy in (
        ("rb1_v11_p2_020_040", make_rb1_policy_v11(0.20, 0.40)),
        ("rb2_v11_p2_0574_0465", make_rb2_policy(0.574, 0.465)),
    ):
        started = time.perf_counter()
        data = init_and_run_loop(policy, n_years=years)
        wall = time.perf_counter() - started
        summaries[label] = summarize(data, wall, None)
        save_run(output, label, data)
        print(label, summaries[label], flush=True)

    base_data: dict[int, dict] = {}
    for horizon in protocol["horizons"]:
        config = MPCConfig(
            horizon_steps=horizon, forecast_mode="perfect",
            health_mode="no_soh", beta_fc=0.0, beta_ely=0.0,
        )
        policy = MPCPolicyV11(config)
        started = time.perf_counter()
        data = init_and_run_loop(policy, n_years=years)
        wall = time.perf_counter() - started
        label = f"mpc_v11_p2_no_soh_h{horizon}_perfect"
        summaries[label] = summarize(data, wall, policy.diagnostics())
        save_run(output, label, data)
        base_data[horizon] = data
        print(label, summaries[label], flush=True)

    # Test nul complet sur le premier horizon demande : memes trajectoires et ledger.
    null_horizon = protocol["horizons"][0]
    null_config = MPCConfig(
        horizon_steps=null_horizon, forecast_mode="perfect",
        health_mode="soh", beta_fc=0.0, beta_ely=0.0,
    )
    null_policy = MPCPolicyV11(null_config)
    started = time.perf_counter()
    null_data = init_and_run_loop(null_policy, n_years=years)
    null_wall = time.perf_counter() - started
    assert_null_exact(base_data[null_horizon], null_data)
    null_label = f"mpc_v11_p2_soh_null_h{null_horizon}_perfect"
    summaries[null_label] = summarize(null_data, null_wall, null_policy.diagnostics())
    summaries[null_label]["null_exact"] = True
    save_run(output, null_label, null_data)

    for label, summary in summaries.items():
        if summary["max_lol"] > 1.0 + 1e-9:
            raise AssertionError(f"{label}: lol brut > 1")
        if summary["max_deficit_balance_residual_w"] > 1e-5:
            raise AssertionError(f"{label}: bilan deficit non ferme")
        diagnostics = summary.get("diagnostics")
        if diagnostics and diagnostics["failures"]:
            raise AssertionError(f"{label}: echec solveur")

    result = {
        "protocol": protocol,
        "fingerprint": fingerprint,
        "output": str(output),
        "runs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "Smoke MPC V11-p=2 -- resultats non scientifiques",
        f"empreinte={fingerprint} ; jours={args.days:g} ; test nul exact=True",
        "label ; steps ; LPSP_pct ; deg_kEUR ; EENS_kWh ; J3_kEUR ; wall_s ; mean_solve_ms ; max_lol",
    ]
    for label, summary in summaries.items():
        diagnostics = summary.get("diagnostics") or {}
        lines.append(
            f"{label} ; {summary['n_steps']} ; {summary['lpsp_pct']:.8f} ; "
            f"{summary['degradation_keur']:.8f} ; {summary['eens_kwh']:.8f} ; "
            f"{summary['j_voll3_keur']:.8f} ; {summary['wall_seconds']:.3f} ; "
            f"{1000.0 * diagnostics.get('mean_solve_seconds', 0.0):.3f} ; "
            f"{summary['max_lol']:.9f}"
        )
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"OK -- smoke et test nul -> {output}")


if __name__ == "__main__":
    main()
