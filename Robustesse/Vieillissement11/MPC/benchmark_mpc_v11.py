"""Screening attribuable d'un an du MPC V11-p=2.

Le protocole compare H=6/H=24 sans ponderation SoH et sonde, a beta=1, les
canaux FC, ELY et combines. Il s'agit d'un screening de formulation, pas encore
du reglage final. Chaque trajectoire est sauvegardee immediatement dans un cache
empreinte ; une relance reprend les points deja termines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Common.cost_fcn_total2 import get_cost_from_ledger  # noqa: E402
from Common.degradation_v11 import MODEL_ID  # noqa: E402
from Common.main_init_and_loop import init_and_run_loop  # noqa: E402
from Common.rb1_policy_v11 import make_rb1_policy_v11  # noqa: E402
from Common.rb2_policy import make_rb2_policy  # noqa: E402
from Common.reliability_metrics import compute_reliability_metrics  # noqa: E402
from MPC.mpc_v11 import DT_H, MPCConfig, MPCPolicyV11  # noqa: E402


VOLL_REPORTING = 3.0
ARRAY_KEYS = (
    "temps", "SoC", "E_h2", "P_bat", "P_fc", "P_ely", "P_dc_load",
    "P_dc_pv", "P_dc_bat", "P_dc_fc", "P_dc_ely", "lol_tab",
    "SoH_bat", "SoH_fc", "SoH_ely", "alpha_fc", "alpha_ely",
    "SoH_fc_operando", "SoH_ely_operando",
)


def _screen_configs() -> list[dict[str, Any]]:
    return [
        {"label": "rb1_v11_p2_020_040", "kind": "rb1"},
        {"label": "rb2_v11_p2_0574_0465", "kind": "rb2"},
        {"label": "mpc_no_soh_h6", "kind": "mpc", "horizon_steps": 6,
         "health_mode": "no_soh", "beta_fc": 0.0, "beta_ely": 0.0},
        {"label": "mpc_no_soh_h24", "kind": "mpc", "horizon_steps": 24,
         "health_mode": "no_soh", "beta_fc": 0.0, "beta_ely": 0.0},
        {"label": "mpc_soh_fc1_h6", "kind": "mpc", "horizon_steps": 6,
         "health_mode": "soh", "beta_fc": 1.0, "beta_ely": 0.0},
        {"label": "mpc_soh_ely1_h6", "kind": "mpc", "horizon_steps": 6,
         "health_mode": "soh", "beta_fc": 0.0, "beta_ely": 1.0},
        {"label": "mpc_soh_both1_h6", "kind": "mpc", "horizon_steps": 6,
         "health_mode": "soh", "beta_fc": 1.0, "beta_ely": 1.0},
        {"label": "mpc_soh_both1_h24", "kind": "mpc", "horizon_steps": 24,
         "health_mode": "soh", "beta_fc": 1.0, "beta_ely": 1.0},
    ]


def _summary(data: dict, wall_seconds: float, diagnostics: dict | None) -> dict:
    reliability = compute_reliability_metrics(data)
    degradation_eur = get_cost_from_ledger(data)
    p_ref = np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"])
    lol = np.asarray(data["lol_tab"], dtype=float)
    deficit = p_ref > 1e-12
    balance = np.zeros_like(p_ref, dtype=float)
    balance[deficit] = (
        np.asarray(data["P_dc_bat"])[deficit]
        + np.asarray(data["P_dc_fc"])[deficit]
        + np.asarray(data["P_dc_ely"])[deficit]
        + lol[deficit] * p_ref[deficit] - p_ref[deficit]
    )
    residual_kw = np.clip(p_ref / 1000.0, 0.0, None)
    excess_kwh = float((residual_kw * np.clip(lol - 1.0, 0.0, None)).sum())
    deficit_balance = balance[deficit]
    shortage_after_lol = np.clip(-deficit_balance, 0.0, None)
    implicit_curtailment = np.clip(deficit_balance, 0.0, None)
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
        "max_deficit_shortage_after_lol_w": float(
            np.max(shortage_after_lol, initial=0.0)),
        "max_implicit_curtailment_w": float(
            np.max(implicit_curtailment, initial=0.0)),
        "implicit_curtailment_kwh": float(
            np.sum(implicit_curtailment) / 1000.0 * DT_H),
        "implicit_curtailment_steps": int(
            np.count_nonzero(implicit_curtailment > 1e-4)),
        "max_lol": float(np.max(lol, initial=0.0)),
        "lol_above_one_steps": int(np.count_nonzero(lol > 1.0 + 1e-9)),
        "excess_beyond_clip_kwh": excess_kwh,
        "ledger_events": len(data["degradation_ledger"]["events"]),
        "diagnostics": diagnostics,
    }


def _save(output: Path, label: str, data: dict, summary: dict,
          config: dict[str, Any]) -> None:
    np.savez_compressed(
        output / f"{label}.npz",
        model_id=np.array(MODEL_ID), ely_stress_exponent=np.array(2.0),
        label=np.array(label), config_json=np.array(json.dumps(config, sort_keys=True)),
        **{key: np.asarray(data[key]) for key in ARRAY_KEYS},
    )
    (output / f"{label}_ledger.json").write_text(
        json.dumps(data["degradation_ledger"], indent=2) + "\n")
    (output / f"{label}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


def _run_one(job: tuple[dict[str, Any], float, str]) -> tuple[str, dict]:
    config, years, output_text = job
    output = Path(output_text)
    label = config["label"]
    if config["kind"] == "rb1":
        policy = make_rb1_policy_v11(0.20, 0.40)
        diagnostics = None
    elif config["kind"] == "rb2":
        policy = make_rb2_policy(0.574, 0.465)
        diagnostics = None
    else:
        forecast_keys = (
            "forecast_mode", "forecast_seed",
            "forecast_sigma_energy_kwh_18h", "forecast_bias_energy_kwh_18h",
            "forecast_error_rho", "forecast_sigma_scale",
            "time_limit_s", "mip_rel_gap",
        )
        forecast_args = {
            key: config[key] for key in forecast_keys if key in config
        }
        mpc_config = MPCConfig(
            horizon_steps=int(config["horizon_steps"]),
            health_mode=str(config["health_mode"]),
            beta_fc=float(config["beta_fc"]), beta_ely=float(config["beta_ely"]),
            **forecast_args,
        )
        policy = MPCPolicyV11(mpc_config)

    started = time.perf_counter()
    data = init_and_run_loop(policy, n_years=years)
    wall = time.perf_counter() - started
    if config["kind"] == "mpc":
        diagnostics = policy.diagnostics()
    summary = _summary(data, wall, diagnostics)
    _save(output, label, data, summary, config)
    return label, summary


def _write_table(output: Path, results: dict[str, dict], configs: list[dict]) -> None:
    lines = [
        "Screening MPC V11-p=2 -- prevision exacte a horizon fini -- 1 an",
        "label ; LPSP_pct ; deg_kEUR ; EENS_kWh ; J3_kEUR ; wall_s ; mean_solve_ms ; failures ; excess_clip_kWh",
    ]
    for config in configs:
        label = config["label"]
        if label not in results:
            continue
        result = results[label]
        diagnostics = result.get("diagnostics") or {}
        lines.append(
            f"{label} ; {result['lpsp_pct']:.8f} ; {result['degradation_keur']:.8f} ; "
            f"{result['eens_kwh']:.8f} ; {result['j_voll3_keur']:.8f} ; "
            f"{result['wall_seconds']:.3f} ; "
            f"{1000.0 * diagnostics.get('mean_solve_seconds', 0.0):.3f} ; "
            f"{diagnostics.get('failures', 0)} ; {result['excess_beyond_clip_kwh']:.8f}"
        )
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    if args.years <= 0.0:
        raise SystemExit("years doit etre positif")

    configs = _screen_configs()
    if args.only:
        wanted = set(args.only)
        configs = [config for config in configs if config["label"] in wanted]
        missing = wanted - {config["label"] for config in configs}
        if missing:
            raise SystemExit("labels inconnus : " + ", ".join(sorted(missing)))
    protocol = {
        "model_id": MODEL_ID,
        "ely_stress_exponent": 2.0,
        "years": args.years,
        "forecast_mode": "perfect",
        "voll_reporting": VOLL_REPORTING,
        "configs": configs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = HERE / "runs" / f"screen_{args.years:g}y_{fingerprint}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    results: dict[str, dict] = {}
    pending = []
    for config in configs:
        cached = output / f"{config['label']}_summary.json"
        trajectory = output / f"{config['label']}.npz"
        ledger = output / f"{config['label']}_ledger.json"
        if cached.exists() and trajectory.exists() and ledger.exists():
            results[config["label"]] = json.loads(cached.read_text())
            print(f"[cache] {config['label']}", flush=True)
        else:
            pending.append((config, args.years, str(output)))
    _write_table(output, results, configs)

    workers = max(1, min(int(args.workers), len(pending) or 1))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_one, job): job[0]["label"] for job in pending}
        for future in as_completed(futures):
            label, result = future.result()
            results[label] = result
            _write_table(output, results, configs)
            print(
                f"[{label}] LPSP={result['lpsp_pct']:.4f}% "
                f"deg={result['degradation_keur']:.3f} kEUR "
                f"J3={result['j_voll3_keur']:.3f} kEUR "
                f"wall={result['wall_seconds']:.0f}s",
                flush=True,
            )

    if len(results) != len(configs):
        raise RuntimeError("screening incomplet")
    for label, result in results.items():
        diagnostics = result.get("diagnostics") or {}
        if diagnostics.get("failures", 0):
            raise RuntimeError(f"{label}: echecs solveur")
        shortage_residual = result.get(
            "max_deficit_shortage_after_lol_w",
            result.get("max_deficit_balance_residual_w", 0.0),
        )
        if shortage_residual > 1e-4:
            raise RuntimeError(f"{label}: deficit non ferme apres LOL")
        if (result["lol_above_one_steps"]
                and result.get("excess_beyond_clip_kwh", 0.0) > 1e-9):
            raise RuntimeError(f"{label}: lol>1")
    print(f"OK -- screening complet -> {output}")


if __name__ == "__main__":
    main()
