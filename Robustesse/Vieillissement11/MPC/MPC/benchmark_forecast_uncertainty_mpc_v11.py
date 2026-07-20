"""Banc MPC V11 pour l'incertitude des previsions de puissance.

Le bruit est calibre sur le backtest historique du projet : erreur d'energie
cumulee a 18 h de biais -2.32 kWh et d'ecart-type 39.38 kWh. Les facteurs
0.5/1/1.5 forment une bande de sensibilite, et les graines sont communes aux
variantes SoH afin de permettre des comparaisons appariees.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_mpc_v11 import MODEL_ID, _run_one


HERE = Path(__file__).resolve().parent
SIGMA_18H_KWH = 39.38
BIAS_18H_KWH = -2.32


def _configs(seeds: list[int], scales: list[float]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    health_cases = (
        ("no_soh", 0.0, 0.0),
        ("soh_both1", 1.0, 1.0),
    )
    for health_label, beta_fc, beta_ely in health_cases:
        common = {
            "kind": "mpc", "horizon_steps": 24,
            "health_mode": "no_soh" if health_label == "no_soh" else "soh",
            "beta_fc": beta_fc, "beta_ely": beta_ely,
            "time_limit_s": 30.0, "mip_rel_gap": 1e-4,
        }
        for mode in ("perfect", "persistence"):
            configs.append({
                **common, "label": f"mpc_{health_label}_h24_{mode}",
                "forecast_mode": mode,
            })
        for scale in scales:
            scale_tag = str(scale).replace(".", "p")
            for seed in seeds:
                configs.append({
                    **common,
                    "label": f"mpc_{health_label}_h24_noisy_s{scale_tag}_r{seed}",
                    "forecast_mode": "noisy", "forecast_seed": seed,
                    "forecast_sigma_energy_kwh_18h": SIGMA_18H_KWH,
                    "forecast_bias_energy_kwh_18h": BIAS_18H_KWH,
                    "forecast_error_rho": 0.8,
                    "forecast_sigma_scale": scale,
                })
    return configs


def _write_outputs(output: Path, configs: list[dict[str, Any]],
                   results: dict[str, dict]) -> None:
    rows = [
        "label\thealth_mode\tforecast_mode\tsigma_scale\tseed\t"
        "lpsp_pct\tdegradation_keur\teens_kwh\tj_voll3_keur\twall_seconds"
    ]
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for config in configs:
        result = results.get(config["label"])
        if result is None:
            continue
        scale = float(config.get("forecast_sigma_scale", 0.0))
        seed = config.get("forecast_seed", "")
        health = str(config["health_mode"])
        mode = str(config["forecast_mode"])
        rows.append(
            f"{config['label']}\t{health}\t{mode}\t{scale:g}\t{seed}\t"
            f"{result['lpsp_pct']:.10g}\t{result['degradation_keur']:.10g}\t"
            f"{result['eens_kwh']:.10g}\t{result['j_voll3_keur']:.10g}\t"
            f"{result['wall_seconds']:.10g}"
        )
        grouped.setdefault((health, mode, scale), []).append(result)
    (output / "points.tsv").write_text("\n".join(rows) + "\n")

    stats = [
        "health_mode\tforecast_mode\tsigma_scale\tn\tmetric\tmean\tstd\tmin\tmax"
    ]
    for (health, mode, scale), group in sorted(grouped.items()):
        for metric in ("lpsp_pct", "degradation_keur", "j_voll3_keur"):
            values = np.asarray([item[metric] for item in group], dtype=float)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            stats.append(
                f"{health}\t{mode}\t{scale:g}\t{len(values)}\t{metric}\t"
                f"{values.mean():.10g}\t{std:.10g}\t"
                f"{values.min():.10g}\t{values.max():.10g}"
            )
    (output / "summary_stats.tsv").write_text("\n".join(stats) + "\n")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[202601, 202602, 202603, 202604, 202605])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    if args.years <= 0 or any(scale < 0 for scale in args.scales):
        raise SystemExit("years doit etre positif et les echelles non negatives")

    configs = _configs(args.seeds, args.scales)
    if args.only:
        wanted = set(args.only)
        configs = [config for config in configs if config["label"] in wanted]
        missing = wanted - {config["label"] for config in configs}
        if missing:
            raise SystemExit("labels inconnus : " + ", ".join(sorted(missing)))
    protocol = {
        "model_id": MODEL_ID, "years": args.years, "horizon_steps": 24,
        "present_measurement": "exact",
        "error_model": "lead_sqrt_AR1_independent_origins",
        "sigma_energy_kwh_18h": SIGMA_18H_KWH,
        "bias_energy_kwh_18h": BIAS_18H_KWH,
        "rho": 0.8, "seeds": args.seeds, "scales": args.scales,
        "configs": configs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = HERE / "runs" / f"forecast_uncertainty_{args.years:g}y_{fingerprint}"
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
        else:
            pending.append((config, args.years, str(output)))
    _write_outputs(output, configs, results)

    workers = max(1, min(args.workers, len(pending) or 1))
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_run_one, job): job[0]["label"] for job in pending}
        for future in as_completed(futures):
            label = futures[future]
            try:
                _, result = future.result()
            except Exception as exc:
                failures[label] = repr(exc)
                print(f"[ECHEC] {label}: {exc}", flush=True)
                continue
            results[label] = result
            _write_outputs(output, configs, results)
            print(f"[{label}] LPSP={result['lpsp_pct']:.4f}% "
                  f"deg={result['degradation_keur']:.3f} kEUR", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures or len(results) != len(configs):
        raise RuntimeError(
            f"banc d'incertitude incomplet : {len(failures)} echec(s), "
            f"{len(results)}/{len(configs)} point(s) termines")
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()
