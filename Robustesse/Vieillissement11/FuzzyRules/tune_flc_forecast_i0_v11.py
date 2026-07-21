"""Screening cinq ans de FLC-IF sous oracle et erreur LSTM empirique."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics

from .flc_forecast_policy_v11 import (
    DEFAULT_BIAS_KWH,
    DEFAULT_HORIZON_STEPS,
    DEFAULT_SIGMA_KWH,
    make_forecast_augmented_flc_policy_v11,
)
from .flc_policy_v11 import make_tuned_expert_flc_policy_v11
from .run_promoted_flc_25y_v11 import (
    DP_REFERENCE,
    _install_reference_profile,
    _json_hash,
    _reference_profile_sha256,
)
from .tune_flc_i0_v11 import _audit_data, _file_sha256, _starts
from .tune_flc_soh_i0_v11 import _replacement_counts, _trajectory_sha256


HERE = Path(__file__).resolve().parent
PROTOCOL_ID = "tune-flc-if-v11-p2-5y-v1-2026-07-21"
YEARS = 5.0
EXPECTED_STEPS = 43799
VOLL = 3.0
STRENGTH_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
NOISE_SEEDS = (20260701, 20260702, 20260703, 20260704)


def _job(kind, strength=0.0, seed=0):
    params = {
        "forecast_strength": float(strength),
        "forecast_scenario": (
            "oracle" if kind in ("oracle", "parent") else "gaussian_iid"
        ),
        "noise_seed": int(seed),
    }
    identity = {"kind": kind, **params}
    prefix = "flc_i0_parent" if kind == "parent" else "flc_if"
    return {
        "candidate_id": f"{prefix}_{_json_hash(identity)}",
        "kind": kind,
        "parameters": params,
    }


def generate_jobs():
    jobs = [_job("parent")]
    jobs.extend(_job("oracle", strength) for strength in STRENGTH_VALUES)
    jobs.extend(
        _job("gaussian_iid", strength, seed)
        for strength in STRENGTH_VALUES[1:]
        for seed in NOISE_SEEDS
    )
    return jobs


def _evaluate_job(payload):
    job, output_dir, fingerprint, reference_path = payload
    output_dir = Path(output_dir)
    result_path = output_dir / f"{job['candidate_id']}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("protocol_fingerprint") == fingerprint:
            return cached

    _install_reference_profile(reference_path)
    if job["kind"] == "parent":
        policy = make_tuned_expert_flc_policy_v11()
        diagnostics = None
    else:
        policy = make_forecast_augmented_flc_policy_v11(**job["parameters"])
        diagnostics = policy.forecast_diagnostics
    started = perf_counter()
    data = init_and_run_loop(
        policy, n_years=YEARS, replacement_accounting="corrected"
    )
    runtime_s = perf_counter() - started
    if len(data["lol_tab"]) != EXPECTED_STEPS:
        raise RuntimeError("nombre de pas inattendu")
    audit = _audit_data(data)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{job['candidate_id']}: {audit['failures']}")
    reliability = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]
    components = {
        key: float(value) for key, value in ledger["total_eur"].items()
    }
    degradation = float(sum(components.values()))
    result = {
        **job,
        "protocol_fingerprint": fingerprint,
        "policy_spec_sha256": policy.flc_metadata["spec_sha256"],
        "trajectory_sha256": _trajectory_sha256(data),
        "metrics": {
            "steps": int(len(data["lol_tab"])),
            "lpsp_pct": float(reliability["lpsp_pct"]),
            "eens_kwh": float(reliability["eens_kwh"]),
            "load_energy_kwh": float(reliability["load_energy_kwh"]),
            "degradation_eur": degradation,
            "degradation_components_eur": components,
            "j3_eur": degradation + VOLL * float(reliability["eens_kwh"]),
            "fc_starts": _starts(data["P_fc"]),
            "ely_starts": _starts(data["P_ely"]),
            "replacement_counts": _replacement_counts(ledger),
            "soc_min": float(np.min(data["SoC"])),
            "soc_max": float(np.max(data["SoC"])),
            "h2_min_kwh": float(np.min(data["E_h2"])),
            "h2_terminal_kwh": float(np.asarray(data["E_h2"])[-1]),
            "runtime_s": runtime_s,
        },
        "forecast_diagnostics": diagnostics() if diagnostics else None,
        "audit": audit,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, result_path)
    return result


def _run_jobs(jobs, output_dir, fingerprint, workers):
    payloads = [
        (job, str(output_dir), fingerprint, str(DP_REFERENCE)) for job in jobs
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_job, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            metrics = result["metrics"]
            print(
                f"[{index:02d}/{len(futures):02d}] {result['candidate_id']} "
                f"LPSP={metrics['lpsp_pct']:.4f}% "
                f"deg={metrics['degradation_eur']/1000.0:.3f} kEUR "
                f"J3={metrics['j3_eur']/1000.0:.3f} kEUR",
                flush=True,
            )
    return results


def _aggregate_noise(results):
    grouped = {}
    for item in results:
        strength = item["parameters"]["forecast_strength"]
        grouped.setdefault(strength, []).append(item)
    aggregate = []
    metric_keys = ("lpsp_pct", "eens_kwh", "degradation_eur", "j3_eur")
    for strength, items in sorted(grouped.items()):
        summary = {}
        for key in metric_keys:
            values = np.asarray([item["metrics"][key] for item in items], dtype=float)
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        aggregate.append({
            "configuration_id": f"flc_if_strength_{_json_hash({'strength': strength})}",
            "forecast_strength": float(strength),
            "seeds": sorted(item["parameters"]["noise_seed"] for item in items),
            "n": len(items),
            "metrics": summary,
            "candidate_ids": sorted(item["candidate_id"] for item in items),
        })
    return aggregate


def _selection(aggregate, parent):
    parent_metrics = parent["metrics"]
    active = [item for item in aggregate if item["forecast_strength"] > 0.0]
    reliability_pool = [
        item for item in active
        if item["metrics"]["degradation_eur"]["mean"]
        <= 1.01 * parent_metrics["degradation_eur"]
    ]
    durability_pool = [
        item for item in active
        if item["metrics"]["lpsp_pct"]["mean"]
        <= parent_metrics["lpsp_pct"] + 0.05
    ]
    roles = {
        "best_active_mean_j3": min(
            active, key=lambda item: item["metrics"]["j3_eur"]["mean"]
        ),
        "reliability_under_1pct_deg": min(
            reliability_pool,
            key=lambda item: item["metrics"]["lpsp_pct"]["mean"],
        ),
        "durability_under_0p05_lpsp": min(
            durability_pool,
            key=lambda item: item["metrics"]["degradation_eur"]["mean"],
        ),
    }
    return {role: item for role, item in roles.items()}


def _write_raw_csv(path, results):
    fields = [
        "candidate_id", "kind", "forecast_strength", "forecast_scenario",
        "noise_seed", "lpsp_pct", "degradation_eur", "eens_kwh", "j3_eur",
        "ely_starts", "precharge_applied_steps", "ely_energy_removed_kwh_dc",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            diagnostics = item["forecast_diagnostics"] or {}
            writer.writerow({
                "candidate_id": item["candidate_id"],
                "kind": item["kind"],
                **item["parameters"],
                "lpsp_pct": item["metrics"]["lpsp_pct"],
                "degradation_eur": item["metrics"]["degradation_eur"],
                "eens_kwh": item["metrics"]["eens_kwh"],
                "j3_eur": item["metrics"]["j3_eur"],
                "ely_starts": item["metrics"]["ely_starts"],
                "precharge_applied_steps": diagnostics.get("precharge_applied_steps"),
                "ely_energy_removed_kwh_dc": diagnostics.get("ely_energy_removed_kwh_dc"),
            })


def _plot(path, oracle, aggregate, parent):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    strengths = [item["parameters"]["forecast_strength"] for item in oracle]
    axes[0].plot(
        strengths, [item["metrics"]["j3_eur"] / 1000.0 for item in oracle],
        "o-", label="Oracle",
    )
    axes[0].errorbar(
        [item["forecast_strength"] for item in aggregate],
        [item["metrics"]["j3_eur"]["mean"] / 1000.0 for item in aggregate],
        yerr=[item["metrics"]["j3_eur"]["std"] / 1000.0 for item in aggregate],
        fmt="s-", capsize=3, label="LSTM iid (moyenne ± écart-type)",
    )
    axes[0].axhline(parent["metrics"]["j3_eur"] / 1000.0,
                    color="0.35", linestyle="--", label="Parent I0")
    axes[0].set_xlabel("Force de pré-charge")
    axes[0].set_ylabel("J3 (kEUR)")
    axes[0].set_title("Scalarisation auxiliaire")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].scatter(parent["metrics"]["lpsp_pct"],
                    parent["metrics"]["degradation_eur"] / 1000.0,
                    marker="X", s=100, label="Parent I0")
    axes[1].plot(
        [item["metrics"]["lpsp_pct"] for item in oracle],
        [item["metrics"]["degradation_eur"] / 1000.0 for item in oracle],
        "o-", label="Oracle",
    )
    axes[1].plot(
        [item["metrics"]["lpsp_pct"]["mean"] for item in aggregate],
        [item["metrics"]["degradation_eur"]["mean"] / 1000.0 for item in aggregate],
        "s-", label="LSTM iid — moyennes",
    )
    axes[1].set_xlabel("LPSP (%)")
    axes[1].set_ylabel("Coût de dégradation (kEUR)")
    axes[1].set_title("Plan des deux objectifs")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("FLC-IF — screening canonique sur cinq ans")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers doit etre positif")
    jobs = generate_jobs()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "replacement_accounting": "corrected",
        "years": YEARS,
        "expected_steps": EXPECTED_STEPS,
        "profile_source": str(DP_REFERENCE.resolve()),
        "profile_source_sha256": _file_sha256(DP_REFERENCE),
        "profile_sha256": _reference_profile_sha256(),
        "parent_candidate_id": "flc_8126e6f729c6",
        "parent_spec_sha256": "71c0531744f2ecf0b6cde6ee97a7ed0ba0d3d2468cebca06caa75643c2bd162d",
        "horizon_steps": DEFAULT_HORIZON_STEPS,
        "bias_kwh": DEFAULT_BIAS_KWH,
        "sigma_kwh": DEFAULT_SIGMA_KWH,
        "threshold_sigma_multiplier_oracle": 0.0,
        "threshold_sigma_multiplier_gaussian": 1.0,
        "min_dwell_steps": 0,
        "strength_values": STRENGTH_VALUES,
        "noise_seeds": NOISE_SEEDS,
        "job_count": len(jobs),
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "scalarization_auxiliary": {"name": "j3_eur", "voll_eur_per_kwh": VOLL},
    }
    fingerprint = _json_hash(manifest)
    output_dir = HERE / "runs" / f"tune_flc_if_5y_{fingerprint}"
    cache_dir = output_dir / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Screening FLC-IF: {len(jobs)} runs, {args.workers} workers", flush=True)
    results = _run_jobs(jobs, cache_dir, fingerprint, args.workers)
    parent = next(item for item in results if item["kind"] == "parent")
    oracle = sorted(
        (item for item in results if item["kind"] == "oracle"),
        key=lambda item: item["parameters"]["forecast_strength"],
    )
    noisy = [item for item in results if item["kind"] == "gaussian_iid"]
    null = next(
        item for item in oracle
        if item["parameters"]["forecast_strength"] == 0.0
    )
    parent_metrics = dict(parent["metrics"])
    null_metrics = dict(null["metrics"])
    parent_metrics.pop("runtime_s", None)
    null_metrics.pop("runtime_s", None)
    null_test = {
        "status": "PASS" if (
            null["trajectory_sha256"] == parent["trajectory_sha256"]
            and null_metrics == parent_metrics
        ) else "FAIL",
        "trajectory_exact": null["trajectory_sha256"] == parent["trajectory_sha256"],
        "metrics_exact_excluding_runtime": null_metrics == parent_metrics,
        "parent_trajectory_sha256": parent["trajectory_sha256"],
        "null_trajectory_sha256": null["trajectory_sha256"],
        "null_candidate_id": null["candidate_id"],
    }
    if null_test["status"] != "PASS":
        raise RuntimeError(f"test nul FLC-IF invalide: {null_test}")

    aggregate = _aggregate_noise(noisy)
    selection = _selection(aggregate, parent)
    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "parent": parent,
        "null_test": null_test,
        "oracle_results": oracle,
        "noise_aggregate": aggregate,
        "selection": selection,
        "results": sorted(results, key=lambda item: item["candidate_id"]),
    }
    (output_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "null_test.json").write_text(
        json.dumps(null_test, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_raw_csv(output_dir / "evaluations.csv", results)
    _plot(output_dir / "screening_5y.png", oracle, aggregate, parent)
    print(f"Resultats: {output_dir}")
    print(json.dumps({"null_test": null_test, "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
