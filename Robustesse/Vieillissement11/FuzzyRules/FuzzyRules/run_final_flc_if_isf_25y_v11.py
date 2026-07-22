"""Evaluation finale 25 ans de FLC-IF et de l'ablation unifiee ISF-IF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics

from .flc_forecast_policy_v11 import (
    DEFAULT_BIAS_KWH,
    DEFAULT_HORIZON_STEPS,
    DEFAULT_SIGMA_KWH,
    make_forecast_augmented_flc_policy_v11,
)
from .run_promoted_flc_25y_v11 import (
    ARRAY_KEYS,
    DP_PARETO,
    DP_REFERENCE,
    _install_reference_profile,
    _json_hash,
    _reference_metrics,
    _reference_profile_sha256,
)
from .run_promoted_flc_soh_25y_v11 import (
    PARENT_CANDIDATE_ID,
    _array_digest,
    _parent_reference,
)
from .tune_flc_i0_v11 import _audit_data, _file_sha256, _starts
from .tune_flc_soh_i0_v11 import _replacement_counts


HERE = Path(__file__).resolve().parent
DEFAULT_TUNING_RUN = HERE / "runs" / "tune_flc_if_5y_7863060e5115"
PROTOCOL_ID = "final-flc-if-isf-v11-p2-25y-v1-2026-07-21"
EXPECTED_STEPS = 218999
VOLL = 3.0
IID_SEEDS = tuple(range(20260701, 20260709))
AR_SEEDS = tuple(range(20260701, 20260705))
ISF_SOH_PARAMETERS = {"soh_strength_fc": 0.0, "soh_strength_ely": 0.025}
METRIC_KEYS = ("lpsp_pct", "eens_kwh", "degradation_eur", "j3_eur")


def _candidate(kind, forecast_strength, scenario, seed=0, noise_rho=0.0,
               soh_strength_fc=0.0, soh_strength_ely=0.0):
    params = {
        "forecast_strength": float(forecast_strength),
        "forecast_scenario": scenario,
        "noise_seed": int(seed),
        "noise_rho": float(noise_rho),
        "soh_strength_fc": float(soh_strength_fc),
        "soh_strength_ely": float(soh_strength_ely),
    }
    return {
        "candidate_id": f"flc_{kind}_{_json_hash({'kind': kind, **params})}",
        "kind": kind,
        "parameters": params,
    }


def generate_jobs(forecast_strength):
    jobs = [
        _candidate("null", 0.0, "oracle"),
        _candidate("oracle_if", forecast_strength, "oracle"),
        _candidate("persistence_if", forecast_strength, "persistence"),
        _candidate(
            "oracle_isf", forecast_strength, "oracle",
            **ISF_SOH_PARAMETERS,
        ),
    ]
    jobs.extend(
        _candidate("iid_if", forecast_strength, "gaussian_iid", seed=seed)
        for seed in IID_SEEDS
    )
    jobs.extend(
        _candidate(
            "iid_isf", forecast_strength, "gaussian_iid", seed=seed,
            **ISF_SOH_PARAMETERS,
        )
        for seed in IID_SEEDS
    )
    jobs.extend(
        _candidate(
            "ar_if", forecast_strength, "gaussian_ar1", seed=seed,
            noise_rho=0.8,
        )
        for seed in AR_SEEDS
    )
    return jobs


def _worker(payload):
    candidate, output_dir, fingerprint, reference_path = payload
    output_dir = Path(output_dir)
    result_path = output_dir / f"{candidate['candidate_id']}.json"
    trajectory_path = output_dir / f"{candidate['candidate_id']}.npz"
    if result_path.exists() and trajectory_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("protocol_fingerprint") == fingerprint:
            return cached

    _install_reference_profile(reference_path)
    policy = make_forecast_augmented_flc_policy_v11(**candidate["parameters"])
    started = perf_counter()
    data = init_and_run_loop(
        policy, n_years=25.0, replacement_accounting="corrected"
    )
    runtime_s = perf_counter() - started
    audit = _audit_data(data)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{candidate['candidate_id']}: {audit['failures']}")
    reliability = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]
    components = {
        key: float(value) for key, value in ledger["total_eur"].items()
    }
    degradation = float(sum(components.values()))
    arrays = {key: np.asarray(data[key]) for key in ARRAY_KEYS}
    metrics = {
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
    }
    temporary_npz = trajectory_path.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary_npz, trajectory_path)
    result = {
        **candidate,
        "protocol_fingerprint": fingerprint,
        "policy_spec_sha256": policy.flc_metadata["spec_sha256"],
        "trajectory_sha256": _array_digest(arrays),
        "metrics": metrics,
        "forecast_diagnostics": policy.forecast_diagnostics(),
        "audit": audit,
        "ledger": ledger,
    }
    temporary_json = result_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, result_path)
    return result


def _run_jobs(jobs, output_dir, fingerprint, workers):
    payloads = [
        (job, str(output_dir), fingerprint, str(DP_REFERENCE)) for job in jobs
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            metrics = result["metrics"]
            print(
                f"[{index:02d}/{len(futures):02d}] {result['kind']} "
                f"seed={result['parameters']['noise_seed']} "
                f"LPSP={metrics['lpsp_pct']:.4f}% "
                f"deg={metrics['degradation_eur']/1000.0:.3f} kEUR "
                f"J3={metrics['j3_eur']/1000.0:.3f} kEUR",
                flush=True,
            )
    return results


def _metric_summary(values):
    values = np.asarray(values, dtype=float)
    n = int(len(values))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
        if n > 1 else 0.0
    )
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _group_statistics(results, kind):
    items = sorted(
        (item for item in results if item["kind"] == kind),
        key=lambda item: item["parameters"]["noise_seed"],
    )
    return {
        "kind": kind,
        "seeds": [item["parameters"]["noise_seed"] for item in items],
        "metrics": {
            key: _metric_summary([item["metrics"][key] for item in items])
            for key in METRIC_KEYS
        },
        "degradation_components_eur": {
            component: _metric_summary([
                item["metrics"]["degradation_components_eur"][component]
                for item in items
            ])
            for component in ("bat", "fc", "ely")
        },
        "ely_starts": _metric_summary([
            item["metrics"]["ely_starts"] for item in items
        ]),
    }


def _paired_statistics(results, kind_a="iid_isf", kind_b="iid_if"):
    a = {
        item["parameters"]["noise_seed"]: item
        for item in results if item["kind"] == kind_a
    }
    b = {
        item["parameters"]["noise_seed"]: item
        for item in results if item["kind"] == kind_b
    }
    seeds = sorted(set(a) & set(b))
    if seeds != sorted(a) or seeds != sorted(b):
        raise RuntimeError("appariement de graines incomplet")
    differences = {
        key: [a[seed]["metrics"][key] - b[seed]["metrics"][key] for seed in seeds]
        for key in METRIC_KEYS
    }
    return {
        "contrast": f"{kind_a}-{kind_b}",
        "seeds": seeds,
        "metrics": {
            key: {
                **_metric_summary(values),
                "gains": int(np.count_nonzero(np.asarray(values) < 0.0)),
            }
            for key, values in differences.items()
        },
        "both_primary_axes_improved": int(sum(
            (a[seed]["metrics"]["lpsp_pct"] < b[seed]["metrics"]["lpsp_pct"])
            and (
                a[seed]["metrics"]["degradation_eur"]
                < b[seed]["metrics"]["degradation_eur"]
            )
            for seed in seeds
        )),
        "raw": [
            {"seed": seed, **{key: differences[key][i] for key in METRIC_KEYS}}
            for i, seed in enumerate(seeds)
        ],
    }


def _decision(iid_if, paired, parent):
    mean_if_j3 = iid_if["metrics"]["j3_eur"]["mean"]
    j3_gain_pct = 100.0 * (
        parent["metrics"]["j3_eur"] - mean_if_j3
    ) / parent["metrics"]["j3_eur"]
    isf_delta = paired["metrics"]["j3_eur"]
    if_status = (
        "promote_if" if j3_gain_pct >= 1.0 else "retain_i0_over_if"
    )
    isf_status = (
        "promote_isf" if (
            isf_delta["ci95_high"] < 0.0
            and paired["both_primary_axes_improved"] > len(paired["seeds"]) / 2
        ) else "retain_if_over_isf"
    )
    return {
        "if_decision": if_status,
        "isf_decision": isf_status,
        "selected_final_information_set": (
            "ISF" if isf_status == "promote_isf"
            else "IF" if if_status == "promote_if" else "I0"
        ),
        "if_mean_j3_gain_pct_vs_i0": j3_gain_pct,
        "isf_minus_if_mean_j3_eur": isf_delta["mean"],
        "isf_minus_if_ci95_j3_eur": [
            isf_delta["ci95_low"], isf_delta["ci95_high"],
        ],
        "j3_materiality_threshold_pct": 1.0,
    }


def _write_raw_csv(path, results):
    fields = [
        "candidate_id", "kind", "seed", "scenario", "rho", "soh_ely",
        "lpsp_pct", "degradation_eur", "eens_kwh", "j3_eur", "ely_starts",
        "precharge_applied_steps", "ely_energy_removed_kwh_dc",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in sorted(results, key=lambda x: (x["kind"], x["parameters"]["noise_seed"])):
            metrics = item["metrics"]
            params = item["parameters"]
            diagnostics = item["forecast_diagnostics"]
            writer.writerow({
                "candidate_id": item["candidate_id"],
                "kind": item["kind"],
                "seed": params["noise_seed"],
                "scenario": params["forecast_scenario"],
                "rho": params["noise_rho"],
                "soh_ely": params["soh_strength_ely"],
                "lpsp_pct": metrics["lpsp_pct"],
                "degradation_eur": metrics["degradation_eur"],
                "eens_kwh": metrics["eens_kwh"],
                "j3_eur": metrics["j3_eur"],
                "ely_starts": metrics["ely_starts"],
                "precharge_applied_steps": diagnostics["precharge_applied_steps"],
                "ely_energy_removed_kwh_dc": diagnostics["ely_energy_removed_kwh_dc"],
            })


def _plot(path, results, statistics, parent, references):
    with np.load(DP_PARETO, allow_pickle=False) as dp:
        dp_lpsp = np.asarray(dp["lpsp"], dtype=float)
        dp_deg = np.asarray(dp["deg_keur"], dtype=float)
        mask = np.asarray(dp["nondominated"], dtype=bool)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    points = {
        "I0": (parent["metrics"]["lpsp_pct"], parent["metrics"]["degradation_eur"] / 1000.0),
        "IF oracle": next(
            (x["metrics"]["lpsp_pct"], x["metrics"]["degradation_eur"] / 1000.0)
            for x in results if x["kind"] == "oracle_if"
        ),
        "IF persistance": next(
            (x["metrics"]["lpsp_pct"], x["metrics"]["degradation_eur"] / 1000.0)
            for x in results if x["kind"] == "persistence_if"
        ),
        "IF LSTM iid": (
            statistics["iid_if"]["metrics"]["lpsp_pct"]["mean"],
            statistics["iid_if"]["metrics"]["degradation_eur"]["mean"] / 1000.0,
        ),
        "ISF LSTM iid": (
            statistics["iid_isf"]["metrics"]["lpsp_pct"]["mean"],
            statistics["iid_isf"]["metrics"]["degradation_eur"]["mean"] / 1000.0,
        ),
        "IF LSTM rho=0,8": (
            statistics["ar_if"]["metrics"]["lpsp_pct"]["mean"],
            statistics["ar_if"]["metrics"]["degradation_eur"]["mean"] / 1000.0,
        ),
    }
    markers = ("P", "o", "v", "s", "D", "^")
    stochastic_groups = {
        "IF LSTM iid": "iid_if",
        "ISF LSTM iid": "iid_isf",
        "IF LSTM rho=0,8": "ar_if",
    }
    for axis_index, axis in enumerate(axes):
        if axis_index == 0:
            axis.plot(dp_lpsp[mask], dp_deg[mask], "o-", color="0.75",
                      markersize=3, linewidth=1, label="Front PD offline")
        for key, label in (("RB1_p2_tuned", "RB1"), ("RB2_p2_tuned", "RB2")):
            axis.scatter(references[key]["lpsp_pct"],
                         references[key]["degradation_eur"] / 1000.0,
                         marker="X", s=80, label=label)
        for (label, point), marker in zip(points.items(), markers):
            if label in stochastic_groups:
                group = statistics[stochastic_groups[label]]["metrics"]
                lpsp = group["lpsp_pct"]
                degradation = group["degradation_eur"]
                axis.errorbar(
                    *point,
                    xerr=[[point[0] - lpsp["ci95_low"]],
                          [lpsp["ci95_high"] - point[0]]],
                    yerr=[[(1000.0 * point[1] - degradation["ci95_low"]) / 1000.0],
                          [(degradation["ci95_high"] - 1000.0 * point[1]) / 1000.0]],
                    marker=marker, markersize=8.5, linestyle="none",
                    capsize=2.5, label=label,
                )
            else:
                axis.scatter(*point, marker=marker, s=75, label=label)
        axis.set_xlabel("LPSP (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coût de dégradation (kEUR)")
    axes[0].set_title("Contexte avec front PD")
    axes[1].set_title("Zoom FLC/RB")
    zoom_x = [value[0] for value in points.values()] + [
        references[key]["lpsp_pct"] for key in ("RB1_p2_tuned", "RB2_p2_tuned")
    ]
    zoom_y = [value[1] for value in points.values()] + [
        references[key]["degradation_eur"] / 1000.0
        for key in ("RB1_p2_tuned", "RB2_p2_tuned")
    ]
    axes[1].set_xlim(min(zoom_x) - 0.025, max(zoom_x) + 0.025)
    axes[1].set_ylim(min(zoom_y) - 0.35, max(zoom_y) + 0.35)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
                  ncol=5, fontsize=8)
    figure.suptitle("Clôture FLC — prévision IF et ablation ISF sur 25 ans")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-run", type=Path, default=DEFAULT_TUNING_RUN)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers doit etre positif")
    tuning_run = args.tuning_run.resolve()
    selection_path = tuning_run / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    forecast_strength = float(
        selection["best_active_mean_j3"]["forecast_strength"]
    )
    jobs = generate_jobs(forecast_strength)
    parent = _parent_reference()
    references = _reference_metrics()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "replacement_accounting": "corrected",
        "years": 25.0,
        "expected_steps": EXPECTED_STEPS,
        "profile_sha256": _reference_profile_sha256(),
        "profile_source": "RB1_p2_tuned arrays from canonical DP reference cache",
        "parent_candidate_id": PARENT_CANDIDATE_ID,
        "parent_reference": parent,
        "tuning_run": str(tuning_run),
        "selection_sha256": _file_sha256(selection_path),
        "selected_forecast_strength": forecast_strength,
        "horizon_steps": DEFAULT_HORIZON_STEPS,
        "bias_kwh": DEFAULT_BIAS_KWH,
        "sigma_kwh": DEFAULT_SIGMA_KWH,
        "iid_seeds": IID_SEEDS,
        "ar_seeds": AR_SEEDS,
        "isf_soh_parameters": ISF_SOH_PARAMETERS,
        "job_count": len(jobs),
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "voll_eur_per_kwh": VOLL,
        "reference_sources": {
            "trajectories": str(DP_REFERENCE.resolve()),
            "trajectories_sha256": _file_sha256(DP_REFERENCE),
            "pareto": str(DP_PARETO.resolve()),
            "pareto_sha256": _file_sha256(DP_PARETO),
        },
    }
    fingerprint = _json_hash(manifest)
    output_dir = HERE / "runs" / f"final_flc_if_isf_25y_{fingerprint}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Final FLC IF/ISF: {len(jobs)} runs, {args.workers} workers", flush=True)
    results = _run_jobs(jobs, output_dir, fingerprint, args.workers)
    results.sort(key=lambda item: item["candidate_id"])

    null = next(item for item in results if item["kind"] == "null")
    with np.load(parent["trajectory"], allow_pickle=False) as parent_cache, np.load(
        output_dir / f"{null['candidate_id']}.npz", allow_pickle=False
    ) as null_cache:
        exact_by_array = {
            key: bool(np.array_equal(parent_cache[key], null_cache[key]))
            for key in ARRAY_KEYS
        }
    parent_metrics = dict(parent["metrics"])
    null_metrics = dict(null["metrics"])
    parent_metrics.pop("runtime_s", None)
    null_metrics.pop("runtime_s", None)
    null_test = {
        "status": "PASS" if (
            all(exact_by_array.values())
            and parent_metrics == null_metrics
            and parent["ledger"] == null["ledger"]
        ) else "FAIL",
        "array_exact_by_key": exact_by_array,
        "metrics_exact_excluding_runtime": parent_metrics == null_metrics,
        "ledger_exact": parent["ledger"] == null["ledger"],
        "null_candidate_id": null["candidate_id"],
    }
    if null_test["status"] != "PASS":
        raise RuntimeError(f"test nul 25 ans invalide: {null_test}")

    statistics = {
        kind: _group_statistics(results, kind)
        for kind in ("iid_if", "iid_isf", "ar_if")
    }
    paired = _paired_statistics(results)
    decision = _decision(statistics["iid_if"], paired, parent)
    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "parent": parent,
        "null_test": null_test,
        "statistics": statistics,
        "paired_isf_minus_if": paired,
        "decision": decision,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, value in (
        ("null_test.json", null_test),
        ("statistics.json", statistics),
        ("paired_isf_minus_if.json", paired),
        ("decision.json", decision),
    ):
        (output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _write_raw_csv(output_dir / "evaluations.csv", results)
    _plot(output_dir / "pareto_25y.png", results, statistics, parent, references)
    print(f"Resultats: {output_dir}")
    print(json.dumps({
        "null_test": null_test,
        "statistics": statistics,
        "paired_isf_minus_if": paired,
        "decision": decision,
        "deterministic": {
            item["kind"]: item["metrics"] for item in results
            if item["kind"] in ("oracle_if", "oracle_isf", "persistence_if")
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
