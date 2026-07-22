"""Promotion 25 ans des compromis FLC-IS SoH issus du screening cinq ans."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from .flc_soh_policy_v11 import make_soh_augmented_flc_policy_v11
from .run_promoted_flc_25y_v11 import (
    ARRAY_KEYS,
    DP_PARETO,
    DP_REFERENCE,
    _install_reference_profile,
    _json_hash,
    _reference_metrics,
    _reference_profile_sha256,
)
from .tune_flc_i0_v11 import _audit_data, _file_sha256, _starts


HERE = Path(__file__).resolve().parent
DEFAULT_TUNING_RUN = HERE / "runs" / "tune_flc_is_soh_5y_5bd34686bda6"
PARENT_RUN = HERE / "runs" / "promoted_flc_25y_5d6c177f02a7"
PARENT_CANDIDATE_ID = "flc_8126e6f729c6"
NULL_PARAMETERS = {"soh_strength_fc": 0.0, "soh_strength_ely": 0.0}
PROTOCOL_ID = "promoted-flc-is-soh-v11-p2-25y-v1-2026-07-21"
VOLL = 3.0
EXPECTED_STEPS = 218999


def _selection_candidates(selection):
    """Dedoublonne les roles preannonces sans modifier leur attribution."""
    grouped = {}
    for role, item in selection.items():
        candidate_id = item["candidate_id"]
        grouped.setdefault(candidate_id, {
            "candidate_id": candidate_id,
            "kind": "soh_active",
            "parameters": item["parameters"],
            "roles": [],
        })["roles"].append(role)
    return sorted(grouped.values(), key=lambda item: item["candidate_id"])


def _null_candidate():
    payload = json.dumps(NULL_PARAMETERS, sort_keys=True, separators=(",", ":"))
    candidate_id = f"flc_is_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"
    return {
        "candidate_id": candidate_id,
        "kind": "null_control",
        "parameters": dict(NULL_PARAMETERS),
        "roles": ["null_control_25y"],
    }


def _replacement_counts(ledger):
    counts = {"bat": 0, "fc": 0, "ely": 0}
    for event in ledger["events"]:
        counts[event["component"]] += 1
    return counts


def _array_digest(arrays):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        values = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


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
    policy = make_soh_augmented_flc_policy_v11(**candidate["parameters"])
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
        "audit": audit,
        "ledger": ledger,
    }
    temporary_json = result_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, result_path)
    return result


def _parent_reference():
    summary_path = PARENT_RUN / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = next(
        item for item in summary["results"]
        if item["candidate_id"] == PARENT_CANDIDATE_ID
    )
    metrics = dict(result["metrics"])
    metrics["replacement_counts"] = _replacement_counts(result["ledger"])
    return {
        "candidate_id": PARENT_CANDIDATE_ID,
        "metrics": metrics,
        "ledger": result["ledger"],
        "trajectory": str((PARENT_RUN / f"{PARENT_CANDIDATE_ID}.npz").resolve()),
        "summary": str(summary_path.resolve()),
        "summary_sha256": _file_sha256(summary_path),
    }


def _relative(delta, reference):
    return 100.0 * float(delta) / float(reference) if reference else None


def _deltas(metrics, parent_metrics):
    values = {}
    for key in ("lpsp_pct", "eens_kwh", "degradation_eur", "j3_eur"):
        delta = float(metrics[key] - parent_metrics[key])
        values[f"{key}_absolute"] = delta
        values[f"{key}_relative_pct"] = _relative(delta, parent_metrics[key])
    values["dominates_parent"] = bool(
        metrics["lpsp_pct"] <= parent_metrics["lpsp_pct"]
        and metrics["degradation_eur"] <= parent_metrics["degradation_eur"]
        and (
            metrics["lpsp_pct"] < parent_metrics["lpsp_pct"]
            or metrics["degradation_eur"] < parent_metrics["degradation_eur"]
        )
    )
    values["dominated_by_parent"] = bool(
        metrics["lpsp_pct"] >= parent_metrics["lpsp_pct"]
        and metrics["degradation_eur"] >= parent_metrics["degradation_eur"]
        and (
            metrics["lpsp_pct"] > parent_metrics["lpsp_pct"]
            or metrics["degradation_eur"] > parent_metrics["degradation_eur"]
        )
    )
    return values


def _decision(active_results):
    nondominated = [
        item for item in active_results
        if not item["deltas_vs_parent"]["dominated_by_parent"]
    ]
    materially_better_j3 = [
        item for item in nondominated
        if item["deltas_vs_parent"]["j3_eur_relative_pct"] <= -1.0
    ]
    if nondominated and materially_better_j3:
        selected = min(materially_better_j3, key=lambda item: item["metrics"]["j3_eur"])
        return {
            "status": "active_candidate_promoted",
            "selected_candidate_id": selected["candidate_id"],
            "reason": "candidat non domine et gain J3 au moins egal a 1 %",
            "j3_materiality_threshold_pct": 1.0,
        }
    reason = (
        "tous les candidats actifs sont domines par le parent I0"
        if not nondominated
        else "aucun candidat non domine n'atteint 1 % de gain J3"
    )
    return {
        "status": "retain_parent_i0",
        "selected_candidate_id": PARENT_CANDIDATE_ID,
        "active_candidate_promoted": None,
        "reason": reason,
        "j3_materiality_threshold_pct": 1.0,
    }


def _null_test(null_result, parent):
    parent_metrics = dict(parent["metrics"])
    null_metrics = dict(null_result["metrics"])
    parent_metrics.pop("runtime_s", None)
    null_metrics.pop("runtime_s", None)
    return {
        "null_candidate_id": null_result["candidate_id"],
        "parent_candidate_id": parent["candidate_id"],
        "metrics_exact_excluding_runtime": null_metrics == parent_metrics,
        "ledger_exact": null_result["ledger"] == parent["ledger"],
    }


def _plot(path, active_results, parent, references):
    with np.load(DP_PARETO, allow_pickle=False) as dp:
        dp_lpsp = np.asarray(dp["lpsp"], dtype=float)
        dp_deg = np.asarray(dp["deg_keur"], dtype=float)
        mask = np.asarray(dp["nondominated"], dtype=bool)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    styles = {
        "RB1_p2_tuned": {"label": "RB1", "marker": "X", "s": 90},
        "RB2_p2_tuned": {"label": "RB2", "marker": "X", "s": 90},
    }
    active_styles = ("o", "s", "^")
    for axis_index, axis in enumerate(axes):
        if axis_index == 0:
            axis.plot(dp_lpsp[mask], dp_deg[mask], "o-", color="0.72",
                      markersize=3, linewidth=1, label="Front PD offline")
        for key, style in styles.items():
            axis.scatter(references[key]["lpsp_pct"],
                         references[key]["degradation_eur"] / 1000.0,
                         marker=style["marker"], s=style["s"], label=style["label"])
        axis.scatter(parent["metrics"]["lpsp_pct"],
                     parent["metrics"]["degradation_eur"] / 1000.0,
                     marker="P", s=115, label="FLC parent I0")
        for item, marker in zip(active_results, active_styles):
            metrics = item["metrics"]
            params = item["parameters"]
            label = (
                f"IS FC={params['soh_strength_fc']:g}, "
                f"ELY={params['soh_strength_ely']:g}"
            )
            axis.scatter(metrics["lpsp_pct"],
                         metrics["degradation_eur"] / 1000.0,
                         marker=marker, s=70, label=label)
        axis.set_xlabel("LPSP (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coût de dégradation (kEUR)")
    axes[0].set_title("Contexte avec front PD")
    axes[1].set_title("Zoom FLC/RB")
    zoom_lpsp = [parent["metrics"]["lpsp_pct"]]
    zoom_deg = [parent["metrics"]["degradation_eur"] / 1000.0]
    for key in styles:
        zoom_lpsp.append(references[key]["lpsp_pct"])
        zoom_deg.append(references[key]["degradation_eur"] / 1000.0)
    for item in active_results:
        zoom_lpsp.append(item["metrics"]["lpsp_pct"])
        zoom_deg.append(item["metrics"]["degradation_eur"] / 1000.0)
    axes[1].set_xlim(min(zoom_lpsp) - 0.012, max(zoom_lpsp) + 0.012)
    axes[1].set_ylim(min(zoom_deg) - 0.22, max(zoom_deg) + 0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, fontsize=8,
    )
    figure.suptitle("FLC-IS SoH — compromis promus sur 25 ans")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_csv(path, results):
    fields = [
        "candidate_id", "roles", "soh_strength_fc", "soh_strength_ely",
        "lpsp_pct", "degradation_eur", "eens_kwh", "j3_eur",
        "delta_lpsp_point", "delta_degradation_eur", "delta_j3_eur",
        "delta_j3_relative_pct", "dominates_parent", "dominated_by_parent",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metrics = item["metrics"]
            deltas = item["deltas_vs_parent"]
            writer.writerow({
                "candidate_id": item["candidate_id"],
                "roles": "/".join(item["roles"]),
                **item["parameters"],
                "lpsp_pct": metrics["lpsp_pct"],
                "degradation_eur": metrics["degradation_eur"],
                "eens_kwh": metrics["eens_kwh"],
                "j3_eur": metrics["j3_eur"],
                "delta_lpsp_point": deltas["lpsp_pct_absolute"],
                "delta_degradation_eur": deltas["degradation_eur_absolute"],
                "delta_j3_eur": deltas["j3_eur_absolute"],
                "delta_j3_relative_pct": deltas["j3_eur_relative_pct"],
                "dominates_parent": deltas["dominates_parent"],
                "dominated_by_parent": deltas["dominated_by_parent"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-run", type=Path, default=DEFAULT_TUNING_RUN)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers doit etre positif")
    tuning_run = args.tuning_run.resolve()
    selection_path = tuning_run / "selection.json"
    tuning_manifest_path = tuning_run / "manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    active_candidates = _selection_candidates(selection)
    candidates = active_candidates + [_null_candidate()]
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
        "voll_eur_per_kwh": VOLL,
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "scalarization_auxiliary": "degradation_eur + 3 * eens_kwh",
        "tuning_protocol_id": tuning_manifest["protocol_id"],
        "tuning_run": str(tuning_run),
        "selection_sha256": _file_sha256(selection_path),
        "active_candidate_ids": [item["candidate_id"] for item in active_candidates],
        "null_control_candidate_id": _null_candidate()["candidate_id"],
        "parent_reference": parent,
        "reference_sources": {
            "trajectories": str(DP_REFERENCE.resolve()),
            "trajectories_sha256": _file_sha256(DP_REFERENCE),
            "pareto": str(DP_PARETO.resolve()),
            "pareto_sha256": _file_sha256(DP_PARETO),
        },
    }
    fingerprint = _json_hash(manifest)
    output_dir = HERE / "runs" / f"promoted_flc_is_soh_25y_{fingerprint}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    payloads = [
        (candidate, str(output_dir), fingerprint, str(DP_REFERENCE))
        for candidate in candidates
    ]
    results = []
    print(
        f"Promotion FLC-IS SoH 25 ans: {len(payloads)} runs, "
        f"{args.workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            metrics = result["metrics"]
            print(
                f"[{index}/{len(futures)}] {result['candidate_id']} "
                f"LPSP={metrics['lpsp_pct']:.4f}% "
                f"deg={metrics['degradation_eur']/1000.0:.3f} kEUR "
                f"J3={metrics['j3_eur']/1000.0:.3f} kEUR",
                flush=True,
            )
    results.sort(key=lambda item: item["candidate_id"])
    null_result = next(item for item in results if item["kind"] == "null_control")
    active_results = [item for item in results if item["kind"] == "soh_active"]
    for item in active_results:
        item["deltas_vs_parent"] = _deltas(item["metrics"], parent["metrics"])

    null_test = _null_test(null_result, parent)
    with np.load(parent["trajectory"], allow_pickle=False) as parent_cache, np.load(
        output_dir / f"{null_result['candidate_id']}.npz", allow_pickle=False
    ) as null_cache:
        equal_by_array = {
            key: bool(np.array_equal(parent_cache[key], null_cache[key]))
            for key in ARRAY_KEYS
        }
        parent_digest = _array_digest(parent_cache)
        null_digest = _array_digest(null_cache)
    null_test.update({
        "array_exact_by_key": equal_by_array,
        "trajectory_exact": all(equal_by_array.values()),
        "parent_trajectory_sha256": parent_digest,
        "null_trajectory_sha256": null_digest,
    })
    null_test["status"] = "PASS" if (
        null_test["trajectory_exact"]
        and null_test["metrics_exact_excluding_runtime"]
        and null_test["ledger_exact"]
    ) else "FAIL"
    if null_test["status"] != "PASS":
        raise RuntimeError(f"test nul 25 ans invalide: {null_test}")

    decision = _decision(active_results)

    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "parent": parent,
        "null_test": null_test,
        "decision": decision,
        "results": results,
        "active_results": active_results,
        "null_control": null_result,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "null_test.json").write_text(
        json.dumps(null_test, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "comparison_vs_parent.csv", active_results)
    _plot(output_dir / "pareto_25y.png", active_results, parent, references)
    print(f"Resultats: {output_dir}")
    print(json.dumps({
        "decision": decision,
        "null_test": null_test,
        "active_results": [
            {
                "candidate_id": item["candidate_id"],
                "parameters": item["parameters"],
                "roles": item["roles"],
                "metrics": item["metrics"],
                "deltas_vs_parent": item["deltas_vs_parent"],
            }
            for item in active_results
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
