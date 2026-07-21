"""Screening cinq ans de la couche SoH FLC-IS avec test nul exact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Mapping
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

from .flc_policy_v11 import make_tuned_expert_flc_policy_v11
from .flc_soh_policy_v11 import make_soh_augmented_flc_policy_v11
from .run_promoted_flc_25y_v11 import (
    DP_REFERENCE,
    _install_reference_profile,
    _reference_profile_sha256,
)
from .tune_flc_i0_v11 import _audit_data, _file_sha256, _starts, nondominated


HERE = Path(__file__).resolve().parent
PROTOCOL_ID = "tune-flc-is-soh-v11-p2-5y-v1-2026-07-21"
VOLL = 3.0
YEARS = 5.0
EXPECTED_STEPS = 43799
STRENGTH_VALUES = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40)
TRAJECTORY_ARRAY_KEYS = (
    "temps", "SoC", "E_h2", "P_bat", "P_fc", "P_ely",
    "P_dc_load", "P_dc_pv", "P_dc_bat", "P_dc_fc", "P_dc_ely",
    "alpha_fc", "alpha_ely", "lol_tab", "SoH_bat", "SoH_fc",
    "SoH_ely", "RUL_fc", "RUL_ely",
)


def _json_hash(value, length=12):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _candidate(strength_fc, strength_ely):
    params = {
        "soh_strength_fc": float(strength_fc),
        "soh_strength_ely": float(strength_ely),
    }
    return {
        "candidate_id": f"flc_is_{_json_hash(params)}",
        "label": (
            f"is_fc_{strength_fc:g}_ely_{strength_ely:g}"
        ),
        "parameters": params,
    }


def generate_candidates():
    return [
        _candidate(strength_fc, strength_ely)
        for strength_fc in STRENGTH_VALUES
        for strength_ely in STRENGTH_VALUES
    ]


def _update_digest(digest, value):
    if isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode())
            _update_digest(digest, value[key])
    elif isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(str(type(value).__name__).encode())
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(json.dumps(value, sort_keys=True).encode())


def _trajectory_sha256(data):
    digest = hashlib.sha256()
    for key in TRAJECTORY_ARRAY_KEYS:
        digest.update(key.encode())
        _update_digest(digest, np.asarray(data[key]))
    _update_digest(digest, data["deg_fc"])
    _update_digest(digest, data["deg_ely"])
    _update_digest(digest, data["degradation_ledger"])
    return digest.hexdigest()


def _replacement_counts(ledger):
    counts = {"bat": 0, "fc": 0, "ely": 0}
    for event in ledger["events"]:
        counts[event["component"]] += 1
    return counts


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
        policy_spec_sha256 = policy.flc_metadata["spec_sha256"]
    else:
        policy = make_soh_augmented_flc_policy_v11(**job["parameters"])
        policy_spec_sha256 = policy.flc_metadata["spec_sha256"]
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
        "policy_spec_sha256": policy_spec_sha256,
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


def _selection(candidates, parent):
    parent_metrics = parent["metrics"]
    nonnull = [
        item for item in candidates
        if any(value > 0.0 for value in item["parameters"].values())
    ]
    reliability_pool = [
        item for item in candidates
        if item["metrics"]["degradation_eur"]
        <= 1.01 * parent_metrics["degradation_eur"]
    ]
    durability_pool = [
        item for item in candidates
        if item["metrics"]["lpsp_pct"]
        <= parent_metrics["lpsp_pct"] + 0.05
    ]
    roles = {
        "minimum_j3": min(candidates, key=lambda item: item["metrics"]["j3_eur"]),
        "best_nonnull_j3": min(nonnull, key=lambda item: item["metrics"]["j3_eur"]),
        "reliability_under_1pct_deg": min(
            reliability_pool, key=lambda item: item["metrics"]["lpsp_pct"]
        ),
        "durability_under_0p05_lpsp": min(
            durability_pool, key=lambda item: item["metrics"]["degradation_eur"]
        ),
    }
    return {
        role: {
            "candidate_id": item["candidate_id"],
            "parameters": item["parameters"],
            "metrics": item["metrics"],
        }
        for role, item in roles.items()
    }


def _write_csv(path, results):
    fields = [
        "candidate_id", "soh_strength_fc", "soh_strength_ely",
        "lpsp_pct", "eens_kwh", "degradation_eur", "j3_eur",
        "bat_deg_eur", "fc_deg_eur", "ely_deg_eur",
        "fc_starts", "ely_starts", "h2_terminal_kwh",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metrics = item["metrics"]
            components = metrics["degradation_components_eur"]
            writer.writerow({
                "candidate_id": item["candidate_id"],
                **item["parameters"],
                "lpsp_pct": metrics["lpsp_pct"],
                "eens_kwh": metrics["eens_kwh"],
                "degradation_eur": metrics["degradation_eur"],
                "j3_eur": metrics["j3_eur"],
                "bat_deg_eur": components["bat"],
                "fc_deg_eur": components["fc"],
                "ely_deg_eur": components["ely"],
                "fc_starts": metrics["fc_starts"],
                "ely_starts": metrics["ely_starts"],
                "h2_terminal_kwh": metrics["h2_terminal_kwh"],
            })


def _plot(path, candidates, front, parent, selection):
    figure, axis = plt.subplots(figsize=(8.1, 5.7), constrained_layout=True)
    axis.scatter(
        [item["metrics"]["lpsp_pct"] for item in candidates],
        [item["metrics"]["degradation_eur"] / 1000.0 for item in candidates],
        s=30, alpha=0.55, label="FLC-IS SoH",
    )
    axis.plot(
        [item["metrics"]["lpsp_pct"] for item in front],
        [item["metrics"]["degradation_eur"] / 1000.0 for item in front],
        "o-", color="black", linewidth=1.1, markersize=4,
        label="Front IS non dominé",
    )
    axis.scatter(
        parent["metrics"]["lpsp_pct"],
        parent["metrics"]["degradation_eur"] / 1000.0,
        marker="X", s=110, label="Parent I0",
    )
    selected_ids = {item["candidate_id"] for item in selection.values()}
    for item in candidates:
        if item["candidate_id"] in selected_ids:
            axis.annotate(
                item["candidate_id"].removeprefix("flc_is_"),
                (item["metrics"]["lpsp_pct"], item["metrics"]["degradation_eur"] / 1000.0),
                xytext=(5, 5), textcoords="offset points", fontsize=7,
            )
    axis.set_xlabel("LPSP (%)")
    axis.set_ylabel("Coût de dégradation (kEUR)")
    axis.set_title("FLC-IS SoH — screening canonique sur cinq ans")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers doit etre positif")
    candidates = generate_candidates()
    parent_job = {
        "candidate_id": "flc_8126e6f729c6_parent_i0",
        "label": "parent_i0",
        "kind": "parent",
        "parameters": {},
    }
    jobs = [
        {**candidate, "kind": "soh"} for candidate in candidates
    ] + [parent_job]
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
        "strength_values": STRENGTH_VALUES,
        "candidate_count": len(candidates),
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "scalarization_auxiliary": {"name": "j3_eur", "voll_eur_per_kwh": VOLL},
    }
    fingerprint = _json_hash(manifest)
    output_dir = HERE / "runs" / f"tune_flc_is_soh_5y_{fingerprint}"
    cache_dir = output_dir / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Screening FLC-IS SoH: {len(jobs)} runs, {args.workers} workers", flush=True)
    results = _run_jobs(jobs, cache_dir, fingerprint, args.workers)
    parent = next(item for item in results if item["kind"] == "parent")
    soh_results = sorted(
        (item for item in results if item["kind"] == "soh"),
        key=lambda item: (
            item["parameters"]["soh_strength_fc"],
            item["parameters"]["soh_strength_ely"],
        ),
    )
    null = next(
        item for item in soh_results
        if item["parameters"] == {
            "soh_strength_fc": 0.0, "soh_strength_ely": 0.0
        }
    )
    null_test = {
        "status": "PASS" if null["trajectory_sha256"] == parent["trajectory_sha256"] else "FAIL",
        "parent_trajectory_sha256": parent["trajectory_sha256"],
        "null_trajectory_sha256": null["trajectory_sha256"],
        "metrics_exact_including_runtime": null["metrics"] == parent["metrics"],
        "null_candidate_id": null["candidate_id"],
    }
    # Le temps d'execution differe et n'est pas un resultat physique du test nul.
    comparable_parent = dict(parent["metrics"])
    comparable_null = dict(null["metrics"])
    comparable_parent.pop("runtime_s", None)
    comparable_null.pop("runtime_s", None)
    null_test["metrics_exact_excluding_runtime"] = comparable_null == comparable_parent
    if (
        null_test["status"] != "PASS"
        or not null_test["metrics_exact_excluding_runtime"]
    ):
        raise RuntimeError(f"test nul FLC-IS invalide: {null_test}")

    front = nondominated(soh_results)
    selection = _selection(soh_results, parent)
    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "parent": parent,
        "null_test": null_test,
        "candidate_count": len(soh_results),
        "pareto_count": len(front),
        "pareto_candidate_ids": [item["candidate_id"] for item in front],
        "selection": selection,
        "results": soh_results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "null_test.json").write_text(
        json.dumps(null_test, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "evaluations.csv", soh_results)
    _write_csv(output_dir / "pareto.csv", front)
    _plot(output_dir / "pareto_5y.png", soh_results, front, parent, selection)
    print(f"Resultats: {output_dir}")
    print(json.dumps({"null_test": null_test, "selection": selection}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
