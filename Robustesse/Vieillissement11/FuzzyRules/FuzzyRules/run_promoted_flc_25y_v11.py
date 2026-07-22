"""Rejeu 25 ans des candidats FLC promus par le screening multiobjectif."""

from __future__ import annotations

import argparse
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

from Common import Init_EMR_MG_v16_python as I
from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics

from .flc_policy_v11 import make_expert_flc_policy_v11
from .tune_flc_i0_v11 import _audit_data, _file_sha256, _starts


HERE = Path(__file__).resolve().parent
V11 = HERE.parent
DEFAULT_TUNING_RUN = HERE / "runs" / "tune_flc_i0_1y_c21a6da6c16c"
DP_REFERENCE = V11 / "DP" / "runs" / "dp_aging_v11_p2_25y_51x51.npz"
DP_LEDGER = V11 / "DP" / "runs" / "dp_aging_v11_p2_25y_51x51_ledgers.json"
DP_PARETO = V11 / "DP" / "runs" / "dp_pareto_v11_p2_25y_51x51_rollout.npz"
PROTOCOL_ID = "promoted-flc-expert-i0-v11-p2-25y-v1-2026-07-21"
VOLL = 3.0
ARRAY_KEYS = (
    "temps", "SoC", "E_h2", "P_bat", "P_fc", "P_ely",
    "P_dc_load", "P_dc_pv", "P_dc_bat", "P_dc_fc", "P_dc_ely",
    "lol_tab", "SoH_bat", "SoH_fc", "SoH_ely",
)


def _json_hash(value, length=12):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _reference_metrics():
    ledgers = json.loads(DP_LEDGER.read_text(encoding="utf-8"))["runs"]
    references = {}
    with np.load(DP_REFERENCE, allow_pickle=False) as cache:
        if str(np.asarray(cache["model_id"]).item()) != MODEL_ID:
            raise RuntimeError("model_id incoherent dans la reference DP")
        for label in ("RB1_p2_tuned", "RB2_p2_tuned"):
            data = {
                "P_dc_load": np.asarray(cache[f"{label}__P_dc_load"]),
                "P_dc_pv": np.asarray(cache[f"{label}__P_dc_pv"]),
                "lol_tab": np.asarray(cache[f"{label}__lol_tab"]),
            }
            reliability = compute_reliability_metrics(data)
            components = {
                key: float(value)
                for key, value in ledgers[label]["total_eur"].items()
            }
            degradation = float(sum(components.values()))
            references[label] = {
                "lpsp_pct": float(reliability["lpsp_pct"]),
                "eens_kwh": float(reliability["eens_kwh"]),
                "load_energy_kwh": float(reliability["load_energy_kwh"]),
                "degradation_eur": degradation,
                "degradation_components_eur": components,
                "j3_eur": degradation + VOLL * float(reliability["eens_kwh"]),
            }
    return references


def _reference_profile_sha256():
    with np.load(DP_REFERENCE, allow_pickle=False) as cache:
        load_ac = (
            np.asarray(cache["RB1_p2_tuned__P_dc_load"], dtype=np.float64)
            * I.CONV["eta"]
        )
        pv_dc = np.asarray(cache["RB1_p2_tuned__P_dc_pv"], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(load_ac.tobytes())
    digest.update(pv_dc.tobytes())
    return digest.hexdigest()


def _install_reference_profile(reference_path):
    """Installe dans le worker le profil exact des references canoniques."""
    with np.load(reference_path, allow_pickle=False) as cache:
        load_dc = np.asarray(cache["RB1_p2_tuned__P_dc_load"], dtype=float)
        pv_dc = np.asarray(cache["RB1_p2_tuned__P_dc_pv"], dtype=float)
    I.LOAD["P_ref"] = load_dc * I.CONV["eta"]
    I.PV["P"] = pv_dc


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
    params = candidate["parameters"]
    nominal_fc_dc = float(I.FC["P_fc_max"] * I.CONV["eta"])
    nominal_ely_dc = float(I.ELY["P_ely_max"] / I.CONV["eta"])
    policy = make_expert_flc_policy_v11(
        fc_ceiling_fraction=params["fc_ceiling_fraction"],
        ely_ceiling_fraction=params["ely_ceiling_fraction"],
        deficit_scale_w=params["deficit_scale_multiplier"] * nominal_fc_dc,
        surplus_scale_w=params["surplus_scale_multiplier"] * nominal_ely_dc,
        output_deadband=params["output_deadband"],
    )
    started = perf_counter()
    data = init_and_run_loop(
        policy, n_years=25.0, replacement_accounting="corrected"
    )
    runtime_s = perf_counter() - started
    audit = _audit_data(data)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{candidate['candidate_id']}: {audit['failures']}")
    reliability = compute_reliability_metrics(data)
    components = {
        key: float(value)
        for key, value in data["degradation_ledger"]["total_eur"].items()
    }
    degradation = float(sum(components.values()))
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
        "soc_min": float(np.min(data["SoC"])),
        "soc_max": float(np.max(data["SoC"])),
        "h2_min_kwh": float(np.min(data["E_h2"])),
        "h2_terminal_kwh": float(np.asarray(data["E_h2"])[-1]),
        "runtime_s": runtime_s,
    }
    arrays = {key: np.asarray(data[key]) for key in ARRAY_KEYS}
    temporary_npz = trajectory_path.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary_npz, trajectory_path)
    result = {
        **candidate,
        "protocol_fingerprint": fingerprint,
        "policy_spec_sha256": policy.flc_metadata["spec_sha256"],
        "metrics": metrics,
        "audit": audit,
        "ledger": data["degradation_ledger"],
    }
    temporary_json = result_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, result_path)
    return result


def _plot(output_path, results, references):
    with np.load(DP_PARETO, allow_pickle=False) as dp:
        dp_lpsp = np.asarray(dp["lpsp"], dtype=float)
        dp_deg = np.asarray(dp["deg_keur"], dtype=float)
        mask = np.asarray(dp["nondominated"], dtype=bool)
    figure, axis = plt.subplots(figsize=(8.3, 5.8), constrained_layout=True)
    axis.plot(dp_lpsp[mask], dp_deg[mask], "o-", color="0.55", markersize=3,
              linewidth=1, label="Front PD offline")
    for key, label, marker in (
        ("RB1_p2_tuned", "RB1", "X"),
        ("RB2_p2_tuned", "RB2", "X"),
    ):
        axis.scatter(
            references[key]["lpsp_pct"], references[key]["degradation_eur"] / 1000.0,
            marker=marker, s=100, label=label,
        )
    for result in results:
        metrics = result["metrics"]
        roles = "/".join(result["roles"])
        axis.scatter(metrics["lpsp_pct"], metrics["degradation_eur"] / 1000.0,
                     s=75, label=f"FLC {roles}")
        axis.annotate(
            result["candidate_id"].removeprefix("flc_"),
            (metrics["lpsp_pct"], metrics["degradation_eur"] / 1000.0),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    axis.set_xlabel("LPSP (%)")
    axis.set_ylabel("Coût de dégradation (kEUR)")
    axis.set_title("FLC experte I0 — points promus sur 25 ans")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-run", type=Path, default=DEFAULT_TUNING_RUN)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers doit etre positif")
    tuning_run = args.tuning_run.resolve()
    selection_path = tuning_run / "selection.json"
    tuning_manifest_path = tuning_run / "manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))

    grouped = {}
    for role, item in selection.items():
        candidate_id = item["candidate_id"]
        grouped.setdefault(candidate_id, {
            "candidate_id": candidate_id,
            "parameters": item["parameters"],
            "roles": [],
        })["roles"].append(role)
    candidates = list(grouped.values())
    references = _reference_metrics()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "replacement_accounting": "corrected",
        "years": 25.0,
        "expected_steps": 218999,
        "profile_sha256": _reference_profile_sha256(),
        "profile_source": "RB1_p2_tuned arrays from canonical DP reference cache",
        "voll_eur_per_kwh": VOLL,
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "tuning_protocol_id": tuning_manifest["protocol_id"],
        "tuning_run": str(tuning_run),
        "selection_sha256": _file_sha256(selection_path),
        "selected_candidate_ids": [item["candidate_id"] for item in candidates],
        "reference_sources": {
            "trajectories": str(DP_REFERENCE.resolve()),
            "trajectories_sha256": _file_sha256(DP_REFERENCE),
            "ledgers": str(DP_LEDGER.resolve()),
            "ledgers_sha256": _file_sha256(DP_LEDGER),
            "pareto": str(DP_PARETO.resolve()),
            "pareto_sha256": _file_sha256(DP_PARETO),
        },
    }
    fingerprint = _json_hash(manifest)
    output_dir = HERE / "runs" / f"promoted_flc_25y_{fingerprint}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    payloads = [
        (candidate, str(output_dir), fingerprint, str(DP_REFERENCE))
        for candidate in candidates
    ]
    results = []
    print(f"Rejeu 25 ans: {len(payloads)} candidats, {args.workers} workers", flush=True)
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
    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "references": references,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(output_dir / "pareto_25y.png", results, references)
    print(f"Resultats: {output_dir}")


if __name__ == "__main__":
    main()
