"""Screening multiobjectif, deterministe et reprenable de la FLC experte I0."""

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

from Common import Init_EMR_MG_v16_python as I
from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics

from .flc_policy_v11 import make_expert_flc_policy_v11


HERE = Path(__file__).resolve().parent
REFERENCE_RUN = HERE / "runs" / "smoke_flc_i0_b7bda4ee3399" / "summary.json"
PROTOCOL_ID = "tune-flc-expert-i0-v11-p2-moo-v1-2026-07-21"
VOLL = 3.0
SEED = 20260721
COARSE_POINTS = 48
LOCAL_STEP_FRACTION = 0.075
PARAMETER_RANGES = {
    "fc_ceiling_fraction": (0.35, 0.90),
    "ely_ceiling_fraction": (0.15, 0.70),
    "deficit_scale_multiplier": (0.60, 1.60),
    "surplus_scale_multiplier": (0.60, 1.60),
    "output_deadband": (0.02, 0.30),
}
ANCHORS = (
    (
        "flc_v1_unoptimized",
        {
            "fc_ceiling_fraction": 1.0,
            "ely_ceiling_fraction": 1.0,
            "deficit_scale_multiplier": 1.0,
            "surplus_scale_multiplier": 1.0,
            "output_deadband": 0.10,
        },
    ),
    (
        "rb2_aligned_caps",
        {
            "fc_ceiling_fraction": 0.574,
            "ely_ceiling_fraction": 0.465,
            "deficit_scale_multiplier": 1.0,
            "surplus_scale_multiplier": 1.0,
            "output_deadband": 0.10,
        },
    ),
    (
        "conservative_h2",
        {
            "fc_ceiling_fraction": 0.55,
            "ely_ceiling_fraction": 0.35,
            "deficit_scale_multiplier": 1.20,
            "surplus_scale_multiplier": 1.20,
            "output_deadband": 0.15,
        },
    ),
    (
        "battery_priority",
        {
            "fc_ceiling_fraction": 0.45,
            "ely_ceiling_fraction": 0.30,
            "deficit_scale_multiplier": 1.40,
            "surplus_scale_multiplier": 1.40,
            "output_deadband": 0.20,
        },
    ),
)


def _json_hash(value, length=12):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_sha256(n_steps):
    digest = hashlib.sha256()
    digest.update(np.asarray(I.LOAD["P_ref"][:n_steps], dtype=np.float64).tobytes())
    digest.update(np.asarray(I.PV["P"][:n_steps], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _canonical_params(params):
    return {key: round(float(params[key]), 8) for key in PARAMETER_RANGES}


def _candidate(label, params, stage):
    params = _canonical_params(params)
    return {
        "candidate_id": f"flc_{_json_hash(params)}",
        "label": str(label),
        "stage": str(stage),
        "parameters": params,
    }


def generate_coarse_candidates(count=COARSE_POINTS, seed=SEED):
    """Plan Latin hypercube fixe, complete par les quatre ancres annoncees."""
    count = int(count)
    if count <= 0:
        raise ValueError("count doit etre positif")
    rng = np.random.default_rng(int(seed))
    dimensions = tuple(PARAMETER_RANGES)
    unit = np.empty((count, len(dimensions)), dtype=float)
    for column in range(len(dimensions)):
        strata = (np.arange(count, dtype=float) + rng.random(count)) / count
        unit[:, column] = strata[rng.permutation(count)]

    candidates = []
    for row in range(count):
        params = {}
        for column, key in enumerate(dimensions):
            lower, upper = PARAMETER_RANGES[key]
            params[key] = lower + unit[row, column] * (upper - lower)
        candidates.append(_candidate(f"lhs_{row:02d}", params, "coarse"))
    candidates.extend(_candidate(label, params, "anchor") for label, params in ANCHORS)
    return _deduplicate_candidates(candidates)


def _deduplicate_candidates(candidates):
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["candidate_id"], candidate)
    return list(unique.values())


def nondominated(results):
    """Retourne le front qui minimise simultanement LPSP et degradation."""
    front = []
    for candidate in results:
        lpsp = candidate["metrics"]["lpsp_pct"]
        degradation = candidate["metrics"]["degradation_eur"]
        dominated = False
        for other in results:
            if other is candidate:
                continue
            other_lpsp = other["metrics"]["lpsp_pct"]
            other_degradation = other["metrics"]["degradation_eur"]
            if (
                other_lpsp <= lpsp and other_degradation <= degradation
                and (other_lpsp < lpsp or other_degradation < degradation)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda item: (
            item["metrics"]["lpsp_pct"],
            item["metrics"]["degradation_eur"],
        ),
    )


def _select_parent_results(results):
    roles = {
        "compromise_j3": min(results, key=lambda item: item["metrics"]["j3_eur"]),
        "reliability": min(results, key=lambda item: item["metrics"]["lpsp_pct"]),
        "durability": min(results, key=lambda item: item["metrics"]["degradation_eur"]),
    }
    selected = []
    seen = set()
    for role, result in roles.items():
        if result["candidate_id"] not in seen:
            selected.append((role, result))
            seen.add(result["candidate_id"])
    return selected


def generate_local_candidates(results, step_fraction=LOCAL_STEP_FRACTION):
    """Voisins coordonnes des trois parents preannonces, au plus 30 points."""
    candidates = []
    for role, parent in _select_parent_results(results):
        for key, (lower, upper) in PARAMETER_RANGES.items():
            step = float(step_fraction) * (upper - lower)
            for direction in (-1.0, 1.0):
                params = dict(parent["parameters"])
                params[key] = min(max(params[key] + direction * step, lower), upper)
                label = f"local_{role}_{key}_{'minus' if direction < 0 else 'plus'}"
                candidates.append(_candidate(label, params, "local"))
    existing = {item["candidate_id"] for item in results}
    return [
        candidate for candidate in _deduplicate_candidates(candidates)
        if candidate["candidate_id"] not in existing
    ]


def _starts(power):
    active = np.abs(np.asarray(power, dtype=float)) > 1e-9
    if active.size == 0:
        return 0
    return int(active[0]) + int(np.count_nonzero(active[1:] & ~active[:-1]))


def _audit_data(data):
    arrays = {
        key: np.asarray(data[key])
        for key in (
            "SoC", "E_h2", "lol_tab", "P_dc_bat", "P_dc_fc", "P_dc_ely",
            "P_dc_load", "P_dc_pv",
        )
    }
    failures = []
    for key, values in arrays.items():
        if not np.all(np.isfinite(values)):
            failures.append(f"{key} non fini")
    if np.min(arrays["SoC"]) < 0.2 - 1e-9 or np.max(arrays["SoC"]) > 0.995 + 1e-9:
        failures.append("SoC hors bornes")
    if np.min(arrays["E_h2"]) < -1e-9 or np.max(arrays["E_h2"]) > 200.0 + 1e-9:
        failures.append("H2 hors bornes")
    if np.min(arrays["lol_tab"]) < -1e-9 or np.max(arrays["lol_tab"]) > 1.0 + 1e-9:
        failures.append("LOL hors bornes")
    simultaneous = (
        (np.abs(arrays["P_dc_fc"]) > 1e-9)
        & (np.abs(arrays["P_dc_ely"]) > 1e-9)
    )
    if np.any(simultaneous):
        failures.append("PEMFC et PEMWE simultanes")
    ledger = data["degradation_ledger"]
    for component in ("bat", "fc", "ely"):
        expected = ledger["retired_eur"][component] + ledger["current_eur"][component]
        if abs(ledger["total_eur"][component] - expected) > 1e-9:
            failures.append(f"ledger {component} incoherent")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "simultaneous_fc_ely_steps": int(np.count_nonzero(simultaneous)),
    }


def _evaluate_candidate(payload):
    candidate, years, cache_dir, protocol_fingerprint = payload
    cache_path = Path(cache_dir) / f"{candidate['candidate_id']}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("protocol_fingerprint") == protocol_fingerprint:
            return cached

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
        policy, n_years=float(years), replacement_accounting="corrected"
    )
    runtime_s = perf_counter() - started
    reliability = compute_reliability_metrics(data)
    components = {
        key: float(value)
        for key, value in data["degradation_ledger"]["total_eur"].items()
    }
    degradation = float(sum(components.values()))
    audit = _audit_data(data)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{candidate['candidate_id']}: {audit['failures']}")
    result = {
        **candidate,
        "protocol_fingerprint": protocol_fingerprint,
        "policy_spec_sha256": policy.flc_metadata["spec_sha256"],
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
            "soc_min": float(np.min(data["SoC"])),
            "soc_max": float(np.max(data["SoC"])),
            "h2_min_kwh": float(np.min(data["E_h2"])),
            "h2_terminal_kwh": float(np.asarray(data["E_h2"])[-1]),
            "runtime_s": runtime_s,
        },
        "audit": audit,
    }
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, cache_path)
    return result


def _run_candidates(candidates, years, cache_dir, fingerprint, workers, title):
    if not candidates:
        return []
    print(f"{title}: {len(candidates)} candidats, {workers} workers", flush=True)
    payloads = [
        (candidate, years, str(cache_dir), fingerprint)
        for candidate in candidates
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_candidate, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            metrics = result["metrics"]
            print(
                f"[{index:02d}/{len(futures):02d}] {result['candidate_id']} "
                f"LPSP={metrics['lpsp_pct']:.4f}% "
                f"deg={metrics['degradation_eur']:.2f} EUR "
                f"J3={metrics['j3_eur']:.2f} EUR",
                flush=True,
            )
    return results


def _references_from_canonical_summary():
    summary = json.loads(REFERENCE_RUN.read_text(encoding="utf-8"))
    if summary["manifest"]["profile_sha256"] != _profile_sha256(8759):
        raise RuntimeError("le profil courant differe du screening FLC canonique")
    references = {}
    for key in ("rb1_v11_p2_020_040", "rb2_v11_p2_0574_0465"):
        source = summary["results"][key]
        references[key] = {
            "lpsp_pct": float(source["lpsp_pct"]),
            "eens_kwh": float(source["eens_kwh"]),
            "degradation_eur": float(source["degradation_eur"]),
            "j3_eur": float(source["unified_voll3_eur"]),
        }
    return references


def _selection(front):
    roles = _select_parent_results(front)
    return {
        role: {
            "candidate_id": result["candidate_id"],
            "parameters": result["parameters"],
            "metrics": result["metrics"],
        }
        for role, result in roles
    }


def _write_csv(path, results):
    fields = [
        "candidate_id", "label", "stage", *PARAMETER_RANGES,
        "lpsp_pct", "eens_kwh", "degradation_eur", "j3_eur",
        "bat_deg_eur", "fc_deg_eur", "ely_deg_eur", "fc_starts",
        "ely_starts", "h2_min_kwh", "h2_terminal_kwh",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            components = metrics["degradation_components_eur"]
            writer.writerow({
                "candidate_id": result["candidate_id"],
                "label": result["label"],
                "stage": result["stage"],
                **result["parameters"],
                "lpsp_pct": metrics["lpsp_pct"],
                "eens_kwh": metrics["eens_kwh"],
                "degradation_eur": metrics["degradation_eur"],
                "j3_eur": metrics["j3_eur"],
                "bat_deg_eur": components["bat"],
                "fc_deg_eur": components["fc"],
                "ely_deg_eur": components["ely"],
                "fc_starts": metrics["fc_starts"],
                "ely_starts": metrics["ely_starts"],
                "h2_min_kwh": metrics["h2_min_kwh"],
                "h2_terminal_kwh": metrics["h2_terminal_kwh"],
            })


def _plot(path, results, front, references, selected):
    figure, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    axis.scatter(
        [item["metrics"]["lpsp_pct"] for item in results],
        [item["metrics"]["degradation_eur"] / 1000.0 for item in results],
        s=24, alpha=0.42, color="tab:blue", label="FLC évaluées",
    )
    axis.plot(
        [item["metrics"]["lpsp_pct"] for item in front],
        [item["metrics"]["degradation_eur"] / 1000.0 for item in front],
        "o-", color="black", linewidth=1.2, markersize=4.5,
        label="Front FLC non dominé",
    )
    reference_labels = {
        "rb1_v11_p2_020_040": "RB1",
        "rb2_v11_p2_0574_0465": "RB2",
    }
    for key, metrics in references.items():
        axis.scatter(
            metrics["lpsp_pct"], metrics["degradation_eur"] / 1000.0,
            marker="X", s=95, label=reference_labels[key],
        )
    selected_ids = {value["candidate_id"] for value in selected.values()}
    for item in front:
        if item["candidate_id"] in selected_ids:
            axis.annotate(
                item["candidate_id"].removeprefix("flc_"),
                (item["metrics"]["lpsp_pct"], item["metrics"]["degradation_eur"] / 1000.0),
                xytext=(5, 5), textcoords="offset points", fontsize=7,
            )
    axis.set_xlabel("LPSP (%)")
    axis.set_ylabel("Coût de dégradation (kEUR)")
    axis.set_title("Calibration FLC experte I0 — screening central d'un an")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--coarse-points", type=int, default=COARSE_POINTS)
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.coarse_points <= 0:
        raise ValueError("workers et coarse-points doivent etre positifs")

    years = 1.0
    expected_steps = 8759
    if len(I.LOAD["P_ref"]) < expected_steps or len(I.PV["P"]) < expected_steps:
        raise RuntimeError("profil annuel incomplet")
    references = _references_from_canonical_summary()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "replacement_accounting": "corrected",
        "objective_primary": ["lpsp_pct", "degradation_eur"],
        "scalarization_auxiliary": {"name": "j3_eur", "voll_eur_per_kwh": VOLL},
        "years": years,
        "expected_steps": expected_steps,
        "profile_sha256": _profile_sha256(expected_steps),
        "reference_summary": str(REFERENCE_RUN.resolve()),
        "reference_summary_sha256": _file_sha256(REFERENCE_RUN),
        "coarse_design": {
            "method": "latin_hypercube",
            "points": int(args.coarse_points),
            "seed": SEED,
            "anchors": [label for label, _ in ANCHORS],
        },
        "local_design": {
            "enabled": not args.skip_local,
            "parent_roles": ["compromise_j3", "reliability", "durability"],
            "coordinate_step_fraction_of_range": LOCAL_STEP_FRACTION,
            "maximum_points": 30,
        },
        "parameter_ranges": PARAMETER_RANGES,
        "fixed": {"membership_functions": "v1", "rule_tables": "v1", "output_points": 401},
    }
    fingerprint = _json_hash(manifest)
    run_dir = HERE / "runs" / f"tune_flc_i0_1y_{fingerprint}"
    cache_dir = run_dir / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    coarse_candidates = generate_coarse_candidates(args.coarse_points, SEED)
    coarse_results = _run_candidates(
        coarse_candidates, years, cache_dir, fingerprint, args.workers,
        "Etage grossier",
    )
    local_results = []
    if not args.skip_local:
        local_candidates = generate_local_candidates(coarse_results)
        local_results = _run_candidates(
            local_candidates, years, cache_dir, fingerprint, args.workers,
            "Raffinement local",
        )
    results = sorted(
        coarse_results + local_results,
        key=lambda item: item["candidate_id"],
    )
    front = nondominated(results)
    selected = _selection(front)
    output = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "references": references,
        "candidate_count": len(results),
        "pareto_count": len(front),
        "results": results,
        "pareto_candidate_ids": [item["candidate_id"] for item in front],
        "selection": selected,
    }
    (run_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(run_dir / "evaluations.csv", results)
    _write_csv(run_dir / "pareto.csv", front)
    _plot(run_dir / "pareto_1y.png", results, front, references, selected)

    print(f"Resultats: {run_dir}")
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
