"""Confirmation 25 ans du réglage MPC retenu après validation aveugle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_mpc_v11 import MODEL_ID, MPC_FORMULATION_ID
from benchmark_tuning_mpc_v11 import (
    MIN_HOLDOUT_GAIN_PCT,
    TUNING_PROTOCOL_ID,
    VOLL_REPORTING,
    _fingerprint,
    _forecast_config,
    _run_batch,
)


HERE = Path(__file__).resolve().parent
LONGRUN_PROTOCOL_ID = "mpc-v11-p2-h24-tuning-longrun-v1-2026-07-21"
DEFAULT_DECISION_RUN = HERE / "runs" / "tune_validation_1y_9c728d3d847a"


def _load_retained_cases(decision_run: Path) -> tuple[
        str, dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    decision_path = decision_run / "decision.json"
    protocol_path = decision_run / "protocol.json"
    if not decision_path.exists() or not protocol_path.exists():
        raise RuntimeError("dossier de décision incomplet")
    decision = json.loads(decision_path.read_text())
    source_protocol = json.loads(protocol_path.read_text())
    if source_protocol.get("tuning_protocol_id") != TUNING_PROTOCOL_ID:
        raise RuntimeError("protocole de tuning source incompatible")
    if not decision.get("retained_tuned_configuration"):
        raise RuntimeError("aucun réglage non-baseline n'a été retenu")
    retained = str(decision["retained_case"])
    physically_valid = set(decision.get("physically_validated_cases", []))
    if retained not in physically_valid:
        raise RuntimeError("le réglage retenu n'est pas physiquement validé")
    source_cases = source_protocol.get("case_parameters")
    if not isinstance(source_cases, dict):
        raise RuntimeError("paramètres des cas absents du protocole source")
    if "baseline" not in source_cases or retained not in source_cases:
        raise RuntimeError("baseline ou réglage retenu absent du protocole source")
    cases = {
        "baseline": dict(source_cases["baseline"]),
        retained: dict(source_cases[retained]),
    }
    return retained, cases, decision, source_protocol


def _longrun_configs(cases: dict[str, dict[str, Any]], retained: str,
                     seeds: list[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for case in ("baseline", retained):
        for seed in seeds:
            configs.append({
                "kind": "mpc",
                "label": f"long_{case}_s1p0_r{seed}",
                "tuning_case": case,
                "tuning_phase": "longrun",
                **cases[case],
                **_forecast_config("noisy", seed=seed, scale=1.0),
            })
    return configs


def _longrun_protocol(decision_run: Path, retained: str,
                      cases: dict[str, dict[str, Any]], seeds: list[int],
                      years: float) -> dict[str, Any]:
    decision_bytes = (decision_run / "decision.json").read_bytes()
    configs = _longrun_configs(cases, retained, seeds)
    return {
        "longrun_protocol_id": LONGRUN_PROTOCOL_ID,
        "source_tuning_protocol_id": TUNING_PROTOCOL_ID,
        "source_validation_run_id": decision_run.name,
        "source_decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "model_id": MODEL_ID,
        "mpc_formulation_id": MPC_FORMULATION_ID,
        "years": float(years),
        "voll_reporting": VOLL_REPORTING,
        "forecast_mode": "noisy",
        "forecast_sigma_scale": 1.0,
        "paired_seeds": seeds,
        "material_gain_threshold_pct": MIN_HOLDOUT_GAIN_PCT,
        "retained_case": retained,
        "case_parameters": cases,
        "configs": configs,
    }


def _compare(output: Path, configs: list[dict[str, Any]],
             results: dict[str, dict], retained: str,
             material_threshold_pct: float) -> dict[str, Any]:
    by_case: dict[str, dict[int, dict]] = {}
    for config in configs:
        by_case.setdefault(config["tuning_case"], {})[
            int(config["forecast_seed"])] = results[config["label"]]
    seeds = sorted(by_case["baseline"])
    if set(seeds) != set(by_case[retained]):
        raise RuntimeError("graines non appariées dans le long run")
    pairs = []
    for seed in seeds:
        baseline = by_case["baseline"][seed]
        tuned = by_case[retained][seed]
        pairs.append({
            "seed": seed,
            "baseline_j3_keur": baseline["j_voll3_keur"],
            "tuned_j3_keur": tuned["j_voll3_keur"],
            "delta_j3_keur": tuned["j_voll3_keur"] - baseline["j_voll3_keur"],
            "gain_j3_pct": 100.0 * (
                1.0 - tuned["j_voll3_keur"] / baseline["j_voll3_keur"]),
            "delta_lpsp_pp": tuned["lpsp_pct"] - baseline["lpsp_pct"],
            "delta_degradation_keur": (
                tuned["degradation_keur"] - baseline["degradation_keur"]),
            "delta_eens_kwh": tuned["eens_kwh"] - baseline["eens_kwh"],
        })
    baseline_mean = mean(
        by_case["baseline"][seed]["j_voll3_keur"] for seed in seeds)
    tuned_mean = mean(
        by_case[retained][seed]["j_voll3_keur"] for seed in seeds)
    aggregate_gain = 100.0 * (1.0 - tuned_mean / baseline_mean)
    comparison = {
        "retained_case": retained,
        "n_paired_seeds": len(seeds),
        "paired_seeds": seeds,
        "baseline_mean_j3_keur": baseline_mean,
        "tuned_mean_j3_keur": tuned_mean,
        "aggregate_gain_j3_pct": aggregate_gain,
        "mean_paired_gain_j3_pct": mean(row["gain_j3_pct"] for row in pairs),
        "tuned_wins": sum(row["delta_j3_keur"] < 0.0 for row in pairs),
        "material_gain_threshold_pct": material_threshold_pct,
        "meets_inherited_materiality_threshold": (
            aggregate_gain >= material_threshold_pct),
        "pairs": pairs,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n")
    rows = [
        "seed\tbaseline_j3_keur\ttuned_j3_keur\tdelta_j3_keur\t"
        "gain_j3_pct\tdelta_lpsp_pp\tdelta_degradation_keur\t"
        "delta_eens_kwh"
    ]
    for row in pairs:
        rows.append(
            f"{row['seed']}\t{row['baseline_j3_keur']:.12g}\t"
            f"{row['tuned_j3_keur']:.12g}\t{row['delta_j3_keur']:.12g}\t"
            f"{row['gain_j3_pct']:.12g}\t{row['delta_lpsp_pp']:.12g}\t"
            f"{row['delta_degradation_keur']:.12g}\t"
            f"{row['delta_eens_kwh']:.12g}"
        )
    (output / "comparison.tsv").write_text("\n".join(rows) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-run", type=Path,
                        default=DEFAULT_DECISION_RUN)
    parser.add_argument("--years", type=float, default=25.0)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.years <= 0.0:
        raise SystemExit("years doit être positif")
    decision_run = args.decision_run.resolve()
    retained, cases, decision, source_protocol = _load_retained_cases(
        decision_run)
    source_seeds = [int(seed) for seed in source_protocol["heldout_seeds"]]
    seeds = source_seeds if args.seeds is None else list(args.seeds)
    if len(set(seeds)) != len(seeds) or set(seeds) != set(source_seeds):
        raise SystemExit("le long run doit reprendre exactement les graines réservées")
    protocol = _longrun_protocol(
        decision_run, retained, cases, seeds, args.years)
    fingerprint = _fingerprint(protocol)
    output = HERE / "runs" / f"tune_longrun_{args.years:g}y_{fingerprint}"
    results, _ = _run_batch(
        output, protocol, args.years, args.workers)
    comparison = _compare(
        output, protocol["configs"], results, retained,
        float(decision["minimum_material_gain_pct"]),
    )
    print(
        f"OK long run {retained}: gain J3={comparison['aggregate_gain_j3_pct']:.3f} % "
        f"({comparison['tuned_wins']}/{comparison['n_paired_seeds']} graines) "
        f"-> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
