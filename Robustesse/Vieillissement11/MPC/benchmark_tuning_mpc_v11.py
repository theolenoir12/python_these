"""Tuning attribuable du MPC H24 V11-p=2 sans ponderation SoH.

Le protocole est preannonce en deux etages. Le screening modifie un seul
hyperparametre a la fois autour de la baseline et classe les cas sur trois
graines nominales de bruit x1. Les trois meilleurs cas non-baseline sont ensuite
valides, avec la baseline, sur deux graines jamais utilisees pour la selection,
aux bruits x0.5/x1/x1.5, en prevision parfaite et sous persistance. Les
combinaisons des deux puis trois meilleurs leviers distincts sont ajoutees a la
validation sans consulter les graines reservees.

Tous les couts publies proviennent du ledger V11 exact. Le tuning agit seulement
sur le surrogate de decision du MPC.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from benchmark_mpc_v11 import MODEL_ID, MPC_FORMULATION_ID, MPCConfig, _run_one
from benchmark_forecast_uncertainty_mpc_v11 import BIAS_18H_KWH, SIGMA_18H_KWH


HERE = Path(__file__).resolve().parent
TUNING_PROTOCOL_ID = "mpc-v11-p2-h24-tuning-v1-2026-07-21"
VOLL_REPORTING = 3.0
DEFAULT_SCREEN_SEEDS = (202601, 202602, 202603)
DEFAULT_VALIDATION_SEEDS = (202604, 202605)
N_FINALISTS = 3
MIN_HOLDOUT_GAIN_PCT = 1.0
MAX_PERFECT_PENALTY_PCT = 1.0
MAX_OTHER_SCENARIO_PENALTY_PCT = 2.0

BASE_PARAMETERS: dict[str, float | int | str] = {
    "horizon_steps": 24,
    "health_mode": "no_soh",
    "beta_fc": 0.0,
    "beta_ely": 0.0,
    "voll_eur_per_kwh": 3.0,
    "terminal_bat_eur_per_kwh": 0.60,
    "terminal_h2_eur_per_kwh": 1.00,
    "battery_wear_scale": 1.0,
    "fc_wear_scale": 1.0,
    "ely_wear_scale": 1.0,
    "fc_dynamic_scale": 1.0,
    "high_soc_hold_eur": 0.0,
    "time_limit_s": 30.0,
    "mip_rel_gap": 1e-4,
}

# Douze perturbations unifactorielles et une baseline. Les valeurs terminales
# restent sous les penalites qui favoriseraient une thesaurisation pendant un
# delestage ; les facteurs d'usure explorent un facteur deux autour du modele.
CASE_CHANGES: tuple[tuple[str, dict[str, float]], ...] = (
    ("baseline", {}),
    ("terminal_bat_0p3", {"terminal_bat_eur_per_kwh": 0.30}),
    ("terminal_bat_1p2", {"terminal_bat_eur_per_kwh": 1.20}),
    ("terminal_h2_0p5", {"terminal_h2_eur_per_kwh": 0.50}),
    ("terminal_h2_1p25", {"terminal_h2_eur_per_kwh": 1.25}),
    ("battery_wear_0p5", {"battery_wear_scale": 0.50}),
    ("battery_wear_2", {"battery_wear_scale": 2.00}),
    ("fc_wear_0p5", {"fc_wear_scale": 0.50}),
    ("fc_wear_2", {"fc_wear_scale": 2.00}),
    ("ely_wear_0p5", {"ely_wear_scale": 0.50}),
    ("ely_wear_2", {"ely_wear_scale": 2.00}),
    ("fc_dynamic_0", {"fc_dynamic_scale": 0.0}),
    ("fc_dynamic_3", {"fc_dynamic_scale": 3.0}),
)


def _case_parameters() -> dict[str, dict[str, float | int | str]]:
    cases: dict[str, dict[str, float | int | str]] = {}
    for name, changes in CASE_CHANGES:
        parameters = dict(BASE_PARAMETERS)
        parameters.update(changes)
        cases[name] = parameters
    return cases


def _forecast_config(mode: str, seed: int | None = None,
                     scale: float = 0.0) -> dict[str, Any]:
    config: dict[str, Any] = {"forecast_mode": mode}
    if mode == "noisy":
        if seed is None:
            raise ValueError("une graine est requise en mode noisy")
        config.update({
            "forecast_seed": int(seed),
            "forecast_sigma_energy_kwh_18h": SIGMA_18H_KWH,
            "forecast_bias_energy_kwh_18h": BIAS_18H_KWH,
            "forecast_error_rho": 0.8,
            "forecast_sigma_scale": float(scale),
        })
    return config


def _screen_configs(cases: dict[str, dict[str, Any]],
                    seeds: list[int]) -> list[dict[str, Any]]:
    configs = []
    for case, parameters in cases.items():
        for seed in seeds:
            configs.append({
                "kind": "mpc",
                "label": f"tune_{case}_train_s1p0_r{seed}",
                "tuning_case": case,
                "tuning_phase": "screen",
                **parameters,
                **_forecast_config("noisy", seed=seed, scale=1.0),
            })
    return configs


def _validation_configs(cases: dict[str, dict[str, Any]],
                        selected: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    configs = []
    for case in selected:
        parameters = cases[case]
        for mode in ("perfect", "persistence"):
            configs.append({
                "kind": "mpc",
                "label": f"tune_{case}_valid_{mode}",
                "tuning_case": case,
                "tuning_phase": "validation",
                **parameters,
                **_forecast_config(mode),
            })
        for scale in (0.5, 1.0, 1.5):
            scale_tag = str(scale).replace(".", "p")
            for seed in seeds:
                configs.append({
                    "kind": "mpc",
                    "label": f"tune_{case}_valid_s{scale_tag}_r{seed}",
                    "tuning_case": case,
                    "tuning_phase": "validation",
                    **parameters,
                    **_forecast_config("noisy", seed=seed, scale=scale),
                })
    return configs


def _fingerprint(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]


def _baseline_source_label(config: dict[str, Any]) -> str:
    mode = str(config["forecast_mode"])
    if mode in {"perfect", "persistence"}:
        return f"mpc_no_soh_h24_{mode}"
    scale_tag = str(float(config["forecast_sigma_scale"])).replace(".", "p")
    return (
        f"mpc_no_soh_h24_noisy_s{scale_tag}_r{int(config['forecast_seed'])}")


def _baseline_reference(run: Path | None,
                        configs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": run.name,
        "mapping": {
            config["label"]: _baseline_source_label(config)
            for config in configs if config["tuning_case"] == "baseline"
        },
    }


def _load_external_baseline(reference: dict[str, Any],
                            config: dict[str, Any],
                            years: float) -> tuple[dict, dict[str, str]]:
    run = HERE / "runs" / str(reference["run_id"])
    source_label = str(reference["mapping"][config["label"]])
    summary_path = run / f"{source_label}_summary.json"
    trajectory_path = run / f"{source_label}.npz"
    ledger_path = run / f"{source_label}_ledger.json"
    if not all(path.exists() for path in (summary_path, trajectory_path, ledger_path)):
        raise RuntimeError(
            f"cache baseline externe incomplet: {run}/{source_label}")
    source_protocol = json.loads((run / "protocol.json").read_text())
    if (source_protocol.get("model_id") != MODEL_ID
            or source_protocol.get("mpc_formulation_id") != MPC_FORMULATION_ID
            or abs(float(source_protocol.get("years", -1.0)) - years) > 1e-12):
        raise RuntimeError("provenance du cache baseline externe incompatible")
    summary = json.loads(summary_path.read_text())
    expected_config = asdict(MPCConfig(**{
        key: config[key]
        for key in MPCConfig.__dataclass_fields__ if key in config
    }))
    actual_config = (summary.get("diagnostics") or {}).get("config")
    if actual_config != expected_config:
        raise RuntimeError(
            f"configuration baseline externe incompatible: {source_label}")
    return summary, {
        "run_id": run.name,
        "source_label": source_label,
        "trajectory": f"runs/{run.name}/{trajectory_path.name}",
        "ledger": f"runs/{run.name}/{ledger_path.name}",
        "summary": f"runs/{run.name}/{summary_path.name}",
    }


def _write_points(output: Path, configs: list[dict[str, Any]],
                  results: dict[str, dict]) -> None:
    rows = [
        "label\ttuning_case\tphase\tforecast_mode\tsigma_scale\tseed\t"
        "lpsp_pct\tdegradation_keur\teens_kwh\tj_voll3_keur\twall_seconds"
    ]
    for config in configs:
        result = results.get(config["label"])
        if result is None:
            continue
        rows.append(
            f"{config['label']}\t{config['tuning_case']}\t"
            f"{config['tuning_phase']}\t{config['forecast_mode']}\t"
            f"{config.get('forecast_sigma_scale', 0):g}\t"
            f"{config.get('forecast_seed', '')}\t{result['lpsp_pct']:.10g}\t"
            f"{result['degradation_keur']:.10g}\t{result['eens_kwh']:.10g}\t"
            f"{result['j_voll3_keur']:.10g}\t{result['wall_seconds']:.10g}"
        )
    (output / "points.tsv").write_text("\n".join(rows) + "\n")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def _invalid_result_reason(result: dict) -> str | None:
    diagnostics = result.get("diagnostics") or {}
    if diagnostics.get("failures", 0):
        return "echec solveur enregistre"
    if result.get("max_deficit_shortage_after_lol_w", 0.0) > 1e-4:
        return "deficit non ferme apres LOL"
    if (result.get("lol_above_one_steps", 0)
            and result.get("excess_beyond_clip_kwh", 0.0) > 1e-9):
        return "lol>1 en deficit"
    return None


def _cache_is_reusable(result: dict) -> bool:
    return _invalid_result_reason(result) is None


def _validate_results(results: dict[str, dict]) -> dict[str, str]:
    invalid: dict[str, str] = {}
    for label, result in results.items():
        reason = _invalid_result_reason(result)
        if reason is not None:
            invalid[label] = reason
    return invalid


def _run_batch(output: Path, protocol: dict[str, Any], years: float,
               workers: int, *, allow_invalid: bool = False
               ) -> tuple[dict[str, dict], dict[str, str]]:
    configs = protocol["configs"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results: dict[str, dict] = {}
    pending = []
    external_manifest: dict[str, dict[str, str]] = {}
    external_reference = protocol.get("baseline_reference")
    for config in configs:
        label = config["label"]
        if (external_reference is not None
                and label in external_reference.get("mapping", {})):
            summary, provenance = _load_external_baseline(
                external_reference, config, years)
            results[label] = summary
            external_manifest[label] = provenance
            continue
        complete = all((output / name).exists() for name in (
            f"{label}.npz", f"{label}_ledger.json", f"{label}_summary.json"))
        if complete:
            cached = json.loads((output / f"{label}_summary.json").read_text())
            if _cache_is_reusable(cached):
                results[label] = cached
            else:
                pending.append((config, years, str(output)))
        else:
            pending.append((config, years, str(output)))
    _write_points(output, configs, results)
    (output / "external_cache_manifest.json").write_text(
        json.dumps(external_manifest, indent=2) + "\n")

    failures: dict[str, str] = {}
    if pending:
        n_workers = max(1, min(int(workers), len(pending)))
        with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {
                pool.submit(_run_one, job): job[0]["label"] for job in pending
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    failures[label] = repr(exc)
                    print(f"[ECHEC] {label}: {exc}", flush=True)
                    continue
                results[label] = result
                _write_points(output, configs, results)
                print(
                    f"[{label}] LPSP={result['lpsp_pct']:.4f}% "
                    f"deg={result['degradation_keur']:.3f} kEUR "
                    f"J3={result['j_voll3_keur']:.3f} kEUR",
                    flush=True,
                )
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    invalid = _validate_results(results)
    (output / "invalid.json").write_text(json.dumps(invalid, indent=2) + "\n")
    if failures or len(results) != len(configs) or (invalid and not allow_invalid):
        raise RuntimeError(
            f"banc incomplet/invalide: {len(results)}/{len(configs)}, "
            f"{len(failures)} echec(s), {len(invalid)} invalide(s)")
    if invalid:
        print(
            f"[AVERTISSEMENT] {len(invalid)} trajectoire(s) invalide(s) "
            "seront exclues du classement",
            flush=True,
        )
    return results, invalid


def _rank_screen(output: Path, configs: list[dict[str, Any]],
                 results: dict[str, dict], n_finalists: int,
                 invalid: dict[str, str] | None = None) -> dict[str, Any]:
    invalid = invalid or {}
    label_to_case = {
        config["label"]: config["tuning_case"] for config in configs
    }
    unknown_invalid = set(invalid) - set(label_to_case)
    if unknown_invalid:
        raise RuntimeError(
            "labels invalides absents du protocole: "
            + ", ".join(sorted(unknown_invalid)))
    excluded_cases: dict[str, dict[str, str]] = {}
    for label, reason in invalid.items():
        excluded_cases.setdefault(label_to_case[label], {})[label] = reason
    (output / "excluded_cases.json").write_text(
        json.dumps(excluded_cases, indent=2) + "\n")
    if "baseline" in excluded_cases:
        raise RuntimeError("la baseline du screening est invalide")

    by_case: dict[str, list[tuple[int, dict]]] = {}
    for config in configs:
        if config["tuning_case"] in excluded_cases:
            continue
        by_case.setdefault(config["tuning_case"], []).append((
            int(config["forecast_seed"]), results[config["label"]]))
    baseline = {seed: result for seed, result in by_case["baseline"]}
    ranking = []
    for case, group in by_case.items():
        ordered = sorted(group)
        values = [result["j_voll3_keur"] for _, result in ordered]
        deltas = [
            100.0 * (result["j_voll3_keur"] / baseline[seed]["j_voll3_keur"] - 1.0)
            for seed, result in ordered
        ]
        ranking.append({
            "tuning_case": case,
            "n": len(values),
            "mean_j3_keur": mean(values),
            "std_j3_keur": stdev(values) if len(values) > 1 else 0.0,
            "mean_lpsp_pct": mean(result["lpsp_pct"] for _, result in ordered),
            "mean_degradation_keur": mean(
                result["degradation_keur"] for _, result in ordered),
            "delta_j3_vs_baseline_pct_mean": mean(deltas),
            "wins_vs_baseline": sum(delta < 0.0 for delta in deltas),
        })
    ranking.sort(key=lambda row: (row["mean_j3_keur"], row["tuning_case"]))
    all_parameters = _case_parameters()
    candidates: list[str] = []
    changed_parameters: set[str] = set()
    target_finalists = min(
        n_finalists,
        sum(row["tuning_case"] != "baseline" for row in ranking),
    )
    for row in ranking:
        case = row["tuning_case"]
        if case == "baseline":
            continue
        changed = [
            key for key, value in all_parameters[case].items()
            if value != BASE_PARAMETERS[key]
        ]
        if len(changed) != 1:
            raise RuntimeError(f"cas non unifactoriel inattendu: {case}")
        if changed[0] in changed_parameters:
            continue
        candidates.append(case)
        changed_parameters.add(changed[0])
        if len(candidates) == target_finalists:
            break
    if len(candidates) != target_finalists:
        raise RuntimeError("nombre insuffisant de leviers distincts")
    validation_cases = {
        "baseline": dict(BASE_PARAMETERS),
        **{case: dict(all_parameters[case]) for case in candidates},
    }
    for count in (2, 3):
        if len(candidates) < count:
            continue
        name = f"combo_top{count}"
        parameters = dict(BASE_PARAMETERS)
        for case in candidates[:count]:
            parameters.update({
                key: value for key, value in all_parameters[case].items()
                if value != BASE_PARAMETERS[key]
            })
        validation_cases[name] = parameters
    selected = list(validation_cases)
    rows = [
        "rank\ttuning_case\tn\tmean_j3_keur\tstd_j3_keur\tmean_lpsp_pct\t"
        "mean_degradation_keur\tdelta_j3_vs_baseline_pct_mean\t"
        "wins_vs_baseline\tselected_for_validation"
    ]
    for rank, row in enumerate(ranking, start=1):
        rows.append(
            f"{rank}\t{row['tuning_case']}\t{row['n']}\t"
            f"{row['mean_j3_keur']:.12g}\t{row['std_j3_keur']:.12g}\t"
            f"{row['mean_lpsp_pct']:.12g}\t"
            f"{row['mean_degradation_keur']:.12g}\t"
            f"{row['delta_j3_vs_baseline_pct_mean']:.12g}\t"
            f"{row['wins_vs_baseline']}\t{int(row['tuning_case'] in selected)}"
        )
    (output / "ranking.tsv").write_text("\n".join(rows) + "\n")
    selection = {
        "selection_metric": "mean_j_voll3_keur_on_training_noisy_x1",
        "n_finalists_nonbaseline": len(candidates),
        "selected_single_factor_cases": candidates,
        "selected_cases": selected,
        "excluded_cases": excluded_cases,
        "validation_case_parameters": validation_cases,
        "ranking": ranking,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def _scenario_key(config: dict[str, Any]) -> tuple[str, float]:
    return (
        str(config["forecast_mode"]),
        float(config.get("forecast_sigma_scale", 0.0)),
    )


def _rank_validation(output: Path, configs: list[dict[str, Any]],
                     results: dict[str, dict], selected: list[str],
                     training_seeds: list[int],
                     invalid: dict[str, str] | None = None) -> dict[str, Any]:
    invalid = invalid or {}
    label_to_case = {
        config["label"]: config["tuning_case"] for config in configs
    }
    unknown_invalid = set(invalid) - set(label_to_case)
    if unknown_invalid:
        raise RuntimeError(
            "labels invalides absents du protocole: "
            + ", ".join(sorted(unknown_invalid)))
    excluded_cases: dict[str, dict[str, str]] = {}
    for label, reason in invalid.items():
        excluded_cases.setdefault(label_to_case[label], {})[label] = reason
    (output / "excluded_validation_cases.json").write_text(
        json.dumps(excluded_cases, indent=2) + "\n")
    if "baseline" in excluded_cases:
        raise RuntimeError("la baseline de validation est invalide")
    valid_selected = [
        case for case in selected if case not in excluded_cases
    ]

    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for config in configs:
        mode, scale = _scenario_key(config)
        grouped.setdefault((config["tuning_case"], mode, scale), []).append(
            results[config["label"]])
    stats: dict[str, dict[tuple[str, float], dict[str, float]]] = {}
    rows = [
        "tuning_case\tforecast_mode\tsigma_scale\tn\tmean_j3_keur\t"
        "std_j3_keur\tmean_lpsp_pct\tmean_degradation_keur\t"
        "delta_j3_vs_baseline_pct"
    ]
    for case in valid_selected:
        stats[case] = {}
        for mode, scale in (("perfect", 0.0), ("persistence", 0.0),
                            ("noisy", 0.5), ("noisy", 1.0), ("noisy", 1.5)):
            group = grouped[(case, mode, scale)]
            values = [result["j_voll3_keur"] for result in group]
            stats[case][(mode, scale)] = {
                "n": len(group),
                "mean_j3_keur": mean(values),
                "std_j3_keur": stdev(values) if len(values) > 1 else 0.0,
                "mean_lpsp_pct": mean(result["lpsp_pct"] for result in group),
                "mean_degradation_keur": mean(
                    result["degradation_keur"] for result in group),
            }
    baseline = stats["baseline"]
    ranking = []
    for case in valid_selected:
        case_stats = stats[case]
        deltas = {
            f"{mode}_{scale:g}": 100.0 * (
                case_stats[(mode, scale)]["mean_j3_keur"]
                / baseline[(mode, scale)]["mean_j3_keur"] - 1.0)
            for mode, scale in case_stats
        }
        for (mode, scale), item in case_stats.items():
            delta = 100.0 * (
                item["mean_j3_keur"] / baseline[(mode, scale)]["mean_j3_keur"] - 1.0)
            rows.append(
                f"{case}\t{mode}\t{scale:g}\t{item['n']}\t"
                f"{item['mean_j3_keur']:.12g}\t{item['std_j3_keur']:.12g}\t"
                f"{item['mean_lpsp_pct']:.12g}\t"
                f"{item['mean_degradation_keur']:.12g}\t{delta:.12g}"
            )
        perfect_penalty = deltas["perfect_0"]
        other_penalty = max(
            deltas["noisy_0.5"], deltas["noisy_1.5"],
            deltas["persistence_0"],
        )
        ranking.append({
            "tuning_case": case,
            "holdout_x1_mean_j3_keur": case_stats[("noisy", 1.0)]["mean_j3_keur"],
            "holdout_x1_gain_vs_baseline_pct": -deltas["noisy_1"],
            "perfect_penalty_vs_baseline_pct": perfect_penalty,
            "max_other_scenario_penalty_vs_baseline_pct": other_penalty,
            "eligible": (
                perfect_penalty <= MAX_PERFECT_PENALTY_PCT
                and other_penalty <= MAX_OTHER_SCENARIO_PENALTY_PCT
            ),
            "scenario_deltas_pct": deltas,
        })
    (output / "validation_stats.tsv").write_text("\n".join(rows) + "\n")
    eligible = [row for row in ranking if row["eligible"]]
    eligible.sort(key=lambda row: (
        row["holdout_x1_mean_j3_keur"], row["tuning_case"]))
    best = eligible[0]
    retained = (
        best["tuning_case"] != "baseline"
        and best["holdout_x1_gain_vs_baseline_pct"] >= MIN_HOLDOUT_GAIN_PCT
    )
    retained_case = best["tuning_case"] if retained else "baseline"
    decision = {
        "primary_metric": "mean_j_voll3_keur_on_heldout_noisy_x1",
        "training_seeds_excluded": training_seeds,
        "minimum_material_gain_pct": MIN_HOLDOUT_GAIN_PCT,
        "max_perfect_penalty_pct": MAX_PERFECT_PENALTY_PCT,
        "max_other_scenario_penalty_pct": MAX_OTHER_SCENARIO_PENALTY_PCT,
        "best_eligible_case": best["tuning_case"],
        "best_holdout_gain_pct": best["holdout_x1_gain_vs_baseline_pct"],
        "retained_case": retained_case,
        "retained_tuned_configuration": retained,
        "physically_validated_cases": valid_selected,
        "excluded_cases": excluded_cases,
        "ranking": ranking,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(
        f"Decision tuning: {retained_case} (meilleur gain holdout x1="
        f"{best['holdout_x1_gain_vs_baseline_pct']:.3f} %)", flush=True)
    return decision


def _screen_protocol(cases: dict[str, dict[str, Any]], seeds: list[int],
                     years: float, baseline_run: Path | None = None) -> dict[str, Any]:
    configs = _screen_configs(cases, seeds)
    return {
        "tuning_protocol_id": TUNING_PROTOCOL_ID,
        "model_id": MODEL_ID,
        "mpc_formulation_id": MPC_FORMULATION_ID,
        "years": years,
        "voll_reporting": VOLL_REPORTING,
        "training_forecast": "noisy_x1",
        "training_seeds": seeds,
        "reserved_validation_seeds": list(DEFAULT_VALIDATION_SEEDS),
        "selection_metric": "mean_j_voll3_keur",
        "n_finalists_nonbaseline": N_FINALISTS,
        "case_parameters": cases,
        "baseline_reference": _baseline_reference(baseline_run, configs),
        "configs": configs,
    }


def _validation_protocol(cases: dict[str, dict[str, Any]], selected: list[str],
                         seeds: list[int], years: float,
                         screen_run: Path,
                         baseline_run: Path | None = None) -> dict[str, Any]:
    configs = _validation_configs(cases, selected, seeds)
    return {
        "tuning_protocol_id": TUNING_PROTOCOL_ID,
        "model_id": MODEL_ID,
        "mpc_formulation_id": MPC_FORMULATION_ID,
        "years": years,
        "voll_reporting": VOLL_REPORTING,
        "screen_run_id": screen_run.name,
        "selected_cases": selected,
        "heldout_seeds": seeds,
        "validation_modes": ["perfect", "persistence", "noisy_x0.5", "noisy_x1", "noisy_x1.5"],
        "decision_rule": {
            "primary_metric": "heldout_noisy_x1_mean_j_voll3_keur",
            "minimum_gain_pct": MIN_HOLDOUT_GAIN_PCT,
            "max_perfect_penalty_pct": MAX_PERFECT_PENALTY_PCT,
            "max_other_scenario_penalty_pct": MAX_OTHER_SCENARIO_PENALTY_PCT,
        },
        "case_parameters": {case: cases[case] for case in selected},
        "baseline_reference": _baseline_reference(baseline_run, configs),
        "configs": configs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "screen", "validation"),
                        default="all")
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--screen-seeds", type=int, nargs="+",
                        default=list(DEFAULT_SCREEN_SEEDS))
    parser.add_argument("--validation-seeds", type=int, nargs="+",
                        default=list(DEFAULT_VALIDATION_SEEDS))
    parser.add_argument("--only-cases", nargs="*", default=None)
    parser.add_argument("--screen-run", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=None)
    args = parser.parse_args()
    if args.years <= 0.0:
        raise SystemExit("years doit etre positif")
    if len(set(args.screen_seeds)) != len(args.screen_seeds):
        raise SystemExit("graines de screening dupliquees")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise SystemExit("graines de validation dupliquees")
    if set(args.screen_seeds) & set(args.validation_seeds):
        raise SystemExit("les graines de selection et validation doivent etre disjointes")

    cases = _case_parameters()
    if args.only_cases:
        unknown = set(args.only_cases) - set(cases)
        if unknown:
            raise SystemExit("cas inconnus : " + ", ".join(sorted(unknown)))
        cases = {case: cases[case] for case in args.only_cases}
        if "baseline" not in cases:
            raise SystemExit("--only-cases doit inclure baseline")

    screen_output: Path | None = args.screen_run
    selection: dict[str, Any] | None = None
    if args.phase in {"all", "screen"}:
        protocol = _screen_protocol(
            cases, args.screen_seeds, args.years, args.baseline_run)
        fingerprint = _fingerprint(protocol)
        screen_output = HERE / "runs" / f"tune_screen_{args.years:g}y_{fingerprint}"
        results, invalid = _run_batch(
            screen_output, protocol, args.years, args.workers,
            allow_invalid=True)
        selection = _rank_screen(
            screen_output, protocol["configs"], results, N_FINALISTS,
            invalid)
        print(f"OK screening tuning -> {screen_output}", flush=True)
    if args.phase == "screen":
        return

    if screen_output is None:
        raise SystemExit("--screen-run est requis pour --phase validation")
    if selection is None:
        screen_protocol = json.loads((screen_output / "protocol.json").read_text())
        if not abs(float(screen_protocol["years"]) - args.years) < 1e-12:
            raise SystemExit("horizon incompatible avec --screen-run")
        selection = json.loads((screen_output / "selection.json").read_text())
    else:
        screen_protocol = protocol
    selected = list(selection["selected_cases"])
    validation_cases = selection.get("validation_case_parameters")
    if not isinstance(validation_cases, dict):
        raise SystemExit("selection.json ne contient pas les parametres de validation")
    if set(selected) != set(validation_cases):
        raise SystemExit("selection.json est incoherent")
    protocol = _validation_protocol(
        validation_cases, selected, args.validation_seeds, args.years,
        screen_output, args.baseline_run)
    fingerprint = _fingerprint(protocol)
    validation_output = (
        HERE / "runs" / f"tune_validation_{args.years:g}y_{fingerprint}")
    results, invalid = _run_batch(
        validation_output, protocol, args.years, args.workers,
        allow_invalid=True)
    _rank_validation(
        validation_output, protocol["configs"], results, selected,
        list(screen_protocol["training_seeds"]), invalid)
    print(f"OK validation tuning -> {validation_output}", flush=True)


if __name__ == "__main__":
    main()
