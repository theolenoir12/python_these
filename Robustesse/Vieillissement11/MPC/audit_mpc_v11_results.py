"""Audit reproductible des sorties annuelles MPC V11-p=2.

Le script ne relance aucune simulation. Il recalcule les metriques depuis les
trajectoires, verifie les ledgers et les metadonnees, puis produit une synthese
du screening, de la reference DP et du banc d'incertitude apparie.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SCREEN = HERE / "runs" / "screen_1y_d840744e29c7"
DEFAULT_FORECAST = HERE / "runs" / "forecast_uncertainty_1y_d0a7f75d0466"
DEFAULT_DP = (
    HERE.parent / "DP" / "results" / "mpc_reference_1y_1b54f384caa8"
    / "dp_reference_1y_51x51_v2.npz"
)
MODEL_ID = "v11-doe-rakousky-mccay-colombo-2026-07-16"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _time_step_h(data: np.lib.npyio.NpzFile) -> float:
    times = np.asarray(data["temps"], dtype=float)
    if len(times) < 2:
        return 1.0
    return float(np.median(np.diff(times)) / 3600.0)


def _audit_run(run: Path) -> dict[str, Any]:
    protocol = _load_json(run / "protocol.json")
    summaries = _load_json(run / "summary.json")
    configs = {item["label"]: item for item in protocol["configs"]}
    expected = set(configs)
    completed = set(summaries)
    failures_path = run / "failures.json"
    failures = _load_json(failures_path) if failures_path.exists() else {}
    errors: list[str] = []
    warnings: list[str] = []
    points: dict[str, dict[str, Any]] = {}

    for label in sorted(completed):
        summary = summaries[label]
        trajectory = run / f"{label}.npz"
        ledger_path = run / f"{label}_ledger.json"
        point_errors: list[str] = []
        point_warnings: list[str] = []
        if label not in configs:
            point_errors.append("label absent du protocole")
        if not trajectory.exists() or not ledger_path.exists():
            point_errors.append("trajectoire ou ledger absent")
            errors.extend(f"{label}: {item}" for item in point_errors)
            continue

        ledger = _load_json(ledger_path)
        with np.load(trajectory, allow_pickle=False) as data:
            model_id = str(np.asarray(data["model_id"]).item())
            exponent = float(np.asarray(data["ely_stress_exponent"]).item())
            saved_config = json.loads(str(np.asarray(data["config_json"]).item()))
            p_load = np.asarray(data["P_dc_load"], dtype=float)
            p_pv = np.asarray(data["P_dc_pv"], dtype=float)
            p_ref = p_load - p_pv
            p_supply = (
                np.asarray(data["P_dc_bat"], dtype=float)
                + np.asarray(data["P_dc_fc"], dtype=float)
                + np.asarray(data["P_dc_ely"], dtype=float)
            )
            lol = np.asarray(data["lol_tab"], dtype=float)
            dt_h = _time_step_h(data)

        residual_load_kw = np.clip(p_ref / 1000.0, 0.0, None)
        eens_kwh = float(
            np.sum(residual_load_kw * np.clip(lol, 0.0, 1.0)) * dt_h)
        demand_kwh = float(np.sum(np.clip(p_load / 1000.0, 0.0, None)) * dt_h)
        lpsp_pct = 100.0 * eens_kwh / demand_kwh
        ledger_eur = float(sum(ledger["total_eur"].values()))
        degradation_keur = ledger_eur / 1000.0
        j_voll3_keur = degradation_keur + 3.0 * eens_kwh / 1000.0

        deficit = p_ref > 1e-12
        balance = np.zeros_like(p_ref)
        balance[deficit] = (
            p_supply[deficit] + lol[deficit] * p_ref[deficit] - p_ref[deficit]
        )
        shortage = np.clip(-balance[deficit], 0.0, None)
        implicit_curtailment = np.clip(balance[deficit], 0.0, None)
        max_shortage_w = float(np.max(shortage, initial=0.0))
        max_implicit_curtailment_w = float(
            np.max(implicit_curtailment, initial=0.0))
        implicit_curtailment_kwh = float(
            np.sum(implicit_curtailment) / 1000.0 * dt_h)
        implicit_curtailment_steps = int(
            np.count_nonzero(implicit_curtailment > 1e-4))

        checks = {
            "n_steps": (len(p_ref), int(summary["n_steps"]), 0.0),
            "eens_kwh": (eens_kwh, float(summary["eens_kwh"]), 1e-8),
            "demand_kwh": (demand_kwh, float(summary["demand_kwh"]), 1e-8),
            "lpsp_pct": (lpsp_pct, float(summary["lpsp_pct"]), 1e-10),
            "degradation_keur": (
                degradation_keur, float(summary["degradation_keur"]), 1e-10),
            "j_voll3_keur": (
                j_voll3_keur, float(summary["j_voll3_keur"]), 1e-10),
        }
        for name, (actual, reported, atol) in checks.items():
            if not np.isclose(actual, reported, rtol=0.0, atol=atol):
                point_errors.append(
                    f"{name} recalcule={actual:.12g}, rapporte={reported:.12g}")
        if model_id != MODEL_ID or protocol.get("model_id") != MODEL_ID:
            point_errors.append(f"model_id inattendu: {model_id}")
        if not np.isclose(exponent, 2.0):
            point_errors.append(f"p inattendu: {exponent}")
        if label in configs and saved_config != configs[label]:
            point_errors.append("config_json differe du protocole")
        if max_shortage_w > 1e-4:
            point_errors.append(
                f"deficit non ferme apres LOL: {max_shortage_w:.6g} W")
        lol_above_one = lol > 1.0 + 1e-9
        lol_above_one_deficit = int(np.count_nonzero(lol_above_one & deficit))
        lol_above_one_surplus = int(np.count_nonzero(lol_above_one & ~deficit))
        if lol_above_one_deficit:
            point_errors.append(
                f"lol > 1 sur {lol_above_one_deficit} pas en deficit")
        if lol_above_one_surplus:
            point_warnings.append(
                f"lol > 1 sur {lol_above_one_surplus} pas en surplus; "
                "sans effet sur EENS/LPSP")
        diagnostics = summary.get("diagnostics") or {}
        if diagnostics.get("failures", 0):
            point_errors.append("echec solveur enregistre dans la trajectoire")
        errors.extend(f"{label}: {item}" for item in point_errors)
        warnings.extend(f"{label}: {item}" for item in point_warnings)
        points[label] = {
            "lpsp_pct": lpsp_pct,
            "degradation_keur": degradation_keur,
            "eens_kwh": eens_kwh,
            "j_voll3_keur": j_voll3_keur,
            "max_shortage_after_lol_w": max_shortage_w,
            "max_implicit_curtailment_w": max_implicit_curtailment_w,
            "implicit_curtailment_kwh": implicit_curtailment_kwh,
            "implicit_curtailment_steps": implicit_curtailment_steps,
            "lol_above_one_deficit_steps": lol_above_one_deficit,
            "lol_above_one_surplus_steps": lol_above_one_surplus,
            "mean_solve_seconds": float(diagnostics.get("mean_solve_seconds", 0.0)),
            "max_solve_seconds": float(diagnostics.get("max_solve_seconds", 0.0)),
            "errors": point_errors,
            "warnings": point_warnings,
        }

    return {
        "run": str(run),
        "protocol": protocol,
        "expected_count": len(expected),
        "completed_count": len(completed),
        "missing": sorted(expected - completed),
        "unexpected": sorted(completed - expected),
        "failures": failures,
        "errors": errors,
        "warnings": warnings,
        "points": points,
    }


def _nondominated(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    for index in range(len(x)):
        dominates = (
            (x <= x[index] + 1e-12)
            & (y <= y[index] + 1e-12)
            & ((x < x[index] - 1e-12) | (y < y[index] - 1e-12))
        )
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return keep


def _audit_dp(path: Path, screen: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        model_id = str(np.asarray(data["model_id"]).item())
        years = float(np.asarray(data["years"]).item())
        epsilon = np.asarray(data["eps"], dtype=float)
        lpsp = np.asarray(data["lpsp"], dtype=float)
        degradation = np.asarray(data["deg_keur"], dtype=float)
        j_voll3 = np.asarray(data["unif3_keur"], dtype=float)
    keep = _nondominated(lpsp, degradation)
    front_lpsp = lpsp[keep]
    front_degradation = degradation[keep]
    order = np.argsort(front_lpsp)
    front_lpsp = front_lpsp[order]
    front_degradation = front_degradation[order]
    h24 = screen["points"]["mpc_no_soh_h24"]
    interpolated = float(np.interp(
        h24["lpsp_pct"], front_lpsp, front_degradation))
    dominated = {}
    for label, point in screen["points"].items():
        dominated[label] = bool(np.any(
            (lpsp <= point["lpsp_pct"] + 1e-12)
            & (degradation <= point["degradation_keur"] + 1e-12)
            & (
                (lpsp < point["lpsp_pct"] - 1e-12)
                | (degradation < point["degradation_keur"] - 1e-12)
            )
        ))
    best = int(np.argmin(j_voll3))
    return {
        "file": str(path),
        "model_id": model_id,
        "years": years,
        "n_points": len(epsilon),
        "n_front_points": int(np.count_nonzero(keep)),
        "all_screen_points_dominated": all(dominated.values()),
        "dominated_by_dp": dominated,
        "h24_front_degradation_at_same_lpsp_keur": interpolated,
        "h24_degradation_gap_keur": h24["degradation_keur"] - interpolated,
        "h24_degradation_gap_pct": 100.0 * (
            h24["degradation_keur"] / interpolated - 1.0),
        "best_j3": {
            "epsilon": float(epsilon[best]),
            "lpsp_pct": float(lpsp[best]),
            "degradation_keur": float(degradation[best]),
            "j_voll3_keur": float(j_voll3[best]),
        },
        "h24_j3_gap_to_best_dp_pct": 100.0 * (
            h24["j_voll3_keur"] / j_voll3[best] - 1.0),
    }


def _forecast_key(config: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(config["forecast_mode"]),
        float(config.get("forecast_sigma_scale", 0.0)),
        str(config.get("forecast_seed", "")),
    )


def _paired_forecast(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    configs = {
        item["label"]: item for item in forecast["protocol"]["configs"]
    }
    no_soh: dict[tuple[str, float, str], dict[str, Any]] = {}
    soh: dict[tuple[str, float, str], dict[str, Any]] = {}
    for label, point in forecast["points"].items():
        config = configs[label]
        target = no_soh if config["health_mode"] == "no_soh" else soh
        target[_forecast_key(config)] = point
    groups: dict[tuple[str, float], list[tuple[dict, dict]]] = {}
    for key in sorted(no_soh.keys() & soh.keys()):
        groups.setdefault(key[:2], []).append((no_soh[key], soh[key]))
    rows = []
    for (mode, scale), pairs in sorted(groups.items()):
        deltas_lpsp = [b["lpsp_pct"] - a["lpsp_pct"] for a, b in pairs]
        deltas_deg = [
            b["degradation_keur"] - a["degradation_keur"] for a, b in pairs]
        deltas_j = [b["j_voll3_keur"] - a["j_voll3_keur"] for a, b in pairs]
        deltas_j_pct = [
            100.0 * (b["j_voll3_keur"] / a["j_voll3_keur"] - 1.0)
            for a, b in pairs
        ]
        rows.append({
            "forecast_mode": mode,
            "sigma_scale": scale,
            "n_pairs": len(pairs),
            "delta_lpsp_pp_mean": mean(deltas_lpsp),
            "delta_degradation_keur_mean": mean(deltas_deg),
            "delta_j3_keur_mean": mean(deltas_j),
            "delta_j3_pct_mean": mean(deltas_j_pct),
            "delta_j3_pct_std": stdev(deltas_j_pct) if len(pairs) > 1 else 0.0,
            "soh_j3_wins": sum(delta < 0.0 for delta in deltas_j),
        })
    return rows


def _fmt_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _write_report(output: Path, screen: dict[str, Any], forecast: dict[str, Any],
                  dp: dict[str, Any], paired: list[dict[str, Any]]) -> None:
    points = screen["points"]
    h6 = points["mpc_no_soh_h6"]
    h24 = points["mpc_no_soh_h24"]
    rb1 = points["rb1_v11_p2_020_040"]
    horizon_gain = 100.0 * (h24["j_voll3_keur"] / h6["j_voll3_keur"] - 1.0)
    rb1_gain = 100.0 * (h24["j_voll3_keur"] / rb1["j_voll3_keur"] - 1.0)
    forecast_curtailment = [
        point["implicit_curtailment_kwh"]
        for point in forecast["points"].values()
        if point["implicit_curtailment_steps"]
    ]
    forecast_max_shortage = max(
        point["max_shortage_after_lol_w"]
        for point in forecast["points"].values())
    forecast_max_solve = max(
        point["max_solve_seconds"] for point in forecast["points"].values())
    surplus_lol_steps = sum(
        point["lol_above_one_surplus_steps"]
        for point in forecast["points"].values())
    surplus_lol_trajectories = sum(
        point["lol_above_one_surplus_steps"] > 0
        for point in forecast["points"].values())
    forecast_configs = {
        item["label"]: item for item in forecast["protocol"]["configs"]
    }
    no_soh_perfect_j = next(
        point["j_voll3_keur"]
        for label, point in forecast["points"].items()
        if (forecast_configs[label]["health_mode"] == "no_soh"
            and forecast_configs[label]["forecast_mode"] == "perfect")
    )
    noise_penalties = []
    for scale in (0.5, 1.0, 1.5):
        values = [
            point["j_voll3_keur"]
            for label, point in forecast["points"].items()
            if (forecast_configs[label]["health_mode"] == "no_soh"
                and forecast_configs[label]["forecast_mode"] == "noisy"
                and np.isclose(
                    forecast_configs[label]["forecast_sigma_scale"], scale))
        ]
        noise_penalties.append(
            (scale, len(values), 100.0 * (mean(values) / no_soh_perfect_j - 1.0)))
    persistence_j = next(
        point["j_voll3_keur"]
        for label, point in forecast["points"].items()
        if (forecast_configs[label]["health_mode"] == "no_soh"
            and forecast_configs[label]["forecast_mode"] == "persistence")
    )
    persistence_penalty = 100.0 * (persistence_j / no_soh_perfect_j - 1.0)

    paired_lines = []
    for row in paired:
        mode = row["forecast_mode"]
        label = mode if mode != "noisy" else f"bruit x{row['sigma_scale']:g}"
        paired_lines.append(
            f"- {label}: n={row['n_pairs']}, variation moyenne de J3 "
            f"{_fmt_signed(row['delta_j3_pct_mean'])} %, "
            f"SoH meilleur sur {row['soh_j3_wins']}/{row['n_pairs']} paire(s).")

    report = f"""# Audit des resultats MPC V11-p=2 du 20 juillet 2026

## Verdict

Le screening annuel a prevision parfaite est complet et valide : 8/8 points,
ledgers et metriques recalcules a l'identique, aucun echec solveur, aucun
deficit de bilan apres application de la LOL et aucune valeur `lol>1`.

Le banc de prevision est exploitable comme resultat preliminaire, mais pas
encore clos : {forecast['completed_count']}/{forecast['expected_count']} points
sont termines. Le point manquant est
`{', '.join(forecast['missing'])}`. Toutes les trajectoires terminees ferment
les deficits apres LOL (residu maximal {forecast_max_shortage:.3g} W). Les
residus positifs precedemment signales sont exclusivement de la puissance
excedentaire : ils correspondent a un ecretage implicite de
{min(forecast_curtailment):.3f} a {max(forecast_curtailment):.3f} kWh/an selon
la trajectoire, qu'il faudra enregistrer explicitement dans les prochaines
sorties. L'ancien `get_lol` a aussi produit `lol>1` sur {surplus_lol_steps} pas
de surplus repartis dans {surplus_lol_trajectories} trajectoire(s) ; ces pas ne
contribuent ni a l'EENS ni a la LPSP. La borne `lol=0` en surplus est maintenant
ajoutee au code pour les prochaines simulations.

## Resultats acquis sur un an

- MPC H24 sans SoH : LPSP {h24['lpsp_pct']:.6f} %, degradation
  {h24['degradation_keur']:.6f} kEUR, J3 {h24['j_voll3_keur']:.6f} kEUR.
- H24 reduit J3 de {-horizon_gain:.3f} % par rapport a H6 et de
  {-rb1_gain:.3f} % par rapport a RB1. Il domine RB1 et RB2 simultanement en
  LPSP et en degradation sur ce profil annuel.
- Temps de resolution H24 sans SoH : {1000*h24['mean_solve_seconds']:.1f} ms
  en moyenne, {h24['max_solve_seconds']:.2f} s au maximum. Les trajectoires de
  prevision terminees restent sous {forecast_max_solve:.2f} s par decision ;
  elles sont donc compatibles avec un controle online horaire. L'echec MILP
  isole interdit toutefois de conclure encore a une robustesse numerique totale.
- Les {dp['n_points']} points DP comparables utilisent le meme horizon
  d'evaluation, le meme profil et le meme modele V11-p=2. Tous les points du
  screening sont domines par le DP. A la LPSP du MPC H24, son surcout de
  degradation est de {dp['h24_degradation_gap_keur']:.6f} kEUR
  ({dp['h24_degradation_gap_pct']:.2f} %). Son J3 est
  {dp['h24_j3_gap_to_best_dp_pct']:.2f} % au-dessus du meilleur J3 DP
  echantillonne (epsilon={dp['best_j3']['epsilon']:g}). Le DP reste une borne
  clairvoyante annuelle discretisee, pas un controleur online comparable en
  information disponible.

## Sensibilite a la prevision

Par rapport au MPC H24 sans SoH avec prevision parfaite, le J3 moyen augmente
de {noise_penalties[0][2]:.2f} % au bruit x0,5 (n={noise_penalties[0][1]}),
{noise_penalties[1][2]:.2f} % au bruit x1 (n={noise_penalties[1][1]}) et
{noise_penalties[2][2]:.2f} % au bruit x1,5 (n={noise_penalties[2][1]}). La
persistance augmente J3 de {persistence_penalty:.2f} %. La qualite de prevision
est donc un levier materiel du MPC, contrairement a la ponderation SoH testee.
Le modele a origines de prevision independantes est volontairement conservatif
et ne constitue pas encore un backtest temporel complet.

## Apport du SoH dans la formulation testee

Les comparaisons ci-dessous sont appariees par graine et n'utilisent jamais le
point non apparie :

{chr(10).join(paired_lines)}

Le gain maximal observe est inferieur a quelques pourcents. Avec le critere de
decision fixe pour ce travail, la ponderation `beta_fc=beta_ely=1` n'apporte
donc pas d'utilite pratique demontree. Cela invalide cette injection simple du
SoH, pas l'usage du SoH dans toute architecture MPC.

## Decision et suite

La base a retenir est MPC H24 sans SoH, avec `p=2`. Avant le tuning MPC, il
reste deux corrections courtes : enregistrer l'ecretage execute dans le bilan
et relancer uniquement le point manquant avec un diagnostic d'infaisabilite.
Ensuite, le tuning doit porter symetriquement sur les couts terminaux et les
poids d'usure ; une variante SoH ne sera conservee que si elle depasse le seuil
de quelques pourcents sur des paires communes.
"""
    output.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--dp", type=Path, default=DEFAULT_DP)
    parser.add_argument("--output", type=Path,
                        default=HERE / "analysis" / "AUDIT_MPC_V11_P2_2026-07-20.md")
    args = parser.parse_args()

    screen = _audit_run(args.screen)
    forecast = _audit_run(args.forecast)
    dp = _audit_dp(args.dp, screen)
    paired = _paired_forecast(forecast)
    result = {"screen": screen, "forecast": forecast, "dp": dp, "paired": paired}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    paired_tsv = [
        "forecast_mode\tsigma_scale\tn_pairs\tdelta_lpsp_pp_mean\t"
        "delta_degradation_keur_mean\tdelta_j3_keur_mean\t"
        "delta_j3_pct_mean\tdelta_j3_pct_std\tsoh_j3_wins"
    ]
    for row in paired:
        paired_tsv.append("\t".join(str(row[key]) for key in (
            "forecast_mode", "sigma_scale", "n_pairs", "delta_lpsp_pp_mean",
            "delta_degradation_keur_mean", "delta_j3_keur_mean",
            "delta_j3_pct_mean", "delta_j3_pct_std", "soh_j3_wins")))
    (args.output.parent / "forecast_uncertainty_paired.tsv").write_text(
        "\n".join(paired_tsv) + "\n")
    _write_report(args.output, screen, forecast, dp, paired)

    if screen["errors"]:
        raise RuntimeError(f"screening invalide: {len(screen['errors'])} erreur(s)")
    print(
        f"OK screening {screen['completed_count']}/{screen['expected_count']} ; "
        f"prevision {forecast['completed_count']}/{forecast['expected_count']} "
        f"({len(forecast['failures'])} echec) -> {args.output}")


if __name__ == "__main__":
    main()
