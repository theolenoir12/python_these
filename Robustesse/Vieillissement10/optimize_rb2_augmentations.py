"""Optimisation reproductible des couches RB2(SoH), RB2(RUL) et RB2(Pred).

La base RB2 reste fixee a ses deux setpoints V10 optimises (0.59, 0.49), afin
que le gain mesure soit attribuable a la couche ajoutee. Chaque grille contient
un cas nul qui retombe exactement sur RB2. L'objectif unique est :

    cout unifie [kEUR] = degradation [kEUR] + 3 EUR/kWh * EENS / 1000.

Exemples :
    python optimize_rb2_augmentations.py --layer suite
    python optimize_rb2_augmentations.py --layer pred --workers 8 --seeds 5
    python optimize_rb2_augmentations.py --layer suite --smoke
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from time import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VOLL_EUR_PER_KWH = 3.0
BASE = {"fc_setpoint": 0.59, "ely_setpoint": 0.49}


def _unique(configs):
    seen = set()
    result = []
    for config in configs:
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def _forecast_config(horizon, target, margin, dwell, enabled=True):
    return {
        "forecast_enabled": enabled,
        "forecast_horizon_h": float(horizon),
        "forecast_soc_target": float(target),
        "forecast_noise_enabled": bool(enabled),
        "forecast_bias_kwh": -2.32,
        "forecast_sigma_kwh": 39.38,
        "forecast_noise_rho": 0.0,
        "forecast_hysteresis_sigma": float(margin),
        "forecast_min_dwell_h": float(dwell),
    }


def build_grid(layer, smoke=False):
    """Construit une grille qui inclut toujours le cas nul RB2."""
    if layer == "rb2_extended":
        fc_values = [0.59, 0.75] if smoke else [0.45, 0.55, 0.59, 0.65, 0.75, 0.85]
        ely_values = [0.49, 0.65] if smoke else [0.35, 0.45, 0.49, 0.55, 0.65, 0.75]
        return [
            {"fc_setpoint": fc, "ely_setpoint": ely}
            for fc, ely in itertools.product(fc_values, ely_values)
        ]

    if layer == "rb2_refined":
        fc_values = [0.59, 0.62] if smoke else [
            round(0.57 + 0.01 * i, 2) for i in range(9)
        ]
        ely_values = [0.49, 0.53] if smoke else [
            round(0.47 + 0.01 * i, 2) for i in range(11)
        ]
        return [
            {"fc_setpoint": fc, "ely_setpoint": ely}
            for fc, ely in itertools.product(fc_values, ely_values)
        ]

    if layer == "rb2_validation25":
        fc_values = [0.60, 0.62] if smoke else [0.59, 0.60, 0.61, 0.62, 0.63]
        ely_values = [0.51, 0.53] if smoke else [0.50, 0.51, 0.52, 0.53, 0.54]
        return [
            {"fc_setpoint": fc, "ely_setpoint": ely}
            for fc, ely in itertools.product(fc_values, ely_values)
        ]

    if layer == "soh":
        values_fc = [0.0, 0.5] if smoke else [0.0, 0.25, 0.5, 1.0, 1.5]
        values_ely = [0.0, 1.5] if smoke else [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
        return [
            dict(BASE, soh_gamma_fc=g_fc, soh_gamma_ely=g_ely)
            for g_fc, g_ely in itertools.product(values_fc, values_ely)
        ]

    if layer == "soh_normalized":
        strengths = [0.0, 0.5] if smoke else [0.0, 0.25, 0.50, 0.75]
        shapes = [2.0] if smoke else [1.0, 2.0, 4.0, 8.0]
        configs = []
        for strength_fc, strength_ely, shape in itertools.product(
            strengths, strengths, shapes
        ):
            configs.append(dict(
                BASE,
                soh_mode="normalized_wear",
                soh_strength_fc=strength_fc,
                soh_strength_ely=strength_ely,
                soh_shape_fc=shape if strength_fc > 0.0 else 1.0,
                soh_shape_ely=shape if strength_ely > 0.0 else 1.0,
            ))
        return _unique(configs)

    if layer == "soh_joint_targeted":
        # Le meilleur RB2 constant (0.59/0.49) est inclus. Les bases ELY plus
        # hautes achètent de la fiabilité en début de vie ; la couche SoH peut
        # ensuite récupérer une partie de leur surcoût de vieillissement.
        configs = [dict(BASE, soh_mode="normalized_wear")]
        for ely_setpoint in ([0.55] if smoke else [0.55, 0.65]):
            configs.append(dict(
                BASE, ely_setpoint=ely_setpoint,
                soh_mode="normalized_wear",
            ))
            for strength_ely in ([0.25] if smoke else [0.25, 0.5, 0.75]):
                for shape in ([4.0] if smoke else [1.0, 2.0, 4.0, 8.0]):
                    configs.append(dict(
                        BASE, ely_setpoint=ely_setpoint,
                        soh_mode="normalized_wear",
                        soh_strength_ely=strength_ely,
                        soh_shape_ely=shape,
                    ))
            if ely_setpoint == 0.55:
                for strength_ely in ([0.25] if smoke else [0.25, 0.5]):
                    for shape in ([8.0] if smoke else [4.0, 8.0]):
                        configs.append(dict(
                            BASE, ely_setpoint=ely_setpoint,
                            soh_mode="normalized_wear",
                            soh_strength_fc=0.25,
                            soh_strength_ely=strength_ely,
                            soh_shape_fc=shape,
                            soh_shape_ely=shape,
                        ))
        return _unique(configs)

    if layer == "soh_isocost_stage5":
        bases = [(0.61, 0.52), (0.61, 0.53), (0.60, 0.52)]
        strengths = [0.05, 0.10] if smoke else [0.05, 0.10, 0.15, 0.20, 0.25]
        shapes = [8.0] if smoke else [1.0, 8.0]
        configs = []
        for fc_setpoint, ely_setpoint in bases:
            base = {
                "fc_setpoint": fc_setpoint,
                "ely_setpoint": ely_setpoint,
                "soh_mode": "normalized_wear",
            }
            configs.append(dict(base))
            for strength, shape in itertools.product(strengths, shapes):
                configs.extend([
                    dict(base, soh_strength_fc=strength, soh_shape_fc=shape),
                    dict(base, soh_strength_ely=strength, soh_shape_ely=shape),
                    dict(
                        base,
                        soh_strength_fc=strength,
                        soh_strength_ely=strength,
                        soh_shape_fc=shape,
                        soh_shape_ely=shape,
                    ),
                ])
        return _unique(configs)

    if layer == "soh_isocost_validation25":
        bases = [(0.61, 0.53), (0.61, 0.52), (0.60, 0.51), (0.59, 0.50)]
        configs = []
        for fc_setpoint, ely_setpoint in bases:
            base = {
                "fc_setpoint": fc_setpoint,
                "ely_setpoint": ely_setpoint,
                "soh_mode": "normalized_wear",
            }
            configs.append(dict(base))
            for strength_ely in [0.05, 0.10, 0.15, 0.20]:
                configs.append(dict(
                    base,
                    soh_strength_ely=strength_ely,
                    soh_shape_ely=1.0,
                ))
            configs.append(dict(
                base, soh_strength_fc=0.05, soh_shape_fc=1.0,
            ))
            for strength in [0.05, 0.10]:
                configs.append(dict(
                    base,
                    soh_strength_fc=strength,
                    soh_strength_ely=strength,
                    soh_shape_fc=1.0,
                    soh_shape_ely=1.0,
                ))
        return _unique(configs)

    if layer == "soh_isocost_asymmetric25":
        values = [0.05, 0.075, 0.10] if smoke else [
            0.05, 0.075, 0.10, 0.125, 0.15
        ]
        return [
            {
                "fc_setpoint": 0.61,
                "ely_setpoint": 0.52,
                "soh_mode": "normalized_wear",
                "soh_strength_fc": strength_fc,
                "soh_strength_ely": strength_ely,
                "soh_shape_fc": 1.0,
                "soh_shape_ely": 1.0,
            }
            for strength_fc, strength_ely in itertools.product(values, values)
        ]

    if layer == "soh_validated25":
        # Grille fine post-validation de la loi batterie. Le cas (0, 0)
        # reproduit exactement RB2(0.59/0.49) ; les faibles modulations sont
        # prioritaires car les grilles precedentes situaient l'iso-cout dans
        # cette zone. RB2(SoH) reste exclusivement une modulation de setpoints
        # H2, sans plafond de puissance ajoute.
        values = [0.0, 0.075] if smoke else [
            0.0, 0.025, 0.05, 0.075, 0.10, 0.15,
        ]
        return [
            {
                **BASE,
                "soh_mode": "normalized_wear",
                "soh_strength_fc": strength_fc,
                "soh_strength_ely": strength_ely,
                "soh_shape_fc": 1.0,
                "soh_shape_ely": 1.0,
            }
            for strength_fc, strength_ely in itertools.product(values, values)
        ]

    if layer == "soh_validated_shapes25":
        # Verification ciblee de la forme temporelle autour des meilleurs
        # points lineaires ; pas de nouvelle exploration exhaustive.
        pairs = [
            (0.0, 0.0), (0.025, 0.0), (0.025, 0.025),
            (0.05, 0.025), (0.05, 0.05),
        ]
        shapes = [2.0] if smoke else [1.0, 2.0, 4.0, 8.0]
        configs = []
        for strength_fc, strength_ely in pairs:
            pair_shapes = [1.0] if strength_fc == strength_ely == 0.0 else shapes
            for shape in pair_shapes:
                configs.append({
                    **BASE,
                    "soh_mode": "normalized_wear",
                    "soh_strength_fc": strength_fc,
                    "soh_strength_ely": strength_ely,
                    "soh_shape_fc": shape,
                    "soh_shape_ely": shape,
                })
        return configs

    if layer == "rul":
        refs_fc = [3000.0] if smoke else [3000.0, 6000.0]
        refs_ely = [8000.0] if smoke else [3000.0, 5000.0, 8000.0]
        gammas_fc = [0.0, 0.05] if smoke else [0.0, 0.05, 0.10]
        gammas_ely = [0.0, 0.05] if smoke else [0.0, 0.05, 0.10, 0.20]
        configs = []
        for r_fc, r_ely, g_fc, g_ely in itertools.product(
            refs_fc, refs_ely, gammas_fc, gammas_ely
        ):
            configs.append(dict(
                BASE, rul_ref_fc_days=r_fc, rul_ref_ely_days=r_ely,
                rul_gamma_fc=g_fc, rul_gamma_ely=g_ely,
            ))
        return _unique(configs)

    if layer == "pred":
        if smoke:
            pred_configs = [
                _forecast_config(18, 0.99, 1.0, 12, False),
                _forecast_config(18, 0.99, 1.0, 12, True),
            ]
        else:
            pred_configs = [_forecast_config(18, 0.99, 1.0, 12, False)]
            pred_configs += [
                _forecast_config(h, target, margin, dwell, True)
                for h, target, margin, dwell in itertools.product(
                    [12, 18, 24, 36], [0.90, 0.99],
                    [0.5, 1.0, 1.5], [0, 12],
                )
            ]
        return [dict(BASE, **config) for config in pred_configs]

    if layer == "all":
        # Grille combinee centree sur les optima individuels V10 observes par
        # les trois phases precedentes. On conserve les cas nuls et un voisin
        # de chaque optimum pour verifier les interactions sans refaire le
        # produit cartesien exhaustif de toutes les grilles.
        soh = (
            [(0.0, 0.0), (0.5, 1.5)] if smoke else
            [(0.0, 0.0), (0.25, 0.0), (0.25, 0.25)]
        )
        rul = (
            [(3000.0, 8000.0, 0.0, 0.0), (3000.0, 8000.0, 0.0, 0.05)]
            if smoke else [
                (3000.0, 3000.0, 0.0, 0.0),
                (3000.0, 3000.0, 0.0, 0.05),
            ]
        )
        pred = (
            [_forecast_config(18, 0.99, 1.0, 12, False),
             _forecast_config(18, 0.99, 1.0, 12, True)]
            if smoke else [
                _forecast_config(18, 0.99, 1.0, 12, False),
                _forecast_config(24, 0.99, 1.5, 0, True),
                _forecast_config(18, 0.90, 1.5, 0, True),
            ]
        )
        configs = []
        for (gs_fc, gs_ely), (rr_fc, rr_ely, gr_fc, gr_ely), forecast in itertools.product(soh, rul, pred):
            configs.append(dict(
                BASE, soh_gamma_fc=gs_fc, soh_gamma_ely=gs_ely,
                rul_ref_fc_days=rr_fc, rul_ref_ely_days=rr_ely,
                rul_gamma_fc=gr_fc, rul_gamma_ely=gr_ely, **forecast,
            ))
        return _unique(configs)

    raise ValueError("couche inconnue : %s" % layer)


def _evaluate(task):
    from Common.main_init_and_loop import init_and_run_loop
    from Common.rb2_policy import make_augmented_rb2_policy
    from Common.reliability_metrics import compute_reliability_metrics

    params, years, seeds = task
    rows = []
    for seed in seeds:
        run_params = dict(params)
        run_params["forecast_seed"] = int(seed)
        policy = make_augmented_rb2_policy(**run_params)
        data = init_and_run_loop(
            policy, n_years=years, replacement_accounting="corrected"
        )
        reliability = compute_reliability_metrics(data)
        ledger = data["degradation_ledger"]
        degradation = sum(ledger["total_eur"].values()) / 1000.0
        unified = degradation + VOLL_EUR_PER_KWH * reliability["eens_kwh"] / 1000.0
        rows.append((
            reliability["lpsp_pct"], reliability["eens_kwh"],
            degradation, unified,
        ))

    n = float(len(rows))
    means = [sum(row[k] for row in rows) / n for k in range(4)]
    std_unified = (
        sum((row[3] - means[3]) ** 2 for row in rows) / n
    ) ** 0.5
    return {
        "params": params,
        "lpsp_pct": means[0],
        "eens_kwh": means[1],
        "degradation_keur": means[2],
        "unified_keur": means[3],
        "unified_std_keur": std_unified,
        "n_seeds": len(rows),
    }


def _write_results(layer, rows, years, output_dir):
    rows = sorted(rows, key=lambda row: row["unified_keur"])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "optimization_%s.csv" % layer)
    txt_path = os.path.join(output_dir, "optimization_%s.txt" % layer)
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "rank", "lpsp_pct", "eens_kwh", "degradation_keur",
            "unified_keur", "unified_std_keur", "n_seeds", "parameters_json",
        ])
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({
                "rank": rank,
                "lpsp_pct": "%.8f" % row["lpsp_pct"],
                "eens_kwh": "%.6f" % row["eens_kwh"],
                "degradation_keur": "%.6f" % row["degradation_keur"],
                "unified_keur": "%.6f" % row["unified_keur"],
                "unified_std_keur": "%.6f" % row["unified_std_keur"],
                "n_seeds": row["n_seeds"],
                "parameters_json": json.dumps(row["params"], sort_keys=True),
            })

    baseline = next((row for row in rows if _is_null(row["params"])), None)
    with open(txt_path, "w", encoding="utf-8") as stream:
        stream.write(
            "# RB2(%s), %d ans, VoLL=%.1f EUR/kWh, LPSP charge totale\n"
            % (layer, years, VOLL_EUR_PER_KWH)
        )
        if baseline:
            stream.write(
                "# Cas nul RB2: LPSP=%.5f%% deg=%.5f kEUR unifie=%.5f kEUR\n"
                % (baseline["lpsp_pct"], baseline["degradation_keur"], baseline["unified_keur"])
            )
        stream.write("# Rang;LPSP[%];EENS[kWh];deg[kEUR];unifie[kEUR];sigma_unifie;parametres\n")
        for rank, row in enumerate(rows, 1):
            stream.write(
                "%d;%.6f;%.3f;%.6f;%.6f;%.6f;%s\n"
                % (rank, row["lpsp_pct"], row["eens_kwh"],
                   row["degradation_keur"], row["unified_keur"],
                   row["unified_std_keur"], json.dumps(row["params"], sort_keys=True))
            )
    return rows[0], csv_path, txt_path


def _is_null(params):
    return (
        float(params.get("soh_gamma_fc", 0.0)) == 0.0
        and float(params.get("soh_gamma_ely", 0.0)) == 0.0
        and float(params.get("soh_strength_fc", 0.0)) == 0.0
        and float(params.get("soh_strength_ely", 0.0)) == 0.0
        and float(params.get("rul_gamma_fc", 0.0)) == 0.0
        and float(params.get("rul_gamma_ely", 0.0)) == 0.0
        and not bool(params.get("forecast_enabled", False))
    )


def run_layer(layer, args):
    grid = build_grid(layer, args.smoke)
    years = 1 if args.smoke else args.years
    uses_prediction = layer in ("pred", "all")
    seeds = list(range(args.seeds)) if uses_prediction else [0]
    tasks = [(params, years, seeds) for params in grid]
    workers = max(1, min(args.workers, len(tasks)))
    print(
        "[%s] %d configurations, %d an(s), %d worker(s), %d graine(s)"
        % (layer, len(tasks), years, workers, len(seeds)), flush=True,
    )
    started = time()
    if workers == 1:
        rows = [_evaluate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_evaluate, tasks))
    best, csv_path, txt_path = _write_results(layer, rows, years, args.output_dir)
    print(
        "[%s] optimum %.5f kEUR, LPSP %.5f%%, deg %.5f kEUR (%.0f s)"
        % (layer, best["unified_keur"], best["lpsp_pct"],
           best["degradation_keur"], time() - started), flush=True,
    )
    print("  %s\n  %s" % (csv_path, txt_path), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", choices=(
            "rb2_extended", "rb2_refined", "rb2_validation25", "soh",
            "soh_normalized", "soh_joint_targeted", "soh_isocost_stage5",
            "soh_isocost_validation25", "soh_isocost_asymmetric25",
            "soh_validated25", "soh_validated_shapes25",
            "rul", "pred", "all", "suite"
        ),
        default="suite",
    )
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir", default=os.path.join(HERE, "Optimization_results")
    )
    args = parser.parse_args()
    if args.years <= 0 or args.workers <= 0 or args.seeds <= 0:
        parser.error("years, workers et seeds doivent etre strictement positifs")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    layers = ("soh", "rul", "pred", "all") if arguments.layer == "suite" else (arguments.layer,)
    for selected_layer in layers:
        run_layer(selected_layer, arguments)
