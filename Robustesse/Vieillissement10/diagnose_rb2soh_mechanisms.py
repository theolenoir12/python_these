"""Diagnostic mecaniste de RB2(SoH) : usage, demarrages et couts par poste."""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


CASES = [
    ("RB2", {}),
    ("SoH_opt_V10", {"soh_gamma_fc": 0.25}),
    ("SoH_historique", {"soh_gamma_fc": 0.5, "soh_gamma_ely": 1.5}),
    ("ELY_g2", {"soh_gamma_ely": 2.0}),
    ("ELY_g5", {"soh_gamma_ely": 5.0}),
    ("ELY_g10", {"soh_gamma_ely": 10.0}),
    ("FC_g5", {"soh_gamma_fc": 5.0}),
    ("FC5_ELY5", {"soh_gamma_fc": 5.0, "soh_gamma_ely": 5.0}),
    ("SoH_Pareto_1p10", {
        "soh_mode": "normalized_wear",
        "soh_strength_fc": 0.25, "soh_strength_ely": 0.25,
        "soh_shape_fc": 1.0, "soh_shape_ely": 1.0,
    }),
    ("SoH_min_deg", {
        "soh_mode": "normalized_wear",
        "soh_strength_fc": 0.50, "soh_strength_ely": 0.75,
        "soh_shape_fc": 1.0, "soh_shape_ely": 1.0,
    }),
]


def _component_usage(data, component):
    from Common.electrochemistry import (
        ely_current_density, ely_pmax, fc_current_density, fc_pmax,
    )

    if component == "fc":
        power = np.abs(np.asarray(data["P_fc"], dtype=float))
        alpha = np.asarray(data["alpha_fc"][:-1], dtype=float)
        pmax = np.asarray(fc_pmax(alpha), dtype=float)
        density = np.asarray(fc_current_density(power, alpha), dtype=float)
    else:
        power = np.abs(np.asarray(data["P_ely"], dtype=float))
        alpha = np.asarray(data["alpha_ely"][:-1], dtype=float)
        pmax = np.asarray(ely_pmax(alpha), dtype=float)
        density = np.asarray(ely_current_density(power, alpha), dtype=float)
    on = power >= 0.0005 * pmax
    starts = int(on[0]) + int(np.sum((~on[:-1]) & on[1:])) if len(on) else 0
    dt_h = 1.0
    return {
        "on_h": float(np.sum(on) * dt_h),
        "starts": starts,
        "energy_kwh": float(np.sum(power) * dt_h / 1000.0),
        "mean_power_on_w": float(np.mean(power[on])) if np.any(on) else 0.0,
        "mean_j_on": float(np.mean(density[on])) if np.any(on) else 0.0,
    }


def _mechanism_costs(data, component):
    from Common import Init_EMR_MG_v16_python as I

    ledger = data["degradation_ledger"]
    deg = data["deg_fc" if component == "fc" else "deg_ely"]
    keys = (
        ("start-stop", "idling", "reversible", "irreversible")
        if component == "fc" else
        ("start-stop", "maintaining", "reversible", "irreversible")
    )
    terminal_indices = [
        int(event["stop_step_exclusive"]) - 1
        for event in ledger["events"] if event["component"] == component
    ]
    current_start = int(ledger["current_start_step"][component])
    if current_start < len(data["P_fc"]):
        terminal_indices.append(len(data["P_fc"]) - 1)
    item = I.FC if component == "fc" else I.ELY
    factor = item["cost"] / ((1.0 - item["SoH_EoL"]) * 100.0)
    return {
        key: float(sum(float(deg[key][idx]) for idx in terminal_indices) * factor / 1000.0)
        for key in keys
    }


def evaluate(case):
    from Common.main_init_and_loop import init_and_run_loop
    from Common.rb2_policy import make_augmented_rb2_policy
    from Common.reliability_metrics import compute_reliability_metrics

    name, parameters = case
    policy = make_augmented_rb2_policy(
        fc_setpoint=0.59, ely_setpoint=0.49,
        **parameters,
    )
    data = init_and_run_loop(policy, n_years=25, replacement_accounting="corrected")
    rel = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]
    component_costs = {
        key: float(value / 1000.0) for key, value in ledger["total_eur"].items()
    }
    replacements = {
        key: sum(event["component"] == key for event in ledger["events"])
        for key in ("bat", "fc", "ely")
    }
    fc_usage = _component_usage(data, "fc")
    ely_usage = _component_usage(data, "ely")
    fc_mechanisms = _mechanism_costs(data, "fc")
    ely_mechanisms = _mechanism_costs(data, "ely")
    degradation = sum(component_costs.values())
    return {
        "name": name,
        "parameters": parameters,
        "lpsp_pct": rel["lpsp_pct"],
        "eens_kwh": rel["eens_kwh"],
        "degradation_keur": degradation,
        "unified_keur": degradation + 3.0 * rel["eens_kwh"] / 1000.0,
        "component_costs_keur": component_costs,
        "replacements": replacements,
        "fc_usage": fc_usage,
        "ely_usage": ely_usage,
        "fc_mechanisms_keur": fc_mechanisms,
        "ely_mechanisms_keur": ely_mechanisms,
    }


def main():
    workers = max(1, min(len(CASES), int(os.environ.get("SLURM_CPUS_PER_TASK", 8))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluate, CASES))
    rows.sort(key=lambda row: row["unified_keur"])

    json_path = os.path.join(HERE, "DIAGNOSTIC_RB2SOH_MECANISMES.json")
    csv_path = os.path.join(HERE, "DIAGNOSTIC_RB2SOH_MECANISMES.csv")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for group in (
            "component_costs_keur", "replacements", "fc_usage", "ely_usage",
            "fc_mechanisms_keur", "ely_mechanisms_keur",
        ):
            for key, value in row[group].items():
                flat["%s.%s" % (group, key)] = value
        flat_rows.append(flat)
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    for row in rows:
        print(
            "%-18s LPSP=%7.4f%% deg=%8.4f unif=%8.4f "
            "FC(on=%6.0fh,start=%5d,cost=%6.2f) ELY(on=%6.0fh,start=%5d,cost=%6.2f)"
            % (
                row["name"],
                row["lpsp_pct"], row["degradation_keur"], row["unified_keur"],
                row["fc_usage"]["on_h"], row["fc_usage"]["starts"],
                row["component_costs_keur"]["fc"],
                row["ely_usage"]["on_h"], row["ely_usage"]["starts"],
                row["component_costs_keur"]["ely"],
            )
        )
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
