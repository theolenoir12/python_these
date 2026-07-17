"""Recherche reproductible d'une RB2(Aging), sans information future."""

from __future__ import annotations

import argparse
import csv
import os
from time import time

import numpy as np

from Common.main_init_and_loop import init_and_run_loop
from Common.rb2_aging_policy import make_aging_rb2_policy
from Common.rb2_policy import make_rb2_policy
from Common.reliability_metrics import compute_reliability_metrics


VOLL_EUR_PER_KWH = 3.0
BASE = {"fc_setpoint": 0.59, "ely_setpoint": 0.49}
DISABLED = {
    **BASE,
    "fc_hold_setpoint": 0.59, "ely_hold_setpoint": 0.49,
    "fc_min_on_h": 0.0, "ely_min_on_h": 0.0,
    "fc_reversible_trigger_uv": float("inf"),
    "ely_reversible_trigger_uv": float("inf"),
    "fc_recovery_h": 0.0, "ely_recovery_h": 0.0,
    "permanent_strength_fc": 0.0, "permanent_strength_ely": 0.0,
}


def _starts(power):
    on = np.abs(np.asarray(power, dtype=float)) > 1e-9
    return int(np.count_nonzero(on[1:] & ~on[:-1]))


def evaluate(label, policy, parameters, years):
    started = time()
    data = init_and_run_loop(policy, n_years=years, replacement_accounting="corrected")
    reliability = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]["total_eur"]
    degradation = float(sum(ledger.values()))
    unified = degradation + VOLL_EUR_PER_KWH * reliability["eens_kwh"]
    row = {
        "label": label, "years": years,
        "eens_kwh": reliability["eens_kwh"],
        "lpsp_pct": reliability["lpsp_pct"],
        "bat_eur": ledger["bat"], "fc_eur": ledger["fc"],
        "ely_eur": ledger["ely"], "degradation_eur": degradation,
        "unified_eur": unified, "fc_starts": _starts(data["P_fc"]),
        "ely_starts": _starts(data["P_ely"]),
        "fc_soh_permanent": float(data["SoH_fc"][-1]),
        "ely_soh_permanent": float(data["SoH_ely"][-1]),
        "fc_soh_operando": float(data["SoH_fc_operando"][-1]),
        "ely_soh_operando": float(data["SoH_ely_operando"][-1]),
        "runtime_s": time() - started,
        "parameters": repr(parameters),
    }
    print(
        f"{label:24s} unified={unified:10.2f} EUR "
        f"deg={degradation:10.2f} EENS={reliability['eens_kwh']:9.2f} "
        f"starts FC/ELY={row['fc_starts']}/{row['ely_starts']} "
        f"({row['runtime_s']:.1f}s)", flush=True,
    )
    return row


def quick_candidates():
    variants = [
        ("aging_disabled", {}),
        ("fc_hold_050_4h", {"fc_hold_setpoint": .50, "fc_min_on_h": 4}),
        ("fc_hold_045_4h", {"fc_hold_setpoint": .45, "fc_min_on_h": 4}),
        ("fc_hold_050_8h", {"fc_hold_setpoint": .50, "fc_min_on_h": 8}),
        ("ely_hold_042_2h", {"ely_hold_setpoint": .42, "ely_min_on_h": 2}),
        ("holds_fc_ely", {
            "fc_hold_setpoint": .50, "fc_min_on_h": 4,
            "ely_hold_setpoint": .42, "ely_min_on_h": 2,
        }),
        ("holds_plus_recovery", {
            "fc_hold_setpoint": .50, "fc_min_on_h": 4,
            "ely_hold_setpoint": .42, "ely_min_on_h": 2,
            "fc_reversible_trigger_uv": 12_000.0, "fc_recovery_h": 2,
            "ely_reversible_trigger_uv": 12_000.0, "ely_recovery_h": 1,
        }),
        ("holds_plus_wear", {
            "fc_hold_setpoint": .50, "fc_min_on_h": 4,
            "ely_hold_setpoint": .42, "ely_min_on_h": 2,
            "permanent_strength_fc": .02, "permanent_strength_ely": .02,
        }),
    ]
    return [(label, {**DISABLED, **changes}) for label, changes in variants]


def full_candidates():
    candidates = quick_candidates()
    for hold in (.45, .50, .55):
        for dwell in (2, 4, 8):
            candidates.append((f"fc_h{hold:.2f}_d{dwell}", {
                **DISABLED, "fc_hold_setpoint": hold, "fc_min_on_h": dwell,
            }))
    for hold in (.38, .42, .46):
        for dwell in (1, 2, 4):
            candidates.append((f"ely_h{hold:.2f}_d{dwell}", {
                **DISABLED, "ely_hold_setpoint": hold, "ely_min_on_h": dwell,
            }))
    for trigger in (6_000.0, 12_000.0, 24_000.0):
        candidates.append((f"fc_recovery_{trigger:.0f}", {
            **DISABLED, "fc_reversible_trigger_uv": trigger, "fc_recovery_h": 2,
        }))
        candidates.append((f"ely_recovery_{trigger:.0f}", {
            **DISABLED, "ely_reversible_trigger_uv": trigger, "ely_recovery_h": 1,
        }))
    for strength in (.01, .02, .05):
        candidates.append((f"wear_{strength:.2f}", {
            **DISABLED, "permanent_strength_fc": strength,
            "permanent_strength_ely": strength,
        }))
    unique = {}
    for label, parameters in candidates:
        unique.setdefault(tuple(sorted(parameters.items())), (label, parameters))
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    rows = [evaluate("RB2", make_rb2_policy(**BASE), BASE, args.years)]
    for label, parameters in (full_candidates() if args.full else quick_candidates()):
        rows.append(evaluate(
            label, make_aging_rb2_policy(**parameters), parameters, args.years
        ))
    baseline = rows[0]["unified_eur"]
    for row in rows:
        row["gain_vs_rb2_pct"] = 100.0 * (baseline - row["unified_eur"]) / baseline
    rows.sort(key=lambda row: row["unified_eur"])

    suffix = str(args.years).replace(".", "p")
    output = os.path.join(os.path.dirname(__file__), f"benchmark_rb2_aging_v11_{suffix}y.csv")
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print("\nClassement:")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:2d}. {row['label']:24s} gain={row['gain_vs_rb2_pct']:+.4f}%")
    print("Resultats:", output)


if __name__ == "__main__":
    main()
