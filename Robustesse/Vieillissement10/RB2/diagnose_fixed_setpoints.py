"""Diagnostic d'une RB2 historique a deux consignes fixes."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from Common import cost_fcn_total2 as C
from Common import main_init_and_loop as M
from Common.reliability_metrics import compute_reliability_metrics
from rb2_policy import make_rb2_policy


def diagnose(fc_frac: float, ely_frac: float, years: int, mode: str) -> dict:
    if mode == "no-h2-aging":
        C.FC_REC["scale"] = 0.0
        C.ELY_REC["scale"] = 0.0
    elif mode == "no-bat-aging":
        M.get_cost_bat = lambda *args, **kwargs: 0.0

    data = M.init_and_run_loop(make_rb2_policy(fc_frac, ely_frac), n_years=years)
    reliability = compute_reliability_metrics(data)

    load = np.clip(np.asarray(data["P_dc_load"], dtype=float) / 1000.0, 0.0, None)
    residual = np.clip(
        (data["P_dc_load"] - data["P_dc_pv"]) / 1000.0, 0.0, None
    )
    unserved = residual * data["lol_tab"]
    bounds = np.linspace(0, len(residual), years + 1, dtype=int)
    yearly = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        denominator = float(load[start:stop].sum())
        yearly.append(
            100.0 * float(unserved[start:stop].sum()) / denominator
            if denominator > 0.0 else 0.0
        )

    replacements = {}
    for component in ("bat", "fc", "ely"):
        soh = np.asarray(data[f"SoH_{component}"])
        replacements[component] = int(np.count_nonzero(np.diff(soh) > 0.05))

    ledger = data["degradation_ledger"]["total_eur"]
    return {
        "policy": {"fc_frac": fc_frac, "ely_frac": ely_frac},
        "years": years,
        "mode": mode,
        **reliability,
        "lpsp_pct_by_year": yearly,
        "soc_min": float(np.min(data["SoC"])),
        "soc_low_steps": int(np.count_nonzero(data["SoC"] <= 0.20001)),
        "h2_min_kwh": float(np.min(data["E_h2"])),
        "h2_empty_steps": int(np.count_nonzero(data["E_h2"] <= 1e-6)),
        "replacements": replacements,
        "degradation_keur": {key: float(value) / 1000.0 for key, value in ledger.items()},
        "first_life_metrics": data.get("first_life_metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc", type=float, default=0.59)
    parser.add_argument("--ely", type=float, default=0.49)
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument(
        "--mode",
        choices=("baseline", "no-h2-aging", "no-bat-aging"),
        default="baseline",
    )
    args = parser.parse_args()
    print(json.dumps(diagnose(args.fc, args.ely, args.years, args.mode), indent=2))


if __name__ == "__main__":
    main()
