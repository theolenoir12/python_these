"""Coarse sweep of the RB2 upper SoC reserve and PEMWE emergency cap."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)

from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from rb2_policy import make_rb2_policy

sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V

FC_BASE = 0.32
ELY_BASE = 0.22
FC_EMERGENCY = 0.85
SOC_LOW = 0.20
SOC_HIGHS = [0.35, 0.50, 0.70, 0.90, 0.995]
ELY_EMERGENCIES = [0.24, 0.40, 0.60, 1.00]
N_WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", min(10, (os.cpu_count() or 2) - 1))))
OUT = os.path.join(HERE, "sweep_soc_reserve_rb2.txt")


def evaluate(args):
    soc_high, ely_emergency = args
    policy = make_rb2_policy(
        FC_BASE, ELY_BASE, FC_EMERGENCY, ely_emergency, SOC_LOW, soc_high
    )
    data = init_and_run_loop(policy, n_years=25)
    planned = np.maximum(
        (np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"])) / 1000, 0
    )
    real = np.maximum(
        (np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"]))
        * (1 - np.asarray(data["lol_tab"])) / 1000, 0
    )
    lpsp = float(np.maximum(planned - real, 0).sum() / planned.sum() * 100)
    degradation = float(get_cost_from_ledger(data) / 1000)
    unified = V.total_cost_keur(lpsp, degradation)
    ledger = data["degradation_ledger"]["total_eur"]
    return (
        soc_high, ely_emergency, lpsp, degradation, unified,
        ledger["bat"] / 1000, ledger["fc"] / 1000, ledger["ely"] / 1000,
        float(np.min(data["E_h2"])),
    )


def main():
    combinations = [(high, cap) for high in SOC_HIGHS for cap in ELY_EMERGENCIES]
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(combinations))) as executor:
        results = list(executor.map(evaluate, combinations))
    results.sort(key=lambda row: row[4])
    best = results[0]
    with open(OUT, "w", encoding="utf-8") as stream:
        stream.write(
            f"# OPTIMUM soc_high={best[0]:.3f}; ely_emergency={best[1]:.3f}; "
            f"LPSP={best[2]:.4f}; deg={best[3]:.4f}; unifie={best[4]:.4f}\n"
        )
        stream.write("rank;soc_high;ely_emergency;LPSP;deg;unified;bat;fc;ely;h2_min\n")
        for rank, row in enumerate(results, 1):
            stream.write(
                f"{rank};{row[0]:.3f};{row[1]:.3f};{row[2]:.4f};{row[3]:.4f};"
                f"{row[4]:.4f};{row[5]:.4f};{row[6]:.4f};{row[7]:.4f};{row[8]:.4f}\n"
            )
            print(
                f"{rank:02d} high={row[0]:.3f} cap={row[1]:.2f} "
                f"LPSP={row[2]:.4f} deg={row[3]:.3f} unified={row[4]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
