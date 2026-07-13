"""Sweep des plafonds de secours de RB2, au cout unifie inchange."""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

N_YEARS = 25
FC_BASE = 0.32
ELY_BASE = 0.22
FC_EMERGENCY = [0.80, 0.85, 0.90]
ELY_EMERGENCY = [0.225, 0.230, 0.235, 0.240, 0.245]
N_WORKERS = max(1, int(os.environ.get(
    "SLURM_CPUS_PER_TASK", min(10, (os.cpu_count() or 2) - 1)
)))
OUT_TXT = os.path.join(HERE, "sweep_reserve_rb2.txt")
OUT_PDF = os.path.join(HERE, "sweep_reserve_rb2.pdf")
OUT_PNG = os.path.join(HERE, "sweep_reserve_rb2.png")


def metrics(data):
    planned = np.maximum(
        (np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"])) / 1000, 0
    )
    real = np.maximum(
        (np.asarray(data["P_dc_load"]) - np.asarray(data["P_dc_pv"]))
        * (1 - np.asarray(data["lol_tab"])) / 1000, 0
    )
    lpsp = float(np.maximum(planned - real, 0).sum() / planned.sum() * 100)
    degradation = float(get_cost_from_ledger(data) / 1000)
    return lpsp, degradation


def evaluate(args):
    fc_emergency, ely_emergency, years = args
    policy = make_rb2_policy(
        FC_BASE, ELY_BASE, fc_emergency, ely_emergency
    )
    data = init_and_run_loop(policy, n_years=years)
    lpsp, degradation = metrics(data)
    unified = V.total_cost_keur(lpsp, degradation)
    ledger = data["degradation_ledger"]["total_eur"]
    ely_life = data["first_life_metrics"]["ely"]
    return (
        fc_emergency, ely_emergency, lpsp, degradation, unified,
        ledger["bat"] / 1000, ledger["fc"] / 1000, ledger["ely"] / 1000,
        ely_life["calendar_years_8760"], ely_life["on_h"],
        float(np.min(data["E_h2"])),
    )


def run(smoke=False):
    years = 2 if smoke else N_YEARS
    if smoke:
        combinations = [(0.7, 0.35, years), (0.8, 0.4, years)]
    else:
        combinations = [
            (float(fc), float(ely), years)
            for fc in FC_EMERGENCY for ely in ELY_EMERGENCY
        ]
    results = []
    t0 = time.time()
    workers = min(N_WORKERS, len(combinations))
    print(
        f"RB2 secours: {len(combinations)} simulations / {years} ans, "
        f"base={FC_BASE:.2f}/{ELY_BASE:.2f}, workers={workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(evaluate, combinations), 1):
            results.append(result)
            print(
                f"[{index:02d}/{len(combinations)}] "
                f"secours={result[0]:.2f}/{result[1]:.2f} "
                f"LPSP={result[2]:.4f}% deg={result[3]:.3f} "
                f"unifie={result[4]:.3f} kEUR",
                flush=True,
            )
    results.sort(key=lambda row: row[4])
    best = results[0]
    print(
        f"OPTIMUM secours={best[0]:.3f}/{best[1]:.3f}, "
        f"LPSP={best[2]:.4f}%, deg={best[3]:.3f}, "
        f"unifie={best[4]:.3f} kEUR ({time.time()-t0:.0f}s)",
        flush=True,
    )
    with open(OUT_TXT, "w", encoding="utf-8") as stream:
        stream.write(
            f"# RB2 reserve - {years} ans; base={FC_BASE:.3f}/{ELY_BASE:.3f}; "
            f"VoLL={V.VOLL_TIERS}\n"
        )
        stream.write(
            f"# OPTIMUM secours={best[0]:.3f}/{best[1]:.3f}; "
            f"LPSP={best[2]:.4f}; deg={best[3]:.4f}; unifie={best[4]:.4f}\n"
        )
        stream.write(
            "rang;fc_emergency;ely_emergency;LPSP(%);deg(kEUR);"
            "unifie(kEUR);bat(kEUR);fc(kEUR);ely(kEUR);"
            "ely_life_years;ely_first_on_h;h2_min_kWh\n"
        )
        for rank, row in enumerate(results, 1):
            stream.write(
                f"{rank};{row[0]:.3f};{row[1]:.3f};{row[2]:.4f};"
                f"{row[3]:.4f};{row[4]:.4f};{row[5]:.4f};{row[6]:.4f};"
                f"{row[7]:.4f};{row[8]:.4f};{row[9]:.0f};{row[10]:.4f}\n"
            )
    return results


def plot(results):
    fc = np.asarray([row[0] for row in results])
    ely = np.asarray([row[1] for row in results])
    unified = np.asarray([row[4] for row in results])
    lpsp = np.asarray([row[2] for row in results])
    best = min(results, key=lambda row: row[4])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    scatter = axes[0].scatter(fc, ely, c=unified, s=90, cmap="viridis_r")
    axes[0].scatter(best[0], best[1], marker="*", s=300, color="crimson")
    axes[0].set(xlabel="PEMFC emergency cap [Pmax fraction]",
                ylabel="PEMWE emergency cap [Pmax fraction]")
    fig.colorbar(scatter, ax=axes[0], label="Unified cost [kEUR]")
    sc2 = axes[1].scatter(lpsp, [row[3] for row in results], c=unified,
                          s=75, cmap="viridis_r")
    axes[1].scatter(best[2], best[3], marker="*", s=300, color="crimson")
    axes[1].set(xlabel="LPSP [%]", ylabel="Degradation cost [kEUR]")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle("RB2: economic setpoints and emergency power caps")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = run(smoke=args.smoke)
    plot(rows)


