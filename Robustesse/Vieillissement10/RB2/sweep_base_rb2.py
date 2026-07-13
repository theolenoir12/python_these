"""Sweep final des consignes normales RB2 avec secours SoC fixes."""

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
FC_EMERGENCY = 0.90
ELY_EMERGENCY = 0.225
FC_BASES = [0.30, 0.31, 0.32, 0.33, 0.34]
ELY_BASES = [0.21, 0.22, 0.23]
N_WORKERS = max(1, int(os.environ.get(
    "SLURM_CPUS_PER_TASK", min(10, (os.cpu_count() or 2) - 1)
)))
OUT_TXT = os.path.join(HERE, "sweep_base_rb2.txt")
OUT_PDF = os.path.join(HERE, "sweep_base_rb2.pdf")
OUT_PNG = os.path.join(HERE, "sweep_base_rb2.png")


def evaluate(args):
    fc_base, ely_base = args
    data = init_and_run_loop(
        make_rb2_policy(fc_base, ely_base, FC_EMERGENCY, ELY_EMERGENCY),
        n_years=N_YEARS,
    )
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
        fc_base, ely_base, lpsp, degradation, unified,
        ledger["bat"] / 1000, ledger["fc"] / 1000, ledger["ely"] / 1000,
        float(np.min(data["E_h2"])),
        data["first_life_metrics"]["ely"]["calendar_years_8760"],
        data["first_life_metrics"]["ely"]["on_h"],
    )


def main():
    combinations = [(fc, ely) for fc in FC_BASES for ely in ELY_BASES]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(combinations))) as executor:
        for index, row in enumerate(executor.map(evaluate, combinations), 1):
            results.append(row)
            print(
                f"[{index:02d}/{len(combinations)}] base={row[0]:.2f}/{row[1]:.2f} "
                f"LPSP={row[2]:.4f}% deg={row[3]:.3f} "
                f"unifie={row[4]:.3f} kEUR",
                flush=True,
            )
    results.sort(key=lambda row: row[4])
    best = results[0]
    print(
        f"OPTIMUM base={best[0]:.3f}/{best[1]:.3f}, "
        f"secours={FC_EMERGENCY:.3f}/{ELY_EMERGENCY:.3f}, "
        f"LPSP={best[2]:.4f}%, deg={best[3]:.3f}, "
        f"unifie={best[4]:.3f} kEUR ({time.time()-t0:.0f}s)",
        flush=True,
    )
    with open(OUT_TXT, "w", encoding="utf-8") as stream:
        stream.write(
            f"# Sweep base RB2 - {N_YEARS} ans; "
            f"secours={FC_EMERGENCY:.3f}/{ELY_EMERGENCY:.3f}; VoLL={V.VOLL_TIERS}\n"
        )
        stream.write(
            f"# OPTIMUM base={best[0]:.3f}/{best[1]:.3f}; "
            f"LPSP={best[2]:.4f}; deg={best[3]:.4f}; unifie={best[4]:.4f}\n"
        )
        stream.write(
            "rang;fc_base;ely_base;LPSP(%);deg(kEUR);unifie(kEUR);"
            "bat(kEUR);fc(kEUR);ely(kEUR);h2_min_kWh;"
            "ely_life_years;ely_first_on_h\n"
        )
        for rank, row in enumerate(results, 1):
            stream.write(
                f"{rank};{row[0]:.3f};{row[1]:.3f};{row[2]:.4f};"
                f"{row[3]:.4f};{row[4]:.4f};{row[5]:.4f};{row[6]:.4f};"
                f"{row[7]:.4f};{row[8]:.4f};{row[9]:.4f};{row[10]:.0f}\n"
            )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(
        [row[2] for row in results], [row[3] for row in results],
        c=[row[4] for row in results], cmap="viridis_r", s=85,
    )
    for row in results:
        ax.annotate(
            f"{row[0]:.2f}/{row[1]:.2f}", (row[2], row[3]),
            xytext=(3, 3), textcoords="offset points", fontsize=7,
        )
    ax.scatter(best[2], best[3], marker="*", s=320, color="crimson")
    ax.set(xlabel="LPSP [%]", ylabel="Degradation cost [kEUR]",
           title="RB2 normal setpoints with SoC emergency dispatch")
    ax.grid(True, ls=":", alpha=0.4)
    fig.colorbar(sc, ax=ax, label="Unified cost [kEUR]")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

