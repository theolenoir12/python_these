"""Classement 25 ans des strategies de base, cout de defaillance inclus."""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from time import time as timer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Analyse_sensibilite")))

from Common.cost_fcn_total2 import ELY_REC
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics
import voll_common as V

SCENARIOS = ("0-100", "25-75", "50-50", "75-25", "100-0", "SoC1", "SoC06", "RB1", "RB2")
N_YEARS = 25
N_WORKERS = max(1, min(len(SCENARIOS), (os.cpu_count() or 2) - 1))
OUT = os.path.join(HERE, "rank_base_strategies_25y.txt")


def run_one(label):
    folder = os.path.join(HERE, label)
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)
    sys.modules.pop("get_optimal_action_RB", None)
    policy = importlib.import_module("get_optimal_action_RB").get_optimal_action_RB
    start = timer()
    data = init_and_run_loop(
        policy, n_years=N_YEARS, replacement_accounting="corrected"
    )
    rel = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]["total_eur"]
    deg = sum(ledger.values()) / 1000.0
    voll = V.voll_eur_per_kwh(rel["lpsp_pct"])
    unified = deg + voll * rel["eens_kwh"] / 1000.0
    result = {
        "label": label,
        **rel,
        "bat_keur": ledger["bat"] / 1000.0,
        "fc_keur": ledger["fc"] / 1000.0,
        "ely_keur": ledger["ely"] / 1000.0,
        "degradation_keur": deg,
        "unified_keur": unified,
    }
    print(
        f"{label:6s} LPSP={rel['lpsp_pct']:7.3f}% "
        f"deg={deg:8.3f} unifie={unified:8.3f} ({timer()-start:.0f}s)",
        flush=True,
    )
    return result


def main():
    print(
        f"Classement {N_YEARS} ans, VoLL={V.voll_eur_per_kwh(0.0)} EUR/kWh, "
        f"a2={ELY_REC['a2']} uV/h, acceleration={ELY_REC['high_current_accel']}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(run_one, SCENARIOS))
    results.sort(key=lambda row: row["unified_keur"])

    with open(OUT, "w", encoding="utf-8") as stream:
        stream.write(
            f"# {N_YEARS} ans; VoLL={V.voll_eur_per_kwh(0.0)} EUR/kWh; "
            f"ELY a2={ELY_REC['a2']} uV/h; "
            f"accel>2Acm2={ELY_REC['high_current_accel']} uV/h/(A/cm2)^2\n"
        )
        stream.write(
            "rang;strategie;EENS(kWh);LPSP(%);"
            "bat(kEUR);fc(kEUR);ely(kEUR);degradation(kEUR);unifie(kEUR)\n"
        )
        for rank, row in enumerate(results, 1):
            stream.write(
                f"{rank};{row['label']};{row['eens_kwh']:.4f};"
                f"{row['lpsp_pct']:.4f};"
                f"{row['bat_keur']:.4f};{row['fc_keur']:.4f};"
                f"{row['ely_keur']:.4f};{row['degradation_keur']:.4f};"
                f"{row['unified_keur']:.4f}\n"
            )
    print(f"Resultats : {OUT}", flush=True)


if __name__ == "__main__":
    main()
