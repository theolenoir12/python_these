"""Extrait et trace le front LPSP/degradation de RB2(SoH)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "Optimization_results" / "optimization_soh_normalized.csv"
DEFAULT_MECHANISMS = ROOT / "DIAGNOSTIC_RB2SOH_MECANISMES.json"


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["lpsp_pct"] = float(row["lpsp_pct"])
        row["eens_kwh"] = float(row["eens_kwh"])
        row["degradation_keur"] = float(row["degradation_keur"])
        row["unified_keur"] = float(row["unified_keur"])
        row["params"] = json.loads(row["parameters_json"])
    return rows


def is_null(row):
    p = row["params"]
    return (
        float(p.get("soh_strength_fc", 0.0)) == 0.0
        and float(p.get("soh_strength_ely", 0.0)) == 0.0
    )


def pareto_front(rows):
    """Points non domines pour la minimisation de LPSP et degradation."""
    front = []
    best_degradation = float("inf")
    for row in sorted(rows, key=lambda r: (r["lpsp_pct"], r["degradation_keur"])):
        if row["degradation_keur"] < best_degradation - 1e-9:
            front.append(row)
            best_degradation = row["degradation_keur"]
    return front


def break_even_voll(baseline, candidate):
    saved_eur = 1000.0 * (
        baseline["degradation_keur"] - candidate["degradation_keur"]
    )
    extra_eens = candidate["eens_kwh"] - baseline["eens_kwh"]
    return saved_eur / extra_eens if saved_eur > 0.0 and extra_eens > 0.0 else float("nan")


def write_report(path, baseline, selected, min_degradation, front, cap, mechanisms=None):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("DIAGNOSTIC RB2(SoH) : TRADE-OFF LPSP / DEGRADATION\n")
        stream.write("===================================================\n\n")
        stream.write("Loi retenue : facteur = 1 - strength * usure_normalisee^shape\n")
        stream.write("usure_normalisee = (1 - SoH) / (1 - SoH_EoL)\n\n")
        stream.write("Reference RB2 :\n")
        stream.write("  LPSP = %.6f %% ; degradation = %.6f kEUR ; unifie = %.6f kEUR\n\n" % (
            baseline["lpsp_pct"], baseline["degradation_keur"], baseline["unified_keur"]
        ))
        stream.write("Point Pareto retenu sous LPSP <= %.2f %% :\n" % cap)
        stream.write("  LPSP = %.6f %% ; degradation = %.6f kEUR ; unifie = %.6f kEUR\n" % (
            selected["lpsp_pct"], selected["degradation_keur"], selected["unified_keur"]
        ))
        stream.write("  delta LPSP = %+.6f point ; delta degradation = %+.6f kEUR (%+.2f %%)\n" % (
            selected["lpsp_pct"] - baseline["lpsp_pct"],
            selected["degradation_keur"] - baseline["degradation_keur"],
            100.0 * (selected["degradation_keur"] / baseline["degradation_keur"] - 1.0),
        ))
        stream.write("  VoLL seuil de rentabilite = %.3f EUR/kWh\n" % break_even_voll(baseline, selected))
        stream.write("  parametres = %s\n\n" % json.dumps(selected["params"], sort_keys=True))
        if mechanisms:
            base_m = next(row for row in mechanisms if row["name"] == "RB2")
            selected_m = next(
                row for row in mechanisms if row["name"] == "SoH_Pareto_1p10"
            )
            stream.write("Decomposition mecaniste RB2 -> point Pareto :\n")
            for component in ("bat", "fc", "ely"):
                before = base_m["component_costs_keur"][component]
                after = selected_m["component_costs_keur"][component]
                stream.write("  %s : %.6f -> %.6f kEUR (delta %+.6f)\n" % (
                    component, before, after, after - before
                ))
            stream.write("  densite moyenne FC active : %.4f -> %.4f A/cm2\n" % (
                base_m["fc_usage"]["mean_j_on"], selected_m["fc_usage"]["mean_j_on"]
            ))
            stream.write("  densite moyenne ELY active : %.4f -> %.4f A/cm2\n" % (
                base_m["ely_usage"]["mean_j_on"], selected_m["ely_usage"]["mean_j_on"]
            ))
            stream.write("  demarrages FC : %d -> %d ; ELY : %d -> %d\n\n" % (
                base_m["fc_usage"]["starts"], selected_m["fc_usage"]["starts"],
                base_m["ely_usage"]["starts"], selected_m["ely_usage"]["starts"],
            ))
        stream.write("Minimum de degradation de la grille :\n")
        stream.write("  LPSP = %.6f %% ; degradation = %.6f kEUR (%+.2f %%)\n" % (
            min_degradation["lpsp_pct"], min_degradation["degradation_keur"],
            100.0 * (min_degradation["degradation_keur"] / baseline["degradation_keur"] - 1.0),
        ))
        stream.write("  parametres = %s\n\n" % json.dumps(min_degradation["params"], sort_keys=True))
        stream.write("Conclusion : le front contient %d points non domines. Avec VoLL=3 EUR/kWh,\n" % len(front))
        stream.write("le minimum du cout unifie reste RB2 ; la couche SoH n'est donc pas\n")
        stream.write("economiquement rentable sous la ponderation actuelle, bien qu'elle reduise\n")
        stream.write("effectivement la degradation.\n")


def plot(path, rows, front, baseline, selected):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        [r["lpsp_pct"] for r in rows], [r["degradation_keur"] for r in rows],
        s=22, alpha=0.35, label="Configurations SoH",
    )
    ax.plot(
        [r["lpsp_pct"] for r in front], [r["degradation_keur"] for r in front],
        "o-", linewidth=1.5, markersize=4, label="Front non domine",
    )
    ax.scatter([baseline["lpsp_pct"]], [baseline["degradation_keur"]],
               marker="*", s=150, label="RB2")
    ax.scatter([selected["lpsp_pct"]], [selected["degradation_keur"]],
               marker="D", s=70, label="RB2(SoH), LPSP <= 1,10 %")
    ax.set_xlabel("LPSP sur la charge totale [%]")
    ax.set_ylabel("Cout de degradation [kEUR]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lpsp-cap", type=float, default=1.10)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    baseline = next(row for row in rows if is_null(row))
    front = pareto_front(rows)
    feasible = [row for row in front if row["lpsp_pct"] <= args.lpsp_cap]
    if not feasible:
        raise ValueError("Aucun point du front sous la contrainte LPSP")
    selected = min(feasible, key=lambda row: row["degradation_keur"])
    min_degradation = min(rows, key=lambda row: row["degradation_keur"])
    mechanisms = None
    if DEFAULT_MECHANISMS.exists():
        with DEFAULT_MECHANISMS.open(encoding="utf-8") as stream:
            mechanisms = json.load(stream)

    report = ROOT / "DIAGNOSTIC_RB2SOH_TRADEOFF.txt"
    write_report(
        report, baseline, selected, min_degradation, front, args.lpsp_cap,
        mechanisms,
    )
    if not args.no_plot:
        plot(ROOT / "RB2SoH_tradeoff.png", rows, front, baseline, selected)
    print(report)
    print(json.dumps(selected["params"], sort_keys=True))


if __name__ == "__main__":
    main()
