"""Dedoublonne, controle et classe les JSONL produits par run_v11_candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from Common import Init_EMR_MG_v16_python as I


def _key(row):
    return json.dumps(
        {
            "kind": row["kind"],
            "params": row.get("params", {}),
            "model": row.get("model", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _read(paths):
    rows = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = _key(row)
                if key in rows:
                    for metric in ("unified_eur", "degradation_eur", "eens_kwh"):
                        if abs(float(row[metric]) - float(rows[key][metric])) > 1e-6:
                            raise ValueError(f"doublon non deterministe {key}: {metric}")
                rows[key] = row
    return list(rows.values())


def _terminal_bus_kwh(row):
    battery_nominal_kwh = (
        I.BAT["parallel_num"] * I.BAT["series_num"]
        * I.BAT["Q_bat"] * I.BAT["v_cell_nom"] / 1000.0
    )
    battery_dc_efficiency = I.CONV["eta"] * I.BAT["eff"]
    fc_best_efficiency = float(max(I.FC["lut"][1])) / 100.0
    hydrogen_dc_efficiency = I.CONV["eta"] * fc_best_efficiency
    return (
        float(row["terminal_soc"]) * float(row["terminal_soh_bat"])
        * battery_nominal_kwh * battery_dc_efficiency
        + float(row["terminal_h2_kwh"]) * hydrogen_dc_efficiency
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference")
    args = parser.parse_args()
    rows = _read(args.inputs)
    for row in rows:
        row["terminal_bus_kwh"] = _terminal_bus_kwh(row)
        row["inventory_adjusted_eur"] = (
            float(row["unified_eur"])
            - float(row["voll_eur_per_kwh"]) * row["terminal_bus_kwh"]
        )
    reference = None
    if args.reference:
        matches = [row for row in rows if row["label"] == args.reference]
        if len(matches) != 1:
            raise ValueError("reference absente ou non unique")
        reference = matches[0]
    rows.sort(key=lambda row: row["unified_eur"])
    fields = (
        "rank", "label", "kind", "params", "model", "unified_eur",
        "degradation_eur", "battery_eur", "fc_eur", "ely_eur", "eens_kwh",
        "lpsp_pct", "terminal_soc", "terminal_h2_kwh", "terminal_bus_kwh",
        "inventory_adjusted_eur", "gain_vs_reference_pct",
        "inventory_adjusted_gain_vs_reference_pct",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            gain = adjusted_gain = ""
            if reference is not None:
                gain = 100.0 * (
                    reference["unified_eur"] - row["unified_eur"]
                ) / reference["unified_eur"]
                adjusted_gain = 100.0 * (
                    reference["inventory_adjusted_eur"]
                    - row["inventory_adjusted_eur"]
                ) / reference["inventory_adjusted_eur"]
            writer.writerow({
                "rank": rank,
                "label": row["label"],
                "kind": row["kind"],
                "params": json.dumps(row.get("params", {}), sort_keys=True),
                "model": json.dumps(row.get("model", {}), sort_keys=True),
                "unified_eur": row["unified_eur"],
                "degradation_eur": row["degradation_eur"],
                "battery_eur": row["battery_eur"],
                "fc_eur": row["fc_eur"],
                "ely_eur": row["ely_eur"],
                "eens_kwh": row["eens_kwh"],
                "lpsp_pct": row["lpsp_pct"],
                "terminal_soc": row["terminal_soc"],
                "terminal_h2_kwh": row["terminal_h2_kwh"],
                "terminal_bus_kwh": row["terminal_bus_kwh"],
                "inventory_adjusted_eur": row["inventory_adjusted_eur"],
                "gain_vs_reference_pct": gain,
                "inventory_adjusted_gain_vs_reference_pct": adjusted_gain,
            })
    print(f"{len(rows)} candidats uniques -> {args.output}")
    for row in rows[:10]:
        print(
            f"{row['label']}: J={row['unified_eur']:.2f}; "
            f"J_inv={row['inventory_adjusted_eur']:.2f}"
        )


if __name__ == "__main__":
    main()
