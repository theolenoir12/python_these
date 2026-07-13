"""Validation scientifique de la comptabilite des remplacements V9_4.

Le cout sur trace complete n'est jamais utilise comme oracle apres un reset.
L'oracle est la somme des segments physiques disjoints, comparee au ledger de
la boucle. Le job nominal de 25 ans est destine au mesocentre.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROBUSTESSE = HERE.parent
for path in (str(HERE), str(ROBUSTESSE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import bench_valeur_info as VI
from Common import cost_fcn_total2 as C
from Common.main_init_and_loop import init_and_run_loop
from Common.main_init_and_loop_maintenance import init_and_run_loop_maintenance
from reproducibility.provenance import (
    acquire_run_lock,
    build_provenance,
    fingerprinted_run_dir,
    provenance_header_lines,
    write_provenance_sidecar,
)


TRACE_KEYS = (
    "SoC", "E_h2", "P_bat", "P_fc", "P_ely", "P_dc_load", "P_dc_pv",
    "P_dc_bat", "P_dc_fc", "P_dc_ely", "lol_tab", "alpha_fc",
    "alpha_ely", "SoH_bat", "SoH_fc", "SoH_ely",
)


def _component_segment_cost(data, component, start, stop):
    if stop <= start:
        return 0.0
    if component == "bat":
        return float(C.get_cost_bat(
            data["P_bat"][start:stop], data["SoC"][start:stop + 1],
            data["SoH_bat"][start:stop],
        ))
    if component == "fc":
        return float(C.get_cost_fc(
            data["alpha_fc"][start:stop], data["P_fc"][start:stop]
        )[0])
    if component == "ely":
        return float(C.get_cost_ely(
            data["alpha_ely"][start:stop], data["P_ely"][start:stop]
        )[0])
    raise ValueError(component)


def replay_disjoint_segments(data):
    """Rejoue chaque unite sur son propre intervalle demi-ouvert."""
    ledger = data["degradation_ledger"]
    result = {}
    for component in ("bat", "fc", "ely"):
        events = [event for event in ledger["events"]
                  if event["component"] == component]
        start = 0
        costs = []
        for event in events:
            stop = event["stop_step_exclusive"]
            cost = _component_segment_cost(data, component, start, stop)
            costs.append(cost)
            if not np.isclose(cost, event["retired_eur"], rtol=1e-10, atol=1e-6):
                raise AssertionError(
                    "%s segment [%d,%d) %.12g != event %.12g"
                    % (component, start, stop, cost, event["retired_eur"])
                )
            start = stop
        costs.append(_component_segment_cost(data, component, start, data["n"]))
        result[component] = {
            "segments_eur": costs,
            "sum_eur": float(sum(costs)),
        }
    return result


def verify_ledger(data):
    ledger = data["degradation_ledger"]
    replay = replay_disjoint_segments(data)
    for component in ("bat", "fc", "ely"):
        closed = ledger["retired_eur"][component] + ledger["current_eur"][component]
        total = ledger["total_eur"][component]
        if not np.all(np.isfinite([closed, total, replay[component]["sum_eur"]])):
            raise AssertionError("valeur non finie dans le ledger %s" % component)
        if not np.isclose(closed, total, rtol=0.0, atol=1e-9):
            raise AssertionError("fermeture ledger invalide pour %s" % component)
        if not np.isclose(replay[component]["sum_eur"], total, rtol=1e-10, atol=1e-6):
            raise AssertionError("somme des segments invalide pour %s" % component)
    return replay


def verify_maintenance_ledger_freeze_aware(data):
    """Oracle online aux frontières, sans rejouer la récupération pendant gel."""
    ledger = data["degradation_ledger"]
    checked = {}
    for component in ("bat", "fc", "ely"):
        events = [event for event in ledger["events"]
                  if event["component"] == component]
        if component == "bat":
            start = 0
            retired = []
            for event in events:
                stop = event["stop_step_exclusive"]
                cost = _component_segment_cost(data, component, start, stop)
                if not np.isclose(cost, event["retired_eur"], rtol=1e-10, atol=1e-6):
                    raise AssertionError("cout batterie correctif invalide")
                retired.append(cost)
                start = stop
            current = _component_segment_cost(data, component, start, data["n"])
        else:
            deg_key = "deg_" + component
            eol = VI.I.FC["SoH_EoL"] if component == "fc" else VI.I.ELY["SoH_EoL"]
            capex = VI.I.FC["cost"] if component == "fc" else VI.I.ELY["cost"]
            retired = []
            for event in events:
                stop = int(event["stop_step_exclusive"])
                if stop <= 0:
                    raise AssertionError("frontiere online invalide")
                deg_pct = float(data[deg_key]["total"][stop - 1])
                cost = deg_pct / ((1 - eol) * 100.0) * capex
                if not np.isfinite(cost) or not np.isclose(
                    cost, event["retired_eur"], rtol=1e-10, atol=1e-6
                ):
                    raise AssertionError("cout online %s invalide a %d" % (component, stop))
                retired.append(cost)
            start = int(ledger["current_start_step"][component])
            if start < data["n"]:
                deg_pct = float(data[deg_key]["total"][data["n"] - 1])
                current = deg_pct / ((1 - eol) * 100.0) * capex
            else:
                current = 0.0
        if not np.isclose(current, ledger["current_eur"][component],
                          rtol=1e-10, atol=1e-6):
            raise AssertionError("cout terminal freeze-aware invalide : %s" % component)
        total = float(sum(retired) + current)
        if not np.isclose(total, ledger["total_eur"][component],
                          rtol=1e-10, atol=1e-6):
            raise AssertionError("total freeze-aware invalide : %s" % component)
        checked[component] = {"retired_eur": retired, "current_eur": current,
                              "total_eur": total}
    return checked


def compare_base_and_instant(base, instant):
    gaps = {}
    for key in TRACE_KEYS:
        a = np.asarray(base[key])
        b = np.asarray(instant[key])
        if a.shape != b.shape:
            raise AssertionError("forme differente pour %s" % key)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError("trace non finie pour %s" % key)
        gap = float(np.max(np.abs(a - b))) if a.size else 0.0
        gaps[key] = gap
        if gap > 1e-12:
            raise AssertionError("instant != base pour %s : %.3e" % (key, gap))
    if "degradation_ledger" in base and "degradation_ledger" in instant:
        for component in ("bat", "fc", "ely"):
            gap = abs(base["degradation_ledger"]["total_eur"][component]
                      - instant["degradation_ledger"]["total_eur"][component])
            if gap > 1e-9:
                raise AssertionError("ledger instant != base pour %s" % component)
    return gaps


def verify_corrective_freezes(data):
    """Verifie puissance nulle, SoH gele et heures-composant par panne dure."""
    failures = data["maintenance"]["failure_log"]
    ledger_events = data["degradation_ledger"]["events"]
    expected_hours = {"fc": 0.0, "ely": 0.0}
    details = []
    ts_h = VI.I.LOAD["Ts"] / 3600.0
    for failure in failures:
        component = failure["component"]
        start = int(failure["state_index"])
        candidates = [event for event in ledger_events
                      if event["component"] == component
                      and event["reason"] == "maintenance_corr"
                      and event["stop_step_exclusive"] >= start]
        stop = min((event["stop_step_exclusive"] for event in candidates),
                   default=data["n"])
        soh = np.asarray(data["SoH_" + component])[start:stop]
        power = np.asarray(data["P_" + component])[start:stop]
        deg = np.asarray(data["deg_" + component]["total"])[max(0, start - 1):stop]
        if not np.all(np.isfinite(soh)) or not np.all(np.isfinite(power)):
            raise AssertionError("gel non fini pour %s" % component)
        if soh.size and float(np.max(np.abs(soh - soh[0]))) > 1e-12:
            raise AssertionError("SoH non gele pour %s sur [%d,%d)" % (component, start, stop))
        if power.size and float(np.max(np.abs(power))) > 1e-12:
            raise AssertionError("puissance non nulle pour %s HS" % component)
        if deg.size and float(np.max(np.abs(deg - deg[0]))) > 1e-12:
            raise AssertionError("cout non gele pour %s HS" % component)
        expected_hours[component] += (stop - start) * ts_h
        details.append((component, start, stop))
    for component in ("fc", "ely"):
        actual = data["maintenance"]["outage_h"][component]
        if not np.isclose(actual, expected_hours[component], rtol=0.0, atol=1e-9):
            raise AssertionError(
                "heures-composant %s : %.12g != %.12g"
                % (component, actual, expected_hours[component])
            )
    return details, expected_hours


def _legacy_metric_on_corrected_trace(data):
    copy = dict(data)
    copy.pop("degradation_ledger", None)
    copy["replacement_accounting"] = "legacy_overlap"
    return VI.metrics(copy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=25.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--no-corrective", action="store_true")
    parser.add_argument("--require-replacements", action="store_true")
    args = parser.parse_args()

    include_legacy = not args.no_legacy
    include_corrective = not args.no_corrective
    provenance = build_provenance(
        "v94_replacement_invariance",
        VI.provenance_files([
            HERE / "check_cost_reset_invariance.py",
            HERE / "Common" / "main_init_and_loop_maintenance.py",
        ]),
        {
            "horizon_years": args.years,
            "strategy": "RB2(SoH)",
            "corrected_accounting": "corrected",
            "include_legacy_overlap": include_legacy,
            "include_corrective_freeze": include_corrective,
        },
        repo_root=HERE.parents[1],
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        run_dir = output.parent
    else:
        run_dir = fingerprinted_run_dir(HERE, "invariance", provenance)
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "report.txt"
    run_lock = acquire_run_lock(run_dir)

    VI.apply_world(VI.NOMINAL_WORLD)
    base = init_and_run_loop(
        VI.load_strategy("RB2(SoH)"), n_years=args.years,
        replacement_accounting="corrected",
    )
    VI.apply_world(VI.NOMINAL_WORLD)
    instant = init_and_run_loop_maintenance(
        VI.load_strategy("RB2(SoH)"), n_years=args.years, policy="instant",
        replacement_accounting="corrected",
    )
    replay_base = verify_ledger(base)
    replay_instant = verify_ledger(instant)
    trace_gaps = compare_base_and_instant(base, instant)
    replacement_count = len(base["degradation_ledger"]["events"])
    if (args.require_replacements or args.years >= 25) and replacement_count == 0:
        raise AssertionError("aucun remplacement observe : invariant de reset non exerce")

    corrective = None
    freeze_details = []
    freeze_hours = {}
    corrective_ledger = {}
    if include_corrective:
        VI.apply_world(VI.NOMINAL_WORLD)
        corrective = init_and_run_loop_maintenance(
            VI.load_strategy("RB2(SoH)"), n_years=args.years,
            visit_period_months=6.0, policy="corrective",
            replacement_accounting="corrected",
        )
        corrective_ledger = verify_maintenance_ledger_freeze_aware(corrective)
        freeze_details, freeze_hours = verify_corrective_freezes(corrective)
        if (args.require_replacements or args.years >= 25) and not freeze_details:
            raise AssertionError("aucune panne dure corrective observee")

    legacy_rows = []
    if include_legacy:
        VI.apply_world(VI.NOMINAL_WORLD)
        legacy_base = init_and_run_loop(
            VI.load_strategy("RB2(SoH)"), n_years=args.years,
            replacement_accounting="legacy_overlap",
        )
        VI.apply_world(VI.NOMINAL_WORLD)
        legacy_instant = init_and_run_loop_maintenance(
            VI.load_strategy("RB2(SoH)"), n_years=args.years, policy="instant",
            replacement_accounting="legacy_overlap",
        )
        compare_base_and_instant(legacy_base, legacy_instant)
        legacy_rows = [VI.metrics(legacy_base), VI.metrics(legacy_instant)]

    metric_base = VI.metrics(base)
    metric_full_trace = _legacy_metric_on_corrected_trace(base)
    with open(output, "w", encoding="utf-8") as stream:
        for line in provenance_header_lines(provenance):
            stream.write(line + "\n")
        stream.write("# Validation comptabilite des remplacements V9_4\n")
        stream.write("horizon_years=%.12g\n" % args.years)
        stream.write("status=OK\n")
        stream.write("replacement_events=%d\n" % replacement_count)
        stream.write("max_trace_gap_base_vs_instant=%.17g\n" % max(trace_gaps.values()))
        for component in ("bat", "fc", "ely"):
            stream.write(
                "ledger_%s_eur=%.17g;segments_%s_eur=%.17g;n_segments=%d\n"
                % (component, base["degradation_ledger"]["total_eur"][component],
                   component, replay_base[component]["sum_eur"],
                   len(replay_base[component]["segments_eur"]))
            )
            stream.write(
                "instant_ledger_%s_eur=%.17g;instant_segments_%s_eur=%.17g\n"
                % (component, instant["degradation_ledger"]["total_eur"][component],
                   component, replay_instant[component]["sum_eur"])
            )
        stream.write("corrected_deg_kEUR=%.17g\n" % metric_base[1])
        stream.write("full_trace_diagnostic_deg_kEUR=%.17g\n" % metric_full_trace[1])
        stream.write("full_trace_minus_ledger_kEUR=%.17g\n"
                     % (metric_full_trace[1] - metric_base[1]))
        stream.write("corrective_freeze_intervals=%r\n" % (freeze_details,))
        stream.write("corrective_component_hours=%r\n" % (freeze_hours,))
        stream.write("corrective_freeze_aware_ledger=%r\n" % (corrective_ledger,))
        if legacy_rows:
            stream.write("legacy_base_metrics=%r\n" % (legacy_rows[0],))
            stream.write("legacy_instant_metrics=%r\n" % (legacy_rows[1],))

    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    write_provenance_sidecar(sidecar, provenance, [output])
    print("Validation OK : %s" % output)
    print("Remplacements observes : %d" % replacement_count)
    print("Biais diagnostic trace complete - ledger : %+.6f kEUR"
          % (metric_full_trace[1] - metric_base[1]))
    run_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
