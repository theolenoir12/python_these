"""Post-traitement apparie des caches bruts P1/P3 corriges, sans simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .paired_stats import (
        bootstrap_cvar_difference_ci,
        bootstrap_mean_ci,
        cvar_high,
        summarize_difference,
    )
    from .provenance import (
        build_provenance,
        fingerprinted_run_dir,
        provenance_header_lines,
        read_provenance_header,
        sha256_file,
        write_provenance_sidecar,
    )
except ImportError:
    from paired_stats import (
        bootstrap_cvar_difference_ci,
        bootstrap_mean_ci,
        cvar_high,
        summarize_difference,
    )
    from provenance import (
        build_provenance,
        fingerprinted_run_dir,
        provenance_header_lines,
        read_provenance_header,
        sha256_file,
        write_provenance_sidecar,
    )


HERE = Path(__file__).resolve().parent
ROBUSTESSE = HERE.parent
V9 = ROBUSTESSE / "Vieillissement9_4"
VOLLS = (1.0, 3.0, 10.0)
C_VISITS = (0.5, 1.5, 3.0)
BOOTSTRAP_SEED = 20260711
FACTOR_KEYS = ("m_fc_a", "m_fc_b", "m_fc_s", "m_ely_a", "m_ely_b", "m_ely_s", "m_bat")


def _seed(label):
    digest = hashlib.sha256((str(BOOTSTRAP_SEED) + "|" + label).encode()).hexdigest()
    return int(digest[:8], 16)


def _artifact_is_recorded(record, path):
    actual = sha256_file(path)
    size = Path(path).stat().st_size
    return any(
        Path(item.get("path", "")).name == Path(path).name
        and item.get("sha256") == actual and item.get("size") == size
        for item in record.get("artifacts", [])
    )


def _load_completed_run_record(path, header_fingerprint):
    path = Path(path)
    raw_sidecar = Path(str(path) + ".provenance.json")
    final_sidecar = path.parent / "provenance.json"
    if not raw_sidecar.is_file() or not final_sidecar.is_file():
        raise ValueError("run non finalise ou sidecar absent : %s" % path)
    raw_record = json.loads(raw_sidecar.read_text(encoding="utf-8"))
    final_record = json.loads(final_sidecar.read_text(encoding="utf-8"))
    for label, record in (("raw", raw_record), ("final", final_record)):
        if record.get("run_fingerprint") != header_fingerprint:
            raise ValueError("fingerprint %s incoherent pour %s" % (label, path))
        if not _artifact_is_recorded(record, path):
            raise ValueError("SHA raw absent/invalide dans sidecar %s" % label)
    return final_record


def _read_raw(path):
    path = Path(path)
    fields = read_provenance_header(path)
    if "run_fingerprint" not in fields:
        raise ValueError("cache sans empreinte : %s" % path)
    run_record = _load_completed_run_record(path, fields["run_fingerprint"])
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    if len(lines) < 2:
        raise ValueError("cache vide : %s" % path)
    header = lines[0].split(";")
    rows = []
    for line in lines[1:]:
        values = line.split(";")
        if len(values) != len(header):
            raise ValueError("ligne brute incomplete dans %s" % path)
        rows.append(dict(zip(header, values)))
    return header, rows, fields["run_fingerprint"], run_record


def _aligned(path, entity_column, entities, metrics):
    header, rows, fingerprint, run_record = _read_raw(path)
    required = {entity_column, "draw", *FACTOR_KEYS, *metrics}
    missing = required.difference(header)
    if missing:
        raise ValueError("colonnes absentes dans %s : %s" % (path, sorted(missing)))
    rows = [row for row in rows if int(row["draw"]) >= 0]
    draws = sorted({int(row["draw"]) for row in rows})
    if draws != list(range(len(draws))):
        raise ValueError("tirages non contigus dans %s" % path)
    by = {(row[entity_column], int(row["draw"])): row for row in rows}
    if len(by) != len(rows):
        raise ValueError("lignes dupliquees dans %s" % path)
    data = {}
    for entity in entities:
        missing_draws = [draw for draw in draws if (entity, draw) not in by]
        if missing_draws:
            raise ValueError("%s incomplet dans %s" % (entity, path))
        data[entity] = {
            metric: np.array([float(by[(entity, draw)][metric]) for draw in draws])
            for metric in metrics
        }
    factors = {
        key: np.array([float(by[(entities[0], draw)][key]) for draw in draws])
        for key in FACTOR_KEYS
    }
    for entity in entities[1:]:
        for draw in draws:
            for key in FACTOR_KEYS:
                if float(by[(entity, draw)][key]) != factors[key][draw]:
                    raise ValueError("CRN internes rompus dans %s" % path)
    return {"draws": np.array(draws), "factors": factors, "data": data,
            "fingerprint": fingerprint, "provenance": run_record}


def load_p1(path):
    return _aligned(
        path, "strat", ("RB2", "RB2(Recale)", "RB2(Sched)", "RB2(SoH)"),
        ("lpsp", "deg", "eens", "uni"),
    )


def load_p3(path):
    return _aligned(
        path, "policy", ("instant", "corrective", "calendar", "rul"),
        ("lpsp", "deg", "eens", "uni0", "nint", "nprev", "waste",
         "wbat", "wfc", "wely", "outfc", "outely"),
    )


def same_crn(reference, other):
    return np.array_equal(reference["draws"], other["draws"]) and all(
        np.array_equal(reference["factors"][key], other["factors"][key])
        for key in FACTOR_KEYS
    )


def require_protocol(dataset, experiment_id, expected_parameters):
    record = dataset["provenance"]
    if record.get("experiment_id") != experiment_id:
        raise ValueError(
            "experiment_id %r, attendu %r"
            % (record.get("experiment_id"), experiment_id)
        )
    parameters = record.get("parameters", {})
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise ValueError(
                "%s=%r, attendu %r pour %s"
                % (key, parameters.get(key), expected, experiment_id)
            )


def verify_numeric_closures(p1, p3_by_tag):
    for strategy, values in p1["data"].items():
        for field, vector in values.items():
            if not np.all(np.isfinite(vector)):
                raise RuntimeError("P1 non fini : %s/%s" % (strategy, field))
        expected = values["deg"] + 3.0 * values["eens"] / 1000.0
        if not np.allclose(values["uni"], expected, rtol=0.0, atol=1e-10):
            raise RuntimeError("fermeture uni P1 invalide : %s" % strategy)
    for tag, dataset in p3_by_tag.items():
        for policy, values in dataset["data"].items():
            for field, vector in values.items():
                if not np.all(np.isfinite(vector)):
                    raise RuntimeError("P3 non fini : %s/%s/%s" % (tag, policy, field))
            expected = values["deg"] + 3.0 * values["eens"] / 1000.0
            if not np.allclose(values["uni0"], expected, rtol=0.0, atol=1e-10):
                raise RuntimeError("fermeture uni0 P3 invalide : %s/%s" % (tag, policy))
            waste = values["wbat"] + values["wfc"] + values["wely"]
            if not np.allclose(values["waste"], waste, rtol=0.0, atol=1e-10):
                raise RuntimeError("fermeture waste invalide : %s/%s" % (tag, policy))


def p1_cost(p1, strategy, voll):
    values = p1["data"][strategy]
    return values["deg"] + voll * values["eens"] / 1000.0


def p3_cost(p3, policy, voll, c_visit):
    values = p3["data"][policy]
    return (values["deg"] + voll * values["eens"] / 1000.0
            + c_visit * values["nint"] + values["waste"])


def contrast(analysis, scenario, voll, c_visit, label, a, b):
    difference = a - b
    seed_label = "%s|%s|%s|%s|%s" % (analysis, scenario, voll, c_visit, label)
    summary = summarize_difference(difference, bootstrap_seed=_seed(seed_label))
    es_marginal_ci = bootstrap_cvar_difference_ci(
        a, b, seed=_seed("ESmarginal|" + seed_label)
    )
    es_difference_ci = bootstrap_cvar_difference_ci(
        difference, np.zeros_like(difference), seed=_seed("ESdifference|" + seed_label)
    )
    return {
        "analysis": analysis, "scenario": scenario, "voll": voll,
        "c_visit": c_visit, "contrast": label, **summary,
        "es90_a_minus_es90_b": cvar_high(a) - cvar_high(b),
        "es90_a_minus_es90_b_ci95_low": es_marginal_ci[0],
        "es90_a_minus_es90_b_ci95_high": es_marginal_ci[1],
        "es90_difference": cvar_high(difference),
        "es90_difference_ci95_low": es_difference_ci[0],
        "es90_difference_ci95_high": es_difference_ci[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True, type=Path)
    parser.add_argument("--p3-t3", required=True, type=Path)
    parser.add_argument("--p3-t6", required=True, type=Path)
    parser.add_argument("--p3-t12", required=True, type=Path)
    parser.add_argument("--p3-t6-m15", required=True, type=Path)
    parser.add_argument("--p3-t6-m2", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=V9)
    args = parser.parse_args()

    p1 = load_p1(args.p1)
    p3 = {
        "T3m": load_p3(args.p3_t3),
        "T6m": load_p3(args.p3_t6),
        "T12m": load_p3(args.p3_t12),
        "T6m_m1.5": load_p3(args.p3_t6_m15),
        "T6m_m2": load_p3(args.p3_t6_m2),
    }
    common_parameters = {
        "horizon_years": 25, "n_worlds": 200, "seed": 2026,
        "multipliers_log_uniform": [0.5, 2.0],
        "factor_keys": list(FACTOR_KEYS), "voll_eur_per_kwh": 3.0,
        "replacement_accounting": "corrected",
    }
    require_protocol(
        p1, "p1_value_information_soh",
        {
            **common_parameters,
            "strategies": ["RB2", "RB2(Recale)", "RB2(Sched)", "RB2(SoH)"],
            "schedule_semantics": "global_nominal_time_no_unit_age_reset",
        },
    )
    for tag, tvisit, margin in (
        ("T3m", 3.0, 1.0), ("T6m", 6.0, 1.0), ("T12m", 12.0, 1.0),
        ("T6m_m1.5", 6.0, 1.5), ("T6m_m2", 6.0, 2.0),
    ):
        require_protocol(
            p3[tag], "p3_rul_maintenance",
            {
                **common_parameters,
                "strategy": "RB2(SoH)",
                "policies": ["instant", "corrective", "calendar", "rul"],
                "visit_period_months": tvisit, "rul_margin": margin,
                "calendar_fraction": 1.0,
                "preventive_scope": ["fc", "ely"],
                "visit_cost_grid_keur": [0.5, 1.5, 3.0],
            },
        )
    if len(p1["draws"]) != 200:
        raise RuntimeError("P1 doit contenir exactement 200 mondes")
    for tag, values in p3.items():
        if not same_crn(p1, values):
            raise RuntimeError("CRN P1/P3 rompus pour %s" % tag)
        if values["provenance"].get("numerical_runtime") != p1["provenance"].get("numerical_runtime"):
            raise RuntimeError("environnements numeriques P1/P3 differents pour %s" % tag)
        p1_inputs = {item["path"]: item["sha256"]
                     for item in p1["provenance"].get("inputs", [])}
        p3_inputs = {item["path"]: item["sha256"]
                     for item in values["provenance"].get("inputs", [])}
        if any(p3_inputs.get(path) != digest for path, digest in p1_inputs.items()):
            raise RuntimeError("sources communes P1/P3 differentes pour %s" % tag)
    verify_numeric_closures(p1, p3)

    # Tests nuls inter-caches, en pleine precision et tirage par tirage.
    for tag, values in p3.items():
        for field_p1, field_p3 in (("lpsp", "lpsp"), ("deg", "deg"), ("eens", "eens")):
            if not np.array_equal(
                p1["data"]["RB2(SoH)"][field_p1], values["data"]["instant"][field_p3]
            ):
                raise RuntimeError("P1 SoH != P3 instant : %s/%s" % (tag, field_p1))
    for tag in ("T3m", "T12m"):
        for field in p3["T6m"]["data"]["instant"]:
            if not np.array_equal(p3["T6m"]["data"]["instant"][field],
                                  p3[tag]["data"]["instant"][field]):
                raise RuntimeError("instant depend de T : %s/%s" % (tag, field))
    for tag in ("T6m_m1.5", "T6m_m2"):
        for policy in ("instant", "corrective", "calendar"):
            for field in p3["T6m"]["data"][policy]:
                if not np.array_equal(p3["T6m"]["data"][policy][field],
                                      p3[tag]["data"][policy][field]):
                    raise RuntimeError("politique non-RUL depend de marge : %s/%s/%s"
                                       % (tag, policy, field))

    records = []
    for voll in VOLLS:
        for label, a_name, b_name in (
            ("Recale-RB2", "RB2(Recale)", "RB2"),
            ("Sched-Recale", "RB2(Sched)", "RB2(Recale)"),
            ("SoH-Sched", "RB2(SoH)", "RB2(Sched)"),
            ("SoH-RB2", "RB2(SoH)", "RB2"),
        ):
            records.append(contrast(
                "P1", "mondes corriges", voll, None, label,
                p1_cost(p1, a_name, voll), p1_cost(p1, b_name, voll),
            ))
    for tag in ("T3m", "T6m", "T12m"):
        for voll in VOLLS:
            for c_visit in C_VISITS:
                for label, a_name, b_name in (
                    ("rul-corrective", "rul", "corrective"),
                    ("rul-calendar", "rul", "calendar"),
                    ("calendar-corrective", "calendar", "corrective"),
                ):
                    records.append(contrast(
                        "P3", tag + "_marge1_corrige", voll, c_visit, label,
                        p3_cost(p3[tag], a_name, voll, c_visit),
                        p3_cost(p3[tag], b_name, voll, c_visit),
                    ))
    for tag, margin in (("T6m_m1.5", 1.5), ("T6m_m2", 2.0)):
        for voll in VOLLS:
            for c_visit in C_VISITS:
                records.append(contrast(
                    "P3_MARGIN", tag + "_corrige", voll, c_visit,
                    "rul(m%g)-rul(m1)" % margin,
                    p3_cost(p3[tag], "rul", voll, c_visit),
                    p3_cost(p3["T6m"], "rul", voll, c_visit),
                ))
                records.append(contrast(
                    "P3_MARGIN", tag + "_corrige", voll, c_visit,
                    "rul(m%g)-corrective" % margin,
                    p3_cost(p3[tag], "rul", voll, c_visit),
                    p3_cost(p3[tag], "corrective", voll, c_visit),
                ))

    provenance = build_provenance(
        "postprocess_p1_p3_corrected",
        [
            Path(__file__), HERE / "paired_stats.py",
            ("inputs/p1_results_raw.tsv", args.p1),
            ("inputs/p3_t3_results_raw.tsv", args.p3_t3),
            ("inputs/p3_t6_results_raw.tsv", args.p3_t6),
            ("inputs/p3_t12_results_raw.tsv", args.p3_t12),
            ("inputs/p3_t6_m15_results_raw.tsv", args.p3_t6_m15),
            ("inputs/p3_t6_m2_results_raw.tsv", args.p3_t6_m2),
        ],
        {"volls": VOLLS, "c_visits_keur": C_VISITS,
         "rul_margins_t6": [1.0, 1.5, 2.0],
         "bootstrap_seed": BOOTSTRAP_SEED, "paired_crn": True,
         "multiplicity": "exploratory_unadjusted"},
        repo_root=V9.parents[1],
    )
    output_dir = fingerprinted_run_dir(args.output_root, "p1_p3_stats", provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    table_path = output_dir / "contrasts.tsv"

    lines = provenance_header_lines(provenance)
    lines += [
        "=" * 100,
        "STATISTIQUES APPARIEES P1/P3 -- MOTEUR CORRIGE ET CACHES PLEINE PRECISION",
        "=" * 100,
        "N=%d mondes CRN ; VoLL=1/3/10 en post-traitement, sans reoptimisation." % len(p1["draws"]),
        "Difference A-B negative : A moins couteuse. ES90(ecart) = pire decile de A-B.",
        "Famille exploratoire de contrastes : IC et tests non ajustes pour multiplicite ;",
        "toute revendication confirmatoire devra pre-specifier ses contrastes ou corriger la famille.",
        "",
        "analysis scenario VoLL Cvis contraste mean[IC95] G/E/P delta_ES90[IC95] ES90(ecart)[IC95]",
    ]
    for record in records:
        lines.append(
            "%s %s %.0f %s %-25s %+.4f[%+.4f,%+.4f] %d/%d/%d "
            "%+.4f[%+.4f,%+.4f] %+.4f[%+.4f,%+.4f]"
            % (record["analysis"], record["scenario"], record["voll"],
               "-" if record["c_visit"] is None else "%.1f" % record["c_visit"],
               record["contrast"], record["mean"], record["mean_ci95_low"],
               record["mean_ci95_high"], record["wins"], record["ties"],
               record["losses"], record["es90_a_minus_es90_b"],
               record["es90_a_minus_es90_b_ci95_low"],
               record["es90_a_minus_es90_b_ci95_high"],
               record["es90_difference"], record["es90_difference_ci95_low"],
               record["es90_difference_ci95_high"])
        )
    lines += ["", "ARGMIN PONCTUELS P3 (descriptifs, pas tests de superiorite)",
              "T VoLL Cvis min_moyenne min_ES90"]
    for tag, values in p3.items():
        for voll in VOLLS:
            for c_visit in C_VISITS:
                costs = {policy: p3_cost(values, policy, voll, c_visit)
                         for policy in values["data"]}
                lines.append("%s %.0f %.1f %s %s" % (
                    tag, voll, c_visit,
                    min(costs, key=lambda key: costs[key].mean()),
                    min(costs, key=lambda key: cvar_high(costs[key])),
                ))

    dlpsp = (p3["T12m"]["data"]["rul"]["lpsp"]
             - p3["T3m"]["data"]["corrective"]["lpsp"])
    ci = bootstrap_mean_ci(dlpsp, seed=_seed("equivalence_lpsp_corrected"))
    lines += ["", "RUL T12m - correctif T3m : delta LPSP=%+.5f pt, IC95 [%+.5f,%+.5f]."
              % (dlpsp.mean(), ci[0], ci[1]),
              "Toute non-inferiorite exige une marge fixee et justifiee a priori.",
              "outfc+outely est une somme d'heures-composant, pas une indisponibilite systeme."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = [
        "analysis", "scenario", "voll", "c_visit", "contrast", "n", "mean",
        "mean_ci95_low", "mean_ci95_high", "sample_sd", "median", "q1", "q3",
        "p05", "p95", "wins", "ties", "losses", "sign_pvalue",
        "es90_a_minus_es90_b", "es90_difference",
        "es90_a_minus_es90_b_ci95_low", "es90_a_minus_es90_b_ci95_high",
        "es90_difference_ci95_low", "es90_difference_ci95_high",
    ]
    with table_path.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for record in records:
            stream.write("\t".join("" if record.get(field) is None else str(record.get(field))
                                   for field in fields) + "\n")
    write_provenance_sidecar(output_dir / "provenance.json", provenance,
                             [report_path, table_path])
    print("Controles CRN/fermeture : OK")
    print("Contrastes : %d" % len(records))
    print("Rapport : %s" % report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
