"""Post-traitement apparie P1/P3, sans nouvelle simulation.

Le script relit uniquement les tableaux bruts versionnes de Vieillissement9_4.
Il reconstruit EENS depuis les couts arrondis a VoLL=3, puis evalue les memes
politiques fixes a VoLL=1/3/10. Il ne s'agit pas d'une reoptimisation.
"""

from __future__ import annotations

import hashlib
import math
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
        provenance_header_lines,
        write_provenance_sidecar,
    )
except ImportError:  # execution directe depuis ce dossier
    from paired_stats import (
        bootstrap_cvar_difference_ci,
        bootstrap_mean_ci,
        cvar_high,
        summarize_difference,
    )
    from provenance import (
        build_provenance,
        provenance_header_lines,
        write_provenance_sidecar,
    )

HERE = Path(__file__).resolve().parent
ROBUSTESSE = HERE.parent
V9 = ROBUSTESSE / "Vieillissement9_4"

P1_PATH = V9 / "valeur_info_25y.txt"
P3_PATHS = {
    "T3m": V9 / "maintenance_25y_T3m.txt",
    "T6m": V9 / "maintenance_25y_T6m.txt",
    "T12m": V9 / "maintenance_25y_T12m.txt",
    "T6m_m1.5": V9 / "maintenance_25y_T6m_m1.5.txt",
    "T6m_m2": V9 / "maintenance_25y_T6m_m2.txt",
}
OUT_TXT = V9 / "STATISTIQUES_APPARIEES_P1_P3.txt"
OUT_TSV = V9 / "STATISTIQUES_APPARIEES_P1_P3.tsv"

VOLLS = (1.0, 3.0, 10.0)
C_VISITS = (0.5, 1.5, 3.0)
BOOTSTRAP_RESAMPLES = 30000
BOOTSTRAP_SEED = 20260711
REFERENCE_VOLL = 3.0


def _stable_seed(label):
    digest = hashlib.sha256((str(BOOTSTRAP_SEED) + "|" + label).encode()).hexdigest()
    return int(digest[:8], 16)


def _read_draw_table(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("draw;"))
    header = lines[start].split(";")
    raw_rows = []
    for line in lines[start + 1 :]:
        if not line or line.startswith("#"):
            break
        parts = line.split(";")
        if len(parts) != len(header) or not parts[0].isdigit():
            break
        raw_rows.append(parts)
    if not raw_rows:
        raise ValueError("Aucun tirage lu dans %s" % path)
    columns = {
        name: np.array([float(row[i]) for row in raw_rows], dtype=float)
        for i, name in enumerate(header)
    }
    columns["draw"] = columns["draw"].astype(int)
    return header, columns


def load_p1(path=P1_PATH):
    header, columns = _read_draw_table(path)
    factors = header[1:8]
    strategies = []
    for name in header[8:]:
        if name.endswith("_lpsp"):
            strategies.append(name[: -len("_lpsp")])
    data = {"draw": columns["draw"], "factors": {k: columns[k] for k in factors}}
    data["strategies"] = {
        strategy: {
            "lpsp": columns[strategy + "_lpsp"],
            "deg": columns[strategy + "_deg"],
            "uni3": columns[strategy + "_uni"],
        }
        for strategy in strategies
    }
    return data


def load_p3(path):
    header, columns = _read_draw_table(path)
    factors = header[1:8]
    policies = []
    for name in header[8:]:
        if name.endswith("_lpsp"):
            policies.append(name[: -len("_lpsp")])
    fields = (
        "lpsp",
        "deg",
        "uni0",
        "nint",
        "nprev",
        "waste",
        "wbat",
        "wfc",
        "wely",
        "outfc",
        "outely",
    )
    data = {"draw": columns["draw"], "factors": {k: columns[k] for k in factors}}
    data["policies"] = {
        policy: {field: columns[policy + "_" + field] for field in fields}
        for policy in policies
    }
    return data


def p1_cost(p1, strategy, voll):
    values = p1["strategies"][strategy]
    eens_term_at_3 = values["uni3"] - values["deg"]
    return values["deg"] + voll / REFERENCE_VOLL * eens_term_at_3


def p3_eens_kwh(p3, policy):
    values = p3["policies"][policy]
    return (values["uni0"] - values["deg"]) * 1000.0 / REFERENCE_VOLL


def p3_cost(p3, policy, voll, c_visit):
    values = p3["policies"][policy]
    eens = p3_eens_kwh(p3, policy)
    return (
        values["uni0"]
        + (voll - REFERENCE_VOLL) * eens / 1000.0
        + c_visit * values["nint"]
        + values["waste"]
    )


def _same_factors(a, b):
    return all(
        key in b["factors"] and np.array_equal(values, b["factors"][key])
        for key, values in a["factors"].items()
    )


def integrity_checks(p1, p3_by_tag):
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    expected_draws = np.arange(len(p1["draw"]))
    add("P1 draws complets", np.array_equal(p1["draw"], expected_draws), "0..%d" % (len(expected_draws) - 1))
    for tag, p3 in p3_by_tag.items():
        add("%s draws complets" % tag, np.array_equal(p3["draw"], expected_draws), "meme N=%d" % len(expected_draws))
        add("%s facteurs CRN=P1" % tag, _same_factors(p1, p3), "7 multiplicateurs arrondis")
        for policy, values in p3["policies"].items():
            closure = values["wbat"] + values["wfc"] + values["wely"]
            add(
                "%s %s waste ferme" % (tag, policy),
                np.max(np.abs(values["waste"] - closure)) <= 0.002,
                "tol=0.002 kEUR (arrondis)",
            )
            add(
                "%s %s nint entier" % (tag, policy),
                np.allclose(values["nint"], np.rint(values["nint"])),
                "",
            )

    p3_main = p3_by_tag["T6m"]
    p1_soh = p1["strategies"]["RB2(SoH)"]
    p3_instant = p3_main["policies"]["instant"]
    identical = all(
        np.array_equal(p1_soh[p1_key], p3_instant[p3_key])
        for p1_key, p3_key in (("lpsp", "lpsp"), ("deg", "deg"), ("uni3", "uni0"))
    )
    add("P1 SoH = P3 instant", identical, "identite tirage par tirage aux arrondis stockes")

    base = p3_by_tag["T6m"]
    for tag in ("T6m_m1.5", "T6m_m2"):
        other = p3_by_tag[tag]
        same = True
        for policy in ("instant", "corrective", "calendar"):
            for field in base["policies"][policy]:
                same &= np.array_equal(base["policies"][policy][field], other["policies"][policy][field])
        add("%s non-RUL = T6m" % tag, same, "ne pas compter comme replication")
    return checks


def _contrast_record(analysis, scenario, voll, c_visit, label, a, b):
    diff = a - b
    seed = _stable_seed("|".join(map(str, (analysis, scenario, voll, c_visit, label))))
    summary = summarize_difference(diff, bootstrap_seed=seed)
    return {
        "analysis": analysis,
        "scenario": scenario,
        "voll": voll,
        "c_visit": c_visit,
        "contrast": label,
        **summary,
        "es90_a_minus_es90_b": cvar_high(a) - cvar_high(b),
        "es90_difference": cvar_high(diff),
    }


def _fmt(value, digits=3):
    return ("%%.%df" % digits) % value


def _fmt_p(value):
    return "%.3g" % value


def analyse():
    p1 = load_p1()
    p3_by_tag = {tag: load_p3(path) for tag, path in P3_PATHS.items()}
    checks = integrity_checks(p1, p3_by_tag)
    if not all(ok for _, ok, _ in checks):
        failures = [name for name, ok, _ in checks if not ok]
        raise RuntimeError("Controles d'integrite en echec : %s" % failures)

    records = []
    p1_pairs = (
        ("Recale-RB2", "RB2(Recale)", "RB2"),
        ("Sched-Recale", "RB2(Sched)", "RB2(Recale)"),
        ("SoH-Sched", "RB2(SoH)", "RB2(Sched)"),
        ("SoH-RB2", "RB2(SoH)", "RB2"),
    )
    for voll in VOLLS:
        for label, a_name, b_name in p1_pairs:
            records.append(
                _contrast_record(
                    "P1", "200 mondes", voll, None, label,
                    p1_cost(p1, a_name, voll), p1_cost(p1, b_name, voll),
                )
            )

    p3_main = p3_by_tag["T6m"]
    p3_pairs = (
        ("rul-corrective", "rul", "corrective"),
        ("rul-calendar", "rul", "calendar"),
        ("calendar-corrective", "calendar", "corrective"),
    )
    for voll in VOLLS:
        for c_visit in C_VISITS:
            for label, a_name, b_name in p3_pairs:
                records.append(
                    _contrast_record(
                        "P3", "T6m_marge1", voll, c_visit, label,
                        p3_cost(p3_main, a_name, voll, c_visit),
                        p3_cost(p3_main, b_name, voll, c_visit),
                    )
                )

    report = []
    report.append("=" * 96)
    report.append("POST-TRAITEMENT STATISTIQUE APPARIE -- P1 DIAGNOSTIC SoH / P3 MAINTENANCE RUL")
    report.append("Sources legacy V9_4 ; aucune nouvelle simulation ; bootstrap apparie N=%d" % BOOTSTRAP_RESAMPLES)
    report.append("=" * 96)
    report.append("")
    report.append("STATUT SCIENTIFIQUE")
    report.append("-" * 96)
    report.append("Ces calculs quantifient les sorties historiques AVANT correction de la metrique segmentee")
    report.append("et du double comptage du pas de remplacement. Ils sont descriptifs/diagnostiques et devront")
    report.append("etre regeneres apres correction du moteur. Les politiques restent celles reglees a VoLL=3 :")
    report.append("le balayage VoLL 1/3/10 n'est pas une reoptimisation.")
    report.append("")
    report.append("CONTROLES D'INTEGRITE")
    report.append("-" * 96)
    for name, ok, detail in checks:
        report.append("[%s] %-48s %s" % ("OK" if ok else "ECHEC", name, detail))

    report.append("")
    report.append("DEFINITIONS")
    report.append("-" * 96)
    report.append("Difference = A-B : negative => A moins couteuse. IC95 = bootstrap percentile de la moyenne.")
    report.append("Gain/equiv/perte = nombres de mondes diff<0 / diff=0 / diff>0 ; test des signes hors egalites.")
    report.append("ES90 marginale = moyenne exacte des 20 pires couts sur 200 ; ES90(ecart) = pire decile du choix A-B.")
    report.append("P1 : EENS=(uni3-deg)*1000/3. P3 : uni0 contient deja deg+3*EENS/1000.")
    report.append("P3 : cout(V,C)=uni0+(V-3)*EENS/1000+C*nint+waste. Aucun CAPEX additionnel n'est ajoute.")

    report.append("")
    report.append("P1 -- DECOMPOSITION CONDITIONNELLE A LA VoLL")
    report.append("-" * 96)
    report.append("VoLL  contraste         mean [IC95] kEUR          median [Q1,Q3]       G/E/P       p_signe")
    for record in [r for r in records if r["analysis"] == "P1"]:
        report.append(
            "%4.0f  %-16s %+7.3f [%+7.3f,%+7.3f]  %+7.3f [%+7.3f,%+7.3f]  %3d/%3d/%3d  %s"
            % (
                record["voll"], record["contrast"], record["mean"],
                record["mean_ci95_low"], record["mean_ci95_high"], record["median"],
                record["q1"], record["q3"], record["wins"], record["ties"],
                record["losses"], _fmt_p(record["sign_pvalue"]),
            )
        )
    report.append("")
    report.append("Lecture : les trois increments ne sont pas des contributions invariantes. Recale-RB2 change")
    report.append("de signe entre VoLL=1 et 3 ; Sched-Recale change de signe entre VoLL=3 et 10. Le capteur")
    report.append("SoH reste favorable en moyenne, mais a VoLL=10 il perd dans %d/200 mondes." % next(
        r["losses"] for r in records if r["analysis"] == "P1" and r["contrast"] == "SoH-Sched" and r["voll"] == 10
    ))

    report.append("")
    report.append("P3 CENTRAL T=6 MOIS -- DEPENDANCE VoLL x C_INTERVENTION")
    report.append("-" * 96)
    report.append("VoLL  Cvis  contraste              mean [IC95] kEUR          median [Q1,Q3]       G/E/P")
    for record in [r for r in records if r["analysis"] == "P3"]:
        report.append(
            "%4.0f  %4.1f  %-22s %+7.3f [%+7.3f,%+7.3f]  %+7.3f [%+7.3f,%+7.3f]  %3d/%3d/%3d"
            % (
                record["voll"], record["c_visit"], record["contrast"], record["mean"],
                record["mean_ci95_low"], record["mean_ci95_high"], record["median"],
                record["q1"], record["q3"], record["wins"], record["ties"], record["losses"],
            )
        )

    report.append("")
    report.append("RISQUE DE NIVEAU vs RISQUE DE L'ECART (VoLL=3, Cvis=1.5)")
    report.append("-" * 96)
    for analysis, label, a, b in (
        ("P1", "SoH-Sched", p1_cost(p1, "RB2(SoH)", 3), p1_cost(p1, "RB2(Sched)", 3)),
        ("P3", "rul-corrective", p3_cost(p3_main, "rul", 3, 1.5), p3_cost(p3_main, "corrective", 3, 1.5)),
        ("P3", "rul-calendar", p3_cost(p3_main, "rul", 3, 1.5), p3_cost(p3_main, "calendar", 3, 1.5)),
    ):
        delta_es = cvar_high(a) - cvar_high(b)
        es_ci = bootstrap_cvar_difference_ci(a, b, seed=_stable_seed("ES|" + analysis + label))
        report.append(
            "%-3s %-19s  delta_ES90=%+7.3f [%+7.3f,%+7.3f] ; ES90(ecart)=%+7.3f kEUR"
            % (analysis, label, delta_es, es_ci[0], es_ci[1], cvar_high(a - b))
        )
    report.append("Une meilleure ES90 marginale ne signifie donc pas que A gagne dans le pire decile de A-B.")

    report.append("")
    report.append("CARTE DES GAGNANTS DE NIVEAU P3 (politiques fixes)")
    report.append("-" * 96)
    report.append("T     VoLL  Cvis   min_moyenne       min_ES90")
    for tag in ("T3m", "T6m", "T12m"):
        p3 = p3_by_tag[tag]
        for voll in VOLLS:
            for c_visit in C_VISITS:
                costs = {p: p3_cost(p3, p, voll, c_visit) for p in p3["policies"]}
                winner_mean = min(costs, key=lambda p: costs[p].mean())
                winner_es = min(costs, key=lambda p: cvar_high(costs[p]))
                report.append("%-5s %4.0f  %4.1f   %-16s  %-16s" % (tag, voll, c_visit, winner_mean, winner_es))

    report.append("")
    report.append("EQUIVALENCE OPERATIONNELLE PROPOSEE : RUL T12m vs CORRECTIF T3m")
    report.append("-" * 96)
    p3_t12 = p3_by_tag["T12m"]
    p3_t3 = p3_by_tag["T3m"]
    dlpsp = p3_t12["policies"]["rul"]["lpsp"] - p3_t3["policies"]["corrective"]["lpsp"]
    lpsp_ci = bootstrap_mean_ci(dlpsp, seed=_stable_seed("equivalence_lpsp"))
    report.append("Delta LPSP moyen = %+.4f pt ; IC95 [%+.4f,%+.4f]." % (dlpsp.mean(), lpsp_ci[0], lpsp_ci[1]))
    report.append("L'IC est inclus dans +/-0.05 pt, marge proposee a justifier a priori dans le manuscrit.")
    for c_visit in C_VISITS:
        diff = p3_cost(p3_t12, "rul", 3, c_visit) - p3_cost(p3_t3, "corrective", 3, c_visit)
        ci = bootstrap_mean_ci(diff, seed=_stable_seed("equivalence_cost|%s" % c_visit))
        report.append("Cvis=%3.1f : delta cout=%+.3f kEUR ; IC95 [%+.3f,%+.3f]" % (c_visit, diff.mean(), ci[0], ci[1]))

    report.append("")
    report.append("LIMITES A CONSERVER AVEC TOUTE CITATION")
    report.append("-" * 96)
    report.append("- Les IC quantifient l'echantillonnage de 200 mondes conditionnel a U_log[0.5,2], pas l'incertitude terrain.")
    report.append("- Les fichiers de periodes/marges reutilisent exactement les memes mondes : ce ne sont pas des repetitions.")
    report.append("- A VoLL differente, on reevalue les politiques reglees a VoLL=3 sans les reoptimiser.")
    report.append("- Les couts arrondis a 0.001 kEUR induisent quelques euros d'erreur, surtout pres des egalites.")
    report.append("- outfc+outely est une somme d'heures-composant, pas une duree d'indisponibilite du site.")
    report.append("- Le moteur legacy double-compte le pas de remplacement et la metrique finale ignore certains resets.")

    provenance = build_provenance(
        "postprocess_p1_p3_legacy_2026_07",
        [Path(__file__), HERE / "paired_stats.py", P1_PATH, *P3_PATHS.values()],
        {
            "volls": VOLLS,
            "c_visits_keur": C_VISITS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "reference_voll": REFERENCE_VOLL,
            "tail_definition": "largest_exact_ceil((1-q)*N)",
        },
    )
    final_lines = provenance_header_lines(provenance) + report
    OUT_TXT.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    fields = [
        "analysis", "scenario", "voll", "c_visit", "contrast", "n", "mean",
        "mean_ci95_low", "mean_ci95_high", "sample_sd", "median", "q1", "q3",
        "iqr", "p05", "p95", "wins", "ties", "losses", "sign_pvalue",
        "es90_a_minus_es90_b", "es90_difference",
    ]
    with OUT_TSV.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for record in records:
            stream.write("\t".join("" if record.get(field) is None else str(record.get(field)) for field in fields) + "\n")

    write_provenance_sidecar(
        str(OUT_TXT) + ".provenance.json", provenance, artifacts=(OUT_TXT, OUT_TSV)
    )
    return checks, records


def main():
    checks, records = analyse()
    print("Controles : %d/%d OK" % (sum(ok for _, ok, _ in checks), len(checks)))
    print("Contrastes : %d" % len(records))
    print("Rapport : %s" % OUT_TXT)
    print("Table  : %s" % OUT_TSV)


if __name__ == "__main__":
    main()
