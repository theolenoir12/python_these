"""Smoke boucle fermée ANFIS TS1 deux branches (V11-p=2).

Compare, à moteur et enseignant identiques, l'ANFIS distillé aux références
RB1/RB2 et à l'arbre distillé (effet de FAMILLE, régresseur flou vs CART). On
rapporte, avec les performances, le NOMBRE DE RÈGLES et le TEMPS DE DÉCISION
amorti (critères de promotion du plan §2.3), et l'ablation I0/IS.

Rappel du cadre : ANFIS reste un apprenant par IMITATION du même maître ; le
chantier a montré (recap §10) que le clonage a un plafond (dérive + capacité).
Ce runner mesure si le lissage flou d'ordre 1 change la donne en boucle fermée.

Ne tourne que sur mésocentre (cache enseignant + simulateur).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy

from .run_smoke_flc_v11 import HERE, VOLL, _evaluate, _profile_hash
from .rl_dataset_v11 import build_dataset
from .rl_tree_policy_v11 import fit_tree, make_tree_policy_v11
from .anfis_policy_v11 import AnfisTwoBranch, make_anfis_policy_v11


def _fit_anfis_policy(dataset, information_set, n_mf, ridge, sigma_scale,
                      deadband_w):
    key = {"I0": "X_i0", "IS": "X_is"}[information_set]
    train = dataset["split"] == "train"
    model = AnfisTwoBranch.fit(
        dataset[key][train], dataset["y"][train], information_set,
        n_mf=n_mf, ridge=ridge, sigma_scale=sigma_scale)
    return make_anfis_policy_v11(model, deadband_w=deadband_w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--n-mf", type=int, default=3,
                        help="fonctions d'appartenance par entrée (3 -> 27 règles/branche)")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--deadband-w", type=float, default=0.0)
    parser.add_argument("--information-sets", nargs="+", default=["I0"],
                        choices=["I0", "IS"])
    parser.add_argument("--with-tree", action="store_true",
                        help="ajoute l'arbre I0 d4 (effet de famille)")
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 < args.days <= 365.0:
        raise ValueError("--days doit appartenir a ]0, 365]")
    years = args.days / 365.0
    expected_steps = max(0, int(round(args.days * 24.0)) - 1)

    dataset = build_dataset()

    policies = {
        "rb1_v11_p2_020_040": make_rb1_policy_v11(0.20, 0.40),
        "rb2_v11_p2_0574_0465": make_rb2_policy(0.574, 0.465),
    }
    meta = {}
    if args.with_tree:
        tree = fit_tree(dataset, "I0", max_depth=args.depth)
        policies[f"rl_tree_i0_d{args.depth}"] = make_tree_policy_v11(
            tree, deadband_w=args.deadband_w)
    for info in args.information_sets:
        pol = _fit_anfis_policy(dataset, info, args.n_mf, args.ridge,
                                args.sigma_scale, args.deadband_w)
        label = f"anfis_{info.lower()}_mf{args.n_mf}"
        policies[label] = pol
        meta[label] = {"rule_count": pol.anfis_metadata["rule_count"],
                       "policy_id": pol.policy_id}

    manifest = {
        "protocol_id": "smoke-anfis-ts1-v11-p2-2026-07-22",
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "voll_eur_per_kwh": VOLL,
        "days_requested": float(args.days),
        "expected_steps": expected_steps,
        "profile_sha256": _profile_hash(expected_steps),
        "n_mf": args.n_mf,
        "ridge": args.ridge,
        "sigma_scale": args.sigma_scale,
        "deadband_w": args.deadband_w,
        "information_sets": list(args.information_sets),
        "anfis_meta": meta,
        "replacement_accounting": "corrected",
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    run_dir = HERE / "runs" / f"smoke_anfis_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, policy in policies.items():
        summary, arrays, _ = _evaluate(policy, years)
        if summary["steps"] != expected_steps:
            raise AssertionError(
                f"{label}: {summary['steps']} pas, attendu {expected_steps}")
        results[label] = summary
        np.savez_compressed(run_dir / f"{label}.npz", **arrays)
        extra = ""
        if label in meta:
            rc = meta[label]["rule_count"]
            us_per_step = summary["runtime_s"] / max(summary["steps"], 1) * 1e6
            extra = (f" regles={rc['deficit']}+{rc['surplus']}"
                     f" t/pas={us_per_step:6.1f}us")
        print(
            f"{label:22s} J3={summary['unified_voll3_eur']:10.3f} "
            f"deg={summary['degradation_eur']:9.3f} "
            f"EENS={summary['eens_kwh']:8.3f} "
            f"LPSP={summary['lpsp_pct']:6.3f}% "
            f"starts={summary['fc_starts']}/{summary['ely_starts']}{extra}",
            flush=True,
        )

    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "fingerprint": fingerprint,
                    "results": results}, indent=2, sort_keys=True) + "\n")
    print(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
