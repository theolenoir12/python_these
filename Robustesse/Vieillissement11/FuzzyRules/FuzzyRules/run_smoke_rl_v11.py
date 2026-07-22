"""Smoke boucle fermee des policies 'regles apprises' (arbre) V11-p=2.

Reutilise exactement le harnais d'evaluation de ``run_smoke_flc_v11`` : meme
moteur ``init_and_run_loop``, meme comptabilite corrigee des remplacements, meme
VoLL de reporting. Compare les arbres distilles (plusieurs profondeurs) aux
references attribuables RB1 ``(0,20 ; 0,40)`` et RB2 ``(0,574 ; 0,465)``.

Rappel de portee : un smoke de quelques jours part de t=0 et reste in-sample
pour un arbre entraine sur les annees 0-14 ; c'est un test d'integration et un
signal grossier (Etape B), pas une conclusion. La selection de profondeur se
fait sur un run plus long et la generalisation sur des profils hors calibration
(Etape C).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--deadband-w", type=float, default=0.0)
    parser.add_argument("--information-set", choices=["I0", "IS"], default="I0")
    parser.add_argument("--min-samples-leaf", type=int, default=200)
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
    tree_meta = {}
    for depth in args.depths:
        tree = fit_tree(dataset, args.information_set, max_depth=depth,
                        min_samples_leaf=args.min_samples_leaf)
        pol = make_tree_policy_v11(tree, deadband_w=args.deadband_w)
        label = f"rl_tree_{args.information_set.lower()}_d{depth}"
        policies[label] = pol
        tree_meta[label] = {"n_leaves": pol.rl_metadata["n_leaves"],
                            "policy_id": pol.policy_id}

    manifest = {
        "protocol_id": "smoke-rl-tree-v11-p2-2026-07-21",
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "voll_eur_per_kwh": VOLL,
        "days_requested": float(args.days),
        "expected_steps": expected_steps,
        "profile_sha256": _profile_hash(expected_steps),
        "information_set": args.information_set,
        "depths": list(args.depths),
        "deadband_w": args.deadband_w,
        "min_samples_leaf": args.min_samples_leaf,
        "dataset_split_years": {k: list(v) for k, v in
                                dataset["split_years"].items()},
        "tree_meta": tree_meta,
        "replacement_accounting": "corrected",
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    run_dir = HERE / "runs" / f"smoke_rl_tree_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, policy in policies.items():
        summary, arrays, _ = _evaluate(policy, years)
        if summary["steps"] != expected_steps:
            raise AssertionError(
                f"{label}: {summary['steps']} pas, attendu {expected_steps}"
            )
        results[label] = summary
        np.savez_compressed(run_dir / f"{label}.npz", **arrays)
        extra = ""
        if label in tree_meta:
            extra = f" leaves={tree_meta[label]['n_leaves']:3d}"
        print(
            f"{label:26s} J3={summary['unified_voll3_eur']:10.3f} "
            f"deg={summary['degradation_eur']:9.3f} "
            f"EENS={summary['eens_kwh']:8.3f} "
            f"LPSP={summary['lpsp_pct']:6.3f}% "
            f"starts={summary['fc_starts']}/{summary['ely_starts']}{extra}",
            flush=True,
        )

    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "fingerprint": fingerprint,
                    "results": results}, indent=2, sort_keys=True) + "\n"
    )
    print(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
