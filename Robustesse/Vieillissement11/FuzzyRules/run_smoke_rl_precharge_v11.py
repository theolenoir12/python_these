"""Smoke boucle fermee de la direction A : arbre I0 + REGLE de precharge.

Question posee (recap section 5, direction A) : la foresight, INUTILE comme
feature de l'arbre (section 2.5), redevient-elle utile comme REGLE de precharge
au-dessus de l'arbre I0 distille, comme elle l'a ete au-dessus de la FLC experte
(FLC-IF, -1,6 % de J3) ?

Comparaison strictement appariee, meme moteur ``init_and_run_loop`` et meme
prevision oracle fournie par la boucle :

  - RB1 (0,20 ; 0,40) et RB2 (0,574 ; 0,465)          : references ;
  - rl_tree_i0_d{d}                                    : arbre I0 nu (domine) ;
  - rl_tree_precharge_i0_d{d}                          : arbre I0 + precharge A ;
  - flc_if_selected                                   : la FLC-IF promue (le
    parent FLC de la MEME regle) pour lire l'effet de famille.

Le test nul (force 0 = arbre nu bit-a-bit) est couvert par les tests unitaires ;
ici on mesure la performance en boucle fermee.
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

from .flc_forecast_policy_v11 import make_selected_if_policy_v11
from .run_smoke_flc_v11 import HERE, VOLL, _evaluate, _profile_hash
from .rl_dataset_v11 import build_dataset
from .rl_tree_policy_v11 import fit_tree, make_tree_policy_v11
from .rl_tree_precharge_policy_v11 import make_tree_precharge_policy_v11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--depth", type=int, default=4,
                        help="profondeur de l'arbre I0 de base (d4 = meilleur "
                             "arbre nu du recap)")
    parser.add_argument("--deadband-w", type=float, default=0.0)
    parser.add_argument("--forecast-scenario", default="oracle",
                        choices=["oracle", "gaussian_iid", "gaussian_ar1",
                                 "persistence"])
    parser.add_argument("--forecast-strength", type=float, default=1.0)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--with-flc-if", action="store_true",
                        help="ajoute la FLC-IF promue (effet de famille)")
    args = parser.parse_args()
    if not 0.0 < args.days <= 365.0:
        raise ValueError("--days doit appartenir a ]0, 365]")
    years = args.days / 365.0
    expected_steps = max(0, int(round(args.days * 24.0)) - 1)

    dataset = build_dataset()
    tree = fit_tree(dataset, "I0", max_depth=args.depth,
                    min_samples_leaf=args.min_samples_leaf)

    base = make_tree_policy_v11(tree, deadband_w=args.deadband_w)
    precharge = make_tree_precharge_policy_v11(
        tree, deadband_w=args.deadband_w,
        forecast_strength=args.forecast_strength,
        forecast_scenario=args.forecast_scenario, noise_seed=args.noise_seed,
    )

    policies = {
        "rb1_v11_p2_020_040": make_rb1_policy_v11(0.20, 0.40),
        "rb2_v11_p2_0574_0465": make_rb2_policy(0.574, 0.465),
        f"rl_tree_i0_d{args.depth}": base,
        f"rl_tree_precharge_i0_d{args.depth}_{args.forecast_scenario}": precharge,
    }
    if args.with_flc_if:
        policies["flc_if_selected"] = make_selected_if_policy_v11(
            forecast_scenario=args.forecast_scenario, noise_seed=args.noise_seed,
        )

    meta = {
        label: {
            "policy_id": getattr(pol, "policy_id", label),
            "information_set": getattr(pol, "information_set", None),
            "n_leaves": getattr(pol, "rl_metadata", {}).get("n_leaves"),
        }
        for label, pol in policies.items()
    }
    manifest = {
        "protocol_id": "smoke-rl-precharge-v11-p2-2026-07-21",
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "voll_eur_per_kwh": VOLL,
        "days_requested": float(args.days),
        "expected_steps": expected_steps,
        "profile_sha256": _profile_hash(expected_steps),
        "base_depth": args.depth,
        "deadband_w": args.deadband_w,
        "forecast_scenario": args.forecast_scenario,
        "forecast_strength": args.forecast_strength,
        "min_samples_leaf": args.min_samples_leaf,
        "noise_seed": args.noise_seed,
        "precharge_spec_sha256": precharge.rl_metadata["spec_sha256"],
        "dataset_split_years": {k: list(v) for k, v in
                                dataset["split_years"].items()},
        "policy_meta": meta,
        "replacement_accounting": "corrected",
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    run_dir = HERE / "runs" / f"smoke_rl_precharge_{fingerprint}"
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
        diag = ""
        if hasattr(policy, "forecast_diagnostics"):
            d = policy.forecast_diagnostics()
            diag = (f" precharge={d['precharge_applied_steps']}"
                    f" H2kept={d['ely_energy_removed_kwh_dc']:.1f}kWh")
        print(
            f"{label:44s} J3={summary['unified_voll3_eur']:10.3f} "
            f"deg={summary['degradation_eur']:9.3f} "
            f"EENS={summary['eens_kwh']:8.3f} "
            f"LPSP={summary['lpsp_pct']:6.3f}% "
            f"starts={summary['fc_starts']}/{summary['ely_starts']}{diag}",
            flush=True,
        )

    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "fingerprint": fingerprint,
                    "results": results}, indent=2, sort_keys=True) + "\n"
    )
    print(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
