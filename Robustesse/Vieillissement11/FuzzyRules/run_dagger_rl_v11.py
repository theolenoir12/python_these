"""DAgger boucle fermee (direction B) : distillation PD -> arbre, V11-p=2.

Boucle : on deroule l'arbre courant sur 1 an (moteur init_and_run_loop), on
RE-ETIQUETTE les etats qu'il a visites avec l'action de la PD (grille de
politique 1 an SoH=1 de dp_core), on agrege ces paires au jeu enseignant, on
re-ajuste l'arbre, on itere. Objectif : voir si corriger la derive de
distribution (la cause diagnostiquee du domino de l'arbre, recap sections 2.2 /
8) rapproche l'arbre I0 des references RB1/RB2 la ou aucune injection de
foresight n'y parvenait.

Oracle = grille de politique PD SoH=1 (sans etat). LIMITE ASSUMEE : le disciple
vieillit un peu au fil de l'annee alors que la grille est au neuf ; sur l'annee 0
l'ecart de SoH reste faible. Une version pleinement aging-aware utiliserait le
rollout de dp_aging (avec etat) et est reportee.

Ne tourne que sur mesocentre (cache enseignant + resolution PD + simulateur).
Soumettre via run_dagger_rl_v11.slurm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from Common.degradation_v11 import ELY_V11, MODEL_ID
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy

from .run_smoke_flc_v11 import HERE, V11, VOLL, _evaluate, _profile_hash
from .rl_dataset_v11 import I0_FEATURES, build_dataset
from .rl_tree_policy_v11 import make_tree_policy_v11
from .rl_dagger_v11 import (
    DiscipleStateLogger,
    build_i0_features,
    dagger_aggregate,
    relabel_with_policy_grid,
)

# dp_core n'est pas un package : on ajoute DP/ au path (comme dp_aging).
_DP_DIR = str(V11 / "DP")
if _DP_DIR not in sys.path:
    sys.path.insert(0, _DP_DIR)


def build_pd_policy_grid(n_iter=3, n_fc=10, n_ely=50, verbose=False):
    """Resout la PD 1 an SoH=1 et renvoie (soc_grid, h2_grid, u, policy)."""
    import dp_core as dpc
    soc_grid = np.linspace(dpc.SOC_LO, dpc.SOC_HI, 51)
    h2_grid = np.linspace(0.0, dpc.E_H2_INIT, 51)
    u = dpc.control_grid(n_fc=n_fc, n_ely=n_ely, extra_u=dpc.v11_control_anchors())
    pre = dpc.precompute_controls(u)
    P_ref, _, _ = dpc.net_reference(dpc.N_YEAR)
    _, policy = dpc.solve_cyclic(soc_grid, h2_grid, u, pre, P_ref,
                                 n_iter=n_iter, verbose=verbose)
    return soc_grid, h2_grid, u, policy


def fit_i0_tree(X, y, sample_weight, max_depth, min_samples_leaf,
                random_state=0):
    """Ajuste un arbre I0 sur des tableaux explicites (agreges DAgger)."""
    tree = DecisionTreeRegressor(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=random_state)
    tree.fit(X, y, sample_weight=sample_weight)
    tree._genial_meta = {
        "information_set": "I0",
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "sample_weight": "dagger",
        "n_leaves": int(tree.get_n_leaves()),
        "features": list(I0_FEATURES),
    }
    return tree


def _fmt(label, summary, extra=""):
    return (f"{label:26s} J3={summary['unified_voll3_eur']:10.3f} "
            f"deg={summary['degradation_eur']:9.3f} "
            f"EENS={summary['eens_kwh']:8.3f} "
            f"LPSP={summary['lpsp_pct']:6.3f}% "
            f"starts={summary['fc_starts']}/{summary['ely_starts']}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5,
                        help="nombre d'iterations DAgger (au-dela de l'arbre 0)")
    parser.add_argument("--deadband-w", type=float, default=0.0)
    parser.add_argument("--visited-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 < args.days <= 365.0:
        raise ValueError("--days doit appartenir a ]0, 365]")
    years = args.days / 365.0
    expected_steps = max(0, int(round(args.days * 24.0)) - 1)

    dataset = build_dataset()
    train = dataset["split"] == "train"
    X0 = dataset["X_i0"][train]
    y0 = dataset["y"][train]

    print("Resolution de la grille de politique PD (1 an, SoH=1)...", flush=True)
    soc_grid, h2_grid, u, policy_grid = build_pd_policy_grid()
    print(f"  grille {policy_grid.shape}, {len(u)} controles.", flush=True)

    references = {
        "rb1_v11_p2_020_040": make_rb1_policy_v11(0.20, 0.40),
        "rb2_v11_p2_0574_0465": make_rb2_policy(0.574, 0.465),
    }
    results = {}
    for label, policy in references.items():
        summary, _, _ = _evaluate(policy, years)
        results[label] = summary
        print(_fmt(label, summary), flush=True)

    # Iteration 0 : arbre distille classique (behavioral cloning).
    tree = fit_i0_tree(X0, y0, None, args.depth, args.min_samples_leaf)
    relabel_batches = []
    dagger_trace = []
    for k in range(args.iterations + 1):
        policy = make_tree_policy_v11(tree, deadband_w=args.deadband_w)
        logger = DiscipleStateLogger(policy)
        summary, _, _ = _evaluate(logger, years)
        if summary["steps"] != expected_steps:
            raise AssertionError(
                f"iter {k}: {summary['steps']} pas, attendu {expected_steps}")
        label = f"dagger_iter{k}_d{args.depth}"
        results[label] = summary
        dagger_trace.append({"iter": int(k), "n_leaves": int(tree.get_n_leaves()),
                             "train_size": int(len(y0) + sum(
                                 len(b[1]) for b in relabel_batches)),
                             **{key: summary[key] for key in
                                ("unified_voll3_eur", "degradation_eur",
                                 "eens_kwh", "lpsp_pct")}})
        print(_fmt(label, summary,
                   extra=f" leaves={tree.get_n_leaves():3d}"), flush=True)

        # Re-etiquetage des etats visites par la PD + agregation + refit.
        visited = logger.visited()
        Xk = build_i0_features(visited)
        yk = relabel_with_policy_grid(visited, policy_grid, u, soc_grid, h2_grid)
        relabel_batches.append((Xk, yk))
        X, y, w = dagger_aggregate(X0, y0, relabel_batches,
                                   visited_weight=args.visited_weight)
        tree = fit_i0_tree(X, y, w, args.depth, args.min_samples_leaf)

    manifest = {
        "protocol_id": "dagger-rl-tree-v11-p2-2026-07-22",
        "model_id": MODEL_ID,
        "ely_stress_exponent": float(ELY_V11["stress_exponent"]),
        "voll_eur_per_kwh": VOLL,
        "days_requested": float(args.days),
        "expected_steps": expected_steps,
        "profile_sha256": _profile_hash(expected_steps),
        "depth": args.depth,
        "min_samples_leaf": args.min_samples_leaf,
        "iterations": args.iterations,
        "deadband_w": args.deadband_w,
        "visited_weight": args.visited_weight,
        "oracle": "pd_policy_grid_1y_soh1 (dp_core)",
        "oracle_caveat": "disciple vieillit; grille au neuf (annee 0, ecart faible)",
        "replacement_accounting": "corrected",
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    run_dir = HERE / "runs" / f"dagger_rl_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "fingerprint": fingerprint,
                    "results": results, "dagger_trace": dagger_trace},
                   indent=2, sort_keys=True) + "\n")

    best = min(dagger_trace, key=lambda r: r["unified_voll3_eur"])
    print(f"\nMeilleure iteration DAgger : #{best['iter']} "
          f"J3={best['unified_voll3_eur']:.3f} "
          f"(arbre 0 : {dagger_trace[0]['unified_voll3_eur']:.3f} ; "
          f"RB1 : {results['rb1_v11_p2_020_040']['unified_voll3_eur']:.3f})")
    print(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
