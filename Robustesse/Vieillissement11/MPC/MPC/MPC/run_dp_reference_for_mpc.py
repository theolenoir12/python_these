"""Calcule un front DP V11 sur le meme horizon que le screening MPC."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DP_DIR = HERE.parent / "DP"
sys.path.insert(0, str(DP_DIR))
sys.path.insert(0, str(HERE.parent))

import dp_pareto as dp  # noqa: E402
from Common.degradation_v11 import MODEL_ID  # noqa: E402


def _scalar_summary(result: dict) -> dict:
    clean = {}
    for key, value in result.items():
        if key.startswith("_"):
            continue
        clean[key] = value.item() if isinstance(value, np.generic) else value
    return clean


def _write(output: Path, results: list[dict], years: float,
           ns: int, nh: int, n_fc: int, n_ely: int, n_iter: int) -> Path:
    ordered = sorted(results, key=lambda item: item["eps"])
    (output / "summary.json").write_text(json.dumps(ordered, indent=2) + "\n")
    rows = [
        "eps\tlpsp_pct\tdegradation_keur\teens_kwh\tj_voll3_keur\t"
        "soh_bat\tsoh_fc\tsoh_ely\twall_seconds"
    ]
    for item in ordered:
        rows.append(
            f"{item['eps']:.10g}\t{item['lpsp']:.10g}\t{item['deg']:.10g}\t"
            f"{item['eens_kwh']:.10g}\t{item['unif3']:.10g}\t"
            f"{item['soh_bat']:.10g}\t{item['soh_fc']:.10g}\t"
            f"{item['soh_ely']:.10g}\t{item['sec']:.10g}"
        )
    (output / "points.tsv").write_text("\n".join(rows) + "\n")
    artifact = output / f"dp_reference_{years:g}y_{ns}x{nh}_v2.npz"
    if ordered:
        col = lambda key: np.asarray([item[key] for item in ordered])
        np.savez_compressed(
            artifact,
            model_id=np.array(MODEL_ID), years=np.array(years),
            ns=np.array(ns), nh=np.array(nh), n_fc=np.array(n_fc),
            n_ely=np.array(n_ely), n_iter=np.array(n_iter),
            eps=col("eps"), lpsp=col("lpsp"), deg_keur=col("deg"),
            eens_kwh=col("eens_kwh"), unif3_keur=col("unif3"),
            soh_bat=col("soh_bat"), soh_fc=col("soh_fc"), soh_ely=col("soh_ely"),
            nondominated=dp._nondominated_mask(col("eens_kwh"), col("deg")),
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--eps", nargs="+", type=float, default=dp.EPSILONS_FULL)
    parser.add_argument("--ns", type=int, default=51)
    parser.add_argument("--nh", type=int, default=51)
    parser.add_argument("--n-fc", type=int, default=10)
    parser.add_argument("--n-ely", type=int, default=50)
    parser.add_argument("--n-iter", type=int, default=3)
    args = parser.parse_args()
    if (args.years <= 0
            or min(args.ns, args.nh, args.n_fc, args.n_ely) <= 1
            or args.n_iter < 1):
        raise SystemExit("horizon, tailles de grille et n_iter invalides")

    protocol = {
        "model_id": MODEL_ID, "years": args.years, "eps": args.eps,
        "ns": args.ns, "nh": args.nh, "n_fc": args.n_fc, "n_ely": args.n_ely,
        "n_iter": args.n_iter,
        "dp_v2": bool(dp.DP_V2), "recompute": "yearly",
        "information": "profil annuel reel connu dans chaque reconstruction",
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = DP_DIR / "results" / f"mpc_reference_{args.years:g}y_{fingerprint}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    results: list[dict] = []
    jobs = []
    for eps in args.eps:
        tag = str(eps).replace(".", "p")
        cached = output / f"eps_{tag}.json"
        if cached.exists():
            results.append(json.loads(cached.read_text()))
        else:
            jobs.append((
                eps, args.ns, args.nh, args.n_fc, args.n_ely,
                args.years, args.n_iter))
    artifact = _write(
        output, results, args.years, args.ns, args.nh, args.n_fc, args.n_ely,
        args.n_iter)

    workers = max(1, min(args.workers, len(jobs) or 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(dp.run_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            raw = future.result()
            result = _scalar_summary(raw)
            tag = str(result["eps"]).replace(".", "p")
            (output / f"eps_{tag}.json").write_text(
                json.dumps(result, indent=2) + "\n")
            results.append(result)
            artifact = _write(
                output, results, args.years, args.ns, args.nh, args.n_fc,
                args.n_ely, args.n_iter)
            print(f"[eps={result['eps']}] LPSP={result['lpsp']:.4f}% "
                  f"deg={result['deg']:.3f} kEUR", flush=True)
    if len(results) != len(args.eps):
        raise RuntimeError("front DP incomplet")
    print(f"OK -> {artifact}")


if __name__ == "__main__":
    main()
