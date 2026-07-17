"""Banc reproductible d'optimisation et de comparaison V11.

Le fichier d'entree est un JSON contenant une liste de candidats :

    [{"label": "rb2_059_049", "kind": "rb2",
      "params": {"fc_setpoint": 0.59, "ely_setpoint": 0.49}}]

Kinds admis : ``rb2``, ``rb2_soh`` et ``rb1``. Chaque simulation utilise le
noyau local V11, le ledger corrige et J = degradation + VoLL * EENS. Les
resultats sont sauvegardes au fil de l'eau dans un JSONL, ce qui permet une
reprise apres interruption sans perdre les candidats deja calcules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Common.main_init_and_loop import init_and_run_loop
from Common.degradation_v11 import ELY_V11, FC_V11, MODEL_ID
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy
from Common.rb2_soh_policy_v11 import make_rb2_soh_policy_v11
from Common.reliability_metrics import compute_reliability_metrics

VOLL_EUR_PER_KWH = 3.0
_MODEL_DEFAULTS = {"ely": dict(ELY_V11), "fc": dict(FC_V11)}


def _assert_local_v11():
    modules = (
        sys.modules[init_and_run_loop.__module__],
        sys.modules[make_rb2_policy.__module__],
        sys.modules[make_rb2_soh_policy_v11.__module__],
        sys.modules[make_rb1_policy_v11.__module__],
    )
    for module in modules:
        path = Path(module.__file__).resolve()
        if HERE not in path.parents:
            raise RuntimeError(f"import hors Vieillissement11: {path}")


def _policy(candidate):
    kind = candidate["kind"]
    params = dict(candidate.get("params", {}))
    if kind == "rb2":
        return make_rb2_policy(**params)
    if kind == "rb2_soh":
        return make_rb2_soh_policy_v11(**params)
    if kind == "rb1":
        return make_rb1_policy_v11(**params)
    if kind == "file":
        folder = str(params.pop("folder"))
        if params:
            raise ValueError(f"parametres file inconnus: {sorted(params)}")
        path = (HERE / folder / "get_optimal_action_RB.py").resolve()
        if HERE not in path.parents or not path.is_file():
            raise ValueError(f"politique hors V11 ou absente: {path}")
        spec = importlib.util.spec_from_file_location(
            f"_v11_policy_{folder.replace('-', '_')}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_optimal_action_RB
    raise ValueError(f"kind inconnu: {kind}")


def _apply_model(candidate):
    """Reinitialise puis applique les variantes de sensibilite du candidat."""
    for name, target in (("ely", ELY_V11), ("fc", FC_V11)):
        target.clear()
        target.update(_MODEL_DEFAULTS[name])
        overrides = dict(candidate.get("model", {}).get(name, {}))
        unknown = set(overrides) - set(target)
        if unknown:
            raise ValueError(f"parametres {name} inconnus: {sorted(unknown)}")
        target.update(overrides)


def _starts(power):
    on = np.abs(np.asarray(power, dtype=float)) > 1e-9
    return int(np.count_nonzero(on & ~np.r_[False, on[:-1]]))


def _on_hours(power, dt_h):
    return float(
        np.count_nonzero(np.abs(np.asarray(power, dtype=float)) > 1e-9) * dt_h
    )


def evaluate(payload):
    candidate, n_years, voll = payload
    _assert_local_v11()
    _apply_model(candidate)
    started = time.time()
    data = init_and_run_loop(
        _policy(candidate), n_years=int(n_years),
        replacement_accounting="corrected",
    )
    reliability = compute_reliability_metrics(data)
    ledger = data["degradation_ledger"]
    parts = ledger["total_eur"]
    degradation = float(sum(parts.values()))
    dt_h = float(data["temps"][1] - data["temps"][0]) / 3600.0
    first = data["first_life_metrics"]
    replacements = {
        component: sum(
            event["component"] == component for event in ledger["events"]
        )
        for component in ("bat", "fc", "ely")
    }
    result = {
        "label": candidate["label"],
        "kind": candidate["kind"],
        "params": candidate.get("params", {}),
        "model": candidate.get("model", {}),
        "model_id": MODEL_ID,
        "effective_model": {"ely": dict(ELY_V11), "fc": dict(FC_V11)},
        "n_years": int(n_years),
        "voll_eur_per_kwh": float(voll),
        "eens_kwh": reliability["eens_kwh"],
        "lpsp_pct": reliability["lpsp_pct"],
        "load_energy_kwh": reliability["load_energy_kwh"],
        "degradation_eur": degradation,
        "battery_eur": float(parts["bat"]),
        "fc_eur": float(parts["fc"]),
        "ely_eur": float(parts["ely"]),
        "unified_eur": degradation + float(voll) * reliability["eens_kwh"],
        "terminal_soc": float(data["SoC"][-1]),
        "terminal_h2_kwh": float(data["E_h2"][-1]),
        "terminal_soh_bat": float(data["SoH_bat"][-1]),
        "terminal_soh_fc": float(data["SoH_fc"][-1]),
        "terminal_soh_ely": float(data["SoH_ely"][-1]),
        "fc_starts": _starts(data["P_fc"]),
        "ely_starts": _starts(data["P_ely"]),
        "fc_on_h": _on_hours(data["P_fc"], dt_h),
        "ely_on_h": _on_hours(data["P_ely"], dt_h),
        "replacements": replacements,
        "first_life": {
            component: {
                key: first[component].get(key)
                for key in (
                    "calendar_h", "on_h", "efph", "energy_kwh", "starts",
                    "eol_reached",
                )
            }
            for component in ("battery", "fc", "ely")
        },
        "elapsed_s": time.time() - started,
    }
    return result


def _candidate_key(candidate):
    return json.dumps(
        {
            "kind": candidate["kind"],
            "params": candidate.get("params", {}),
            "model": candidate.get("model", {}),
        },
        sort_keys=True, separators=(",", ":"),
    )


def _read_completed(output):
    completed = set()
    if not output.exists():
        return completed
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                completed.add(_candidate_key(row))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--voll", type=float, default=VOLL_EUR_PER_KWH)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    _assert_local_v11()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    labels = [candidate["label"] for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("labels candidats non uniques")
    completed = _read_completed(args.output)
    pending = [candidate for candidate in candidates if _candidate_key(candidate) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"V11 local; {len(pending)}/{len(candidates)} candidats; "
        f"{args.years} ans; {min(args.workers, max(1, len(pending)))} workers",
        flush=True,
    )
    if not pending:
        return

    payloads = [(candidate, args.years, args.voll) for candidate in pending]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        futures = {pool.submit(evaluate, payload): payload[0] for payload in payloads}
        with args.output.open("a", encoding="utf-8") as stream:
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                print(
                    f"[{index}/{len(pending)}] {result['label']}: "
                    f"J={result['unified_eur']:.2f} EUR; "
                    f"deg={result['degradation_eur']:.2f}; "
                    f"EENS={result['eens_kwh']:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
