"""Verifie si les caches ont rencontre l'ancienne borne de variation fautive.

Avant la correction du 20 juillet, la variation FC/ELY du premier pas etait
bornee par la capacite courante. Une puissance executee a l'heure precedente
legerement superieure a cette nouvelle capacite pouvait donc interdire un arret
complet. Ce diagnostic determine si des trajectoires terminees ont rencontre
ce cas ; elles ne doivent etre conservees que si le compteur reste nul.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Common import Init_EMR_MG_v16_python as I  # noqa: E402
from Common.electrochemistry import ely_pmax, fc_pmax  # noqa: E402


ETA = float(I.CONV["eta"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or HERE / "analysis" / "delta_bound_cache_audit.tsv"

    rows = [
        "label\tfc_previous_above_new_cap_steps\t"
        "ely_previous_above_new_cap_steps\tmax_fc_excess_w\tmax_ely_excess_w"
    ]
    totals = {"fc_steps": 0, "ely_steps": 0, "trajectories": 0}
    protocol = json.loads((args.run / "protocol.json").read_text())
    configs = {item["label"]: item for item in protocol["configs"]}
    for path in sorted(args.run.glob("*.npz")):
        if configs.get(path.stem, {}).get("kind", "mpc") != "mpc":
            continue
        with np.load(path, allow_pickle=False) as data:
            fc = np.asarray(data["P_dc_fc"], dtype=float)
            ely = np.abs(np.asarray(data["P_dc_ely"], dtype=float))
            alpha_fc = np.asarray(data["alpha_fc"], dtype=float)[1:len(fc)]
            alpha_ely = np.asarray(data["alpha_ely"], dtype=float)[1:len(ely)]
        fc_cap = 0.999 * ETA * np.asarray(fc_pmax(alpha_fc), dtype=float)
        ely_cap = 0.999 * np.asarray(ely_pmax(alpha_ely), dtype=float) / ETA
        fc_excess = fc[:-1] - fc_cap
        ely_excess = ely[:-1] - ely_cap
        fc_steps = int(np.count_nonzero(fc_excess > 1e-8))
        ely_steps = int(np.count_nonzero(ely_excess > 1e-8))
        totals["fc_steps"] += fc_steps
        totals["ely_steps"] += ely_steps
        totals["trajectories"] += 1
        rows.append(
            f"{path.stem}\t{fc_steps}\t{ely_steps}\t"
            f"{float(np.max(fc_excess, initial=0.0)):.12g}\t"
            f"{float(np.max(ely_excess, initial=0.0)):.12g}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    output.with_suffix(".json").write_text(json.dumps(totals, indent=2) + "\n")
    if totals["fc_steps"] or totals["ely_steps"]:
        raise RuntimeError(
            "au moins un cache termine a rencontre l'ancienne borne de variation")
    print(
        f"OK -- {totals['trajectories']} trajectoires, aucun contact avec "
        f"l'ancienne borne -> {output}")


if __name__ == "__main__":
    main()
