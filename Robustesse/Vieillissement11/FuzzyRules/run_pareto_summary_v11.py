"""Complète la synthèse Pareto avec des points de référence recalculés au MÊME
harnais (init_and_run_loop, comptabilité corrigée), pour un overlay rigoureux.

Le front PD tracé par ``plot_pareto_families_v11`` est l'export canonique de la
PD (autre chemin de comptabilité que le ledger réalisé). Ce runner évalue, via
EXACTEMENT le même harnais que les stratégies :
  - la PD centrale (DPPolicy rollout, eps=3) -> point PD harnais-cohérent ;
  - la FLC-I0 réglée promue (parent des extensions), absente de la table de
    campagne car réglée dans une session antérieure.
Les points obtenus sont AJOUTÉS à la table de campagne (pareto_points_v11.tsv) ->
table fusionnée écrite dans figures_synthese/, puis figures régénérées.

Ne tourne que sur mésocentre (cache enseignant, résolution PD, simulateur).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .run_smoke_flc_v11 import HERE, V11, _evaluate
from .flc_policy_v11 import make_tuned_expert_flc_policy_v11
from .plot_pareto_families_v11 import DEFAULT_POINTS, load_points

_DP_DIR = str(V11 / "DP")
if _DP_DIR not in sys.path:
    sys.path.insert(0, _DP_DIR)


def _row(label, famille, aug, info, summary, source):
    deg_keur = summary["degradation_eur"] / 1000.0
    j3 = summary["unified_voll3_eur"]
    return (f"{label}\t{famille}\t{aug}\t{info}\t{summary['lpsp_pct']:.3f}\t"
            f"{summary['eens_kwh']:.3f}\t{deg_keur:.6f}\t{j3:.3f}\t{source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--with-pd", action="store_true", default=True,
                        help="recalcule le point PD via DPPolicy (peut être long)")
    parser.add_argument("--no-pd", dest="with_pd", action="store_false")
    parser.add_argument("--points", default=str(DEFAULT_POINTS))
    args = parser.parse_args()
    years = args.days / 365.0

    extra_rows = []

    # FLC-I0 réglée promue (parent I0). Peu coûteux, très fiable.
    flc = make_tuned_expert_flc_policy_v11()
    summary, _, _ = _evaluate(flc, years)
    extra_rows.append(_row("FLC-I0 réglée", "flc", "none", "I0", summary,
                           "pareto_summary"))
    print(_row("FLC-I0 réglée", "flc", "none", "I0", summary, "pareto_summary"))

    # Point PD harnais-cohérent (DPPolicy rollout, eps=3).
    if args.with_pd:
        from dp_aging import DPPolicy
        pd = DPPolicy(rollout=True, verbose=False)
        summary, _, _ = _evaluate(pd, years)
        extra_rows.append(_row("PD (eps=3, harnais)", "pd", "none", "-",
                               summary, "pareto_summary"))
        print(_row("PD (eps=3, harnais)", "pd", "none", "-", summary,
                   "pareto_summary"))

    # Table fusionnée = campagne + points recalculés.
    outdir = HERE / "figures_synthese"
    outdir.mkdir(parents=True, exist_ok=True)
    base_lines = Path(args.points).read_text(encoding="utf-8").rstrip("\n")
    merged = outdir / "pareto_points_merged.tsv"
    merged.write_text(base_lines + "\n" + "\n".join(extra_rows) + "\n",
                      encoding="utf-8")
    print(f"\nTable fusionnée : {merged}")

    # Régénère les figures depuis la table fusionnée.
    from . import plot_pareto_families_v11 as plotmod
    sys.argv = ["plot", "--points", str(merged), "--outdir", str(outdir)]
    plotmod.main()


if __name__ == "__main__":
    main()
