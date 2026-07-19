"""
=============================================================================
dp_gridcheck.py -- CONTROLE DE ROBUSTESSE EN GRILLE  (a lancer sur le mesocentre)
=============================================================================

Verifie que le gain PD vs RB1/RB2 V11 p=2 (1 an, etat neuf) ne depend PAS de la finesse de
discretisation, et que la petite LPSP de la PD n'est pas un simple artefact de
grille. Resout la PD pour plusieurs resolutions (SoC x E_h2 x controles), EN
PARALLELE (un coeur par configuration, ProcessPoolExecutor facon analyse de
sensibilite), puis ecrit un tableau comparatif.

Sortie : runs/dp_gridcheck_v11_p2.txt  (+ echo console dans le .out SLURM)

Lance (sur le cluster) :  sbatch run_dp.slurm dp_gridcheck.py
Lance (local, NON recommande -- long) : python dp_gridcheck.py
=============================================================================
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

import dp_core as dp   # noyau : grilles, backward, forward, metriques

# Configurations a comparer : (Ns_soc, Nh_h2, n_fc, n_ely, n_iter)
CONFIGS = [
    (21,  21,  5,  20, 3),
    (31,  31,  7,  30, 3),
    (41,  41,  8,  40, 3),
    (51,  51, 10,  50, 3),
    (71,  71, 12,  60, 3),
]


def solve_one(cfg):
    """Resout la PD pour une configuration de grille et renvoie ses metriques."""
    Ns, Nh, n_fc, n_ely, n_iter = cfg
    t0 = time.time()
    soc_grid = np.linspace(dp.SOC_LO, dp.SOC_HI, Ns)
    h2_grid  = np.linspace(0.0, dp.E_H2_INIT, Nh)
    u = dp.control_grid(
        n_fc=n_fc, n_ely=n_ely, extra_u=dp.v11_control_anchors())
    pre = dp.precompute_controls(u)
    P_ref_net, _, _ = dp.net_reference(dp.N_YEAR)

    _, policy = dp.solve_cyclic(soc_grid, h2_grid, u, pre, P_ref_net,
                                n_iter=n_iter, verbose=False)
    data_dp = dp.forward_sim(dp.make_dp_policy(soc_grid, h2_grid, u, policy))
    m = dp.metrics(data_dp)
    m.update(dict(Ns=Ns, Nh=Nh, Nu=len(u), n_iter=n_iter,
                  secs=time.time() - t0))
    return m


def main():
    print("=" * 78)
    print(" CONTROLE DE GRILLE -- PD vs RB1/RB2 V11 p=2 (1 an, etat neuf)")
    print("=" * 78)

    # --- references best-vs-best V11 p=2 (une fois) ---
    references = {}
    for label, policy in (("RB1", dp.rb1_policy), ("RB2", dp.rb2_policy)):
        metrics = dp.metrics(dp.forward_sim(policy))
        references[label] = metrics
        print(f" {label} : LPSP {metrics['lpsp']:.4f}%  "
              f"deg {metrics['deg_keur']:.3f}  "
              f"UNIFIE {metrics['unified_keur']:.3f} kEUR")
    best_label, best_reference = min(
        references.items(), key=lambda item: item[1]['unified_keur'])

    # --- PD pour chaque grille, en parallele ---
    n_workers = min(len(CONFIGS), int(os.environ.get('SLURM_CPUS_PER_TASK',
                                                      os.cpu_count() or 1)))
    print(f" Lancement de {len(CONFIGS)} configs sur {n_workers} workers...\n")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(solve_one, c): c for c in CONFIGS}
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r['Ns'])

    # --- Tableau ---
    header = (f"{'SoCxH2':>10} {'Nu':>4} {'LPSP%':>8} {'deg_kE':>9} "
              f"{'LPS_kE':>8} {'UNIF_kE':>9} {'gain%':>7} {'sec':>7}")
    lines = [header, "-" * len(header)]
    for label, metrics in references.items():
        lines.append(
            f"{label:>10} {'-':>4} {metrics['lpsp']:8.4f} "
            f"{metrics['deg_keur']:9.3f} {metrics['lps_keur']:8.3f} "
            f"{metrics['unified_keur']:9.3f} {'-':>7} {'-':>7}")
    for r in results:
        gain = ((best_reference['unified_keur'] - r['unified_keur'])
                / best_reference['unified_keur'] * 100)
        lines.append(
            f"{str(r['Ns'])+'x'+str(r['Nh']):>10} {r['Nu']:4d} {r['lpsp']:8.4f} "
            f"{r['deg_keur']:9.3f} {r['lps_keur']:8.3f} {r['unified_keur']:9.3f} "
            f"{gain:7.1f} {r['secs']:7.0f}")
    lines.append(f"reference de gain: {best_label}; p_ELY=2")
    table = "\n".join(lines)
    print("\n" + table)

    out_dir = os.path.join(_THIS, "runs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dp_gridcheck_v11_p2.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\n Resultats -> {os.path.join(out_dir, 'dp_gridcheck_v11_p2.txt')}")


if __name__ == "__main__":
    main()
