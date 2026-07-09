"""
=============================================================================
dp_gridcheck.py -- CONTROLE DE ROBUSTESSE EN GRILLE  (a lancer sur le mesocentre)
=============================================================================

Verifie que le gain PD vs RB2 (1 an, etat neuf) ne depend PAS de la finesse de
discretisation, et que la petite LPSP de la PD n'est pas un simple artefact de
grille. Resout la PD pour plusieurs resolutions (SoC x E_h2 x controles), EN
PARALLELE (un coeur par configuration, ProcessPoolExecutor facon analyse de
sensibilite), puis ecrit un tableau comparatif.

Sortie : results/dp_gridcheck.txt  (+ echo console dans le .out SLURM)

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
    (41,  41,  8,  40, 3),
    (51,  51, 10,  50, 3),
    (71,  71, 12,  60, 3),
    (91,  91, 14,  75, 3),
    (111, 111, 16, 90, 3),
    (131, 131, 18, 110, 3),
]


def solve_one(cfg):
    """Resout la PD pour une configuration de grille et renvoie ses metriques."""
    Ns, Nh, n_fc, n_ely, n_iter = cfg
    t0 = time.time()
    soc_grid = np.linspace(dp.SOC_LO, dp.SOC_HI, Ns)
    h2_grid  = np.linspace(0.0, dp.E_H2_INIT, Nh)
    u = dp.control_grid(n_fc=n_fc, n_ely=n_ely)
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
    print(" CONTROLE DE ROBUSTESSE EN GRILLE -- PD vs RB2 (1 an, etat neuf)")
    print("=" * 78)

    # --- RB2 baseline (une fois, peu couteux) ---
    data_rb = dp.forward_sim(dp.rb2_policy)
    m_rb = dp.metrics(data_rb)
    print(f" RB2 : LPSP {m_rb['lpsp']:.4f}%  deg {m_rb['deg_keur']:.3f}  "
          f"UNIFIE {m_rb['unified_keur']:.3f} kEUR")

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
    lines.append(f"{'RB2':>10} {'-':>4} {m_rb['lpsp']:8.4f} {m_rb['deg_keur']:9.3f} "
                 f"{m_rb['lps_keur']:8.3f} {m_rb['unified_keur']:9.3f} {'-':>7} {'-':>7}")
    for r in results:
        gain = (m_rb['unified_keur'] - r['unified_keur']) / m_rb['unified_keur'] * 100
        lines.append(
            f"{str(r['Ns'])+'x'+str(r['Nh']):>10} {r['Nu']:4d} {r['lpsp']:8.4f} "
            f"{r['deg_keur']:9.3f} {r['lps_keur']:8.3f} {r['unified_keur']:9.3f} "
            f"{gain:7.1f} {r['secs']:7.0f}")
    table = "\n".join(lines)
    print("\n" + table)

    out_dir = os.path.join(_THIS, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dp_gridcheck.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\n Resultats -> {os.path.join(out_dir, 'dp_gridcheck.txt')}")


if __name__ == "__main__":
    main()
