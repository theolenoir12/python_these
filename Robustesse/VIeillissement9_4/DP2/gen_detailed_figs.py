"""
gen_detailed_figs.py -- pour QUELQUES eps du front de Pareto PD, re-simule la
politique PD-sequentielle (re-solve DP + boucle 25 ans EXACTE) et produit les
MEMES figures que les autres strategies via Common.main_plot.run_main_plot
(notamment everything_combined_v2_2.pdf et all_aging_2.pdf).

Les figures vont dans :  figures_pareto_meso/eps_<val>/Figures/<n>h/
(meme arborescence que les dossiers de strategie RB2(...)/Figures/...).

Avant les figures, on VALIDE la reproduction locale : LPSP/deg recalcules doivent
coller aux valeurs stockees dans results_meso/dp_pareto_25y_51x51.npz.

Lancer :  python gen_detailed_figs.py 3            (un eps)
          python gen_detailed_figs.py 0.05 0.2 3 50 (plusieurs, sequentiel)
"""
import os
import sys
import time

# IMPORTANT : fixer le dossier de donnees AVANT tout import (sinon le setdefault
# de dp_aging/dp_pareto pointerait vers le chemin mesocentre Linux, absent ici).
# setdefault (et non '=') : sur le mesocentre, le .slurm exporte deja
# GENIAL_DATA_DIR=$WORK/genial_data -> on NE l'ecrase PAS. En local Windows, la
# variable n'est pas definie -> on tombe sur le chemin Data local.
os.environ.setdefault('GENIAL_DATA_DIR', r'C:\Users\tlenoi01\Doctorat\Data')

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))            # Vieillissement8/
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', 'RB2')))    # RB2/
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', '..', 'Analyse_sensibilite')))

import dp_core as dpc
from dp_aging import DPPolicy
from sens_common import metrics as sens_metrics
from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot

VOLL_REF = 3.0
# Front de Pareto pour la validation : results_meso/ (copie locale des sorties
# meso) en priorite, sinon results/ (sorties directes sur le mesocentre).
FRONT_NPZ = os.path.join(_THIS, "results_meso", "dp_pareto_25y_51x51.npz")
if not os.path.exists(FRONT_NPZ):
    FRONT_NPZ = os.path.join(_THIS, "results", "dp_pareto_25y_51x51.npz")
OUT_ROOT = os.path.join(_THIS, "figures_pareto_meso")


def _stored(eps):
    """(lpsp, deg, eens, unif3) stockes pour cet eps, ou None si absent."""
    if not os.path.exists(FRONT_NPZ):
        return None
    d = np.load(FRONT_NPZ)
    k = np.argmin(np.abs(d['eps'] - eps))
    if abs(d['eps'][k] - eps) > 1e-6:
        return None
    return (float(d['lpsp'][k]), float(d['deg_keur'][k]),
            float(d['eens_kwh'][k]), float(d['unif3_keur'][k]))


def run_eps(eps):
    tag = f"eps_{eps:g}".replace('.', 'p')
    strat_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(strat_dir, exist_ok=True)

    print("=" * 70)
    print(f" PD detaillee  eps={eps:g}  -> {strat_dir}")
    print("=" * 70, flush=True)

    t0 = time.time()
    dpc.VOLL = float(eps)
    pol = DPPolicy(51, 51, 10, 50, recompute='yearly', verbose=True)
    data = init_and_run_loop(pol)
    sim_s = time.time() - t0

    # --- validation reproduction ---
    lpsp, deg = sens_metrics(data)
    ref = _stored(eps)
    print(f"\n[reproduction eps={eps:g}]  LPSP {lpsp:.4f}%  deg {deg:.3f} kE  "
          f"({sim_s:.0f}s, {pol.n_rebuild} rebuilds)")
    if ref is not None:
        print(f"   stocke (npz)       LPSP {ref[0]:.4f}%  deg {ref[1]:.3f} kE  "
              f"-> dLPSP {lpsp-ref[0]:+.4f}  ddeg {deg-ref[1]:+.3f}")

    # --- figures (everything_combined_v2_2, all_aging_2, etc.) ---
    import matplotlib
    matplotlib.use("Agg")
    run_main_plot(data, strategy_name=strat_dir)
    print(f" figures eps={eps:g} -> {strat_dir}\\Figures\\{data['n']}h\\", flush=True)


def main():
    eps_list = [float(a) for a in sys.argv[1:]] or [3.0]
    for e in eps_list:
        run_eps(e)


if __name__ == "__main__":
    main()
