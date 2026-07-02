import sys
import os
from time import time as timer

# 1. Ajout des chemins : Predictions/ (pour Common, boucle forecast) + dossier courant
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..', 'Prédictions')))
sys.path.insert(0, _here)

# 2. Imports des modules communs (boucle FORKEE : prevision omnisciente 48h)
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
from Common.main_plot import run_main_plot

# 3. Import de la strategie du dossier courant
from get_optimal_action_RB import get_optimal_action_RB
import get_optimal_action_RB as strat


def main():
    print(f"--- Simulation {os.path.basename(_here)} [prevision 48h + bruit backtest] ---")
    start_time = timer()

    global data
    if hasattr(strat, 'set_noise_seed'):
        strat.set_noise_seed(0)
    if hasattr(strat, 'reset'):
        strat.reset()
    data = init_and_run_loop_forecast(get_optimal_action_RB, H_forecast=48)

    run_main_plot(data)

    end_time = timer()
    print(f"--- Simulation terminée en {end_time - start_time:.2f} s ---")


if __name__ == "__main__":
    main()
