import sys
import os
from time import time as timer

# 1. Chemins : Predictions/ (Common, boucle forecast) + dossier courant
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..', 'Prédictions')))
sys.path.insert(0, _here)

from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
from Common.main_plot import run_main_plot
import Common.get_lol as gl

from get_optimal_action_RB import get_optimal_action_RB
import get_optimal_action_RB as strat


def main():
    print("--- Simulation RB2(Ultime) [gammas SoH + plafond SoC vieilli + pre-charge ±1σ] ---")
    start_time = timer()

    # Plafond SoC vieillissant : applique par Common/get_lol (run standalone)
    gl.SOC_MAX_AGED_GAIN = strat.SOC_WIN_GAIN

    global data
    strat.set_noise_seed(0)
    strat.reset()
    data = init_and_run_loop_forecast(get_optimal_action_RB, H_forecast=48)

    run_main_plot(data)

    end_time = timer()
    print(f"--- Simulation terminée en {end_time - start_time:.2f} s ---")


if __name__ == "__main__":
    main()
