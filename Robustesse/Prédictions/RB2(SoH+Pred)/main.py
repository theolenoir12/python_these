import sys
import os
import time
from time import time as timer

# 1. Ajout du chemin pour trouver 'Common'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Imports des modules communs
#    /!\ Boucle FORKEE : fournit a la strategie la prevision omnisciente 48h.
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
from Common.main_plot import run_main_plot

# 3. Import de TA stratégie spécifique (dans le dossier actuel)
from get_optimal_action_RB import get_optimal_action_RB


def main():
    print("--- Démarrage RB2(SoH+Pred) [base SoH + pre-charge prevision 48h] ---")
    start_time = timer()

    global data
    # ÉTAPE 1 : Faire tourner le code (H_forecast=48 pas = 48h avec Ts=3600)
    data = init_and_run_loop_forecast(get_optimal_action_RB, H_forecast=48)

    # ÉTAPE 2 : Visualisation et résultats
    run_main_plot(data)

    end_time = timer()
    print(f"--- Simulation terminée en {end_time - start_time:.2f} s ---")


if __name__ == "__main__":
    main()
