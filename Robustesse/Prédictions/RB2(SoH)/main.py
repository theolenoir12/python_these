import sys
import os
import time
from time import time as timer

# 1. Ajout du chemin pour trouver 'Common'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Imports des modules communs
from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot

# 3. Import de TA stratégie spécifique (dans le dossier actuel)
from get_optimal_action_RB import get_optimal_action_RB

def main():
    print("--- Démarrage de la simulation ---")
    start_time = timer()

    # ÉTAPE 1 : Faire tourner le code
    global data
    data = init_and_run_loop(get_optimal_action_RB)

    # ÉTAPE 2 : Visualisation et résultats
    run_main_plot(data)

    end_time = timer()
    print(f"--- Simulation terminée en {end_time - start_time:.2f} s ---")

if __name__ == "__main__":
    main()