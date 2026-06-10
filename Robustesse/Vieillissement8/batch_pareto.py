import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from time import time as timer
  
    
# --- CONFIGURATION ---
# Liste des configurations : (Nom du dossier, Label pour le plot)
# Assure-toi que le nom du fichier de stratégie dans ces dossiers est toujours le même
# ou modifie la logique d'import plus bas.
scenarios = [
    ("0-100", "0-100"),
    ("25-75", "25-75"),
    ("50-50", "50-50"),
    ("75-25", "75-25"),
    ("100-0", "100-0"),
    ("RB2",   "RB2"), 
    ("RB2(SoH)",   "RB2(SoH)"), 
    ("RB1",   "RB1"),
    ("SoC1",   "SoC1"), 
    ("SoC06",   "SoC06")
    # ("RB2(RUL)",  "RB2(RUL)")
]

STRATEGY_FILENAME = "get_optimal_action_RB" # Le nom du fichier .py SANS .py
STRATEGY_FUNC_NAME = "get_optimal_action_RB" # Le nom de la fonction dedans

# --- IMPORTS COMMON ---
# On ajoute le chemin courant pour trouver Common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot

def run_batch():
    results = []
    labels = []

    print(f"--- Démarrage du Batch sur {len(scenarios)} scénarios ---")

    for folder_name, label in scenarios:
        print(f"\nTraitement du scénario : {label} (Dossier: {folder_name})")
        
        # 1. Construction du chemin vers le dossier spécifique
        folder_path = os.path.abspath(folder_name)
        
        if not os.path.exists(folder_path):
            print(f"ERREUR: Le dossier {folder_path} n'existe pas. Ignoré.")
            continue

        # 2. Importation dynamique de la stratégie
        # C'est la partie "magique" : on force Python à charger le fichier de CE dossier
        if folder_path not in sys.path:
            sys.path.insert(0, folder_path)
        
        try:
            # On importe le module
            module = importlib.import_module(STRATEGY_FILENAME)
            # IMPORTANT : On force le rechargement car sinon Python garde la stratégie du dossier précédent en mémoire
            importlib.reload(module) 
            
            # On récupère la fonction
            get_action_func = getattr(module, STRATEGY_FUNC_NAME)
            
        except ImportError as e:
            print(f"Erreur d'import dans {folder_name}: {e}")
            sys.path.pop(0)
            continue
            
        # 3. Exécution de la simulation (comme dans main.py)
        data = init_and_run_loop(get_action_func)
        
        # 4. Récupération des métriques (LPSP %, Cost k€)
        # Note: on peut désactiver l'affichage des plots individuels si on veut aller vite
        # en modifiant run_main_plot pour accepter un argument 'silent=True' par exemple
        lpsp, cost = run_main_plot(data,strategy_name=folder_name) 
        
        results.append([lpsp, cost])
        labels.append(label)
        
        # Nettoyage du path
        sys.path.pop(0)

    # Convertir en numpy array pour le plot
    points = np.array(results)
    
    return points, labels

# --- FONCTION DE PLOT (Ta version adaptée) ---
def plot_pareto(points, labels_list):
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Affichage des points calculés
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], color='royalblue', s=60, alpha=0.8)

        # Ajout des labels
        for i, label in enumerate(labels_list):
            x, y = points[i, 0], points[i, 1]
            
            # # Tes conditions spécifiques de placement
            if 'RB2(SoH)' in label: # "in" permet d'être plus souple
                ax.text(x + 0.05, y - 0.8, label, fontsize=14, color='black', verticalalignment='top')
            # elif label == 'RB2':
            #     ax.text(x - 1.5, y + 2.5, label, fontsize=14, color='black', verticalalignment='top')  
            # elif 'SoC06' in label:
            #     ax.text(x - 3, y + 2.5, label, fontsize=14, color='black', verticalalignment='top')  
            # elif 'SoH' in label:
            #     ax.text(x + 2.5, y - 0.8, label, fontsize=14, color='black', va='top', ha='center')  
            else:
            #     # Placement standard
                ax.text(x + 0.5, y + 0.5, label, fontsize=14, color='black')

    # Style de l'axe
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Degradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    

    plt.tight_layout()
    plt.savefig("pareto_ems_batch_15y.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    points, labels = run_batch()
    print("\n--- Résultats Finaux ---")
    print("Labels:", labels)
    print("Points (LPSP, Cost):", points)
    
    points = np.vstack([points, [0, 0]])
    labels.append("Ideal")
    
    plot_pareto(points, labels)
    
# Sauvegarde des résultats
results_file = "batch_results_summary_15y.txt"
with open(results_file, "w", encoding="utf-8") as f:
    f.write("Label;LPSP(%);Cost(kEUR)\n") # En-tête
    for label, (lpsp, cost) in zip(labels, points):
        f.write(f"{label};{lpsp:.4f};{cost:.4f}\n")

print(f"✅ Résultats sauvegardés dans : {results_file}")