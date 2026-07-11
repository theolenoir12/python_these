import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from time import time as timer
from concurrent.futures import ProcessPoolExecutor


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
    ("RB1_costopt_v8_020_035", "RB1-costopt-V8"),
    ("SoC1",   "SoC1"),
    ("SoC06",   "SoC06")
    # ("RB2(RUL)",  "RB2(RUL)")
]

STRATEGY_FILENAME = "get_optimal_action_RB" # Le nom du fichier .py SANS .py
STRATEGY_FUNC_NAME = "get_optimal_action_RB" # Le nom de la fonction dedans

# Nombre de workers (1 process par scénario, borné au nb de cœurs - 1)
N_WORKERS = max(1, min(len(scenarios), (os.cpu_count() or 2) - 1))

# --- IMPORTS COMMON ---
# On ajoute le chemin courant pour trouver Common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Common.main_init_and_loop import init_and_run_loop


def _compute_metrics(data):
    """Reproduit EXACTEMENT le calcul (LPSP %, coût k€) de run_main_plot, sans aucun plot.
    -> Extrait des lignes 47-49 / 330-331 / 436-477 de Common/main_plot.py."""
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]

    # LPSP (lignes 330-331 + 436-439)
    P_planned = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    P_real    = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0

    ledger = data.get("degradation_ledger")
    if ledger is None:
        raise RuntimeError("batch V9_4 corrige sans degradation_ledger")
    cost_keur = sum(ledger["total_eur"].values()) / 1000.0
    return float(lpsp), float(cost_keur)


def run_one(args):
    """Worker : (folder_name, label) -> (label, lpsp, cost). Charge dynamiquement la
    stratégie du dossier, lance la simulation, calcule les métriques. Aucun plot."""
    folder_name, label = args
    folder_path = os.path.abspath(folder_name)
    if not os.path.exists(folder_path):
        print(f"ERREUR: Le dossier {folder_path} n'existe pas. Ignoré.", flush=True)
        return label, np.nan, np.nan

    # On place CE dossier en tête de sys.path et on purge un éventuel module homonyme
    # d'un scénario précédent (worker réutilisé) -> on importe bien le bon fichier.
    if folder_path in sys.path:
        sys.path.remove(folder_path)
    sys.path.insert(0, folder_path)
    sys.modules.pop(STRATEGY_FILENAME, None)
    try:
        module = importlib.import_module(STRATEGY_FILENAME)
        get_action_func = getattr(module, STRATEGY_FUNC_NAME)
    except ImportError as e:
        print(f"Erreur d'import dans {folder_name}: {e}", flush=True)
        return label, np.nan, np.nan

    t0 = timer()
    data = init_and_run_loop(get_action_func, replacement_accounting="corrected")
    lpsp, cost = _compute_metrics(data)
    print(f"  [OK] {label:10s} -> LPSP {lpsp:7.4f}%  cost {cost:8.3f} k€  ({timer()-t0:.0f}s)", flush=True)
    return label, lpsp, cost


def run_batch():
    print(f"--- Démarrage du Batch sur {len(scenarios)} scénarios ({N_WORKERS} workers) ---", flush=True)
    t0 = timer()
    results = []
    labels = []
    # ex.map conserve l'ordre des scénarios -> alignement points/labels garanti
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for label, lpsp, cost in ex.map(run_one, scenarios):
            if np.isnan(lpsp):
                continue
            results.append([lpsp, cost])
            labels.append(label)
    print(f"--- Batch terminé en {timer()-t0:.0f}s ---", flush=True)

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
    results_file = "batch_results_summary_25y.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Label;LPSP(%);Cost(kEUR)\n") # En-tête
        for label, (lpsp, cost) in zip(labels, points):
            f.write(f"{label};{lpsp:.4f};{cost:.4f}\n")

    print(f"✅ Résultats sauvegardés dans : {results_file}")
