"""
batch_optimize_rb2.py
=====================
Optimisation des DEUX puissances de fonctionnement de la regle RB2 :
    - P_fc_set  : puissance delivree par la PEMFC quand le deficit la depasse
    - P_ely_set : puissance absorbee par le PEMWE quand le surplus la depasse
(valeurs codees en dur dans RB2/get_optimal_action_RB.py = 950 W et 9000 W).

Chaque couple (P_fc_set, P_ely_set) -> une simulation -> un point (LPSP %, cout k€).
On balaie une grille (par defaut 5x5 = 25 sims, dans le budget 20-30), sur 15 ans,
en parallele. On identifie le sous-ensemble Pareto NON-DOMINE, puis on designe le
"meilleur" point par une SOMME PONDEREE des objectifs normalises min-max
(poids LPSP/cout reglables ci-dessous ; 50/50 par defaut).

NB : on optimise les setpoints pour le DIMENSIONNEMENT COURANT (valeurs de Init).
     Rien n'est ecrit dans les fichiers : le script affiche/sauvegarde le resultat.

Lancement :  python batch_optimize_rb2.py   (depuis le dossier Vieillissement8)
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol

# ======================= CONFIGURATION =======================
HORIZON_YEARS = 15           # horizon de simulation (le coeur fixe T = SIM['Tend']*10)
GRID_FC   = np.linspace(300.0, 1500.0, 5)     # plage P_fc_set  [W]  (P_fc_max ~ 1578 W)
GRID_ELY  = np.linspace(500.0, 2500.0, 5)   # plage P_ely_set [W]  (P_ely_max ~ 16587 W)
W_LPSP    = 0.5              # poids LPSP   (somme = 1)
W_COST    = 0.5             # poids cout de degradation
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT   = "optim_rb2_setpoints_15y.txt"
OUT_PDF   = "optim_rb2_setpoints_15y.pdf"
# =============================================================


def make_rb2(P_fc_set, P_ely_set):
    """Fabrique une fonction d'action RB2 identique a l'originale mais parametree
    par les deux puissances de fonctionnement."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if P_tot_ref_t > 0:
            if P_tot_ref_t > P_fc_set:
                P_dc_bat_t = P_tot_ref_t - P_fc_set
                P_dc_fc_t  = P_fc_set
                P_dc_ely_t = 0
            else:
                P_dc_bat_t = P_tot_ref_t
                P_dc_fc_t  = 0
                P_dc_ely_t = 0
        if P_tot_ref_t < 0:
            if P_tot_ref_t < -P_ely_set:
                P_dc_bat_t = P_tot_ref_t + P_ely_set
                P_dc_fc_t  = 0
                P_dc_ely_t = -P_ely_set
            else:
                P_dc_bat_t = P_tot_ref_t
                P_dc_fc_t  = 0
                P_dc_ely_t = 0
        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t
        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol
    return get_optimal_action_RB


def _lpsp_cost(data):
    """Reproduit EXACTEMENT le calcul (LPSP %, cout k€) de run_main_plot, sans plotting."""
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    P_planned = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    P_real    = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    tot = p.sum()
    lpsp = (np.clip(p - r, 0, None).sum() / tot * 100) if tot > 0 else 0.0
    cost_keur = get_cost_total(data["alpha_fc"][:-1], data["P_fc"], data["alpha_ely"][:-1],
                               data["P_ely"], data["P_bat"], data["SoC"],
                               I.LOAD, I.BAT, I.FC, I.ELY, data["SoH_bat"][:-1]) / 1000
    return float(lpsp), float(cost_keur)


def evaluate(args):
    """Worker : lance une simulation pour (P_fc_set, P_ely_set) et renvoie le point objectif."""
    P_fc_set, P_ely_set = args
    I.SIM['Tend'] = HORIZON_YEARS / 10.0 * 3600 * 24 * 365   # coeur : T = SIM['Tend']*10
    data = init_and_run_loop(make_rb2(P_fc_set, P_ely_set))
    lpsp, cost = _lpsp_cost(data)
    return (P_fc_set, P_ely_set, lpsp, cost)


def pareto_mask(pts):
    """pts : array (N,2) [LPSP, cost], minimisation. Renvoie le masque des non-domines."""
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for k in range(n):
            if k == i:
                continue
            # k domine i si k <= i sur les deux et < sur au moins un
            if pts[k, 0] <= pts[i, 0] and pts[k, 1] <= pts[i, 1] and \
               (pts[k, 0] < pts[i, 0] or pts[k, 1] < pts[i, 1]):
                keep[i] = False
                break
    return keep


def main():
    combos = [(fc, el) for fc in GRID_FC for el in GRID_ELY]
    print(f"--- Optimisation RB2 : {len(combos)} simulations sur {HORIZON_YEARS} ans "
          f"({N_WORKERS} workers) ---", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(evaluate, combos), 1):
            results.append(res)
            print(f"  [{i:2d}/{len(combos)}] P_fc={res[0]:6.0f}W P_ely={res[1]:7.0f}W "
                  f"-> LPSP={res[2]:6.3f}%  cout={res[3]:8.3f} k€", flush=True)
    print(f"--- {len(combos)} sims en {time.time()-t0:.0f}s ---", flush=True)

    arr = np.array(results)                      # colonnes : Pfc, Pely, LPSP, cost
    obj = arr[:, 2:4]                            # [LPSP, cost]
    mask = pareto_mask(obj)

    # Normalisation min-max sur les points evalues, puis somme ponderee (50/50 def.)
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    objn = (obj - lo) / span
    score = W_LPSP * objn[:, 0] + W_COST * objn[:, 1]
    best = int(np.argmin(score))

    # --- Sauvegarde texte ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("P_fc_set(W);P_ely_set(W);LPSP(%);Cost(kEUR);Pareto;Score\n")
        order = np.argsort(score)
        for k in order:
            f.write(f"{arr[k,0]:.0f};{arr[k,1]:.0f};{arr[k,2]:.4f};{arr[k,3]:.4f};"
                    f"{int(mask[k])};{score[k]:.4f}\n")

    # --- Plot Pareto (convention batch_pareto : LPSP en x, degradation k€ en y) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(obj[~mask, 0], obj[~mask, 1], c="lightgray", s=55, label="domines")
    ax.scatter(obj[mask, 0],  obj[mask, 1],  c="royalblue", s=70, label="front non-domine")
    ax.scatter(obj[best, 0],  obj[best, 1],  c="crimson", s=160, marker="*",
               zorder=5, label="meilleur (50/50)")
    for k in range(len(arr)):
        ax.annotate(f"{arr[k,0]:.0f}/{arr[k,1]:.0f}", (obj[k, 0], obj[k, 1]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points", color="black")
    ax.scatter([0], [0], c="green", marker="x", s=80, label="ideal (0,0)")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Degradation [k€]")
    ax.set_title(f"RB2 — setpoints PEMFC/PEMWE (15 ans)\nannotations = P_fc/P_ely [W]")
    ax.grid(True, ls="--", alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # --- Resume console ---
    print("\n=== FRONT NON-DOMINE ===", flush=True)
    for k in np.where(mask)[0][np.argsort(obj[mask, 0])]:
        print(f"  P_fc={arr[k,0]:6.0f}W  P_ely={arr[k,1]:7.0f}W  |  "
              f"LPSP={arr[k,2]:6.3f}%  cout={arr[k,3]:8.3f} k€", flush=True)
    print("\n=== MEILLEUR POINT (somme ponderee {:.0f}/{:.0f}) ===".format(W_LPSP*100, W_COST*100))
    print(f"  >>> P_fc_set = {arr[best,0]:.0f} W   |   P_ely_set = {arr[best,1]:.0f} W")
    print(f"      LPSP = {arr[best,2]:.3f} %   |   cout degradation = {arr[best,3]:.3f} k€ (15 ans)")
    print(f"  (defaut actuel RB2 : P_fc_set=950 W, P_ely_set=9000 W)")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
