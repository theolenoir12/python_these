"""
sweep_setpoints_rb2.py
======================
Ré-optimisation des setpoints FIXES de RB2 (fractions de Pmax, SANS SoH) par
minimisation du COUT UNIFIE :

    cout_unifie [kEUR] = degradation [kEUR] + cout_LPSP [kEUR]
    cout_LPSP  = VoLL * energie_non_fournie   (voll_common.py, VoLL=3 par defaut)

On balaie une grille (fc_frac, ely_frac), on simule chaque couple sur N_YEARS,
on calcule (LPSP %, deg k€) puis le cout unifie, et on classe. Le meilleur couple
est celui a reporter dans RB2/get_optimal_action_RB.py (lignes P_fc_set/P_ely_set)
avant de retracer le front de Pareto complet.

Sorties (dans RB2/) :
  - sweep_setpoints_rb2.txt : tous les couples, TRIES par cout unifie croissant
  - sweep_setpoints_rb2.pdf/.png : nuage (LPSP, deg), couleur = cout unifie,
    etoile = optimum, pointilles = iso-cout unifie.

Lancement (depuis Vieillissement9/RB2/) :
    python sweep_setpoints_rb2.py            # sweep complet + figure
    python sweep_setpoints_rb2.py --smoke    # 4 sims, horizon court (2 ans)
    python sweep_setpoints_rb2.py --replot    # refait la figure depuis le .txt
Regler N_WORKERS via SLURM_CPUS_PER_TASK sur le mesocentre.
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement9/RB2
PARENT = os.path.dirname(HERE)                               # .../Vieillissement9
sys.path.insert(0, PARENT)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from Common.reliability_metrics import compute_reliability_metrics
from rb2_policy import make_rb2_policy
sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V                                       # cout unifie (VoLL)

# ======================= CONFIGURATION =======================
N_YEARS   = 25
# Grille de fractions de Pmax a balayer (elargir/raffiner selon le besoin).
FC_FRACS  = np.round(np.arange(0.57, 0.59 + 1e-9, 0.01), 3)   # setpoint FC
ELY_FRACS = np.round(np.arange(0.45, 0.49 + 1e-9, 0.02), 3)   # setpoint ELY
_N_AVAIL  = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb2.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb2.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb2.png")
# =============================================================


make_rb2_frac = make_rb2_policy  # alias conserve pour les anciens scripts

def _metrics(data):
    """(LPSP charge totale %, degradation kEUR)."""
    lpsp = compute_reliability_metrics(data)["lpsp_pct"]
    deg = get_cost_from_ledger(data) / 1000.0
    return float(lpsp), float(deg)


def _eval(args):
    fc, ely, n_years = args
    data = init_and_run_loop(make_rb2_frac(fc, ely), n_years=n_years)
    lpsp, deg = _metrics(data)
    return (fc, ely, lpsp, deg, V.total_cost_keur(lpsp, deg))   # dernier = cout unifie


def run_sweep(smoke=False):
    ny = 2 if smoke else N_YEARS
    if smoke:
        combos = [(0.44, 0.30, ny), (0.44, 0.34, ny), (0.48, 0.30, ny), (0.40, 0.32, ny)]
    else:
        combos = [(fc, el, ny) for fc in FC_FRACS for el in ELY_FRACS]
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2 : {len(combos)} sims / {ny} ans ({nw} workers) ; VoLL={V.VOLL_TIERS} ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(_eval, combos), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(combos)}] fc={r[0]:.3f} ely={r[1]:.3f}"
                  f" -> LPSP={r[2]:6.3f}%  deg={r[3]:7.3f}  UNIF={r[4]:8.3f} k€", flush=True)
    res.sort(key=lambda r: r[4])                                # tri par cout unifie
    best = res[0]
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)
    print(f">>> OPTIMUM : fc_frac={best[0]:.3f}  ely_frac={best[1]:.3f}"
          f"  (LPSP={best[2]:.3f}%  deg={best[3]:.3f} k€  UNIF={best[4]:.3f} k€)", flush=True)
    print(f"    -> dans RB2/get_optimal_action_RB.py :  P_fc_set={best[0]:.3f}*P_fc_max ;"
          f"  P_ely_set={best[1]:.3f}*P_ely_max", flush=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Sweep setpoints RB2 (fixes) - {ny} ans ; VoLL={V.VOLL_TIERS}\n")
        f.write(f"# P_fc_max={I.FC['P_fc_max']:.2f} W  P_ely_max={I.ELY['P_ely_max']:.2f} W\n")
        f.write(f"# OPTIMUM : fc_frac={best[0]:.3f} ely_frac={best[1]:.3f} -> unifie={best[4]:.4f} kEUR\n")
        f.write("rang;fc_frac;ely_frac;LPSP(%);deg(kEUR);unifie(kEUR)\n")
        for i, r in enumerate(res, 1):
            f.write(f"{i};{r[0]:.3f};{r[1]:.3f};{r[2]:.4f};{r[3]:.4f};{r[4]:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)
    return res


def _read_txt():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("rang"):
                continue
            p = line.strip().split(";")
            if len(p) == 6:
                rows.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return rows


def plot():
    rows = _read_txt()
    if not rows:
        print("Rien a tracer (txt vide).", flush=True); return
    lpsp = np.array([r[2] for r in rows]); deg = np.array([r[3] for r in rows])
    unif = np.array([r[4] for r in rows])
    best = min(rows, key=lambda r: r[4])
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                         "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 6))
    # iso-cout unifie : deg = C - slope*LPSP  (slope = cout d'1 point de LPSP)
    slope = V.cost_lpsp_keur(1.0)
    xs = np.linspace(lpsp.min(), lpsp.max() + 1e-6, 50)
    for C in np.linspace(unif.min(), unif.max(), 7):
        ax.plot(xs, C - slope * xs, ls=':', color='0.7', lw=0.8, zorder=0)
    sc = ax.scatter(lpsp, deg, c=unif, cmap='viridis_r', s=55, zorder=3)
    for fc, el, lp, dg, un in rows:
        ax.annotate(f"{fc:.2f}/{el:.2f}", (lp, dg), fontsize=6, color='0.3',
                    xytext=(3, 3), textcoords="offset points", zorder=3)
    ax.scatter(best[2], best[3], marker='*', s=340, color='crimson',
               edgecolors='white', linewidths=0.8, zorder=5,
               label=f"optimum {best[0]:.3f}/{best[1]:.3f} ({best[4]:.2f} k€)")
    fig.colorbar(sc, ax=ax, label="coût unifié [k€]")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='upper right')
    ax.set_title(f"Setpoints RB2 : minimisation du coût unifié ({N_YEARS} ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUT_PNG, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure : {OUT_PDF}\n         {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 sims, horizon court")
    ap.add_argument("--replot", action="store_true", help="refait la figure depuis le .txt")
    args = ap.parse_args()
    if args.replot:
        plot()
    else:
        run_sweep(smoke=args.smoke)
        plot()
