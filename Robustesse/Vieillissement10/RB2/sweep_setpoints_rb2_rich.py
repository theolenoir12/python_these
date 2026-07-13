"""
sweep_setpoints_rb2_rich.py
===========================
Sweep V10 des consignes de base de la VRAIE politique RB2 du dossier
(rb2_policy.make_rb2_policy : consignes economiques + secours de faisabilite aux
bornes de SoC). Contrairement a sweep_setpoints_rb2.py (politique SIMPLE
make_rb2_frac, sans secours SoC), on balaie ici (fc_base, ely_base) A EMERGENCY
FIXE (FC_EMERGENCY, ELY_EMERGENCY = valeurs documentees du dossier), sur une
grille ELARGIE, et on classe par COUT UNIFIE :

    cout_unifie [kEUR] = degradation [kEUR] + VoLL * energie_non_fournie
                       = voll_common.total_cost_keur(LPSP, deg)   (VoLL=3 EUR/kWh)

deg et LPSP sont calcules EXACTEMENT comme batch_pareto / sweep_setpoints_rb2 :
  deg  = get_cost_from_ledger(data)/1000  ( = sum(ledger['total_eur'])/1000 )
  LPSP = 100 * sum(clip(P_planned-P_real,0)) / sum(clip(P_planned,0))
-> empilable avec batch_pareto.

Sorties (dans RB2/, noms DISTINCTS pour ne rien ecraser) :
  - sweep_setpoints_rb2_rich.txt  : tous les couples, tries par cout unifie
  - sweep_setpoints_rb2_rich.pdf/.png : nuage (LPSP, deg), couleur = cout unifie

Lancement (python = anaconda) :
  python sweep_setpoints_rb2_rich.py            # sweep complet + figure
  python sweep_setpoints_rb2_rich.py --time1    # 1 simu 25 ans chronometree, n'ecrit rien
  python sweep_setpoints_rb2_rich.py --smoke    # 4 sims 2 ans -> fichier _SMOKE, n'ecrase rien
  python sweep_setpoints_rb2_rich.py --replot   # refait la figure depuis le .txt
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement10/RB2
PARENT = os.path.dirname(HERE)                               # .../Vieillissement10
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from rb2_policy import make_rb2_policy                       # VRAIE politique RB2 (rich)
sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V                                      # cout unifie (VoLL)

# ======================= CONFIGURATION =======================
N_YEARS   = 25
# Consignes de secours FIXES (= valeurs documentees RB2/get_optimal_action_RB.py).
FC_EMERGENCY  = 0.90
ELY_EMERGENCY = 0.225
# Grille ELARGIE des consignes de base (fractions de Pmax).
FC_BASES  = np.round(np.arange(0.22, 0.44 + 1e-9, 0.02), 3)   # 0.22 .. 0.44
ELY_BASES = np.round(np.arange(0.16, 0.30 + 1e-9, 0.02), 3)   # 0.16 .. 0.30
_N_AVAIL  = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb2_rich.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb2_rich.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb2_rich.png")
# =============================================================


def _metrics(data):
    """(LPSP %, deg kEUR) STRICTEMENT comme batch_pareto / sweep_setpoints_rb2."""
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    Pp = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    Pr = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(Pp, 0, None), np.clip(Pr, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    deg = get_cost_from_ledger(data) / 1000.0
    return float(lpsp), float(deg)


def _eval(args):
    fc, ely, n_years = args
    policy = make_rb2_policy(fc, ely, FC_EMERGENCY, ELY_EMERGENCY)
    data = init_and_run_loop(policy, n_years=n_years)
    lpsp, deg = _metrics(data)
    return (fc, ely, lpsp, deg, V.total_cost_keur(lpsp, deg))


def run_sweep(smoke=False):
    ny = 2 if smoke else N_YEARS
    if smoke:
        combos = [(0.30, 0.22, ny), (0.34, 0.22, ny), (0.28, 0.20, ny), (0.38, 0.24, ny)]
        out_txt = os.path.join(HERE, "sweep_setpoints_rb2_rich_SMOKE.txt")
    else:
        combos = [(float(fc), float(el), ny) for fc in FC_BASES for el in ELY_BASES]
        out_txt = OUT_TXT
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2 RICH : {len(combos)} sims / {ny} ans ({nw} workers) ; "
          f"emergency=({FC_EMERGENCY},{ELY_EMERGENCY}) ; VoLL={V.VOLL_TIERS} ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(_eval, combos), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(combos)}] fc={r[0]:.3f} ely={r[1]:.3f}"
                  f" -> LPSP={r[2]:6.3f}%  deg={r[3]:7.3f}  UNIF={r[4]:8.3f} k€", flush=True)
    res.sort(key=lambda r: r[4])
    best = res[0]
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)
    print(f">>> OPTIMUM : fc_base={best[0]:.3f}  ely_base={best[1]:.3f}"
          f"  (LPSP={best[2]:.3f}%  deg={best[3]:.3f} k€  UNIF={best[4]:.3f} k€)", flush=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# Sweep setpoints RB2 RICH (base, secours SoC) - {ny} ans ; VoLL={V.VOLL_TIERS}\n")
        f.write(f"# emergency=({FC_EMERGENCY},{ELY_EMERGENCY}) ; "
                f"P_fc_max={I.FC['P_fc_max']:.2f} W  P_ely_max={I.ELY['P_ely_max']:.2f} W\n")
        f.write(f"# OPTIMUM : fc_base={best[0]:.3f} ely_base={best[1]:.3f} -> unifie={best[4]:.4f} kEUR\n")
        f.write("rang;fc_base;ely_base;LPSP(%);deg(kEUR);unifie(kEUR)\n")
        for i, r in enumerate(res, 1):
            f.write(f"{i};{r[0]:.3f};{r[1]:.3f};{r[2]:.4f};{r[3]:.4f};{r[4]:.4f}\n")
    print(f"Ecrit : {out_txt}", flush=True)
    return res


def time1():
    """Une seule simu 25 ans (couple de reference), chronometree, n'ecrit rien."""
    fc, ely = 0.31, 0.22
    print(f"--- TIME1 : 1 simu {N_YEARS} ans, fc_base={fc} ely_base={ely} "
          f"emergency=({FC_EMERGENCY},{ELY_EMERGENCY}) ---", flush=True)
    t0 = time.time()
    r = _eval((fc, ely, N_YEARS))
    dt = time.time() - t0
    print(f"    LPSP={r[2]:.4f}%  deg={r[3]:.4f} k€  UNIF={r[4]:.4f} k€", flush=True)
    print(f"    1 simu 25 ans = {dt:.1f}s", flush=True)
    return dt


def _read_txt(path=None):
    path = path or OUT_TXT
    rows = []
    with open(path, encoding="utf-8") as f:
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
    ax.set_title(f"RB2 rich (secours SoC) : minimisation du coût unifié ({N_YEARS} ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUT_PNG, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure : {OUT_PDF}\n         {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 sims 2 ans -> fichier _SMOKE")
    ap.add_argument("--time1", action="store_true", help="1 simu 25 ans chronometree, n'ecrit rien")
    ap.add_argument("--replot", action="store_true", help="refait la figure depuis le .txt")
    args = ap.parse_args()
    if args.time1:
        time1()
    elif args.replot:
        plot()
    else:
        run_sweep(smoke=args.smoke)
        plot()
