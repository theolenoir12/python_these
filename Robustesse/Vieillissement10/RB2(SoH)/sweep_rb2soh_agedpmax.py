"""
sweep_rb2soh_agedpmax.py
========================
NOUVELLE nature de la regle SoH, inspiree de l'optimum global (DP sequentielle).

Constat DP (dp_aging PD_seq) : les deux composants H2 operent a FRACTION CONSTANTE
de leur capacite VIEILLIE P_*_max_t (et non de la capacite nominale x SoH^gamma) :
    ELY : |P_dc_ely|_p95 / P_ely_max_t = 0.335  (constant sur 25 ans)
    FC  : |P_dc_fc|_p95  / P_fc_max_t  ~ 0.72-0.84
La baisse apparente en fraction du nominal = SEULEMENT le plafond physique qui
retrecit avec l'age, pas un derating volontaire.

=> Regle testee ici (structure inchangee : batterie variable, H2 = offset) :
    P_fc_set  = c_fc  * P_fc_max_t     (capacite VIEILLIE, passee en argument)
    P_ely_set = c_ely * P_ely_max_t
Le SoH entre donc par P_*_max_t (deja calcule dans la boucle), a fraction constante.

Balaie (c_ely) x (c_fc). Point repere DP-inspire : (c_ely=0.335, c_fc=0.72).
Chaque combinaison -> 1 sim 25 ans -> (LPSP %, deg k€) + cout unifie (VoLL=3).

Sortie : sweep_rb2soh_agedpmax.txt + figure. Lancement (depuis RB2(SoH)/) :
    python sweep_rb2soh_agedpmax.py            # sweep + figure
    python sweep_rb2soh_agedpmax.py --smoke     # 2 sims, horizon court
    python sweep_rb2soh_agedpmax.py --replot     # figure depuis le .txt
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from Common.get_lol import get_lol
# VoLL officiel (Analyse_sensibilite)
sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V

# ======================= CONFIGURATION =======================
N_YEARS    = 25
CELY_FRACS = [0.300, 0.335, 0.370]          # fraction de P_ely_max_t (DP ~0.335)
# NB : c_fc eleve (~0.72, niveau DP) fait EXPLOSER la LPSP dans la regle myope
# (vide le reservoir H2 sans anticipation) -> on reste dans la zone utile ~0.44.
CFC_FRACS  = [0.400, 0.440, 0.480, 0.550]   # fraction de P_fc_max_t (vieilli)
_N_AVAIL   = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT    = os.path.join(HERE, "sweep_rb2soh_agedpmax.txt")
OUT_PDF    = os.path.join(HERE, "sweep_rb2soh_agedpmax.pdf")
OUT_PNG    = os.path.join(HERE, "sweep_rb2soh_agedpmax.png")
FIXED_TXT  = os.path.abspath(os.path.join(HERE, "..", "RB2", "sweep_setpoints_rb2.txt"))
DP_POINT   = (0.3057, 57.836)               # PD_seq (LPSP%, deg k€) pour situer
# =============================================================


def make_rule(c_fc, c_ely):
    """Regle RB2(SoH) nouvelle forme : offsets = fraction constante de la capacite
    VIEILLIE P_*_max_t. Plafonds H2 et get_lol identiques a l'original."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = c_fc  * P_fc_max_t
        P_ely_set = c_ely * P_ely_max_t

        dt_h         = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff']  * I.CONV['eta'] * 1000
        P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (I.ELY['eff'] * I.CONV['eta']) * 1000

        if P_tot_ref_t > 0:
            P_fc_avail = min(P_fc_set, P_fc_h2_max)
            if P_tot_ref_t > P_fc_avail:
                P_dc_fc_t  = P_fc_avail
                P_dc_bat_t = P_tot_ref_t - P_fc_avail
            else:
                P_dc_fc_t  = 0
                P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
        if P_tot_ref_t < 0:
            P_ely_avail = min(P_ely_set, P_ely_h2_max)
            if P_tot_ref_t < -P_ely_avail:
                P_dc_ely_t = -P_ely_avail
                P_dc_bat_t = P_tot_ref_t + P_ely_avail
            else:
                P_dc_ely_t = 0
                P_dc_bat_t = P_tot_ref_t
            P_dc_fc_t = 0

        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t
        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol
    return get_optimal_action_RB


def _compute_metrics(data):
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc  = data["alpha_fc"][:-1]
    alpha_ely = data["alpha_ely"][:-1]
    SoH_bat   = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)),
            SoH_bat[~np.isnan(SoH_bat)])
    P_planned = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    P_real    = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    load = np.clip(np.asarray(P_dc_load, dtype=float) / 1000.0, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / load.sum() * 100) if load.sum() > 0 else 0.0
    cost_keur = get_cost_from_ledger(data) / 1000.0
    return float(lpsp), float(cost_keur)


def _eval(args):
    c_fc, c_ely, n_years = args
    data = init_and_run_loop(make_rule(c_fc, c_ely), n_years=n_years)
    lpsp, cost = _compute_metrics(data)
    return (c_fc, c_ely, lpsp, cost, V.total_cost_keur(lpsp, cost))


def run_sweep(smoke=False):
    n_years = 2 if smoke else N_YEARS
    if smoke:
        combos = [(0.440, 0.335, n_years), (0.720, 0.335, n_years)]
    else:
        combos = [(cf, ce, n_years) for cf in CFC_FRACS for ce in CELY_FRACS]
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2(SoH) aged-Pmax : {len(combos)} sims / {n_years} ans ({nw} workers) ---",
          flush=True)
    t0 = time.time(); results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, res in enumerate(ex.map(_eval, combos), 1):
            results.append(res)
            print(f"  [{i:2d}/{len(combos)}] c_fc={res[0]:.3f} c_ely={res[1]:.3f} -> "
                  f"LPSP={res[2]:6.3f}%  deg={res[3]:7.3f}  UNIF={res[4]:7.3f} k€", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)

    results.sort(key=lambda r: r[4])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# RB2(SoH) forme aged-Pmax : P_*_set = c * P_*_max_t (vieilli) - {n_years} ans\n")
        f.write(f"# VoLL={V.VOLL_TIERS}  unifie = deg + {V.cost_lpsp_keur(1.0):.4f}*LPSP\n")
        f.write("rang;c_fc;c_ely;LPSP(%);deg(kEUR);total(kEUR)\n")
        for i, (cf, ce, lp, dg, tot) in enumerate(results, 1):
            f.write(f"{i};{cf:.3f};{ce:.3f};{lp:.4f};{dg:.4f};{tot:.4f}\n")
    best = results[0]
    print(f"\nMIN aged-Pmax : c_fc={best[0]:.3f} c_ely={best[1]:.3f} -> {best[4]:.3f} k€", flush=True)
    print(f"Ecrit : {OUT_TXT}", flush=True)


def _read_fixed_best():
    if not os.path.exists(FIXED_TXT):
        return None
    best = None
    with open(FIXED_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 5 or p[0] == "RB2(SoH)":
                continue
            tot = V.total_cost_keur(float(p[3]), float(p[4]))
            if best is None or tot < best[2]:
                best = (float(p[3]), float(p[4]), tot)
    return best


def plot():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("rang"):
                continue
            p = line.strip().split(";")
            if len(p) != 6:
                continue
            rows.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 17, "axes.titlesize": 14, "legend.fontsize": 10,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    # RB2 fixe (contexte gris)
    if os.path.exists(FIXED_TXT):
        first = True
        with open(FIXED_TXT, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("kind"):
                    continue
                q = line.strip().split(";")
                if len(q) != 5 or q[0] == "RB2(SoH)":
                    continue
                ax.scatter(float(q[3]), float(q[4]), color="0.75", s=35, zorder=1,
                           label="RB2 fixe" if first else None); first = False
    fb = _read_fixed_best()
    if fb:
        ax.scatter(fb[0], fb[1], marker="s", color="darkorange", s=90, zorder=4,
                   label=f"meilleur RB2 fixe = {fb[2]:.1f} k€")
    # points aged-Pmax, colores par c_fc
    cfs = sorted(set(r[0] for r in rows))
    cmap = {c: col for c, col in zip(cfs, plt.cm.plasma(np.linspace(0.1, 0.8, len(cfs))))}
    for cf in cfs:
        pts = sorted([r for r in rows if r[0] == cf], key=lambda r: r[1])
        ax.plot([p[2] for p in pts], [p[3] for p in pts], "-o", color=cmap[cf], ms=6,
                label=f"$c_{{FC}}$={cf:.2f}")
        for p in pts:
            ax.annotate(f"{p[1]:.2f}", (p[2], p[3]), fontsize=7, color=cmap[cf],
                        xytext=(3, 3), textcoords="offset points")
    best = min(rows, key=lambda r: r[4])
    ax.scatter(best[2], best[3], marker="*", color="crimson", s=320, zorder=6,
               edgecolors="white", linewidths=0.8,
               label=f"MIN aged-Pmax ({best[0]:.2f}/{best[1]:.2f}) = {best[4]:.1f} k€")
    ax.scatter(*DP_POINT, marker="P", color="green", s=140, zorder=5,
               edgecolors="white", linewidths=0.6, label="DP séquentielle (réf.)")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="best", framealpha=0.92)
    ax.set_title("Règle aged-Pmax : $P_*=c\\cdot P_{*,max}(t)$ (25 ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure : {OUT_PDF}\n         {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        plot()
    else:
        run_sweep(smoke=args.smoke)
        plot()
