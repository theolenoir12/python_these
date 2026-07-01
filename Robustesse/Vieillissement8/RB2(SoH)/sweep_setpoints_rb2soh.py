"""
sweep_setpoints_rb2soh.py
=========================
Pendant de sweep_setpoints_rb2.py, mais pour RB2(SoH) : on balaie les parametres
de la REGLE polynomiale en fonction du vieillissement, pour mesurer l'AMPLITUDE
des gains reellement apportes par l'exploitation du SoH.

Regle RB2(SoH) (dans RB2(SoH)/get_optimal_action_RB.py) :
    P_fc_set  = f_FC  * Pmax_fc  * SoH_fc  ** gamma_FC     (gamma_FC = 0 : FC non module)
    P_ely_set = f_ELY * Pmax_ely * SoH_ely ** gamma_ELY    (gamma_ELY = 0.5 actuel)

On balaie (f_ELY base) x (gamma_ELY). Cas particuliers reperes :
    - gamma_ELY = 0            -> setpoint FIXE (test-nul : l'info SoH n'agit plus)
    - (f_ELY=0.320, gamma=0.5) -> RB2(SoH) NOMINAL actuel
Le FC reste fixe (f_FC=0.440, gamma_FC=0) : la degradation FC est pilotee par le
start-stop, insensible au niveau de puissance -> moduler par SoH_fc n'aide pas.
Editable en tete si on veut tester malgre tout.

Chaque combinaison -> 1 simulation 25 ans -> 1 point (LPSP %, cout deg k€).
Sortie : sweep_setpoints_rb2soh.txt (colonnes avec les gammas) + figure de base.
Le classement par cout unifie (VoLL) et la comparaison au meilleur RB2 fixe se
font dans rank_rb2soh_unified.py (post-traitement, apres rapatriement).

Lancement (depuis Vieillissement8/RB2/) :
    python sweep_setpoints_rb2soh.py            # sweep complet + figure
    python sweep_setpoints_rb2soh.py --smoke    # smoke-test (2 sims, horizon court)
    python sweep_setpoints_rb2soh.py --replot    # refait la figure depuis le .txt
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement8/RB2
PARENT = os.path.dirname(HERE)                               # .../Vieillissement8
sys.path.insert(0, PARENT)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol

# ======================= CONFIGURATION =======================
N_YEARS   = 25
FC_FRAC   = 0.440       # base FC (fixe ; FC non module par le SoH)
GAMMA_FC  = 0.0         # exposant SoH_fc (0 = pas de modulation FC)
ELY_FRACS = [0.300, 0.310, 0.320, 0.340]   # bases f_ELY a tester
GAMMA_ELYS = [0.0, 0.5, 1.0, 2.0]          # exposants gamma_ELY (0 = setpoint fixe)
_N_AVAIL  = int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1))
_N_AVAIL  = max(1, _N_AVAIL)
OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb2soh.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb2soh.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb2soh.png")
# =============================================================


def make_rb2soh(fc_frac, gamma_fc, ely_frac, gamma_ely):
    """Action RB2(SoH) IDENTIQUE a l'originale (plafonds H2 inclus), parametree
    par la base et l'exposant SoH de chaque convertisseur."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = fc_frac  * I.FC['P_fc_max']   * SoH_fc_t  ** gamma_fc
        P_ely_set = ely_frac * I.ELY['P_ely_max'] * SoH_ely_t ** gamma_ely

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
    """(LPSP %, cout k€) EXACTEMENT comme batch_pareto._compute_metrics."""
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
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    cost_keur = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
                               I.LOAD, I.BAT, I.FC, I.ELY, SoH_bat) / 1000
    return float(lpsp), float(cost_keur)


def _eval(args):
    fc, gfc, el, gel, n_years = args
    data = init_and_run_loop(make_rb2soh(fc, gfc, el, gel), n_years=n_years)
    lpsp, cost = _compute_metrics(data)
    return (fc, gfc, el, gel, lpsp, cost)


def run_sweep(smoke=False):
    n_years = 2 if smoke else N_YEARS
    if smoke:
        combos = [(FC_FRAC, GAMMA_FC, 0.320, 0.5, n_years),
                  (FC_FRAC, GAMMA_FC, 0.320, 0.0, n_years)]
    else:
        combos = [(FC_FRAC, GAMMA_FC, el, gel, n_years)
                  for el in ELY_FRACS for gel in GAMMA_ELYS]
    n_workers = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2(SoH) : {len(combos)} sims sur {n_years} ans ({n_workers} workers) ---",
          flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_eval, combos), 1):
            results.append(res)
            print(f"  [{i:2d}/{len(combos)}] fc={res[0]:.3f}^{res[1]:.1f} "
                  f"ely={res[2]:.3f}^{res[3]:.2f} -> LPSP={res[4]:6.3f}%  "
                  f"cout={res[5]:8.3f} k€", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Sweep regle RB2(SoH) : f_ELY base x gamma_ELY - {n_years} ans\n")
        f.write(f"# f_FC={FC_FRAC} gamma_FC={GAMMA_FC} fixes ; "
                f"P_ely_set = f_ELY*Pmax*SoH_ely^gamma_ELY\n")
        f.write(f"# P_fc_max={I.FC['P_fc_max']:.2f} W  P_ely_max={I.ELY['P_ely_max']:.2f} W\n")
        f.write("kind;fc_frac;gamma_fc;ely_frac;gamma_ely;LPSP(%);Cost(kEUR)\n")
        for fc, gfc, el, gel, lpsp, cost in results:
            if abs(el-0.320) < 1e-9 and abs(gel-0.5) < 1e-9:
                tag = "RB2(SoH)_nominal"
            elif abs(gel-0.0) < 1e-9:
                tag = "fixed(gamma0)"        # setpoint fixe : test-nul
            else:
                tag = "RB2(SoH)_sweep"
            f.write(f"{tag};{fc:.3f};{gfc:.2f};{el:.3f};{gel:.3f};{lpsp:.4f};{cost:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)


def plot():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 7:
                continue
            rows.append((p[0], float(p[3]), float(p[4]), float(p[5]), float(p[6])))
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 18, "axes.titlesize": 15, "legend.fontsize": 11,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- overlay RB2 FIXE (nuage gris) + RB2 nominal, depuis ../RB2/ ---
    fixed_txt = os.path.abspath(os.path.join(HERE, "..", "RB2", "sweep_setpoints_rb2.txt"))
    if os.path.exists(fixed_txt):
        first = True
        with open(fixed_txt, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("kind"):
                    continue
                q = line.strip().split(";")
                if len(q) != 5 or q[0] == "RB2(SoH)":
                    continue
                lp, dg = float(q[3]), float(q[4])
                ax.scatter(lp, dg, color="0.72", s=40, zorder=1,
                           label="RB2 fixe (balayage)" if first else None)
                first = False
                if q[0] == "RB2_nominal":
                    ax.scatter(lp, dg, marker="s", color="0.15", s=70, zorder=4,
                               label="RB2 nominal nu (0.450/0.330)")
                    ax.annotate("RB2", (lp, dg), fontsize=11, color="0.15", weight="bold",
                                path_effects=LABEL_STROKE, xytext=(5, 4),
                                textcoords="offset points", zorder=4)

    bases = sorted(set(r[1] for r in rows))
    cmap = {b: c for b, c in zip(bases, plt.cm.viridis(np.linspace(0.1, 0.8, len(bases))))}
    for b in bases:
        pts = sorted([r for r in rows if r[1] == b], key=lambda r: r[2])
        xs = [p[3] for p in pts]; ys = [p[4] for p in pts]
        ax.plot(xs, ys, '-o', color=cmap[b], alpha=0.8, ms=5,
                label=f"$f_{{ELY}}$={b:.3f}")
        for p in pts:
            ax.annotate(f"$\\gamma$={p[2]:.1f}", (p[3], p[4]), fontsize=7,
                        color=cmap[b], xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.92, fontsize=10)
    ax.set_title("Balayage règle RB2(SoH) : base $f_{ELY}$ × exposant $\\gamma_{ELY}$ (25 ans)")
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
