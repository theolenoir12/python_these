"""
sweep_soh_attribution.py
========================
BALAYAGE EXHAUSTIF de l'augmentation ATTRIBUABLE de RB2 par le SoH.

Cadre (contrainte Théo) : RB2 n'a que 2 organes de commande (P_fc_set, P_ely_set,
la batterie prend le reste). La SEULE augmentation admissible sans ajouter de
composante = rendre ces 2 setpoints fonction du SoH, avec h(SoH=1)=1 (test-nul :
a l'etat neuf on retombe EXACTEMENT sur RB2). On balaie donc :

    P_fc_set  = c_fc  * Pmax_fc  * SoH_fc  ^ gamma_fc
    P_ely_set = c_ely * Pmax_ely * SoH_ely ^ gamma_ely     (Pmax NOMINAL)

gamma>0 : setpoint baisse en vieillissant ; gamma=0 : constant (= PAS de SoH, la
reference "sans augmentation") ; gamma<0 : setpoint MONTE en vieillissant (test :
durcir l'ELY tant qu'il est jeune pour se constituer une reserve H2) ; gamma~3 :
equivaut a suivre la capacite VIEILLIE Pmax_t (car Pmax_t ~ Pmax_nom*SoH^2.9).

ATTRIBUTION propre : pour chaque base, la ligne gamma_ely=0 est la version
CONSTANTE (sans SoH). Le gain de la meilleure version SoH-modulee sur la meilleure
constante = valeur PURE, 100% attribuable au SoH, du levier.

Sortie : classement par cout unifie (VoLL=3) + figure unifie vs gamma_ely par base.
Lancement (depuis RB2(SoH)/) :
    python sweep_soh_attribution.py           # sweep + figure
    python sweep_soh_attribution.py --smoke    # 2 sims horizon court
    python sweep_soh_attribution.py --replot    # figure depuis .txt
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
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol
sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V

# ======================= CONFIGURATION =======================
N_YEARS   = 25
# --- branche ELY (le levier) : base x exposant, FC constant a 0.44 ---
CELY   = [0.300, 0.310, 0.320, 0.335]
GELY   = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
CFC0, GFC0 = 0.440, 0.0
# --- sonde FC<-SoH_fc : au meilleur ELY presume, on module aussi le FC ---
FC_PROBE = [(0.440, 1.0, 0.310, 2.0), (0.440, 2.0, 0.310, 2.0), (0.440, -1.0, 0.310, 2.0)]
# --- reference RB2 simple (base 0.45/0.33, aucun SoH) ---
PLAIN_RB2 = (0.450, 0.0, 0.330, 0.0)
_N_AVAIL  = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT   = os.path.join(HERE, "sweep_soh_attribution.txt")
OUT_PDF   = os.path.join(HERE, "sweep_soh_attribution.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_soh_attribution.png")
# =============================================================


def make_rule(c_fc, g_fc, c_ely, g_ely):
    def act(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
            RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = c_fc  * I.FC['P_fc_max']   * SoH_fc_t  ** g_fc
        P_ely_set = c_ely * I.ELY['P_ely_max'] * SoH_ely_t ** g_ely
        dt_h = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff']  * I.CONV['eta'] * 1000
        P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (I.ELY['eff'] * I.CONV['eta']) * 1000
        if P_tot_ref_t > 0:
            P_fc_avail = min(P_fc_set, P_fc_h2_max)
            if P_tot_ref_t > P_fc_avail:
                P_dc_fc_t = P_fc_avail; P_dc_bat_t = P_tot_ref_t - P_fc_avail
            else:
                P_dc_fc_t = 0; P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
        if P_tot_ref_t < 0:
            P_ely_avail = min(P_ely_set, P_ely_h2_max)
            if P_tot_ref_t < -P_ely_avail:
                P_dc_ely_t = -P_ely_avail; P_dc_bat_t = P_tot_ref_t + P_ely_avail
            else:
                P_dc_ely_t = 0; P_dc_bat_t = P_tot_ref_t
            P_dc_fc_t = 0
        if 'FC' in defaillances and P_tot_ref_t > 0: P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0: P_dc_bat_t = P_tot_ref_t
        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        return get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t, E_h2_init,
                       P_fc_max_t, P_ely_max_t, SoH_bat_t)
    return act


def _metrics(data):
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    SoC = data["SoC"]; af = data["alpha_fc"][:-1]; ae = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1: SoH_bat[k-1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)), SoH_bat[~np.isnan(SoH_bat)])
    Pp = np.array([(a-b)/1000 for a, b in zip(P_dc_load, P_dc_pv)])
    Pr = np.array([(a-b)*(1-c)/1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(Pp, 0, None), np.clip(Pr, 0, None)
    lpsp = (np.clip(p-r, 0, None).sum()/p.sum()*100) if p.sum() > 0 else 0.0
    deg = get_cost_total(af, P_fc, ae, P_ely, P_bat, SoC, I.LOAD, I.BAT, I.FC, I.ELY, SoH_bat)/1000
    return float(lpsp), float(deg)


def _eval(args):
    c_fc, g_fc, c_ely, g_ely, n_years = args
    d = init_and_run_loop(make_rule(c_fc, g_fc, c_ely, g_ely), n_years=n_years)
    lp, dg = _metrics(d)
    return (c_fc, g_fc, c_ely, g_ely, lp, dg, V.total_cost_keur(lp, dg))


def run_sweep(smoke=False):
    ny = 2 if smoke else N_YEARS
    if smoke:
        combos = [(CFC0, GFC0, 0.310, 0.0, ny), (CFC0, GFC0, 0.310, 2.0, ny)]
    else:
        combos = [(CFC0, GFC0, ce, ge, ny) for ce in CELY for ge in GELY]
        combos += [(cf, gf, ce, ge, ny) for (cf, gf, ce, ge) in FC_PROBE]
        combos += [(*PLAIN_RB2, ny)]
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep SoH-attribution : {len(combos)} sims / {ny} ans ({nw} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(_eval, combos), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(combos)}] c_fc={r[0]:.3f}^{r[1]:+.1f} c_ely={r[2]:.3f}^{r[3]:+.1f}"
                  f" -> LPSP={r[4]:6.3f}%  deg={r[5]:6.2f}  UNIF={r[6]:7.3f}", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)
    res.sort(key=lambda r: r[6])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# SoH-attribution : P_*=c*Pmax_nom*SoH^gamma - {ny} ans ; VoLL={V.VOLL_TIERS}\n")
        f.write("rang;c_fc;gamma_fc;c_ely;gamma_ely;LPSP(%);deg(kEUR);total(kEUR)\n")
        for i, r in enumerate(res, 1):
            f.write(f"{i};{r[0]:.3f};{r[1]:.2f};{r[2]:.3f};{r[3]:.2f};{r[4]:.4f};{r[5]:.4f};{r[6]:.4f}\n")
    # attribution : meilleure constante (tous gamma=0) vs meilleure SoH-modulee
    cst = [r for r in res if r[1] == 0.0 and r[3] == 0.0]
    soh = [r for r in res if not (r[1] == 0.0 and r[3] == 0.0)]
    best_cst = min(cst, key=lambda r: r[6]); best_soh = min(soh, key=lambda r: r[6])
    with open(OUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"\n# meilleure CONSTANTE (sans SoH) : c_fc={best_cst[0]:.3f} c_ely={best_cst[2]:.3f}"
                f" -> {best_cst[6]:.4f} kEUR\n")
        f.write(f"# meilleure SoH-MODULEE : c_fc={best_soh[0]:.3f}^{best_soh[1]:+.1f}"
                f" c_ely={best_soh[2]:.3f}^{best_soh[3]:+.1f} -> {best_soh[6]:.4f} kEUR\n")
        f.write(f"# >>> GAIN PUR ATTRIBUABLE AU SoH = {best_cst[6]-best_soh[6]:+.4f} kEUR "
                f"({100*(best_cst[6]-best_soh[6])/best_cst[6]:+.2f}%)\n")
    print(f"\nMeilleure CONSTANTE   : {best_cst[6]:.3f} kEUR (c_ely={best_cst[2]:.3f})")
    print(f"Meilleure SoH-modulee : {best_soh[6]:.3f} kEUR "
          f"(c_ely={best_soh[2]:.3f}^{best_soh[3]:+.1f}, c_fc^{best_soh[1]:+.1f})")
    print(f">>> GAIN PUR ATTRIBUABLE AU SoH = {best_cst[6]-best_soh[6]:+.3f} kEUR "
          f"({100*(best_cst[6]-best_soh[6])/best_cst[6]:+.2f}%)")
    print(f"Ecrit : {OUT_TXT}", flush=True)


def plot():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("rang"):
                continue
            p = line.strip().split(";")
            if len(p) != 8: continue
            rows.append(tuple(float(x) for x in p[1:]))   # cfc,gfc,cely,gely,lpsp,deg,tot
    # on trace la branche principale (c_fc=0.44, gamma_fc=0)
    main = [r for r in rows if abs(r[0]-CFC0) < 1e-9 and r[1] == 0.0]
    plain = min((r for r in rows if abs(r[0]-0.450) < 1e-9), default=None, key=lambda r: r[6]) \
        if any(abs(r[0]-0.450) < 1e-9 for r in rows) else None
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 17, "axes.titlesize": 14, "legend.fontsize": 10,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4, 6))
    bases = sorted(set(r[2] for r in main))
    cmap = {b: c for b, c in zip(bases, plt.cm.viridis(np.linspace(0.1, 0.85, len(bases))))}
    for b in bases:
        pts = sorted([r for r in main if r[2] == b], key=lambda r: r[3])
        ax.plot([p[3] for p in pts], [p[6] for p in pts], "-o", color=cmap[b], ms=6,
                label=f"$c_{{ELY}}$={b:.3f}")
    # gamma=0 = ligne "sans SoH"
    ax.axvline(0.0, color="0.6", ls=":", lw=1.4)
    ax.text(0.05, ax.get_ylim()[1], "  γ=0 : constant (sans SoH)", color="0.4",
            fontsize=9, va="top")
    if plain:
        ax.axhline(plain[6], color="0.3", ls="--", lw=1.6,
                   label=f"RB2 simple (0.45/0.33) = {plain[6]:.1f} k€")
    best = min(main, key=lambda r: r[6])
    ax.scatter([best[3]], [best[6]], marker="*", color="crimson", s=300, zorder=6,
               edgecolors="white", linewidths=0.8,
               label=f"meilleure SoH-modulée = {best[6]:.1f} k€")
    ax.set_xlabel(r"exposant $\gamma_{ELY}$")
    ax.set_ylabel("Coût unifié [k€]")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="best", framealpha=0.92)
    ax.set_title("Augmentation attribuable de RB2 par le SoH (exhaustif, 25 ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure : {OUT_PDF}\n         {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    if a.replot: plot()
    else:
        run_sweep(smoke=a.smoke); plot()
