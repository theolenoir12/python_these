"""
sweep_setpoints_rb2soh.py
=========================
Ré-optimisation des setpoints de RB2(SoH) par minimisation du COUT UNIFIE
(degradation + VoLL*LPSP), pendant SoH-module de sweep_setpoints_rb2.py.

Regle RB2(SoH) (cf. get_optimal_action_RB.py) : les 2 setpoints H2 sont module
par le SoH, avec h(SoH=1)=1 (a l'etat neuf on retombe EXACTEMENT sur RB2) :

    P_fc_set  = c_fc  * Pmax_fc(nom)  * SoH_fc  ^ gamma_fc
    P_ely_set = c_ely * Pmax_ely(nom) * SoH_ely ^ gamma_ely

gamma > 0 : setpoint baisse en vieillissant (derating) ; gamma = 0 : constant
(= PAS de SoH -> reference d'attribution) ; gamma < 0 : setpoint monte (durcir
tant que le composant est jeune). NB : gamma ~ 3 revient a suivre la capacite
VIEILLIE Pmax_t (car Pmax_t ~ Pmax_nom * SoH^~2.9), ce que teste par ailleurs
sweep_rb2soh_agedpmax.py avec la regle P = c * Pmax_t.

On balaie la grille (c_fc, gamma_fc, c_ely, gamma_ely), on simule chaque combo
sur N_YEARS, on calcule (LPSP %, deg k€) puis le cout unifie, et on classe.
Le meilleur combo est a reporter dans RB2(SoH)/get_optimal_action_RB.py.

ATTRIBUTION : les combos gamma_fc=gamma_ely=0 sont les versions CONSTANTES (sans
SoH). L'ecart entre la meilleure constante et le meilleur SoH-module = gain PUR,
100% attribuable a l'exploitation du SoH.

NB modele de cout : avec la nouvelle fonction PEMFC (reversible/irreversible,
dependante du courant), moduler la FC par SoH_fc a MAINTENANT un effet -> on
balaie donc aussi gamma_fc (contrairement a l'ancien modele start-stop, ou le FC
etait fige a gamma_fc=0). En V8 l'optimum etait gamma_fc=1, gamma_ely=2.

Sorties (dans RB2(SoH)/) :
  - sweep_setpoints_rb2soh.txt : tous les combos, TRIES par cout unifie croissant
  - sweep_setpoints_rb2soh.pdf/.png : nuage (LPSP, deg), couleur = cout unifie,
    etoile = optimum, pointilles = iso-cout unifie.

Lancement (depuis Vieillissement9/RB2(SoH)/) :
    python sweep_setpoints_rb2soh.py            # sweep complet + figure
    python sweep_setpoints_rb2soh.py --smoke    # 4 sims, horizon court (2 ans)
    python sweep_setpoints_rb2soh.py --replot    # refait la figure depuis le .txt
Regler N_WORKERS via SLURM_CPUS_PER_TASK sur le mesocentre.
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement9/RB2(SoH)
PARENT = os.path.dirname(HERE)                               # .../Vieillissement9
sys.path.insert(0, PARENT)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from Common.get_lol import get_lol
sys.path.insert(0, os.path.abspath(os.path.join(PARENT, "..", "Analyse_sensibilite")))
import voll_common as V                                       # cout unifie (VoLL)

# ======================= CONFIGURATION =======================
N_YEARS   = 25
# Grille des 4 parametres (elargir/raffiner selon le besoin). gamma=0 => constant
# (reference d'attribution) ; inclure 0 dans chaque liste gamma pour l'obtenir.
FC_FRACS   = [0.400]                 # c_fc  : base FC (fraction de Pmax nominal)
GAMMA_FC   = [0.0, 0.5, 1.0, 1.5, 2.0]         # exposant sur SoH_fc
ELY_FRACS  = [0.260]   # c_ely : base ELY
GAMMA_ELY  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]    # exposant sur SoH_ely
_N_AVAIL   = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT    = os.path.join(HERE, "sweep_setpoints_rb2soh.txt")
OUT_PDF    = os.path.join(HERE, "sweep_setpoints_rb2soh.pdf")
OUT_PNG    = os.path.join(HERE, "sweep_setpoints_rb2soh.png")
# =============================================================


def make_rule(c_fc, g_fc, c_ely, g_ely):
    """Action RB2(SoH) : setpoints H2 = base * Pmax_nom * SoH^gamma (batterie = reste)."""
    def act(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
            RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = c_fc  * I.FC['P_fc_max']   * SoH_fc_t  ** g_fc
        P_ely_set = c_ely * I.ELY['P_ely_max'] * SoH_ely_t ** g_ely
        dt_h = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff']  * I.CONV['eta'] * 1000
        P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (I.ELY['eff'] * I.CONV['eta']) * 1000
        P_dc_fc_t = P_dc_ely_t = 0
        if P_tot_ref_t > 0:
            P_fc_avail = min(P_fc_set, P_fc_h2_max)
            if P_tot_ref_t > P_fc_avail:
                P_dc_fc_t = P_fc_avail; P_dc_bat_t = P_tot_ref_t - P_fc_avail
            else:
                P_dc_bat_t = P_tot_ref_t
        elif P_tot_ref_t < 0:
            P_ely_avail = min(P_ely_set, P_ely_h2_max)
            if P_tot_ref_t < -P_ely_avail:
                P_dc_ely_t = -P_ely_avail; P_dc_bat_t = P_tot_ref_t + P_ely_avail
            else:
                P_dc_bat_t = P_tot_ref_t
        else:
            P_dc_bat_t = P_tot_ref_t
        if 'FC' in defaillances and P_tot_ref_t > 0: P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0: P_dc_bat_t = P_tot_ref_t
        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        return get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t, E_h2_init,
                       P_fc_max_t, P_ely_max_t, SoH_bat_t)
    return act


def _metrics(data):
    """(LPSP %, deg k€) comme batch_pareto (interpolation NaN de SoH_bat)."""
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
    deg = get_cost_from_ledger(data) / 1000.0
    return float(lpsp), float(deg)


def _eval(args):
    c_fc, g_fc, c_ely, g_ely, n_years = args
    data = init_and_run_loop(make_rule(c_fc, g_fc, c_ely, g_ely), n_years=n_years)
    lpsp, deg = _metrics(data)
    return (c_fc, g_fc, c_ely, g_ely, lpsp, deg, V.total_cost_keur(lpsp, deg))


def run_sweep(smoke=False):
    ny = 2 if smoke else N_YEARS
    if smoke:
        combos = [(0.44, 0.0, 0.31, 0.0, ny), (0.44, 1.0, 0.31, 2.0, ny),
                  (0.44, 0.0, 0.31, 2.0, ny), (0.44, 2.0, 0.31, 1.0, ny)]
    else:
        combos = [(cf, gf, ce, ge, ny) for cf in FC_FRACS for gf in GAMMA_FC
                  for ce in ELY_FRACS for ge in GAMMA_ELY]
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2(SoH) : {len(combos)} sims / {ny} ans ({nw} workers) ; VoLL={V.VOLL_TIERS} ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(_eval, combos), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(combos)}] c_fc={r[0]:.3f}^{r[1]:+.1f} c_ely={r[2]:.3f}^{r[3]:+.1f}"
                  f" -> LPSP={r[4]:6.3f}%  deg={r[5]:7.3f}  UNIF={r[6]:8.3f} k€", flush=True)
    res.sort(key=lambda r: r[6])                              # tri par cout unifie
    best = res[0]
    # attribution : meilleure CONSTANTE (gamma_fc=gamma_ely=0) vs meilleure SoH-modulee
    cst = [r for r in res if r[1] == 0.0 and r[3] == 0.0]
    soh = [r for r in res if not (r[1] == 0.0 and r[3] == 0.0)]
    best_cst = min(cst, key=lambda r: r[6]) if cst else None
    best_soh = min(soh, key=lambda r: r[6]) if soh else None
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)
    print(f">>> OPTIMUM : c_fc={best[0]:.3f}^{best[1]:+.1f}  c_ely={best[2]:.3f}^{best[3]:+.1f}"
          f"  (LPSP={best[4]:.3f}%  deg={best[5]:.3f} k€  UNIF={best[6]:.3f} k€)", flush=True)
    print(f"    -> dans RB2(SoH)/get_optimal_action_RB.py :"
          f"  P_fc_set={best[0]:.3f}*FC['P_fc_max']*SoH_fc_t**{best[1]:.0f} ;"
          f"  P_ely_set={best[2]:.3f}*ELY['P_ely_max']*SoH_ely_t**{best[3]:.0f}", flush=True)
    if best_cst and best_soh:
        gain = best_cst[6] - best_soh[6]
        print(f">>> GAIN PUR ATTRIBUABLE AU SoH = {gain:+.3f} k€ ({100*gain/best_cst[6]:+.2f}%)"
              f"  [meilleure constante {best_cst[6]:.3f} vs SoH-modulee {best_soh[6]:.3f}]", flush=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Sweep setpoints RB2(SoH) : P=c*Pmax_nom*SoH^gamma - {ny} ans ; VoLL={V.VOLL_TIERS}\n")
        f.write(f"# P_fc_max={I.FC['P_fc_max']:.2f} W  P_ely_max={I.ELY['P_ely_max']:.2f} W\n")
        f.write(f"# OPTIMUM : c_fc={best[0]:.3f}^{best[1]:+.1f} c_ely={best[2]:.3f}^{best[3]:+.1f}"
                f" -> unifie={best[6]:.4f} kEUR\n")
        if best_cst and best_soh:
            f.write(f"# meilleure CONSTANTE : c_fc={best_cst[0]:.3f} c_ely={best_cst[2]:.3f} -> {best_cst[6]:.4f} kEUR\n")
            f.write(f"# meilleure SoH-MODULEE : c_fc={best_soh[0]:.3f}^{best_soh[1]:+.1f} c_ely={best_soh[2]:.3f}^{best_soh[3]:+.1f} -> {best_soh[6]:.4f} kEUR\n")
            f.write(f"# >>> GAIN ATTRIBUABLE AU SoH = {best_cst[6]-best_soh[6]:+.4f} kEUR\n")
        f.write("rang;c_fc;gamma_fc;c_ely;gamma_ely;LPSP(%);deg(kEUR);unifie(kEUR)\n")
        for i, r in enumerate(res, 1):
            f.write(f"{i};{r[0]:.3f};{r[1]:.2f};{r[2]:.3f};{r[3]:.2f};{r[4]:.4f};{r[5]:.4f};{r[6]:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)
    return res


def _read_txt():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("rang"):
                continue
            p = line.strip().split(";")
            if len(p) == 8:
                rows.append(tuple(float(x) for x in p[1:]))   # cfc,gfc,cely,gely,lpsp,deg,unif
    return rows


def plot():
    rows = _read_txt()
    if not rows:
        print("Rien a tracer (txt vide).", flush=True); return
    lpsp = np.array([r[4] for r in rows]); deg = np.array([r[5] for r in rows])
    unif = np.array([r[6] for r in rows])
    best = min(rows, key=lambda r: r[6])
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                         "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 6))
    slope = V.cost_lpsp_keur(1.0)                              # iso-cout : deg = C - slope*LPSP
    xs = np.linspace(lpsp.min(), lpsp.max() + 1e-6, 50)
    for C in np.linspace(unif.min(), unif.max(), 7):
        ax.plot(xs, C - slope * xs, ls=':', color='0.7', lw=0.8, zorder=0)
    sc = ax.scatter(lpsp, deg, c=unif, cmap='viridis_r', s=55, zorder=3)
    for cf, gf, ce, ge, lp, dg, un in rows:
        ax.annotate(f"{ce:.2f}^{ge:.0f}", (lp, dg), fontsize=6, color='0.3',
                    xytext=(3, 3), textcoords="offset points", zorder=3)
    ax.scatter(best[4], best[5], marker='*', s=340, color='crimson',
               edgecolors='white', linewidths=0.8, zorder=5,
               label=f"optimum c_fc={best[0]:.2f}^{best[1]:.0f} / c_ely={best[2]:.2f}^{best[3]:.0f}"
                     f"  ({best[6]:.2f} k€)")
    fig.colorbar(sc, ax=ax, label="coût unifié [k€]")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Setpoints RB2(SoH) : minimisation du coût unifié ({N_YEARS} ans)")
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
