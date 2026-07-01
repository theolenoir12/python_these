"""
sweep_rul_attribution.py
========================
Point RB2(RUL) HONNETE (cost-min, meme socle que RB2/RB2(SoH) a base 0.310).
Le sweep_rul.py existant figeait la base a 0.320 -> comparaison biaisee. Ici on
balaie AUSSI la base c_ely (et c_fc), avec la modulation RUL de RB2(RUL) :
    f_ely     = min(RUL_ely / RUL_REF, 1) ^ EXP_ELY
    P_fc_set  = c_fc  * Pmax_fc
    P_ely_set = c_ely * Pmax_ely * f_ely
EXP_ELY=0 -> RB2 constant (test-nul). FC non modulee (design RB2(RUL)).
Metrique = cout unifie (deg + VoLL*EENS, VoLL=3, voll_common) IDENTIQUE au reste.

Sortie : sweep_rul_attribution.txt (classement) ; affiche le MIN.
Grille par defaut petite (9 sims) pour tourner en local ; l'elargir pour le meso.
Lancement (depuis Predictions/) : python sweep_rul_attribution.py
"""
import sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))         # .../Predictions
sys.path.insert(0, HERE)
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Analyse_sensibilite")))
import voll_common as V

# ======================= CONFIGURATION =======================
# Balayage THOROUGH : setpoint ELY (socle) x exposant x seuil RUL_ref.
# RUL_ref decide QUAND le derating s'active (RUL<ref) : le sweeper est essentiel
# pour donner sa vraie chance au levier RUL. exp=0 = RB2 constant (indep. de ref).
CELY     = [0.300, 0.310, 0.320]                 # socle ELY (elargir au meso)
EXPS     = [0.0, 0.05, 0.10, 0.20, 0.50]         # exposant RUL
RUL_REFS = [1000.0, 2000.0, 3000.0]              # seuil RUL [jours]
CFC      = 0.440                                  # base FC (= meilleur RB2 fixe)
_N_AVAIL = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))
OUT_TXT  = os.path.join(HERE, "sweep_rul_attribution.txt")
# =============================================================


def make_rule(c_fc, c_ely, rul_ref, exp_ely):
    def act(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
            RUL_ely_t, SoH_fc_t, SoH_ely_t):
        f_ely = min(max(RUL_ely_t, 0.0) / rul_ref, 1.0) ** exp_ely
        P_fc_set  = c_fc  * I.FC['P_fc_max']
        P_ely_set = c_ely * I.ELY['P_ely_max'] * f_ely
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
    c_fc, c_ely, ref, exp = args
    d = init_and_run_loop(make_rule(c_fc, c_ely, ref, exp))
    lp, dg = _metrics(d)
    return (c_fc, c_ely, ref, exp, lp, dg, V.total_cost_keur(lp, dg))


def main():
    # exp=0 est independant de RUL_ref (f_ely=1) -> un seul combo par c_ely.
    combos = []
    for ce in CELY:
        for ex in EXPS:
            if ex == 0.0:
                combos.append((CFC, ce, RUL_REFS[0], ex))
            else:
                for ref in RUL_REFS:
                    combos.append((CFC, ce, ref, ex))
    nw = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2(RUL) attribution : {len(combos)} sims 25 ans ({nw} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(_eval, combos), 1):
            res.append(r)
            print(f"  [{i}/{len(combos)}] c_fc={r[0]:.3f} c_ely={r[1]:.3f} ref={r[2]:.0f} exp={r[3]:.2f}"
                  f" -> LPSP={r[4]:.3f}% deg={r[5]:.2f} UNIF={r[6]:.3f}", flush=True)
    print(f"--- {time.time()-t0:.0f}s ---", flush=True)
    res.sort(key=lambda r: r[6])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# RB2(RUL) attribution : P_ely=c_ely*Pmax*min(RUL/ref,1)^exp ; refs={RUL_REFS} ; VoLL={V.VOLL_TIERS}\n")
        f.write("rang;c_fc;c_ely;RUL_ref;exp_ely;LPSP(%);deg(kEUR);total(kEUR)\n")
        for i, r in enumerate(res, 1):
            f.write(f"{i};{r[0]:.3f};{r[1]:.3f};{r[2]:.0f};{r[3]:.3f};{r[4]:.4f};{r[5]:.4f};{r[6]:.4f}\n")
    b = res[0]
    best_mod = min((r for r in res if r[3] > 0.0), key=lambda r: r[6], default=None)
    best_cst = min((r for r in res if r[3] == 0.0), key=lambda r: r[6])
    print(f"\n>>> MEILLEUR RB2(RUL) global : c_ely={b[1]:.3f} exp={b[3]:.2f} ref={b[2]:.0f}"
          f" -> (LPSP {b[4]:.4f} %, deg {b[5]:.4f} k€), UNIF {b[6]:.4f} k€", flush=True)
    print(f"    meilleur SANS RUL (exp=0)    : c_ely={best_cst[1]:.3f} -> UNIF {best_cst[6]:.4f} k€")
    if best_mod:
        print(f"    meilleur AVEC RUL (exp>0)    : c_ely={best_mod[1]:.3f} exp={best_mod[3]:.2f}"
              f" ref={best_mod[2]:.0f} -> UNIF {best_mod[6]:.4f} k€")
        print(f"    >>> gain du levier RUL = {best_cst[6]-best_mod[6]:+.4f} k€ "
              f"({100*(best_cst[6]-best_mod[6])/best_cst[6]:+.3f} %)  "
              f"[<0 => le RUL DEGRADE]")
    with open(OUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"\n# MIN global : c_ely={b[1]:.3f} exp={b[3]:.2f} ref={b[2]:.0f} -> {b[6]:.4f} kEUR\n")
        f.write(f"# MIN sans RUL (exp=0) : {best_cst[6]:.4f} kEUR (c_ely={best_cst[1]:.3f})\n")
        if best_mod:
            f.write(f"# MIN avec RUL (exp>0) : {best_mod[6]:.4f} kEUR "
                    f"(c_ely={best_mod[1]:.3f} exp={best_mod[3]:.2f} ref={best_mod[2]:.0f})\n")
            f.write(f"# gain levier RUL = {best_cst[6]-best_mod[6]:+.4f} kEUR "
                    f"({100*(best_cst[6]-best_mod[6])/best_cst[6]:+.3f} %)\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
