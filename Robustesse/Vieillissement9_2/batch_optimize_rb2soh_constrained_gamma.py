"""
batch_optimize_rb2soh_constrained_gamma.py
==========================================
RECHERCHE DES MEILLEURS SETPOINTS de RB2(SoH) GENERALISEE (avec exposant gamma)
qui MINIMISENT LA DEGRADATION sous CONTRAINTE de LPSP, en visant a BATTRE un
SEUL point RB2 de reference.

    Reference RB2 (figee)  : setpoints (0.45, 0.33)
                             -> LPSP_REF = 2.454 %   deg_REF = 65.42 kEUR
    Probleme               : min   deg(c_fc, c_ely, g_fc, g_ely)      [kEUR]
                             s.c.  LPSP <= LPSP_REF

Famille generalisee (niche RB2 a g=0, RB2(SoH) a g=1) :
    P_fc_set  = c_fc  * P_fc_max  * SoH_fc^g_fc
    P_ely_set = c_ely * P_ely_max * SoH_ely^g_ely

Tout le reste de la regle (plafond H2, defaillances) est identique aux fichiers
RB2 / RB2(SoH). MEME moteur (Common.main_init_and_loop via sens_common) et MEME
estimateur metrics() (deg ET lpsp) que la reference -> comparaison equitable.

Strategie de recherche :
  PHASE 1 : grille autour du coin serre, gamma en degre de liberte.
  PHASE 2 : raffinement qui PINCE la frontiere LPSP=LPSP_REF autour du meilleur
            point faisable (on pousse c_ely vers le bas tant que LPSP <= cap).

Dimensionnement FIXE (Init), horizon = main_init_and_loop (25 ans).

Lancer depuis Vieillissement8 :
    ~/miniconda3/envs/simu_env/bin/python batch_optimize_rb2soh_constrained_gamma.py
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_SENS = os.path.normpath(os.path.join(HERE, os.pardir, "Analyse_sensibilite"))
if _SENS not in sys.path:
    sys.path.insert(0, _SENS)

from sens_common import I, init_and_run_loop, metrics, lifetimes  # noqa: E402
from Common.get_lol import get_lol                                # noqa: E402

# ============================ CONFIGURATION ============================
REF_RB2 = (0.450, 0.330)   # point RB2 de reference (LPSP_CAP = sa LPSP, deg = a battre)
LPSP_TOL = 1e-3            # tolerance sur la contrainte [pts de %]

# --- PHASE 1 : grille autour du coin serre (gamma en DOF) ---
GRID_C_FC  = [0.45, 0.475, 0.50]
GRID_C_ELY = [0.31, 0.32, 0.33, 0.34, 0.35]
GRID_G_FC  = [0.0, 1.0, 2.0]
GRID_G_ELY = [0.0, 1.0, 2.0]
# -> 3 x 5 x 3 x 3 = 135 simulations.

# --- PHASE 2 : raffinement qui pince la frontiere autour du meilleur faisable ---
REFINE = True
C_STEPS  = [-0.010, -0.005, 0.005]   # on pousse surtout c_ely vers le bas (deg plus faible)
G_NEIGH  = [0.5]                     # on teste aussi des gamma intermediaires autour du best

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_rb2soh_constrained_gamma_25y.txt"
OUT_PDF = "optim_rb2soh_constrained_gamma_25y.pdf"
# ======================================================================


def make_rule(c_fc, c_ely, g_fc, g_ely):
    """P_set = c * P_max * SoH^g. g=0 -> RB2 ; g=1 -> RB2(SoH) ; reste identique."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = c_fc  * I.FC['P_fc_max']   * max(SoH_fc_t, 0.0)  ** g_fc
        P_ely_set = c_ely * I.ELY['P_ely_max'] * max(SoH_ely_t, 0.0) ** g_ely

        dt_h         = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff'] * I.CONV['eta'] * 1000
        P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (I.ELY['eff'] * I.CONV['eta']) * 1000

        if P_tot_ref_t > 0:
            P_fc_avail = min(P_fc_set, P_fc_h2_max)
            if P_tot_ref_t > P_fc_avail:
                P_dc_fc_t, P_dc_bat_t = P_fc_avail, P_tot_ref_t - P_fc_avail
            else:
                P_dc_fc_t, P_dc_bat_t = 0, P_tot_ref_t
            P_dc_ely_t = 0
        if P_tot_ref_t < 0:
            P_ely_avail = min(P_ely_set, P_ely_h2_max)
            if P_tot_ref_t < -P_ely_avail:
                P_dc_ely_t, P_dc_bat_t = -P_ely_avail, P_tot_ref_t + P_ely_avail
            else:
                P_dc_ely_t, P_dc_bat_t = 0, P_tot_ref_t
            P_dc_fc_t = 0

        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t

        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        return get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t, E_h2_init,
                       P_fc_max_t, P_ely_max_t, SoH_bat_t)
    return get_optimal_action_RB


def evaluate(params):
    """Worker (picklable) : (c_fc, c_ely, g_fc, g_ely) -> resultats via metrics()."""
    c_fc, c_ely, g_fc, g_ely = params
    data = init_and_run_loop(make_rule(c_fc, c_ely, g_fc, g_ely))
    lpsp, deg = metrics(data)
    lifb, liff, life = lifetimes(data)
    return dict(c_fc=c_fc, c_ely=c_ely, g_fc=g_fc, g_ely=g_ely,
                lpsp=float(lpsp), deg=float(deg),
                life_bat=lifb, life_fc=liff, life_ely=life)


def fmt_life(x):
    return f"{x:5.1f}" if x is not None else " >hor"


def feasible_pts(res, cap):
    return [r for r in res if r['lpsp'] <= cap + LPSP_TOL]


def best_feasible(res, cap):
    feas = feasible_pts(res, cap)
    return min(feas, key=lambda r: r['deg']) if feas else None


def run_pool(param_list, title, cap):
    print(f"\n--- {title} : {len(param_list)} simulations ({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            flag = "OK " if r['lpsp'] <= cap + LPSP_TOL else "x  "
            print(f"  [{i:3d}/{len(param_list)}] c=({r['c_fc']:.3f},{r['c_ely']:.3f}) "
                  f"g=({r['g_fc']:.1f},{r['g_ely']:.1f}) -> LPSP {r['lpsp']:6.3f}% {flag}"
                  f"deg {r['deg']:6.2f}k€ | vie E {fmt_life(r['life_ely'])}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    print("=== RB2(SoH) generalisee (gamma) : min deg s.c. LPSP <= LPSP(RB2 ref) — 25 ans ===", flush=True)
    print(f"    Moteur + metrics() communs | dim FIXE | {N_WORKERS} workers", flush=True)

    # --- Reference RB2 (cap + cible deg), calculee EN INTERNE (auto-coherent) ---
    ref = evaluate((REF_RB2[0], REF_RB2[1], 0.0, 0.0))
    cap = ref['lpsp']
    print(f"    Reference RB2 {REF_RB2} : LPSP {ref['lpsp']:.4f}%  deg {ref['deg']:.2f}k€  "
          f"(vie B/F/E {fmt_life(ref['life_bat'])}/{fmt_life(ref['life_fc'])}/{fmt_life(ref['life_ely'])})", flush=True)
    print(f"    => CONTRAINTE : LPSP <= {cap:.4f}% ; OBJECTIF : deg < {ref['deg']:.2f}k€", flush=True)

    # --- PHASE 1 : grille ---
    p1 = [(cf, ce, gf, ge) for cf in GRID_C_FC for ce in GRID_C_ELY
          for gf in GRID_G_FC for ge in GRID_G_ELY]
    r1 = run_pool(p1, "PHASE 1 — grille (coin serre, gamma en DOF)", cap)

    # --- PHASE 2 : raffinement qui pince la frontiere autour du meilleur faisable ---
    r2 = []
    if REFINE:
        b = best_feasible(r1, cap)
        if b is not None:
            cf0, ce0, gf0, ge0 = b['c_fc'], b['c_ely'], b['g_fc'], b['g_ely']
            cf_set = sorted({round(cf0 + d, 4) for d in [0.0] + C_STEPS})
            ce_set = sorted({round(ce0 + d, 4) for d in [0.0] + C_STEPS})
            gf_set = sorted({max(0.0, gf0 + d) for d in [0.0] + ([-1] + G_NEIGH if gf0 >= 1 else G_NEIGH)})
            ge_set = sorted({max(0.0, ge0 + d) for d in [0.0] + ([-1] + G_NEIGH if ge0 >= 1 else G_NEIGH)})
            seen = {(r['c_fc'], r['c_ely'], r['g_fc'], r['g_ely']) for r in r1}
            p2 = [(cf, ce, gf, ge) for cf in cf_set for ce in ce_set
                  for gf in gf_set for ge in ge_set
                  if (cf, ce, gf, ge) not in seen]
            if p2:
                r2 = run_pool(p2, f"PHASE 2 — raffinement autour de "
                              f"c=({cf0:.3f},{ce0:.3f}) g=({gf0:.1f},{ge0:.1f})", cap)
        else:
            print("\n[!] Aucun point faisable en phase 1 — pas de raffinement.", flush=True)

    allres = r1 + r2
    best = best_feasible(allres, cap)
    # Pour reference : meilleur RB2(SoH) "classique" (g=1,1) faisable.
    best_soh1 = best_feasible([r for r in allres if r['g_fc'] == 1.0 and r['g_ely'] == 1.0], cap)

    # --- Sauvegarde txt (faisables tries par deg croissant en tete) ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# RB2(SoH) generalisee gamma — min deg s.c. LPSP<=cap — 25 ans — metrics() commun\n")
        f.write(f"# REF RB2{REF_RB2} : LPSP_cap={cap:.4f}%  deg_cible={ref['deg']:.2f}kEUR\n")
        f.write("phase;c_fc;c_ely;g_fc;g_ely;faisable;LPSP_%;deg_kEUR;vie_bat;vie_fc;vie_ely\n")
        rows = [("1", r) for r in r1] + [("2", r) for r in r2]
        for ph, r in sorted(rows, key=lambda x: (x[1]['lpsp'] > cap + LPSP_TOL, x[1]['deg'])):
            feas = int(r['lpsp'] <= cap + LPSP_TOL)
            f.write(f"{ph};{r['c_fc']:.3f};{r['c_ely']:.3f};{r['g_fc']:.1f};{r['g_ely']:.1f};"
                    f"{feas};{r['lpsp']:.4f};{r['deg']:.2f};"
                    f"{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")

    # --- Figure : (LPSP, deg) ; faisable colore par g_ely ; ref + optimum ---
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lp = np.array([r['lpsp'] for r in allres]); dg = np.array([r['deg'] for r in allres])
    feas = lp <= cap + LPSP_TOL
    ax.scatter(lp[~feas], dg[~feas], c="lightgray", s=40, label="LPSP > contrainte")
    if feas.any():
        sc = ax.scatter(lp[feas], dg[feas], c=[r['g_ely'] for r, k in zip(allres, feas) if k],
                        cmap="viridis", s=75, label="faisable")
        plt.colorbar(sc, label="γ_ely")
    ax.axhline(ref['deg'], color="black", ls=":", lw=1.2, label=f"deg RB2 ref = {ref['deg']:.1f}k€")
    ax.axvline(cap, color="red", ls="--", lw=1.5, label=f"LPSP_cap = {cap:.3f}%")
    ax.scatter(ref['lpsp'], ref['deg'], c="black", marker="s", s=130, zorder=5,
               label=f"RB2 ref ({REF_RB2[0]},{REF_RB2[1]})")
    if best is not None:
        ax.scatter(best['lpsp'], best['deg'], c="crimson", marker="*", s=300, zorder=6,
                   label="optimum RB2(SoH+γ)")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût total de dégradation [k€]")
    ax.set_title("RB2(SoH) généralisée (γ) — min dégradation sous contrainte LPSP ≤ LPSP(RB2)\n"
                 "faisable = à gauche du trait rouge ; on cherche sous le trait noir")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # --- Resume console : LE verdict ---
    print("\n" + "=" * 80)
    print(f"VERDICT — min deg sous LPSP <= {cap:.4f}%  (cible a battre : RB2 = {ref['deg']:.2f}k€)")
    print("=" * 80)
    if best is None:
        print("  [!] AUCUN point faisable trouve. Elargir la grille / relacher la contrainte.")
    else:
        d = best['deg']; gain = (ref['deg'] - d) / ref['deg'] * 100
        verdict = "BAT RB2" if d < ref['deg'] - 1e-9 else ("EGALITE" if abs(d - ref['deg']) <= 1e-9 else "NE BAT PAS RB2")
        print(f"  Optimum  : c=({best['c_fc']:.3f},{best['c_ely']:.3f})  "
              f"γ=({best['g_fc']:.1f},{best['g_ely']:.1f})")
        print(f"             LPSP {best['lpsp']:.4f}%   deg {d:.2f}k€   "
              f"({gain:+.2f}% vs RB2)  -> {verdict}")
        print(f"             vie B/F/E {fmt_life(best['life_bat'])}/{fmt_life(best['life_fc'])}/"
              f"{fmt_life(best['life_ely'])} ans")
        if best_soh1 is not None:
            print(f"  (RB2(SoH) classique g=1,1 faisable : c=({best_soh1['c_fc']:.3f},"
                  f"{best_soh1['c_ely']:.3f}) LPSP {best_soh1['lpsp']:.3f}% deg {best_soh1['deg']:.2f}k€)")
        feas_sorted = sorted(feasible_pts(allres, cap), key=lambda r: r['deg'])[:6]
        print("\n  Meilleurs faisables (deg croissant) :")
        for r in feas_sorted:
            print(f"    c=({r['c_fc']:.3f},{r['c_ely']:.3f}) γ=({r['g_fc']:.1f},{r['g_ely']:.1f}) : "
                  f"LPSP {r['lpsp']:.4f}%  deg {r['deg']:.2f}k€  vie E {fmt_life(r['life_ely'])}")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
