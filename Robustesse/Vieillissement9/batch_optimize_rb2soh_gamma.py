"""
batch_optimize_rb2soh_gamma.py
==============================
GENERALISATION de RB2(SoH) par un EXPOSANT gamma sur le SoH, qui NICHE RB2 :

    P_set = coef * P_max * SoH^gamma          (par composant FC et ELY)
        gamma = 0  -> P_set = coef * P_max            == RB2          (setpoint constant)
        gamma = 1  -> P_set = coef * P_max * SoH       == RB2(SoH) actuelle

Comme RB2 (gamma=0) et RB2(SoH) (gamma=1) sont des CAS PARTICULIERS de cette
famille, balayes sur LE MEME moteur (Common.main_init_and_loop via sens_common)
et LA MEME metrique metrics(), la famille gamma DOMINE les deux par construction
au sens du probleme contraint :

    min   deg(coef_fc, coef_ely, gamma_fc, gamma_ely)   [kEUR]
    s.c.  LPSP <= LPSP_CAP

-> pour TOUTE borne LPSP_CAP, deg*(famille gamma) <= deg*(RB2) puisque RB2 est
   toujours faisable (gamma=0). L'interet : montrer que l'optimum choisit
   gamma* > 0 des qu'il y a de la marge de LPSP, et CHIFFRER le gain. La valeur
   de gamma* mesure la "valeur de l'information SoH".

On balaie UNE fois toute la grille (coef_fc, coef_ely, gamma_fc, gamma_ely) ;
RB2 = sous-ensemble {gamma_fc=gamma_ely=0}, RB2(SoH) = {gamma_fc=gamma_ely=1}.
Tout le post-traitement (fronts, optima par borne) se fait sur ce seul balayage.

Dimensionnement FIXE (Init), horizon = main_init_and_loop (25 ans).

Lancer depuis Vieillissement8 :
    ~/miniconda3/envs/simu_env/bin/python batch_optimize_rb2soh_gamma.py
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
# Grille de balayage. La famille contient RB2 (g=0) et RB2(SoH) (g=1).
# IMPORTANT : la grille DOIT contenir le point optimal de RB2 (0.450,0.330) et
# celui de RB2(SoH) pour que les baselines soient optimisees a EGALITE avec la
# famille gamma (sinon le "gain vs RB2" est gonfle par une RB2 mal echantillonnee).
GRID_F_FC   = [0.40, 0.45, 0.475, 0.50, 0.55]
GRID_F_ELY  = [0.30, 0.32, 0.33, 0.34, 0.36, 0.40]
GRID_G_FC   = [0.0, 1.0, 2.0, 3.0]
GRID_G_ELY  = [0.0, 1.0, 2.0, 3.0]
# -> 5 x 6 x 4 x 4 = 480 simulations.

# Setpoints de reference de RB2 -> fixe la borne LPSP "nominale" (auto-coherent).
REF_RB2 = (0.450, 0.330)
# Bornes LPSP auxquelles on compare les optima (la 1re ~ coin serre de RB2).
# 'auto' sera remplacee par la LPSP de RB2 a REF_RB2 (calculee en interne).
CAPS = ["auto", 2.6, 2.9, 3.2, 3.6, 5.0]
LPSP_TOL = 1e-3

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_rb2soh_gamma_25y.txt"
OUT_PDF = "optim_rb2soh_gamma_25y.pdf"
# ======================================================================


def make_rule(coef_fc, coef_ely, g_fc, g_ely):
    """Regle generalisee : P_set = coef * P_max * SoH^gamma. Identique aux
    fichiers RB2/RB2(SoH) pour le reste. g=0 -> RB2 ; g=1 -> RB2(SoH)."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        # SoH^gamma : gamma=0 -> 1 (constant, RB2) ; gamma>0 -> recule avec l'usure.
        P_fc_set  = coef_fc  * I.FC['P_fc_max']   * max(SoH_fc_t, 0.0)  ** g_fc
        P_ely_set = coef_ely * I.ELY['P_ely_max'] * max(SoH_ely_t, 0.0) ** g_ely

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


def is_rb2(r):     return r['g_fc'] == 0.0 and r['g_ely'] == 0.0
def is_rb2soh(r):  return r['g_fc'] == 1.0 and r['g_ely'] == 1.0


def pareto_front(pts):
    """Front non-domine (minimise LPSP ET deg). pts = list of (lpsp, deg, meta)."""
    pts = sorted(pts, key=lambda t: (t[0], t[1]))
    front, best = [], float("inf")
    for lp, dg, meta in pts:
        if dg < best - 1e-9:
            front.append((lp, dg, meta)); best = dg
    return front


def best_feasible(res, cap):
    feas = [r for r in res if r['lpsp'] <= cap + LPSP_TOL]
    return min(feas, key=lambda r: r['deg']) if feas else None


def run_pool(param_list):
    print(f"\n--- Balayage famille gamma : {len(param_list)} simulations "
          f"({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            tag = "RB2   " if is_rb2(r) else ("RB2SoH" if is_rb2soh(r) else "      ")
            print(f"  [{i:3d}/{len(param_list)}] {tag} c=({r['c_fc']:.3f},{r['c_ely']:.3f}) "
                  f"g=({r['g_fc']:.0f},{r['g_ely']:.0f}) -> LPSP {r['lpsp']:6.3f}% "
                  f"deg {r['deg']:6.2f}k€", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    print("=== RB2(SoH) generalisee SoH^gamma (niche RB2 a gamma=0) — 25 ans ===", flush=True)
    print(f"    Moteur + metrics() communs | dim FIXE | {N_WORKERS} workers", flush=True)
    print(f"    Grille : {len(GRID_F_FC)}x{len(GRID_F_ELY)}x{len(GRID_G_FC)}x{len(GRID_G_ELY)} "
          f"= {len(GRID_F_FC)*len(GRID_F_ELY)*len(GRID_G_FC)*len(GRID_G_ELY)} sims", flush=True)

    params = [(cf, ce, gf, ge) for cf in GRID_F_FC for ce in GRID_F_ELY
              for gf in GRID_G_FC for ge in GRID_G_ELY]
    res = run_pool(params)

    # Borne nominale = LPSP de RB2 a REF_RB2 (calculee si pas deja dans la grille).
    ref = next((r for r in res if (r['c_fc'], r['c_ely']) == REF_RB2 and is_rb2(r)), None)
    if ref is None:
        ref = evaluate((REF_RB2[0], REF_RB2[1], 0.0, 0.0))
    caps = [ref['lpsp'] if c == "auto" else float(c) for c in CAPS]

    sub_rb2  = [r for r in res if is_rb2(r)]
    sub_soh  = [r for r in res if is_rb2soh(r)]

    # -------- Sauvegarde txt --------
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# RB2(SoH) generalisee SoH^gamma — 25 ans — estimateur metrics() commun\n")
        f.write(f"# RB2 = sous-ensemble (g_fc=g_ely=0) ; RB2(SoH) = (g_fc=g_ely=1)\n")
        f.write(f"# REF_RB2={REF_RB2} -> LPSP_nominale={ref['lpsp']:.4f}%\n")
        f.write("c_fc;c_ely;g_fc;g_ely;LPSP_%;deg_kEUR;vie_bat;vie_fc;vie_ely\n")
        for r in sorted(res, key=lambda x: (x['lpsp'], x['deg'])):
            f.write(f"{r['c_fc']:.3f};{r['c_ely']:.3f};{r['g_fc']:.1f};{r['g_ely']:.1f};"
                    f"{r['lpsp']:.4f};{r['deg']:.2f};{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")
        f.write("\n# === Optima contraints par borne LPSP : RB2 vs RB2(SoH) vs famille gamma ===\n")
        f.write("cap_%;deg_RB2;deg_RB2SoH;deg_gamma;g_fc*;g_ely*;c_fc*;c_ely*;gain_vs_RB2_%\n")
        for cap in caps:
            bR = best_feasible(sub_rb2, cap); bS = best_feasible(sub_soh, cap)
            bG = best_feasible(res, cap)
            if bG is None:
                f.write(f"{cap:.4f};(aucun faisable)\n"); continue
            gain = (bR['deg'] - bG['deg']) / bR['deg'] * 100 if bR else float('nan')
            f.write(f"{cap:.4f};{bR['deg'] if bR else float('nan'):.2f};"
                    f"{bS['deg'] if bS else float('nan'):.2f};{bG['deg']:.2f};"
                    f"{bG['g_fc']:.0f};{bG['g_ely']:.0f};{bG['c_fc']:.3f};{bG['c_ely']:.3f};"
                    f"{gain:+.2f}\n")

    # -------- Figure : 3 fronts (RB2, RB2(SoH), famille gamma) ; gamma colore --------
    fig, ax = plt.subplots(figsize=(9, 6))
    fR = pareto_front([(r['lpsp'], r['deg'], r) for r in sub_rb2])
    fS = pareto_front([(r['lpsp'], r['deg'], r) for r in sub_soh])
    fG = pareto_front([(r['lpsp'], r['deg'], r) for r in res])
    ax.scatter([r['lpsp'] for r in res], [r['deg'] for r in res],
               c="lightgray", s=14, alpha=0.5, zorder=1)
    if fR:
        ax.plot([p[0] for p in fR], [p[1] for p in fR], "-o", c="tab:blue",
                lw=2, ms=5, label="RB2 (γ=0)")
    if fS:
        ax.plot([p[0] for p in fS], [p[1] for p in fS], "-o", c="tab:green",
                lw=2, ms=5, label="RB2(SoH) (γ=1)")
    if fG:
        gx = [p[0] for p in fG]; gy = [p[1] for p in fG]
        gcol = [p[2]['g_ely'] for p in fG]
        ax.plot(gx, gy, "-", c="crimson", lw=2.2, zorder=4, label="Famille γ (front)")
        sc = ax.scatter(gx, gy, c=gcol, cmap="autumn_r", s=85, edgecolors="k",
                        zorder=5, vmin=0, vmax=max(GRID_G_ELY))
        plt.colorbar(sc, label="γ_ely optimal au point du front")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût total de dégradation [k€]")
    ax.set_title("Famille γ (P=c·Pmax·SoHᵞ) niche RB2 (γ=0) et RB2(SoH) (γ=1)\n"
                 "même moteur, même métrique — le front γ domine les deux (25 ans)")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # -------- Resume console : LE verdict par borne --------
    print("\n" + "=" * 86)
    print("VERDICT — min deg sous contrainte LPSP <= cap (estimateur commun) :")
    print("  borne |  deg RB2 | deg RB2(SoH) | deg famille γ | γ*(fc,ely) | gain vs RB2")
    print("-" * 86)
    for cap in caps:
        bR = best_feasible(sub_rb2, cap); bS = best_feasible(sub_soh, cap)
        bG = best_feasible(res, cap)
        if bG is None:
            print(f"  {cap:5.3f} | (aucun point faisable)"); continue
        dR = bR['deg'] if bR else float('nan')
        dS = bS['deg'] if bS else float('nan')
        gain = (dR - bG['deg']) / dR * 100 if bR else float('nan')
        print(f"  {cap:5.3f} | {dR:8.2f} | {dS:12.2f} | {bG['deg']:13.2f} | "
              f"({bG['g_fc']:.0f},{bG['g_ely']:.0f})      | {gain:+6.2f}%")
    print("=" * 86)
    print("Lecture : gamma*(fc,ely)=(0,0) => l'optimum EST RB2 (aucun gain possible) ;")
    print("          gamma*>0 => l'info SoH abaisse strictement la degradation a iso-LPSP.")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
