"""
batch_optimize_constrained_compare.py
=====================================
COMPARAISON EQUITABLE RB2 vs RB2(SoH) au sens du meme probleme contraint :

    min   deg(coef_fc, coef_ely)              [cout total de degradation, kEUR]
    s.c.  LPSP(coef_fc, coef_ely) <= LPSP_CAP

POUR LES DEUX FAMILLES, sur LE MEME moteur (Common.main_init_and_loop via
sens_common) et LE MEME estimateur metrics() (deg ET lpsp). C'est la correction
du biais de batch_optimize_rb2soh.py, qui appelait get_cost_total directement sur
SoH_bat BRUT (sans l'interpolation aux remplacements faite par metrics()), alors
que le cote RB2 passait deja par metrics() -> les deux deg n'etaient pas
comparables.

Seule difference entre les deux familles (tout le reste est identique aux fichiers
RB2/ et RB2(SoH)/get_optimal_action_RB.py) :
    RB2      : P_set = coef * P_max
    RB2(SoH) : P_set = coef * P_max * SoH_t

LPSP_CAP est ancre sur RB2 a ses setpoints de reference (REF_RB2), calcule EN
INTERNE avec metrics() pour etre auto-coherent. Les deux familles doivent alors
faire LPSP <= LPSP_CAP, et on compare leur plancher de degradation faisable.

Dimensionnement FIXE (Init), horizon = main_init_and_loop (25 ans).

Lancer depuis Vieillissement8 :
    ~/miniconda3/envs/simu_env/bin/python batch_optimize_constrained_compare.py
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# --- Chemins : ce dossier + Analyse_sensibilite (sens_common) ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_SENS = os.path.normpath(os.path.join(HERE, os.pardir, "Analyse_sensibilite"))
if _SENS not in sys.path:
    sys.path.insert(0, _SENS)

# Moteur + estimateurs STRICTEMENT identiques a l'analyse de sensibilite :
# metrics() renvoie (LPSP %, deg kEUR) avec l'interpolation SoH_bat aux
# remplacements -> deg comparable entre les deux familles.
from sens_common import I, init_and_run_loop, metrics, lifetimes  # noqa: E402
from Common.get_lol import get_lol                                # noqa: E402

# ============================ CONFIGURATION ============================
# Setpoints de reference de RB2 (= fichier RB2/get_optimal_action_RB.py).
# LPSP_CAP sera la LPSP de RB2 a ce point (calculee plus bas). RB2 y est donc
# faisable a l'egalite ; on regarde si chaque famille peut descendre la deg
# SANS depasser cette LPSP.
REF_RB2 = (0.450, 0.330)
LPSP_TOL = 1e-3            # tolerance sur la contrainte (pts de %)
LPSP_CAP_OVERRIDE = None  # mettre un float [%] pour forcer la borne au lieu de REF_RB2

# Grille de balayage (fraction de P_max), commune aux deux familles. On encadre
# largement la frontiere de faisabilite reperee dans les .txt existants.
GRID_F_FC  = [0.40, 0.45, 0.475, 0.50, 0.55]
GRID_F_ELY = [0.30, 0.32, 0.33, 0.34, 0.36]

REFINE      = True
REFINE_STEP = 0.01        # pas fin pour pincer la frontiere autour du meilleur faisable

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_constrained_compare_25y.txt"
OUT_PDF = "optim_constrained_compare_25y.pdf"
FAMILIES = ("RB2", "RB2(SoH)")
# ======================================================================


def make_rule(family, coef_fc, coef_ely):
    """Regle parametree, IDENTIQUE aux fichiers de chaque famille. La seule
    difference est le facteur SoH applique aux setpoints pour RB2(SoH)."""
    soh = (family == "RB2(SoH)")

    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if soh:
            P_fc_set  = coef_fc  * I.FC['P_fc_max']   * SoH_fc_t
            P_ely_set = coef_ely * I.ELY['P_ely_max'] * SoH_ely_t
        else:
            P_fc_set  = coef_fc  * I.FC['P_fc_max']
            P_ely_set = coef_ely * I.ELY['P_ely_max']

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
    """Worker (picklable) : (family, coef_fc, coef_ely) -> resultats.
    deg ET lpsp via metrics() -> estimateur unique pour les deux familles."""
    family, coef_fc, coef_ely = params
    data = init_and_run_loop(make_rule(family, coef_fc, coef_ely))
    lpsp, deg = metrics(data)                       # LPSP %, deg kEUR (estimateur commun)
    lifb, liff, life = lifetimes(data)
    return dict(family=family, f_fc=coef_fc, f_ely=coef_ely,
                lpsp=float(lpsp), deg=float(deg),
                life_bat=lifb, life_fc=liff, life_ely=life)


def fmt_life(x):
    return f"{x:5.1f}" if x is not None else " >hor"


def run_pool(param_list, title):
    print(f"\n--- {title} : {len(param_list)} simulations ({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(param_list)}] {r['family']:8s} f=({r['f_fc']:.3f},"
                  f"{r['f_ely']:.3f}) -> LPSP {r['lpsp']:6.3f}% | deg {r['deg']:6.2f}k€ "
                  f"| vie B/F/E {fmt_life(r['life_bat'])}/{fmt_life(r['life_fc'])}/"
                  f"{fmt_life(r['life_ely'])}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def best_feasible(res, cap):
    feas = [r for r in res if r['lpsp'] <= cap + LPSP_TOL]
    return min(feas, key=lambda r: r['deg']) if feas else None


def optimize_family(family, cap):
    """Grille + raffinement local autour du meilleur faisable (min deg s.c. LPSP<=cap)."""
    p1 = [(family, ff, fe) for ff in GRID_F_FC for fe in GRID_F_ELY]
    r1 = run_pool(p1, f"{family} PHASE 1 — grille")

    r2 = []
    if REFINE:
        b = best_feasible(r1, cap)
        if b is not None:
            ff0, fe0 = b['f_fc'], b['f_ely']
            ff_set = sorted({round(max(0.05, ff0 + d * REFINE_STEP), 4) for d in (-1, 0, 1)})
            fe_set = sorted({round(max(0.05, fe0 + d * REFINE_STEP), 4) for d in (-2, -1, 0, 1)})
            seen = {(r['f_fc'], r['f_ely']) for r in r1}
            p2 = [(family, ff, fe) for ff in ff_set for fe in fe_set if (ff, fe) not in seen]
            if p2:
                r2 = run_pool(p2, f"{family} PHASE 2 — raffinement ({ff0:.3f},{fe0:.3f})")
        else:
            print(f"\n[!] {family} : aucun point faisable en phase 1.", flush=True)
    return r1 + r2


def main():
    print("=== Comparaison EQUITABLE RB2 vs RB2(SoH) : min deg s.c. LPSP <= cap ===", flush=True)
    print(f"    Moteur + metrics() communs | dim FIXE | 25 ans | {N_WORKERS} workers", flush=True)
    print(f"    FC P_max={I.FC['P_fc_max']/1000:.2f} kW | ELY P_max={I.ELY['P_ely_max']/1000:.2f} kW", flush=True)

    # --- 1) Borne LPSP : RB2 a ses setpoints de reference (auto-coherent) ---
    if LPSP_CAP_OVERRIDE is not None:
        cap = float(LPSP_CAP_OVERRIDE)
        ref = None
        print(f"    LPSP_CAP forcee = {cap:.4f} %", flush=True)
    else:
        ref = evaluate(("RB2",) + REF_RB2)
        cap = ref['lpsp']
        print(f"    Reference RB2 {REF_RB2} -> LPSP {ref['lpsp']:.4f}%  deg {ref['deg']:.2f}k€", flush=True)
        print(f"    => LPSP_CAP = {cap:.4f} % (les deux familles doivent rester <= a ca)", flush=True)

    # --- 2) Optimisation contrainte de chaque famille, MEME cap ---
    results = {fam: optimize_family(fam, cap) for fam in FAMILIES}
    best = {fam: best_feasible(results[fam], cap) for fam in FAMILIES}

    # --- 3) Sauvegarde txt ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Comparaison contrainte RB2 vs RB2(SoH) — min deg s.c. LPSP<=CAP — 25 ans\n")
        f.write(f"# LPSP_CAP={cap:.4f}%  (ancre = RB2{REF_RB2})  estimateur=metrics() commun\n")
        f.write("famille;coef_fc;coef_ely;faisable;LPSP_%;deg_kEUR;vie_bat;vie_fc;vie_ely\n")
        for fam in FAMILIES:
            for r in sorted(results[fam], key=lambda x: x['deg']):
                feas = int(r['lpsp'] <= cap + LPSP_TOL)
                f.write(f"{fam};{r['f_fc']:.3f};{r['f_ely']:.3f};{feas};{r['lpsp']:.4f};"
                        f"{r['deg']:.2f};{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")

    # --- 4) Figure : nuages (LPSP, deg) des deux familles + cap + optima ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"RB2": "tab:blue", "RB2(SoH)": "tab:green"}
    for fam in FAMILIES:
        lp = np.array([r['lpsp'] for r in results[fam]])
        dg = np.array([r['deg'] for r in results[fam]])
        feas = lp <= cap + LPSP_TOL
        ax.scatter(lp[feas], dg[feas], c=colors[fam], s=55, label=f"{fam} (faisable)")
        ax.scatter(lp[~feas], dg[~feas], facecolors="none", edgecolors=colors[fam],
                   s=45, alpha=0.5)
        if best[fam] is not None:
            ax.scatter(best[fam]['lpsp'], best[fam]['deg'], c=colors[fam], marker="*",
                       s=320, edgecolors="k", zorder=6,
                       label=f"{fam} optimum  deg={best[fam]['deg']:.1f}k€")
    ax.axvline(cap, color="red", ls="--", lw=1.5, label=f"LPSP_cap = {cap:.3f}%")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût total de dégradation [k€]")
    ax.set_title("RB2 vs RB2(SoH) — min dégradation sous contrainte LPSP (25 ans)\n"
                 "même moteur, même métrique ; faisable = à gauche de la ligne rouge")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # --- 5) Resume console : LE verdict ---
    print("\n" + "=" * 78)
    print(f"VERDICT — min deg sous contrainte LPSP <= {cap:.4f}% (estimateur commun)")
    print("=" * 78)
    for fam in FAMILIES:
        b = best[fam]
        if b is None:
            print(f"  {fam:8s} : AUCUN point faisable dans la grille.")
            continue
        print(f"  {fam:8s} : optimum ({b['f_fc']:.3f},{b['f_ely']:.3f}) "
              f"LPSP {b['lpsp']:.3f}%  deg {b['deg']:.2f}k€  "
              f"| vie B/F/E {fmt_life(b['life_bat'])}/{fmt_life(b['life_fc'])}/{fmt_life(b['life_ely'])}")
    if all(best[f] is not None for f in FAMILIES):
        d_rb2, d_soh = best["RB2"]['deg'], best["RB2(SoH)"]['deg']
        gain = (d_rb2 - d_soh) / d_rb2 * 100
        verdict = "RB2(SoH) MEILLEURE" if d_soh < d_rb2 - 1e-9 else \
                  ("EGALITE" if abs(d_soh - d_rb2) <= 1e-9 else "RB2 meilleure")
        print("-" * 78)
        print(f"  Δdeg = {d_rb2 - d_soh:+.2f} k€  ({gain:+.2f}% en faveur de RB2(SoH))  -> {verdict}")
        for fam in FAMILIES:
            top = sorted([r for r in results[fam] if r['lpsp'] <= cap + LPSP_TOL],
                         key=lambda r: r['deg'])[:3]
            print(f"  {fam:8s} top faisables : " + " | ".join(
                f"({r['f_fc']:.3f},{r['f_ely']:.3f}) {r['deg']:.1f}k€@{r['lpsp']:.3f}%" for r in top))
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
