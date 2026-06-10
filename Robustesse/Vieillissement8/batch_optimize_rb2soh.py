"""
batch_optimize_rb2soh.py
========================
MEILLEURS SETPOINTS de RB2(SoH) qui MINIMISENT LA DEGRADATION sous CONTRAINTE de LPSP.

Probleme :
    min   deg(coef_fc, coef_ely)            [cout total de degradation FC+ELY+BAT, k€]
    s.c.  LPSP(coef_fc, coef_ely) <= LPSP_MAX        (= LPSP de RB2 = 2.6921 %)

On ne fait varier que les DEUX coefficients de RB2(SoH) :
    P_fc_set  = coef_fc  * FC['P_fc_max']  * SoH_fc_t      (0.475 dans le fichier)
    P_ely_set = coef_ely * ELY['P_ely_max'] * SoH_ely_t    (0.33  dans le fichier)
Le reste de la regle (scaling x SoH, plafond H2 -> batterie, defaillances) est IDENTIQUE
au fichier RB2(SoH)/get_optimal_action_RB.py — seuls ces 2 nombres bougent.

Dimensionnement FIXE (Init courant). Horizon = celui de main_init_and_loop (25 ans) : on
NE touche pas a SIM['Tend'].

Plages de recherche calees sur optim_rb2soh_setpoints_25y.txt :
  - coef_ely PILOTE la degradation (deg croit fort avec lui). Le plancher (~50 k€,
    coef_ely<=0.27) est INFAISABLE (LPSP>2.8 %). L'optimum contraint est sur la frontiere
    LPSP=LPSP_MAX, au plus petit coef_ely admissible (~0.32-0.33).
  - coef_fc a un optimum en U pour la LPSP autour de 0.475 (c'est la qu'on a le plus de
    marge LPSP pour abaisser coef_ely). -> on raffine autour de cette zone.

Lance depuis le dossier Vieillissement8, comme main.py :
    python batch_optimize_rb2soh.py
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol

# ============================ CONFIGURATION ============================
LPSP_MAX = 2.7127                    # contrainte dure [%] : LPSP de RB2 a ne pas depasser
LPSP_TOL = 1e-4                      # tolerance numerique sur la contrainte

DISCOUNT_RATE = 0.05                 # r : uniquement pour le report NPC (n'affecte pas l'objectif)
BASE_F_FC, BASE_F_ELY = 0.475, 0.33  # couple de reference (= setpoints RB2)

# --- PHASE 1 : grille fine autour de la frontiere de faisabilite ---
# (resserree sur coef_fc ~ 0.475 et coef_ely ~ 0.30-0.36, d'apres le fichier exporte)
GRID_F_FC  = [0.42, 0.45, 0.475, 0.50, 0.53]
GRID_F_ELY = [0.30, 0.31, 0.32, 0.33, 0.34, 0.36]
# -> Phase 1 = 5 x 6 = 30 simulations (le couple de base y est inclus)

# --- PHASE 2 : raffinement local autour du meilleur point FAISABLE ---
REFINE = True
REFINE_STEP = 0.01                   # pas fin (+/- sur chaque axe) pour pincer la frontiere
# -> Phase 2 = jusqu'a 3 x 3 = 9 simulations  (total <= ~39)

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_rb2soh_constrained_25y.txt"
OUT_PDF = "optim_rb2soh_constrained_25y.pdf"
# ======================================================================


def make_rb2soh(coef_fc, coef_ely):
    """Regle RB2(SoH) IDENTIQUE au fichier, parametree par les 2 seuls coefficients."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = coef_fc  * I.FC['P_fc_max']  * SoH_fc_t
        P_ely_set = coef_ely * I.ELY['P_ely_max'] * SoH_ely_t

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


def annuity_factor(r, N):
    return 1.0 if r <= 0 else (1 - (1 + r) ** (-N)) / (r * N)


def _lpsp_unserved(data):
    P_dc_load, P_dc_pv, lol = data["P_dc_load"], data["P_dc_pv"], data["lol_tab"]
    Pp = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    Pr = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol)])
    p, rr = np.clip(Pp, 0, None), np.clip(Pr, 0, None)
    unmet = np.clip(p - rr, 0, None)
    lpsp = (unmet.sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    e_unserved = unmet.sum() * I.LOAD['Ts'] / 3600.0
    return lpsp, e_unserved


def _lifetimes(data):
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        out.append(rep[0] * yr if len(rep) > 0 else None)
    return out  # [bat, fc, ely]


def _bop_fixe():
    kwh_bat = I.BAT['series_num'] * I.BAT['parallel_num'] * I.BAT['Q_bat'] * I.BAT['v_cell_nom'] / 1000.0
    return (I.BAT['CAPEX'] * kwh_bat - I.BAT['cost']) \
        + (I.FC['CAPEX'] * I.FC['P_fc_max'] / 1000.0 - I.FC['cost']) \
        + (I.ELY['CAPEX'] * I.ELY['P_ely_max'] / 1000.0 - I.ELY['cost'])


def evaluate(params):
    """Worker : (coef_fc, coef_ely) -> resultats. Dimensionnement FIXE (Init).
    OBJECTIF = deg (cout total de degradation, k€) ; CONTRAINTE = lpsp <= LPSP_MAX."""
    f_fc, f_ely = params
    strat = make_rb2soh(f_fc, f_ely)
    data = init_and_run_loop(strat)

    N = len(data["P_fc"]) * I.LOAD['Ts'] / (3600 * 24 * 365)
    AF = annuity_factor(DISCOUNT_RATE, N)

    lpsp, e_unserved = _lpsp_unserved(data)
    deg = get_cost_total(data["alpha_fc"][:-1], data["P_fc"], data["alpha_ely"][:-1],
                         data["P_ely"], data["P_bat"], data["SoC"],
                         I.LOAD, I.BAT, I.FC, I.ELY, data["SoH_bat"][:-1])
    bop = _bop_fixe()
    deg_k = deg / 1000.0                          # OBJECTIF : cout total de degradation [k€]
    npc = (bop + AF * deg) / 1000.0               # info : cout net actualise
    lifb, liff, life = _lifetimes(data)
    return dict(f_fc=f_fc, f_ely=f_ely, N=N,
                lpsp=float(lpsp), deg=float(deg_k), npc=float(npc), bop=bop / 1000.0,
                e_unserved=float(e_unserved),
                feasible=bool(lpsp <= LPSP_MAX + LPSP_TOL),
                life_bat=lifb, life_fc=liff, life_ely=life)


def best_feasible(res):
    """Index du min deg PARMI les points faisables (lpsp <= LPSP_MAX). None si aucun."""
    feas = [i for i, r in enumerate(res) if r['feasible']]
    if not feas:
        return None
    return min(feas, key=lambda i: res[i]['deg'])


def fmt_life(x):
    return f"{x:5.1f}" if x is not None else " >hor"


def run_pool(param_list, title):
    print(f"\n--- {title} : {len(param_list)} simulations ({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            flag = "OK " if r['feasible'] else "x  "
            print(f"  [{i:2d}/{len(param_list)}] f=({r['f_fc']:.3f},{r['f_ely']:.3f}) -> "
                  f"LPSP {r['lpsp']:6.3f}% {flag}|  deg {r['deg']:6.1f}k€  NPC {r['npc']:6.1f} "
                  f"| vie B/F/E {fmt_life(r['life_bat'])}/{fmt_life(r['life_fc'])}/"
                  f"{fmt_life(r['life_ely'])}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    print(f"=== RB2(SoH) : min DEGRADATION sous contrainte LPSP <= {LPSP_MAX:.4f} % "
          f"(= LPSP RB2) — dim FIXE, 25 ans ===", flush=True)
    print(f"    Dimensionnement (Init) : BAT parallel_num={I.BAT['parallel_num']:.4g} | "
          f"FC P_max={I.FC['P_fc_max']/1000:.2f} kW | ELY P_max={I.ELY['P_ely_max']/1000:.2f} kW", flush=True)
    print(f"    Couple de reference    : (coef_fc, coef_ely) = ({BASE_F_FC}, {BASE_F_ELY})", flush=True)

    # -------- PHASE 1 : grille fine (+ base garantie presente) --------
    p1 = [(ff, fe) for ff in GRID_F_FC for fe in GRID_F_ELY]
    if (BASE_F_FC, BASE_F_ELY) not in p1:
        p1.append((BASE_F_FC, BASE_F_ELY))
    r1 = run_pool(p1, "PHASE 1 — grille autour de la frontiere")

    # -------- PHASE 2 : raffinement autour du meilleur FAISABLE --------
    r2 = []
    if REFINE:
        ib = best_feasible(r1)
        if ib is not None:
            b = r1[ib]
            ff0, fe0 = b['f_fc'], b['f_ely']
            # On pince la frontiere : autour du meilleur faisable, et un cran plus bas en
            # coef_ely (pour tester si la frontiere autorise un coef_ely encore plus petit).
            ff_set = sorted({round(max(0.05, ff0 + d * REFINE_STEP), 4) for d in (-1, 0, 1)})
            fe_set = sorted({round(max(0.05, fe0 + d * REFINE_STEP), 4) for d in (-2, -1, 0, 1)})
            seen = {(r['f_fc'], r['f_ely']) for r in r1}
            p2 = [(ff, fe) for ff in ff_set for fe in fe_set if (ff, fe) not in seen]
            if p2:
                r2 = run_pool(p2, f"PHASE 2 — raffinement autour de ({ff0:.3f},{fe0:.3f})")
        else:
            print("\n[!] Aucun point FAISABLE en phase 1 — pas de raffinement.", flush=True)

    allres = r1 + r2

    # Reperes
    ref = min(r1, key=lambda r: (abs(r['f_fc'] - BASE_F_FC) + abs(r['f_ely'] - BASE_F_ELY)))
    ib = best_feasible(allres)
    best = allres[ib] if ib is not None else None

    # -------- Sauvegarde txt --------
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# RB2(SoH) min deg s.c. LPSP<={LPSP_MAX}% — dim FIXE — N~{allres[0]['N']:.2f}ans "
                f"BoP_fixe={allres[0]['bop']:.2f}k€ (constant)\n")
        f.write(f"# base=({BASE_F_FC},{BASE_F_ELY})\n")
        f.write("phase;coef_fc;coef_ely;faisable;LPSP_%;deg_k€;NPC_k€;E_nonservie_kWh;"
                "vie_bat;vie_fc;vie_ely\n")
        for ph, rs in (("1", r1), ("2", r2)):
            for r in rs:
                f.write(f"{ph};{r['f_fc']:.3f};{r['f_ely']:.3f};{int(r['feasible'])};"
                        f"{r['lpsp']:.4f};{r['deg']:.2f};{r['npc']:.2f};{r['e_unserved']:.0f};"
                        f"{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")

    # -------- Plot : (LPSP %, deg k€) + contrainte verticale --------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lp = np.array([r['lpsp'] for r in allres]); dg = np.array([r['deg'] for r in allres])
    feas = np.array([r['feasible'] for r in allres])
    ax.scatter(lp[~feas], dg[~feas], c="lightgray", s=55, label="LPSP > contrainte")
    sc = ax.scatter(lp[feas], dg[feas], c=[r['f_ely'] for r, k in zip(allres, feas) if k],
                    cmap="viridis", s=80, label="faisable")
    ax.axvline(LPSP_MAX, color="red", ls="--", lw=1.5, label=f"LPSP_max = {LPSP_MAX:.3f}% (RB2)")
    ax.scatter(ref['lpsp'], ref['deg'], c="black", marker="s", s=130, zorder=5,
               label=f"base ({BASE_F_FC},{BASE_F_ELY})")
    if best is not None:
        ax.scatter(best['lpsp'], best['deg'], c="crimson", marker="*", s=280, zorder=6,
                   label=f"optimum (min deg faisable)")
    plt.colorbar(sc, label="coef_ely (fraction P_ely_max)")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût total de dégradation [k€]")
    ax.set_title("RB2(SoH) — min dégradation sous contrainte LPSP (25 ans)\n"
                 "zone admissible = à gauche de la ligne rouge")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # -------- Resume console --------
    print("\n" + "=" * 74)
    print(f"BASE  (coef_fc,coef_ely)=({ref['f_fc']:.3f},{ref['f_ely']:.3f}) : "
          f"LPSP {ref['lpsp']:.3f}%  {'(faisable)' if ref['feasible'] else '(INFAISABLE)'}  "
          f"deg {ref['deg']:.1f}k€  | vie ELY {fmt_life(ref['life_ely'])} ans")
    print("=" * 74)
    if best is None:
        print("[!] AUCUN couple FAISABLE (LPSP <= contrainte) dans la grille exploree.")
        cl = min(allres, key=lambda r: r['lpsp'])
        print(f"    Point le plus proche : ({cl['f_fc']:.3f},{cl['f_ely']:.3f}) "
              f"LPSP {cl['lpsp']:.3f}%  deg {cl['deg']:.1f}k€")
        print(f"    -> Elargir GRID_F_FC / GRID_F_ELY ou relacher LPSP_MAX.")
    else:
        dd = (best['deg'] - ref['deg']) / ref['deg'] * 100 if ref['deg'] else float('nan')
        print(f">>> OPTIMUM (min coût de dégradation, LPSP <= {LPSP_MAX:.3f}%) :")
        print(f"      coef_fc = {best['f_fc']:.3f}   coef_ely = {best['f_ely']:.3f}")
        print(f"      LPSP {best['lpsp']:.3f}%   deg {best['deg']:.1f}k€  ({dd:+.1f}% vs base)   "
              f"NPC {best['npc']:.1f}k€")
        print(f"      vie  BAT {fmt_life(best['life_bat'])} | FC {fmt_life(best['life_fc'])} | "
              f"ELY {fmt_life(best['life_ely'])} ans")
        # Top 3 faisables par deg croissant (pour voir la marge de la frontiere)
        feas_sorted = sorted([r for r in allres if r['feasible']], key=lambda r: r['deg'])[:5]
        print("\n    Meilleurs faisables (deg croissant) :")
        for r in feas_sorted:
            print(f"      ({r['f_fc']:.3f},{r['f_ely']:.3f}) : LPSP {r['lpsp']:.3f}%  "
                  f"deg {r['deg']:.1f}k€  vie ELY {fmt_life(r['life_ely'])}")
        print("\n    Pour appliquer : dans RB2(SoH)/get_optimal_action_RB.py, ne change que")
        print(f"       0.475  ->  {best['f_fc']:.3f}      (ligne P_fc_set)")
        print(f"       0.33   ->  {best['f_ely']:.3f}      (ligne P_ely_set)")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
