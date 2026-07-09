"""
batch_optimize_rb2_voll.py
==========================
MEILLEURS SETPOINTS de RB2 au sens du COUT UNIFIE (degradation + LPSP financiarise
par la VOLL definie dans Robustesse/Analyse_sensibilite).

Probleme (minimisation libre, un seul objectif) :
    min   cout_total(coef_fc, coef_ely) = deg [kEUR]  +  clps [kEUR]
ou
    deg   = cout total de degradation FC+ELY+BAT          (sens_common.metrics)
    clps  = cout financier de l'energie non servie,        (sens_common.lps_cost_keur)
            evalue PAS A PAS : sum_t VoLL(LPS(t)) * E_unserved(t),
            avec les paliers VoLL de voll_common.VOLL_TIERS.
    cout_total = voll_common.total_cost_keur(lpsp, deg, clps).

On ne fait varier que les DEUX setpoints de RB2 :
    P_fc_set  = coef_fc  * FC['P_fc_max']      (0.475 dans le fichier, SANS le *0.9)
    P_ely_set = coef_ely * ELY['P_ely_max']    (0.33  dans le fichier, SANS le *0.9)
Le reste de la regle RB2 (plafond H2 -> batterie, gestion des defaillances) est
RIGOUREUSEMENT IDENTIQUE a RB2/get_optimal_action_RB.py.

IMPORTANT — coherence avec l'analyse de sensibilite : on REUTILISE telles quelles
metrics(), lps_cost_keur() et total_cost_keur() des modules de Analyse_sensibilite.
Aucune redefinition de la VOLL ici : si tu changes VOLL_TIERS la-bas, l'optimum suit.

Dimensionnement FIXE (valeurs de Init). Horizon = celui de main_init_and_loop
(25 ans) : on NE touche pas a SIM['Tend'] (l'energie de reference de la VOLL y est
calee).

Lancer depuis le dossier Vieillissement8, comme main.py :
    python batch_optimize_rb2_voll.py
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# --- Chemins : ce dossier (Common) + Analyse_sensibilite (sens_common/voll_common) ---
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
_SENS = os.path.normpath(os.path.join(HERE, os.pardir, "Analyse_sensibilite"))
if _SENS not in sys.path:
    sys.path.insert(0, _SENS)

# On importe le cout UNIFIE directement depuis les codes d'analyse de sensibilite
# -> definitions strictement identiques (LPSP, deg, clps pas-a-pas, paliers VoLL).
from sens_common import I, init_and_run_loop, metrics, lps_cost_keur   # noqa: E402
from voll_common import total_cost_keur, voll_eur_per_kwh, VOLL_TIERS  # noqa: E402
from Common.get_lol import get_lol                                     # noqa: E402

# ============================ CONFIGURATION ============================
BASE_F_FC, BASE_F_ELY = 0.475, 0.33   # setpoints de reference (= valeurs du fichier RB2)

# --- PHASE 1 : grille de balayage des deux setpoints (fraction de P_max) ---
GRID_F_FC  = [0.20, 0.30, 0.40, 0.475, 0.55, 0.65]
GRID_F_ELY = [0.15, 0.25, 0.33, 0.40, 0.50, 0.60]
# -> Phase 1 = 6 x 6 = 36 simulations (le couple de base y est inclus).

# --- PHASE 2 : raffinement local autour du meilleur point (cout total min) ---
REFINE      = True
REFINE_STEP = 0.025                   # pas fin (+/- sur chaque axe)
# -> Phase 2 = jusqu'a 3 x 3 = 9 simulations (total <= ~45).

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_rb2_voll_25y.txt"
OUT_PDF = "optim_rb2_voll_25y.pdf"
# ======================================================================


def make_rb2(coef_fc, coef_ely):
    """Regle RB2 IDENTIQUE a RB2/get_optimal_action_RB.py, parametree par les deux
    SEULS setpoints (sans le *0.9 : voir consigne). Tout le reste est inchange."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        # Setpoints (les deux variables d'optimisation)
        P_fc_set  = coef_fc  * I.FC['P_fc_max']
        P_ely_set = coef_ely * I.ELY['P_ely_max']

        # Plafonds imposes par l'etat du reservoir H2 sur ce pas de temps
        dt_h         = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff'] * I.CONV['eta'] * 1000
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


def _lifetimes(data):
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        out.append(float(rep[0] * yr) if len(rep) > 0 else None)
    return out  # [bat, fc, ely]


def evaluate(params):
    """Worker : (coef_fc, coef_ely) -> resultats. Dimensionnement FIXE (Init).
    OBJECTIF = cout total unifie = deg + clps (kEUR), via les fonctions de
    l'analyse de sensibilite (definitions strictement identiques)."""
    coef_fc, coef_ely = params
    data = init_and_run_loop(make_rb2(coef_fc, coef_ely))

    lpsp, deg = metrics(data)                       # LPSP %, deg kEUR (== batch_pareto)
    clps      = lps_cost_keur(data)                 # cout LPS pas-a-pas kEUR (VoLL)
    total     = total_cost_keur(lpsp, deg, clps)    # = deg + clps

    lifb, liff, life = _lifetimes(data)
    return dict(coef_fc=coef_fc, coef_ely=coef_ely,
                lpsp=float(lpsp), deg=float(deg), clps=float(clps), total=float(total),
                life_bat=lifb, life_fc=liff, life_ely=life)


def fmt_life(x):
    return f"{x:5.1f}" if x is not None else " >hor"


def run_pool(param_list, title):
    print(f"\n--- {title} : {len(param_list)} simulations ({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(param_list)}] f=({r['coef_fc']:.3f},{r['coef_ely']:.3f}) -> "
                  f"LPSP {r['lpsp']:6.3f}% | deg {r['deg']:6.1f}  clps {r['clps']:6.1f}  "
                  f"=> TOTAL {r['total']:7.1f} k€ | vie B/F/E {fmt_life(r['life_bat'])}/"
                  f"{fmt_life(r['life_fc'])}/{fmt_life(r['life_ely'])}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    tiers = " ; ".join(
        (f"LPSP<{int(t*100)}%:{v}" if t is not None else f">=:{v}") for t, v in VOLL_TIERS)
    print("=== RB2 : meilleurs setpoints au sens du COUT UNIFIE (deg + LPSP*VoLL) — "
          "dim FIXE, 25 ans ===", flush=True)
    print(f"    Dimensionnement (Init) : FC P_max={I.FC['P_fc_max']/1000:.2f} kW | "
          f"ELY P_max={I.ELY['P_ely_max']/1000:.2f} kW", flush=True)
    print(f"    Paliers VoLL [EUR/kWh] : {tiers}", flush=True)
    print(f"    Couple de reference    : (coef_fc, coef_ely) = ({BASE_F_FC}, {BASE_F_ELY})", flush=True)

    # -------- PHASE 1 : grille (+ base garantie presente) --------
    p1 = [(ff, fe) for ff in GRID_F_FC for fe in GRID_F_ELY]
    if (BASE_F_FC, BASE_F_ELY) not in p1:
        p1.append((BASE_F_FC, BASE_F_ELY))
    r1 = run_pool(p1, "PHASE 1 — grille")

    # -------- PHASE 2 : raffinement local autour du meilleur (cout total min) --------
    r2 = []
    if REFINE:
        b = min(r1, key=lambda r: r['total'])
        ff0, fe0 = b['coef_fc'], b['coef_ely']
        ff_set = sorted({round(max(0.02, ff0 + d * REFINE_STEP), 4) for d in (-1, 0, 1)})
        fe_set = sorted({round(max(0.02, fe0 + d * REFINE_STEP), 4) for d in (-1, 0, 1)})
        seen = {(r['coef_fc'], r['coef_ely']) for r in r1}
        p2 = [(ff, fe) for ff in ff_set for fe in fe_set if (ff, fe) not in seen]
        if p2:
            r2 = run_pool(p2, f"PHASE 2 — raffinement autour de ({ff0:.3f},{fe0:.3f})")

    allres = r1 + r2

    # Reperes
    ref  = min(r1, key=lambda r: (abs(r['coef_fc'] - BASE_F_FC) + abs(r['coef_ely'] - BASE_F_ELY)))
    best = min(allres, key=lambda r: r['total'])

    # -------- Sauvegarde txt (trie par cout total croissant) --------
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# RB2 min cout unifie (deg+clps) — dim FIXE — 25 ans — VoLL_TIERS={VOLL_TIERS}\n")
        f.write(f"# base=({BASE_F_FC},{BASE_F_ELY})\n")
        f.write("phase;coef_fc;coef_ely;LPSP_%;deg_kEUR;clps_kEUR;total_kEUR;"
                "vie_bat;vie_fc;vie_ely\n")
        rows = ([("1", r) for r in r1] + [("2", r) for r in r2])
        for ph, r in sorted(rows, key=lambda x: x[1]['total']):
            f.write(f"{ph};{r['coef_fc']:.3f};{r['coef_ely']:.3f};{r['lpsp']:.4f};"
                    f"{r['deg']:.2f};{r['clps']:.2f};{r['total']:.2f};"
                    f"{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")

    # -------- Plot : LPSP (x) vs deg (y), couleur = cout total, etoile = optimum --------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lp  = np.array([r['lpsp'] for r in allres])
    dg  = np.array([r['deg'] for r in allres])
    tot = np.array([r['total'] for r in allres])
    sc = ax.scatter(lp, dg, c=tot, cmap="viridis_r", s=80, label="setpoints testes")
    ax.scatter(ref['lpsp'], ref['deg'], c="black", marker="s", s=130, zorder=5,
               label=f"base ({BASE_F_FC},{BASE_F_ELY})")
    ax.scatter(best['lpsp'], best['deg'], c="crimson", marker="*", s=300, zorder=6,
               label="optimum (cout total min)")
    plt.colorbar(sc, label="Cout total unifie [k€]  (deg + LPSP*VoLL)")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût total de dégradation [k€]")
    ax.set_title("RB2 — setpoints minimisant le coût unifié (deg + LPSP financiarisé VoLL)\n"
                 "25 ans, dimensionnement fixe")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # -------- Resume console --------
    print("\n" + "=" * 78)
    print(f"BASE  (coef_fc,coef_ely)=({ref['coef_fc']:.3f},{ref['coef_ely']:.3f}) : "
          f"LPSP {ref['lpsp']:.3f}%  deg {ref['deg']:.1f}  clps {ref['clps']:.1f}  "
          f"=> TOTAL {ref['total']:.1f} k€")
    print("=" * 78)
    dd = (best['total'] - ref['total']) / ref['total'] * 100 if ref['total'] else float('nan')
    print(">>> OPTIMUM (cout unifie deg + LPSP*VoLL minimal) :")
    print(f"      coef_fc = {best['coef_fc']:.3f}   ->  P_fc_set  = "
          f"{best['coef_fc']*I.FC['P_fc_max']:.0f} W")
    print(f"      coef_ely= {best['coef_ely']:.3f}   ->  P_ely_set = "
          f"{best['coef_ely']*I.ELY['P_ely_max']:.0f} W")
    print(f"      LPSP {best['lpsp']:.3f}%   deg {best['deg']:.1f} k€   "
          f"clps {best['clps']:.1f} k€   => TOTAL {best['total']:.1f} k€  ({dd:+.1f}% vs base)")
    print(f"      vie  BAT {fmt_life(best['life_bat'])} | FC {fmt_life(best['life_fc'])} | "
          f"ELY {fmt_life(best['life_ely'])} ans")

    top = sorted(allres, key=lambda r: r['total'])[:5]
    print("\n    Top 5 (cout total croissant) :")
    for r in top:
        print(f"      ({r['coef_fc']:.3f},{r['coef_ely']:.3f}) : LPSP {r['lpsp']:.3f}%  "
              f"deg {r['deg']:.1f}  clps {r['clps']:.1f}  TOTAL {r['total']:.1f} k€")

    print("\n    Pour appliquer : dans RB2/get_optimal_action_RB.py, ne change que")
    print(f"       P_fc_set  = 0.475 ... ->  P_fc_set  = {best['coef_fc']:.3f} * FC['P_fc_max']")
    print(f"       P_ely_set = 0.33  ... ->  P_ely_set = {best['coef_ely']:.3f} * ELY['P_ely_max']")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
