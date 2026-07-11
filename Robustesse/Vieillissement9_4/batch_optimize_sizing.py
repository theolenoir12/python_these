"""
batch_optimize_sizing.py
========================
Optimisation TECHNICO-ECONOMIQUE du microgrid pour la regle RB2.

Objectifs (a minimiser) :  (LPSP %,  Cout Net Actualise NPC k€)
   NPC = BoP(taille)  +  AF(r,N) * get_cost_total
   - BoP   = somme_composants (CAPEX_complet - cout_remplacement) : paye UNE FOIS,
             proportionnel a la taille -> penalise le surdimensionnement (le BoP de
             la FC/ELY est lourd, celui de la batterie quasi nul : asymetrie reelle).
   - AF    = facteur d'annuite (present-worth) = (1-(1+r)^-N)/(r*N) : actualise le
             cout de degradation etale sur l'horizon (standard, type HOMER).
   - get_cost_total : cout de degradation (modele de vieillissement), inchange.

Variables (5) :
   - 3 TAILLES : batterie [kWh], PEMFC [kW], PEMWE [kW]
   - 2 SETPOINTS exprimes en FRACTION de Pmax (BoL) : f_fc, f_ely  (independants de
     la taille). P_fc_set = f_fc * P_fc_max ; P_ely_set = f_ely * P_ely_max.

Strategie etagee (lisible, ~budget 50-60 sims) :
   PHASE 1 : grille des 3 TAILLES (setpoints fixes a F_FC0/F_ELY0) -> Pareto (LPSP,NPC).
   PHASE 2 : autour du MEILLEUR dimensionnement, raffinement des 2 setpoints.
Le "meilleur" point = min de la somme ponderee des 2 objectifs normalises (50/50 def.).

PV impose (non variable). Aucune contrainte d'autonomie : le role du H2 emerge du cout.
Rien n'est ecrit dans Init : on patche en memoire. Lancer depuis le dossier Vieillissement8 :
   python batch_optimize_sizing.py
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
from Common.cost_fcn_total2 import get_cost_from_ledger
from Common.get_lol import get_lol

# ============================ CONFIGURATION ============================
HORIZON_YEARS = 15
DISCOUNT_RATE = 0.05                  # r reel (essaie 0.03 / 0.08 en sensibilite ; 0 = somme brute)

# Plages de TAILLES (en unites physiques)
GRID_BAT_KWH = [30.0, 50.0, 70.0, 90.0]      # batterie
GRID_FC_KW   = [1.5, 2.5, 3.5]               # PEMFC
GRID_ELY_KW  = [4.0, 8.0, 12.0, 16.0]        # PEMWE
# -> Phase 1 = 4 x 3 x 4 = 48 simulations

# Setpoints (fraction de Pmax) fixes pour la phase 1
F_FC0, F_ELY0 = 0.60, 0.50
# Raffinement phase 2
GRID_F_FC  = [0.4, 0.6, 0.8]
GRID_F_ELY = [0.3, 0.5, 0.7]
# -> Phase 2 = 3 x 3 = 9 simulations  (total ~57)

# --- Monetisation de la LPSP (VOLL, Value of Lost Load) ---
# Cout de l'energie non fournie [EUR/kWh]. Valeur SOCIO-ECONOMIQUE (site insulaire
# isole) : plus haute que le diesel evite (~0.4). Etudes UE/ilots : ~1-30 EUR/kWh.
# Defaut 5 EUR/kWh -> A AJUSTER/justifier selon ta source. (lineaire ici ; la
# convexite "depth-dependent VOLL" viendra dans un 2e temps.)
VOLL_EUR_PER_KWH = 5.0

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT_TXT = "optim_sizing_15y.txt"
OUT_PDF = "optim_sizing_15y.pdf"
# ======================================================================

# Valeurs "par cellule/stack unitaire" figees au demarrage (independantes de la taille)
_BAT_KWH_PER_PAR = I.BAT['series_num'] * I.BAT['Q_bat'] * I.BAT['v_cell_nom'] / 1000.0


def _pmax_fc(n_series):
    im = I.FC['i_fc_max']      # = 238.8252 * n_parallel (fige par Init, lineaire en n_series)
    return im * n_series * (I.FC['E_0'] - I.FC['R'] * im / I.FC['n_parallel']
        - I.A * I.FC['T'] * np.log((im / I.S / I.FC['n_parallel'] + I.j_in) / I.FC['j_0'])
        - I.B * I.FC['T'] * np.log(1 - im / I.S / I.FC['n_parallel'] / I.FC['j_L']))

def _pmax_ely(n_series):
    im = I.ELY['i_ely_max']    # = 732.6 * n_parallel
    return im * n_series * (I.ELY['E_0'] + I.ELY['R'] * im / I.ELY['n_parallel']
        + I.A * I.ELY['T'] * np.log((im / I.S / I.ELY['n_parallel'] + I.j_in) / I.ELY['j_0'])
        + I.B * I.ELY['T'] * np.log(1 - im / I.S / I.ELY['n_parallel'] / I.ELY['j_L']))

_P1_FC  = float(_pmax_fc(1.0))     # puissance par n_series (W)
_P1_ELY = float(_pmax_ely(1.0))


def apply_sizing(bat_kwh, fc_kw, ely_kw):
    """Patche les dicts Init pour le dimensionnement demande et recompute les couts
    derives (Pmax, cost) comme le fait Init. Renvoie (P_fc_max_BoL, P_ely_max_BoL)."""
    # --- batterie ---
    I.BAT['parallel_num'] = bat_kwh / _BAT_KWH_PER_PAR
    I.BAT['cost'] = 0.9 * I.BAT['CAPEX'] * I.BAT['series_num'] * I.BAT['parallel_num'] \
                    * I.BAT['Q_bat'] * I.BAT['v_cell_nom'] / 1000.0
    # --- PEMFC ---
    I.FC['n_series'] = fc_kw * 1000.0 / _P1_FC
    I.FC['P_fc_max'] = _pmax_fc(I.FC['n_series'])
    I.FC['cost'] = I.FC['CAPEX_stack'] * I.FC['P_fc_max'] * 0.9 / 1000.0
    # --- PEMWE ---
    I.ELY['n_series'] = ely_kw * 1000.0 / _P1_ELY
    I.ELY['P_ely_max'] = _pmax_ely(I.ELY['n_series'])
    I.ELY['cost'] = I.ELY['CAPEX_stack'] * I.ELY['P_ely_max'] * 0.9 / 1000.0
    return I.FC['P_fc_max'], I.ELY['P_ely_max']


def make_rb2(P_fc_set, P_ely_set):
    """Regle RB2 identique a l'originale, parametree par les 2 setpoints (W absolus)."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if P_tot_ref_t > 0:
            if P_tot_ref_t > P_fc_set:
                P_dc_bat_t, P_dc_fc_t, P_dc_ely_t = P_tot_ref_t - P_fc_set, P_fc_set, 0
            else:
                P_dc_bat_t, P_dc_fc_t, P_dc_ely_t = P_tot_ref_t, 0, 0
        if P_tot_ref_t < 0:
            if P_tot_ref_t < -P_ely_set:
                P_dc_bat_t, P_dc_fc_t, P_dc_ely_t = P_tot_ref_t + P_ely_set, 0, -P_ely_set
            else:
                P_dc_bat_t, P_dc_fc_t, P_dc_ely_t = P_tot_ref_t, 0, 0
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

_AF = annuity_factor(DISCOUNT_RATE, HORIZON_YEARS)


def _lpsp_unserved(data):
    """Renvoie (LPSP %, energie non fournie [kWh]) sur tout l'horizon."""
    P_dc_load, P_dc_pv, lol = data["P_dc_load"], data["P_dc_pv"], data["lol_tab"]
    Pp = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])      # kW
    Pr = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol)])
    p, rr = np.clip(Pp, 0, None), np.clip(Pr, 0, None)
    unmet = np.clip(p - rr, 0, None)                                          # kW non servi
    lpsp = (unmet.sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    e_unserved = unmet.sum() * I.LOAD['Ts'] / 3600.0                         # kWh sur l'horizon
    return lpsp, e_unserved

def _lifetimes(data):
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        out.append(rep[0] * yr if len(rep) > 0 else None)
    return out  # [bat, fc, ely]


def evaluate(params):
    """Worker : (bat_kwh, fc_kw, ely_kw, f_fc, f_ely) -> resultats."""
    bat_kwh, fc_kw, ely_kw, f_fc, f_ely = params
    I.SIM['Tend'] = HORIZON_YEARS / 10.0 * 3600 * 24 * 365
    Pfc_max, Pely_max = apply_sizing(bat_kwh, fc_kw, ely_kw)
    strat = make_rb2(f_fc * Pfc_max, f_ely * Pely_max)
    data = init_and_run_loop(strat)
    lpsp, e_unserved = _lpsp_unserved(data)
    deg = get_cost_from_ledger(data)
    bop = (I.BAT['CAPEX'] * bat_kwh - I.BAT['cost']) \
        + (I.FC['CAPEX'] * fc_kw - I.FC['cost']) \
        + (I.ELY['CAPEX'] * ely_kw - I.ELY['cost'])
    npc = (bop + _AF * deg) / 1000.0                       # k€ : axe "degradation" du Pareto
    voll_cost = _AF * VOLL_EUR_PER_KWH * e_unserved / 1000.0   # k€ : cout d'indisponibilite actualise
    total_econ = npc + voll_cost                          # k€ : metrique unifiee (sert a choisir l'optimum)
    lifb, liff, life = _lifetimes(data)
    return dict(bat=bat_kwh, fc=fc_kw, ely=ely_kw, f_fc=f_fc, f_ely=f_ely,
                lpsp=float(lpsp), npc=float(npc), bop=bop / 1000.0, deg=deg / 1000.0,
                e_unserved=float(e_unserved), voll_cost=float(voll_cost),
                total_econ=float(total_econ), life_bat=lifb, life_fc=liff, life_ely=life)


def pareto_mask(obj):
    n = len(obj); keep = np.ones(n, bool)
    for i in range(n):
        for k in range(n):
            if k != i and obj[k, 0] <= obj[i, 0] and obj[k, 1] <= obj[i, 1] \
               and (obj[k, 0] < obj[i, 0] or obj[k, 1] < obj[i, 1]):
                keep[i] = False; break
    return keep

def best_econ(res):
    """Index du meilleur point = min du cout total economique (NPC + cout VOLL).
    La VOLL remplace la ponderation arbitraire : c'est elle qui arbitre LPSP vs degradation."""
    return int(np.argmin([r['total_econ'] for r in res]))

def fmt_life(x):
    return f"{x:5.1f}" if x is not None else " >hor"


def run_pool(param_list, title):
    print(f"\n--- {title} : {len(param_list)} simulations ({N_WORKERS} workers) ---", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            print(f"  [{i:2d}/{len(param_list)}] BAT {r['bat']:4.0f}kWh FC {r['fc']:.1f}kW "
                  f"ELY {r['ely']:4.1f}kW f=({r['f_fc']:.2f},{r['f_ely']:.2f}) -> "
                  f"LPSP {r['lpsp']:6.3f}%  NPC {r['npc']:6.1f}  +VOLL {r['voll_cost']:6.1f} "
                  f"= {r['total_econ']:6.1f}k€ | vie B/F/E "
                  f"{fmt_life(r['life_bat'])}/{fmt_life(r['life_fc'])}/{fmt_life(r['life_ely'])}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    print(f"=== Optimisation technico-economique RB2 — horizon {HORIZON_YEARS} ans, "
          f"r={DISCOUNT_RATE:.0%} (AF={_AF:.3f}), VOLL={VOLL_EUR_PER_KWH:.1f} €/kWh ===", flush=True)

    # -------- PHASE 1 : dimensionnement --------
    p1 = [(b, f, e, F_FC0, F_ELY0) for b in GRID_BAT_KWH for f in GRID_FC_KW for e in GRID_ELY_KW]
    r1 = run_pool(p1, "PHASE 1 — dimensionnement (setpoints fixes)")
    obj1 = np.array([[r['lpsp'], r['npc']] for r in r1])     # axes Pareto (% / k€) — CONSERVE
    mask1 = pareto_mask(obj1)                                # front non-domine (LPSP, NPC)
    best1 = best_econ(r1)                                    # optimum = min cout total (VOLL)
    bb = r1[best1]
    print(f"\n>>> Meilleur dimensionnement (phase 1, min cout total) : BAT {bb['bat']:.0f}kWh / "
          f"FC {bb['fc']:.1f}kW / ELY {bb['ely']:.1f}kW  (total {bb['total_econ']:.1f} k€)", flush=True)

    # -------- PHASE 2 : setpoints autour du meilleur dimensionnement --------
    p2 = [(bb['bat'], bb['fc'], bb['ely'], ff, fe) for ff in GRID_F_FC for fe in GRID_F_ELY]
    r2 = run_pool(p2, "PHASE 2 — raffinement setpoints")
    final = r2[best_econ(r2)]

    # -------- Sauvegarde --------
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# horizon={HORIZON_YEARS}ans r={DISCOUNT_RATE} AF={_AF:.4f} "
                f"VOLL={VOLL_EUR_PER_KWH}EUR/kWh\n")
        f.write("phase;BAT_kWh;FC_kW;ELY_kW;f_fc;f_ely;LPSP_%;E_nonservie_kWh;NPC_k€;"
                "VOLL_cost_k€;total_econ_k€;BoP_k€;deg_k€;vie_bat;vie_fc;vie_ely\n")
        for ph, rs in (("1", r1), ("2", r2)):
            for r in rs:
                f.write(f"{ph};{r['bat']:.0f};{r['fc']:.2f};{r['ely']:.2f};{r['f_fc']:.2f};"
                        f"{r['f_ely']:.2f};{r['lpsp']:.4f};{r['e_unserved']:.0f};{r['npc']:.2f};"
                        f"{r['voll_cost']:.2f};{r['total_econ']:.2f};{r['bop']:.2f};{r['deg']:.2f};"
                        f"{r['life_bat']};{r['life_fc']};{r['life_ely']}\n")

    # -------- Plot : on GARDE le Pareto (LPSP %, NPC k€) ; l'optimum economique y est marque --------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(obj1[:, 0], obj1[:, 1], c=[r['ely'] for r in r1], cmap="viridis", s=70)
    ax.scatter(obj1[mask1, 0], obj1[mask1, 1], facecolors="none", edgecolors="red",
               s=140, linewidths=1.6, label="front non-domine (LPSP, NPC)")
    ax.scatter(obj1[best1, 0], obj1[best1, 1], c="crimson", marker="*", s=240,
               zorder=5, label=f"optimum économique (VOLL={VOLL_EUR_PER_KWH:.0f} €/kWh)")
    plt.colorbar(sc, label="Taille PEMWE [kW]")
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("NPC [k€]  (BoP + dégradation actualisée)")
    ax.set_title(f"RB2 — Pareto dimensionnement ({HORIZON_YEARS} ans, r={DISCOUNT_RATE:.0%})\n"
                 f"étoile = min coût total (NPC + VOLL·énergie non servie)")
    ax.grid(True, ls="--", alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight"); plt.close()

    # -------- Resume --------
    print("\n" + "=" * 72)
    print(f"RECOMMANDATION FINALE — min COUT TOTAL (VOLL = {VOLL_EUR_PER_KWH:.1f} €/kWh)")
    print("=" * 72)
    print(f"  Batterie : {final['bat']:.0f} kWh")
    print(f"  PEMFC    : {final['fc']:.2f} kW   (setpoint f_fc = {final['f_fc']:.2f})")
    print(f"  PEMWE    : {final['ely']:.2f} kW   (setpoint f_ely = {final['f_ely']:.2f})")
    print(f"  --> LPSP = {final['lpsp']:.3f} %  ({final['e_unserved']:.0f} kWh non servis sur {HORIZON_YEARS} ans)")
    print(f"  --> COUT TOTAL = {final['total_econ']:.1f} k€  =  NPC {final['npc']:.1f} "
          f"(BoP {final['bop']:.1f} + dég.act {_AF*final['deg']:.1f})  +  VOLL {final['voll_cost']:.1f}")
    print(f"  --> Durees de vie  BAT {fmt_life(final['life_bat'])} | FC {fmt_life(final['life_fc'])}"
          f" | ELY {fmt_life(final['life_ely'])} ans")
    print(f"\nResultats : {OUT_TXT}  |  Figure : {OUT_PDF}")


if __name__ == "__main__":
    main()
