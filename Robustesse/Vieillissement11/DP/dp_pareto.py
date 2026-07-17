"""
=============================================================================
dp_pareto.py -- FRONTIERE DE PARETO degradation <-> fiabilite (EMS optimal PD)
=============================================================================

Trace la frontiere de Pareto entre les DEUX axes du cout :
    axe 1 = degradation des composants   [kEUR]
    axe 2 = energie non servie (EENS)     [kWh]   (== fiabilite ; LPSP)

PRINCIPE
--------
Dans la PD, le poids relatif des deux axes EST la valeur du cout de l'energie
non servie (VOLL) utilisee DANS LA RESOLUTION (dp_core.soc_step_and_lpsp ->
cost_bl = cout_batterie + VOLL*energie_non_servie). En faisant varier ce poids
  epsilon := VOLL_resolution  (EUR/kWh)
on obtient une politique PD differente a chaque fois :
  - epsilon petit  -> la PD ignore presque la fiabilite -> MIN degradation,
                      LPSP eleve   (coin "pas cher / peu fiable")
  - epsilon grand  -> la PD paie cher tout delestage     -> MIN LPSP,
                      degradation plus elevee (coin "cher / tres fiable")
Chaque politique est ENSUITE EVALUEE sur la vraie boucle (vieillissement,
get_lol, metriques EXACTS) -> un point (deg_kEUR, EENS_kWh) reel. Le balayage
de epsilon trace la frontiere.

NB cap. mesocentre : chaque resolution PD est MONO-THREAD (cf run corrige :
user~=real). On lance donc tous les epsilon EN PARALLELE, 1 par coeur
(OMP_NUM_THREADS=1 force dans le .slurm pour ne pas sur-souscrire).

V2 (defaut, DP_PARETO_V2=0 pour revenir au v1) : la PD sequentielle utilise
la projection du vieillissement (seuils de degradation prices au Pmax projete
fin d'annee) et le rollout exact (choix du controle par minimisation du cout
exact du pas au vieillissement courant + V(t+1) interpole). Motivation : au
run v1 (job 210452), la derive intra-annuelle du seuil 0.8*P_fc_max coutait
~10 kEUR de degradation FC "high" non pricee (57 000 h, 10 remplacements FC
sur 25 ans a eps=3) + ~2 kEUR d'irreversible ELY au-dessus du genou 30% --
c'est exactement ce qui permettait a RB2(SoH_all)/RB2(SoH_all+Pred) de passer
sous le front. Sorties v2 taggees *_v2 (les resultats v1 ne sont pas ecrases).

RAPPEL THESE : le VoLL=3 reste la valeur de REFERENCE (colonne UNIF@VoLL3 du
tableau, comparable au run principal). epsilon est ICI un parametre de
sensibilite explicite (la frontiere de Pareto demandee), pas un changement du
VoLL de reference.

Sortie : results/dp_pareto_<grid>.txt          (tableau lisible)
         results/dp_pareto_<grid>.npz          (front compact : eps, deg, eens, lpsp, ...)
         results/dp_pareto_traj_<grid>.npz      (trajectoires lol/SoH par eps, pour analyse fine)

Lancer : python dp_pareto.py            (smoke : 3 eps, horizon court)
         python dp_pareto.py full       (15 eps, 25 ans -- pour le mesocentre)
=============================================================================
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))            # Vieillissement8/
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', 'RB2')))    # RB2/
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', '..', 'Analyse_sensibilite')))
os.environ.setdefault('GENIAL_DATA_DIR', '/home/theo/Documents/Doctorat/Data')

import dp_core as dpc
from dp_core import N_YEAR, TS_H
from dp_aging import DPPolicy
from sens_common import metrics as sens_metrics            # (LPSP %, deg kEUR), SANS VoLL
from Common.main_init_and_loop import init_and_run_loop
from Common.reliability_metrics import compute_reliability_metrics
from get_optimal_action_RB import get_optimal_action_RB

# ---------------------------------------------------------------------------
# Grille de epsilon (= VOLL de resolution, EUR/kWh). Dense au coude (bas), 3 inclus.
# Le run 210452 (v1) montrait un TROU realise entre eps=0.1 (LPSP 5.6%) et
# eps=0.2 (LPSP 1.8%) -- precisement la zone ou RB2(SoH_all) et RB2(SoH_all+Pred)
# passaient sous la corde du front. On densifie donc [0.1, 0.35].
# ---------------------------------------------------------------------------
EPSILONS_FULL  = [0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75,
                  1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0]
EPSILONS_SMOKE = [0.2, 3.0, 30.0]

VOLL_REF = 3.0   # VoLL de reference these (pour la colonne UNIF@VoLL3 comparable)

# --- Variante v2 : projection du vieillissement + rollout exact (dp_aging) ---
# Corrige la derive intra-annuelle des seuils de degradation (P_high FC, genou
# 30% ELY, capacite batterie) qui coutait ~10 kEUR de "FC high" silencieux au
# run v1 eps=3 (cf. decomposition get_cost_fc sur dp_aging_25y_51x51.npz).
# DP_PARETO_V2=0 pour retrouver exactement le comportement v1 (A/B).
DP_V2 = os.environ.get('DP_PARETO_V2', '1') != '0'


# ---------------------------------------------------------------------------
# Metriques realisees a partir d'une trajectoire (independantes de epsilon)
# ---------------------------------------------------------------------------
def realized_metrics(data):
    """LPSP %, deg kEUR (officiels), EENS kWh, demande totale kWh -- depuis la traj."""
    _, deg = sens_metrics(data)                          # kEUR (sans VoLL)
    rel = compute_reliability_metrics(data)
    return rel['lpsp_pct'], deg, rel['eens_kwh'], rel['load_energy_kwh']


def n_repl(s):
    return int(np.sum(np.diff(s) > 0.05))


# ---------------------------------------------------------------------------
# Un point de la frontiere : resout + evalue la PD-seq pour un epsilon donne
# (fonction top-level -> picklable par ProcessPoolExecutor ; fork herite des
#  donnees LOAD/PV deja chargees ; dpc.VOLL mute est ISOLE dans le process fils)
# ---------------------------------------------------------------------------
def run_one(args):
    eps, Ns, Nh, n_fc, n_ely, n_years = args
    t0 = time.time()
    dpc.VOLL = float(eps)              # <-- poids fiabilite DANS la resolution PD
    pol = DPPolicy(Ns, Nh, n_fc, n_ely, recompute='yearly', verbose=False,
                   aging_proj=DP_V2, rollout=DP_V2)
    if n_years and n_years != 25:
        data = init_and_run_loop(pol, n_years=n_years)   # smoke (Common modifie)
    else:
        data = init_and_run_loop(pol)                    # 25 ans (defaut)
    lpsp, deg, eens, demand = realized_metrics(data)
    unif3 = deg + VOLL_REF * eens / 1000.0               # kEUR @ VoLL=3 reference
    out = dict(
        eps=float(eps), lpsp=lpsp, deg=deg, eens_kwh=eens, demand_kwh=demand,
        unif3=unif3,
        soh_bat=float(data['SoH_bat'][-1]), soh_fc=float(data['SoH_fc'][-1]),
        soh_ely=float(data['SoH_ely'][-1]),
        repl_bat=n_repl(data['SoH_bat']), repl_fc=n_repl(data['SoH_fc']),
        repl_ely=n_repl(data['SoH_ely']),
        n_rebuild=pol.n_rebuild, sec=time.time() - t0,
        # trajectoires pour analyse fine (lol + SoH ; load/pv stockes une fois a part)
        _lol=data['lol_tab'].astype(np.float32),
        _soh_bat=data['SoH_bat'].astype(np.float32),
        _soh_fc=data['SoH_fc'].astype(np.float32),
        _soh_ely=data['SoH_ely'].astype(np.float32),
        _E_h2=data['E_h2'].astype(np.float32),
    )
    return out


# ---------------------------------------------------------------------------
def main():
    full = len(sys.argv) > 1 and sys.argv[1] == 'full'
    if full:
        EPS = EPSILONS_FULL
        Ns, Nh, n_fc, n_ely, n_years = 51, 51, 10, 50, 25
    else:
        EPS = EPSILONS_SMOKE
        Ns, Nh, n_fc, n_ely, n_years = 25, 25, 7, 24, 6     # smoke (Common doit avoir n_years)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(EPS)))
    n_workers = max(1, min(n_workers, len(EPS)))

    print("=" * 70)
    print(f" FRONTIERE DE PARETO  deg <-> EENS   ({'FULL 25 ans' if full else 'SMOKE'})")
    print(f" grille {Ns}x{Nh}  n_fc={n_fc} n_ely={n_ely}  | {len(EPS)} epsilon | "
          f"{n_workers} workers paralleles")
    print(f" epsilon (EUR/kWh) = {EPS}")
    print(f" variante : {'V2 (projection vieillissement + rollout)' if DP_V2 else 'V1 (legacy)'}")
    print("=" * 70, flush=True)

    out_dir = os.path.join(_THIS, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{n_years}y_{Ns}x{Nh}" + ("_v2" if DP_V2 else "")

    # --- RB2 : point de reference unique (independant de epsilon) -------------
    t0 = time.time()
    data_rb = init_and_run_loop(get_optimal_action_RB) if (n_years == 25) \
        else init_and_run_loop(get_optimal_action_RB, n_years=n_years)
    rb_lpsp, rb_deg, rb_eens, rb_demand = realized_metrics(data_rb)
    rb_unif3 = rb_deg + VOLL_REF * rb_eens / 1000.0
    print(f"[RB2] LPSP {rb_lpsp:.4f}%  deg {rb_deg:.3f} kE  EENS {rb_eens:.1f} kWh  "
          f"UNIF@3 {rb_unif3:.3f} kE   ({time.time()-t0:.0f}s)", flush=True)

    # --- balayage epsilon EN PARALLELE --------------------------------------
    jobs = [(e, Ns, Nh, n_fc, n_ely, n_years) for e in EPS]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[eps={r['eps']:<6.3g}] LPSP {r['lpsp']:.4f}%  deg {r['deg']:.3f} kE  "
                  f"EENS {r['eens_kwh']:8.1f} kWh  UNIF@3 {r['unif3']:.3f} kE  "
                  f"(rebuild {r['n_rebuild']}, {r['sec']:.0f}s)", flush=True)
            _write_partial(out_dir, tag, results,
                           (rb_lpsp, rb_deg, rb_eens, rb_unif3))   # sauvegarde incrementale

    results.sort(key=lambda d: d['eps'])

    # --- ecriture finale : tableau + npz front compact + npz trajectoires ----
    _write_partial(out_dir, tag, results, (rb_lpsp, rb_deg, rb_eens, rb_unif3))

    eps_a = np.array([r['eps'] for r in results])
    def col(k): return np.array([r[k] for r in results])
    np.savez_compressed(
        os.path.join(out_dir, f"dp_pareto_{tag}.npz"),
        eps=eps_a, lpsp=col('lpsp'), deg_keur=col('deg'), eens_kwh=col('eens_kwh'),
        unif3_keur=col('unif3'), soh_bat=col('soh_bat'), soh_fc=col('soh_fc'),
        soh_ely=col('soh_ely'), repl_bat=col('repl_bat'), repl_fc=col('repl_fc'),
        repl_ely=col('repl_ely'),
        nondominated=_nondominated_mask(col('eens_kwh'), col('deg')),
        RB2_lpsp=rb_lpsp, RB2_deg_keur=rb_deg, RB2_eens_kwh=rb_eens,
        RB2_unif3_keur=rb_unif3, demand_kwh=rb_demand)

    np.savez_compressed(
        os.path.join(out_dir, f"dp_pareto_traj_{tag}.npz"),
        eps=eps_a,
        P_dc_load=data_rb['P_dc_load'].astype(np.float32),   # entree commune (1 fois)
        P_dc_pv=data_rb['P_dc_pv'].astype(np.float32),
        RB2_lol=data_rb['lol_tab'].astype(np.float32),
        **{f"lol_{i}": r['_lol'] for i, r in enumerate(results)},
        **{f"soh_bat_{i}": r['_soh_bat'] for i, r in enumerate(results)},
        **{f"soh_fc_{i}": r['_soh_fc'] for i, r in enumerate(results)},
        **{f"soh_ely_{i}": r['_soh_ely'] for i, r in enumerate(results)},
        **{f"E_h2_{i}": r['_E_h2'] for i, r in enumerate(results)})

    print(f"\n Resultats -> {out_dir}/dp_pareto_{tag}.txt (+ .npz, + _traj.npz)", flush=True)


def _nondominated_mask(eens, deg):
    """Masque des points NON-DOMINES dans le plan (EENS, deg) (minimiser les 2).
    Les points domines restent des politiques PD valides mais ne font pas
    partie du front (on les trace en creux / on ne relie que les non-domines)."""
    eens = np.asarray(eens); deg = np.asarray(deg)
    n = len(eens)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (eens <= eens[i]) & (deg <= deg[i]) \
            & ((eens < eens[i]) | (deg < deg[i]))
        if dom.any():
            mask[i] = False
    return mask


def _write_partial(out_dir, tag, results, rb):
    """Ecrit le tableau lisible (re-ecrit a chaque point fini -> resultats partiels surs)."""
    rb_lpsp, rb_deg, rb_eens, rb_unif3 = rb
    rs = sorted(results, key=lambda d: d['eps'])
    nd = _nondominated_mask([r['eens_kwh'] for r in rs], [r['deg'] for r in rs])
    L = ["=" * 78,
         f" FRONTIERE DE PARETO  degradation <-> EENS   (25 ans, PD sequentielle"
         f"{', V2 proj+rollout' if DP_V2 else ''})",
         f" epsilon = poids fiabilite (VoLL) DANS la resolution PD [EUR/kWh]",
         f" '*' = point DOMINE (exclu du front non-domine)",
         "=" * 78,
         f" {'eps':>6} {'LPSP%':>8} {'deg_kE':>8} {'EENS_kWh':>10} {'UNIF@3_kE':>10}"
         f" {'SoHbat':>7} {'SoHfc':>6} {'SoHely':>6} {'rebuild':>8}"]
    for r, ok in zip(rs, nd):
        L.append(f" {r['eps']:6.3g} {r['lpsp']:8.4f} {r['deg']:8.3f} {r['eens_kwh']:10.1f}"
                 f" {r['unif3']:10.3f} {r['soh_bat']:7.3f} {r['soh_fc']:6.3f}"
                 f" {r['soh_ely']:6.3f} {r['n_rebuild']:8d}" + ("" if ok else "  *"))
    L += ["-" * 78,
          f" {'RB2':>6} {rb_lpsp:8.4f} {rb_deg:8.3f} {rb_eens:10.1f} {rb_unif3:10.3f}"
          f"   (reference, hors PD)",
          "-" * 78,
          " Lecture : eps croissant -> la PD privilegie la fiabilite (EENS v) au prix",
          " de la degradation (deg ^). Le coude = meilleur compromis. UNIF@3 = cout",
          " unifie a VoLL=3 (reference these), comparable au run principal."]
    with open(os.path.join(out_dir, f"dp_pareto_{tag}.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
