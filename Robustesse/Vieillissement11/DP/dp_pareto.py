"""Frontiere degradation-fiabilite de la PD sequentielle V11 (nominal p=2).

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

Le cout interne et le reporting utilisent le dommage permanent V11, le ledger
corrige et la metrique de fiabilite commune. Les sorties legacy de l'ancien
modele ne sont jamais ecrasees.

RAPPEL THESE : le VoLL=3 reste la valeur de REFERENCE (colonne UNIF@VoLL3 du
tableau, comparable au run principal). epsilon est ICI un parametre de
sensibilite explicite (la frontiere de Pareto demandee), pas un changement du
VoLL de reference.

Sortie : runs/dp_pareto_v11_p2_<grid>.*

Lancer : python dp_pareto.py            (smoke : 3 eps, horizon court)
         python dp_pareto.py full       (15 eps, 25 ans -- pour le mesocentre)
=============================================================================
"""
import os
import sys
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..')))
os.environ.setdefault('GENIAL_DATA_DIR', '/home/theo/Documents/Doctorat/Data')

import dp_core as dpc
from dp_core import N_YEAR, TS_H
from dp_aging import (
    DPPolicy, realized_metrics as realized_metrics_v11, replacement_counts,
)
from Common.main_init_and_loop import init_and_run_loop
from Common.rb1_policy_v11 import make_rb1_policy_v11
from Common.rb2_policy import make_rb2_policy

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

# Variante online : projection de capacite batterie + rollout au cout V11 courant.
DP_V2 = os.environ.get('DP_PARETO_V2', '1') != '0'


# ---------------------------------------------------------------------------
# Metriques realisees a partir d'une trajectoire (independantes de epsilon)
# ---------------------------------------------------------------------------
def realized_metrics(data):
    """Tuple historique construit a partir des metriques canoniques V11."""
    metrics = realized_metrics_v11(data, voll=VOLL_REF)
    return (metrics['lpsp'], metrics['deg'], metrics['eens_kwh'],
            metrics['demand_kwh'])


# ---------------------------------------------------------------------------
# Un point de la frontiere : resout + evalue la PD-seq pour un epsilon donne
# (fonction top-level -> picklable par ProcessPoolExecutor ; fork herite des
#  donnees LOAD/PV deja chargees ; dpc.VOLL mute est ISOLE dans le process fils)
# ---------------------------------------------------------------------------
def run_one(args):
    eps, Ns, Nh, n_fc, n_ely, n_years, n_iter = args
    t0 = time.time()
    dpc.VOLL = float(eps)              # <-- poids fiabilite DANS la resolution PD
    pol = DPPolicy(Ns, Nh, n_fc, n_ely, n_iter=n_iter,
                   recompute='yearly', verbose=False,
                   aging_proj=DP_V2, rollout=DP_V2)
    if n_years and n_years != 25:
        data = init_and_run_loop(pol, n_years=n_years)   # smoke (Common modifie)
    else:
        data = init_and_run_loop(pol)                    # 25 ans (defaut)
    lpsp, deg, eens, demand = realized_metrics(data)
    replacements = replacement_counts(data)
    unif3 = deg + VOLL_REF * eens / 1000.0               # kEUR @ VoLL=3 reference
    out = dict(
        eps=float(eps), lpsp=lpsp, deg=deg, eens_kwh=eens, demand_kwh=demand,
        unif3=unif3,
        soh_bat=float(data['SoH_bat'][-1]), soh_fc=float(data['SoH_fc'][-1]),
        soh_ely=float(data['SoH_ely'][-1]),
        repl_bat=replacements['bat'], repl_fc=replacements['fc'],
        repl_ely=replacements['ely'],
        n_rebuild=pol.n_rebuild, sec=time.time() - t0,
        # trajectoires pour analyse fine (lol + SoH ; load/pv stockes une fois a part)
        _lol=data['lol_tab'].astype(np.float32),
        _soh_bat=data['SoH_bat'].astype(np.float32),
        _soh_fc=data['SoH_fc'].astype(np.float32),
        _soh_ely=data['SoH_ely'].astype(np.float32),
        _E_h2=data['E_h2'].astype(np.float32),
        _ledger=data['degradation_ledger'],
    )
    return out


# ---------------------------------------------------------------------------
def main():
    full = len(sys.argv) > 1 and sys.argv[1] == 'full'
    if full:
        EPS = EPSILONS_FULL
        Ns, Nh, n_fc, n_ely, n_years, n_iter = 51, 51, 10, 50, 25, 3
    else:
        EPS = EPSILONS_SMOKE
        Ns, Nh, n_fc, n_ely, n_years, n_iter = 7, 7, 3, 6, 1, 1

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(EPS)))
    n_workers = max(1, min(n_workers, len(EPS)))

    print("=" * 70)
    print(f" FRONTIERE DE PARETO  deg <-> EENS   ({'FULL 25 ans' if full else 'SMOKE'})")
    print(f" grille {Ns}x{Nh}  n_fc={n_fc} n_ely={n_ely}  | {len(EPS)} epsilon | "
          f"{n_workers} workers paralleles")
    print(f" epsilon (EUR/kWh) = {EPS}")
    print(f" modele : V11 p=2, ledger corrige ; "
          f"{'projection capacite + rollout' if DP_V2 else 'lookup annuel'}")
    print("=" * 70, flush=True)

    out_dir = os.path.join(_THIS, "runs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"v11_p2_{n_years}y_{Ns}x{Nh}" + ("_rollout" if DP_V2 else "_lookup")

    # --- references best-vs-best V11 p=2 (independantes de epsilon) ---------
    references = {}
    reference_data = {}
    reference_ledgers = {}
    for label, policy in (
        ("RB1(0.20,0.40)", make_rb1_policy_v11(0.20, 0.40)),
        ("RB2(0.574,0.465)", make_rb2_policy(0.574, 0.465)),
    ):
        t0 = time.time()
        data_ref = (init_and_run_loop(policy) if n_years == 25
                    else init_and_run_loop(policy, n_years=n_years))
        lpsp, deg, eens, demand = realized_metrics(data_ref)
        unif3 = deg + VOLL_REF * eens / 1000.0
        references[label] = (lpsp, deg, eens, unif3, demand)
        reference_data[label] = data_ref
        reference_ledgers[label] = data_ref['degradation_ledger']
        print(f"[{label}] LPSP {lpsp:.4f}%  deg {deg:.3f} kE  "
              f"EENS {eens:.1f} kWh  UNIF@3 {unif3:.3f} kE   "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- balayage epsilon EN PARALLELE --------------------------------------
    jobs = [(e, Ns, Nh, n_fc, n_ely, n_years, n_iter) for e in EPS]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[eps={r['eps']:<6.3g}] LPSP {r['lpsp']:.4f}%  deg {r['deg']:.3f} kE  "
                  f"EENS {r['eens_kwh']:8.1f} kWh  UNIF@3 {r['unif3']:.3f} kE  "
                  f"(rebuild {r['n_rebuild']}, {r['sec']:.0f}s)", flush=True)
            _write_partial(
                out_dir, tag, results, references, reference_ledgers)

    results.sort(key=lambda d: d['eps'])

    # --- ecriture finale : tableau + npz front compact + npz trajectoires ----
    _write_partial(out_dir, tag, results, references, reference_ledgers)

    eps_a = np.array([r['eps'] for r in results])
    def col(k): return np.array([r[k] for r in results])
    np.savez_compressed(
        os.path.join(out_dir, f"dp_pareto_{tag}.npz"),
        eps=eps_a, lpsp=col('lpsp'), deg_keur=col('deg'), eens_kwh=col('eens_kwh'),
        unif3_keur=col('unif3'), soh_bat=col('soh_bat'), soh_fc=col('soh_fc'),
        soh_ely=col('soh_ely'), repl_bat=col('repl_bat'), repl_fc=col('repl_fc'),
        repl_ely=col('repl_ely'),
        nondominated=_nondominated_mask(col('eens_kwh'), col('deg')),
        RB1_lpsp=references['RB1(0.20,0.40)'][0],
        RB1_deg_keur=references['RB1(0.20,0.40)'][1],
        RB1_eens_kwh=references['RB1(0.20,0.40)'][2],
        RB1_unif3_keur=references['RB1(0.20,0.40)'][3],
        RB2_lpsp=references['RB2(0.574,0.465)'][0],
        RB2_deg_keur=references['RB2(0.574,0.465)'][1],
        RB2_eens_kwh=references['RB2(0.574,0.465)'][2],
        RB2_unif3_keur=references['RB2(0.574,0.465)'][3],
        demand_kwh=references['RB1(0.20,0.40)'][4],
        model_id=np.array(dpc.MODEL_ID),
        ely_stress_exponent=np.array(dpc.NOMINAL_ELY_STRESS_EXPONENT))

    np.savez_compressed(
        os.path.join(out_dir, f"dp_pareto_traj_{tag}.npz"),
        eps=eps_a,
        P_dc_load=reference_data['RB1(0.20,0.40)']['P_dc_load'].astype(np.float32),
        P_dc_pv=reference_data['RB1(0.20,0.40)']['P_dc_pv'].astype(np.float32),
        RB1_lol=reference_data['RB1(0.20,0.40)']['lol_tab'].astype(np.float32),
        RB2_lol=reference_data['RB2(0.574,0.465)']['lol_tab'].astype(np.float32),
        **{f"lol_{i}": r['_lol'] for i, r in enumerate(results)},
        **{f"soh_bat_{i}": r['_soh_bat'] for i, r in enumerate(results)},
        **{f"soh_fc_{i}": r['_soh_fc'] for i, r in enumerate(results)},
        **{f"soh_ely_{i}": r['_soh_ely'] for i, r in enumerate(results)},
        **{f"E_h2_{i}": r['_E_h2'] for i, r in enumerate(results)})

    print(f"\n Resultats -> {out_dir}/dp_pareto_{tag}.txt "
          f"(+ .npz, + dp_pareto_traj_{tag}.npz, + _ledgers.json)", flush=True)


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


def _write_partial(out_dir, tag, results, references, reference_ledgers):
    """Ecrit le tableau lisible (re-ecrit a chaque point fini -> resultats partiels surs)."""
    rs = sorted(results, key=lambda d: d['eps'])
    nd = _nondominated_mask([r['eens_kwh'] for r in rs], [r['deg'] for r in rs])
    L = ["=" * 78,
         f" FRONTIERE DE PARETO V11 p=2  degradation <-> EENS  (PD sequentielle"
         f"{', rollout' if DP_V2 else ', lookup'})",
         f" epsilon = poids fiabilite (VoLL) DANS la resolution PD [EUR/kWh]",
         f" '*' = point DOMINE (exclu du front non-domine)",
         "=" * 78,
         f" {'eps':>6} {'LPSP%':>8} {'deg_kE':>8} {'EENS_kWh':>10} {'UNIF@3_kE':>10}"
         f" {'SoHbat':>7} {'SoHfc':>6} {'SoHely':>6} {'rebuild':>8}"]
    for r, ok in zip(rs, nd):
        L.append(f" {r['eps']:6.3g} {r['lpsp']:8.4f} {r['deg']:8.3f} {r['eens_kwh']:10.1f}"
                 f" {r['unif3']:10.3f} {r['soh_bat']:7.3f} {r['soh_fc']:6.3f}"
                 f" {r['soh_ely']:6.3f} {r['n_rebuild']:8d}" + ("" if ok else "  *"))
    L.append("-" * 78)
    for label, (lpsp, deg, eens, unif3, _demand) in references.items():
        L.append(f" {label:>18} {lpsp:8.4f} {deg:8.3f} {eens:10.1f} "
                 f"{unif3:10.3f}   (reference, hors PD)")
    L += ["-" * 78,
          " Lecture : eps croissant -> la PD privilegie la fiabilite (EENS v) au prix",
          " de la degradation (deg ^). Le coude = meilleur compromis. UNIF@3 = cout",
          " unifie a VoLL=3 (reference these), comparable au run principal."]
    with open(os.path.join(out_dir, f"dp_pareto_{tag}.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, f"dp_pareto_{tag}_ledgers.json"), "w") as f:
        json.dump({
            'model_id': dpc.MODEL_ID,
            'ely_stress_exponent': dpc.NOMINAL_ELY_STRESS_EXPONENT,
            'references': reference_ledgers,
            'points': {str(result['eps']): result['_ledger'] for result in rs},
        }, f, indent=2)


if __name__ == "__main__":
    main()
