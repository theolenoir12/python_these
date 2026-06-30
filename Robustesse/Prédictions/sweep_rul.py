# -*- coding: utf-8 -*-
"""
sweep_rul.py -- BALAYAGE des parametres de RB2(RUL) : (RUL_ELY_REF, EXP_ELY).
==============================================================================
But : produire le tableau de reglage de RB2(RUL) (analogue du balayage gamma de
RB2(SoH)). Chaque couple (RUL_ELY_REF, EXP_ELY) est evalue sur l'horizon complet
de 25 ans (fige dans Common/main_init_and_loop.py) ; on releve LPSP [%], cout de
degradation [kEUR] et cout total unifie [kEUR] = deg + VoLL * EENS.

Le cas EXP_ELY = 0 donne f_ely = 1 partout -> RB2(RUL) coincide EXACTEMENT avec
RB2 nu (propriete de test-nul) ; il sert de reference et est independant de
RUL_ELY_REF, donc evalue une seule fois.

La strategie n'est PAS modifiee : RUL_ELY_REF et EXP_ELY sont des globales du
module RB2(RUL)/get_optimal_action_RB.py, fixees par run (meme mecanisme que
sens_pred_noise.py qui fixe SIGMA_INJECT_KWH).

Parallelisme : 1 processus Python par couple via ProcessPoolExecutor, dimensionne
par SLURM_CPUS_PER_TASK sur le mesocentre (cf. run_sweep_rul.slurm).

Sorties (dans Predictions/) :
    sweep_rul.csv  -- une ligne par couple (brut, pour re-tracer / re-analyser)
    sweep_rul.txt  -- tableau lisible avec ecarts a RB2 (pour le manuscrit)

Usage :
    python sweep_rul.py                  # balayage complet de la grille ci-dessous
    python sweep_rul.py --replot         # relit le CSV, refait juste le .txt
    sbatch run_sweep_rul.slurm           # sur le mesocentre
"""
import os, sys, csv, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VOLL = 3.0   # EUR/kWh -- coherent avec le manuscrit (cout total unifie)

# --- Grille de balayage (editable) ------------------------------------------
# RUL_ELY_REF : seuil de RUL [jours] sous lequel le derating ELY s'active.
# EXP_ELY     : exposant de la loi puissance (0 = pas de derating = RB2 nu).
REF_GRID = [500.0, 750.0, 1000.0, 1500.0, 2000.0]   # [jours]
EXP_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

OUT_CSV = os.path.join(HERE, "sweep_rul.csv")
OUT_TXT = os.path.join(HERE, "sweep_rul.txt")
STRAT_FOLDER = "RB2(RUL)"


# ============================ CALCUL ============================
def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + folder.replace("(", "").replace(")", ""),
        os.path.join(HERE, folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
    """data -> (lpsp [%], deg [kEUR], eens [kWh sur l'horizon])."""
    from Common.main_init_and_loop import LOAD, BAT, FC, ELY
    from Common.cost_fcn_total2 import get_cost_total
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc = data["alpha_fc"][:-1]; alpha_ely = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)), SoH_bat[~np.isnan(SoH_bat)])
    P_planned = (P_dc_load - P_dc_pv) / 1000.0
    P_real    = (P_dc_load - P_dc_pv) * (1 - lol) / 1000.0
    p, r  = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp  = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    dt_h  = LOAD['Ts'] / 3600.0
    eens  = float(np.clip(p - r, 0, None).sum() * dt_h)            # kWh
    deg   = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely,
                           P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(deg), eens


def evaluate(task):
    """Worker picklable : un couple (ref, exp) -> indicateurs 25 ans."""
    from Common.main_init_and_loop import init_and_run_loop
    s = _load(STRAT_FOLDER)
    s.RUL_ELY_REF = task["ref"]
    s.EXP_ELY     = task["exp"]
    data = init_and_run_loop(s.get_optimal_action_RB)
    lpsp, deg, eens = metrics(data)
    total = deg + VOLL * eens / 1000.0
    return dict(ref=task["ref"], exp=task["exp"],
                lpsp=lpsp, deg=deg, eens=eens, total=total)


def build_tasks():
    """Grille (ref x exp), avec exp=0 (=RB2 nu) evalue une seule fois."""
    tasks, seen0 = [], False
    for exp in EXP_GRID:
        for ref in REF_GRID:
            if exp == 0.0:
                if seen0:
                    continue
                seen0 = True
                tasks.append(dict(ref=REF_GRID[0], exp=0.0))
            else:
                tasks.append(dict(ref=ref, exp=exp))
    return tasks


def compute():
    workers_env = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = int(workers_env) if workers_env else max(1, (os.cpu_count() or 2) - 1)
    tasks = build_tasks()

    print("=" * 78)
    print("BALAYAGE RB2(RUL) : (RUL_ELY_REF, EXP_ELY) -- %d couples, 25 ans, %d workers"
          % (len(tasks), workers))
    print("  REF_GRID =", REF_GRID)
    print("  EXP_GRID =", EXP_GRID, flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, tasks):
            rows.append(r)
            print("  ref=%6.0f exp=%4.2f | LPSP=%.3f%%  deg=%.2f kEUR  total=%.2f kEUR"
                  % (r["ref"], r["exp"], r["lpsp"], r["deg"], r["total"]), flush=True)
    print("  (%d runs en %.0fs)" % (len(tasks), time.time() - t0), flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["RUL_ELY_REF", "EXP_ELY", "LPSP_pct", "deg_kEUR", "EENS_kWh", "total_kEUR"])
        for r in rows:
            w.writerow(["%.0f" % r["ref"], "%.3f" % r["exp"], "%.4f" % r["lpsp"],
                        "%.4f" % r["deg"], "%.2f" % r["eens"], "%.4f" % r["total"]])
    print("  brut -> %s" % OUT_CSV)
    return rows


# ============================ TABLEAU LISIBLE ============================
def write_table(rows):
    # reference = le couple exp=0
    ref_row = next((r for r in rows if r["exp"] == 0.0), None)
    with open(OUT_TXT, "w") as f:
        f.write("# Balayage RB2(RUL) -- (RUL_ELY_REF, EXP_ELY), horizon 25 ans, VoLL=%.1f EUR/kWh\n" % VOLL)
        if ref_row:
            f.write("# Reference RB2 nu (EXP_ELY=0) : LPSP=%.3f%%  deg=%.2f kEUR  total=%.2f kEUR\n"
                    % (ref_row["lpsp"], ref_row["deg"], ref_row["total"]))
        f.write("#\n")
        f.write("RUL_ref[j] ; EXP ; LPSP[%] ; dLPSP[pts] ; deg[kEUR] ; ddeg[%] ; total[kEUR] ; dtotal[%]\n")
        for r in sorted(rows, key=lambda x: (x["exp"], x["ref"])):
            if ref_row:
                dlpsp  = r["lpsp"]  - ref_row["lpsp"]
                ddeg   = (r["deg"]   - ref_row["deg"])   / ref_row["deg"]   * 100
                dtotal = (r["total"] - ref_row["total"]) / ref_row["total"] * 100
            else:
                dlpsp = ddeg = dtotal = float("nan")
            f.write("%9.0f ; %.2f ; %7.3f ; %+8.2f ; %8.2f ; %+6.1f ; %9.2f ; %+7.1f\n"
                    % (r["ref"], r["exp"], r["lpsp"], dlpsp, r["deg"], ddeg, r["total"], dtotal))
    print("  tableau -> %s" % OUT_TXT)


def load_csv():
    rows = []
    with open(OUT_CSV, newline="") as f:
        for d in csv.DictReader(f):
            rows.append(dict(ref=float(d["RUL_ELY_REF"]), exp=float(d["EXP_ELY"]),
                             lpsp=float(d["LPSP_pct"]), deg=float(d["deg_kEUR"]),
                             eens=float(d["EENS_kWh"]), total=float(d["total_kEUR"])))
    return rows


if __name__ == "__main__":
    if "--replot" in sys.argv:
        write_table(load_csv())
    else:
        write_table(compute())
