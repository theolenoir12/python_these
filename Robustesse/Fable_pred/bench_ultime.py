# -*- coding: utf-8 -*-
"""
bench_ultime.py -- BANC D'ESSAI du RB2 ULTIME (Fable Predictions).
===================================================================
Empile les leviers valides : socle cost-min + gammas RB2(SoH) (1,2) + plafond
SoC vieillissant (g=0.2) + pre-charge previsionnelle +-1sigma. Compare, sous le
MEME bruit (CRN), sur l'indicateur unifie VoLL=3 :

    RB2 socle                 (ancrage absolu, attendu 80.102)
    Unifiee (base, test nul)  (gammas + plafond, sans prevision : attendu 78.336)
    RB2(SoH+Pred) (g=0)       (reference actuelle de la these : attendu ~77.67)
    RB2 ULTIME                (le point final : cible < 77.67)
    (+ --omni : bornes a prevision parfaite)

USAGE
-----
    python bench_ultime.py                    # N=8 graines, 25 ans
    python bench_ultime.py 200 25 --omni      # complet mesocentre
    python bench_ultime.py --quick            # fumee : N=2, 1 an
    python bench_ultime.py 64 25 --sweep target   # cible de pre-charge x gel
    python bench_ultime.py 64 25 --sweep hpre     # horizon de pre-charge
    sbatch run_meso_ultime.slurm 200 25 --omni

Sorties (dans Fable Predictions/) : bench_ultime*.txt + *_cloud.csv.
"""
import os, sys, csv, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.abspath(os.path.join(HERE, "..", "Prédictions"))
for _p in (HERE, PRED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VOLL    = 3.0
MC_SEED = 2026
ULT     = os.path.join(HERE, "RB2(Ultime)")
G_WIN   = 0.2            # plafond SoC vieillissant retenu (sweep_fable_socwin_fine)

# (label, overrides). Cles speciales "_lol:<PARAM>" -> Common.get_lol.
BENCH_STRATS = [
    ("RB2 socle",            {"ENABLE": False, "NOISE_ENABLE": False,
                              "GAMMA_FC": 0.0, "GAMMA_ELY": 0.0}),
    ("Unifiee (test nul)",   {"ENABLE": False, "NOISE_ENABLE": False,
                              "_lol:SOC_MAX_AGED_GAIN": G_WIN}),
    ("RB2(SoH+Pred) (g=0)",  {}),
    ("RB2 ULTIME",           {"_lol:SOC_MAX_AGED_GAIN": G_WIN}),
]
OMNI_STRATS = [
    ("SoH+Pred omni (g=0)",  {"NOISE_ENABLE": False}),
    ("ULTIME omni",          {"NOISE_ENABLE": False, "_lol:SOC_MAX_AGED_GAIN": G_WIN}),
]

# Cible de pre-charge x gel, sur le RB2 ULTIME complet.
SWEEP_TARGET = [  # (SOC_TARGET_MODE, SOC_TARGET, MIN_DWELL)
    ("ceiling", None, 12), ("ceiling", None, 0),
    ("fixed",   0.99, 12), ("fixed",  0.95, 12), ("fixed", 0.90, 12),
    ("fixed",   0.90, 0),
]
SWEEP_HPRE = [12, 18, 24]     # horizon de pre-charge, sur le RB2 ULTIME complet

LOL_DEFAULTS = {"SOC_MAX_AGED_GAIN": 0.0, "LOL_COMBINED": False}


def _load():
    spec = importlib.util.spec_from_file_location(
        "strat_RB2Ultime", os.path.join(ULT, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
    """(LPSP %, deg kEUR, EENS kWh, demarrages ELY) -- conventions bench_fable."""
    from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY
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
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    eens = np.clip(p - r, 0, None).sum() * (LOAD['Ts'] / 3600.0)   # [kWh]
    deg  = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely,
                          P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    on = np.abs(P_ely) > 1.0
    ely_starts = int(np.sum(on[1:] & ~on[:-1]))
    return float(lpsp), float(deg), float(eens), ely_starts


def evaluate(task):
    from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
    import Common.get_lol as _gl
    for k, v in LOL_DEFAULTS.items():
        setattr(_gl, k, task["overrides"].get("_lol:" + k, v))
    s = _load()
    for k, v in task["overrides"].items():
        if not k.startswith("_lol:"):
            setattr(s, k, v)
    s.set_noise_seed(task["seed"])
    s.reset()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                      n_years=task["ny"])
    lpsp, deg, eens, ely_starts = metrics(data)
    total = deg + VOLL * eens / 1000.0
    out = dict(task)
    out.update(lpsp=lpsp, deg=deg, eens=eens, total=total, ely_starts=ely_starts)
    return out


def run_all(strat_list, n_seeds, ny, tag):
    tasks = []
    for label, overrides in strat_list:
        deterministic = overrides.get("NOISE_ENABLE", True) is False
        seeds = [MC_SEED] if deterministic else [MC_SEED + i for i in range(n_seeds)]
        for sd in seeds:
            tasks.append(dict(label=label, overrides=overrides, seed=sd, ny=ny))

    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    nw = min(nw, len(tasks))
    print(f"[bench_ultime] {len(tasks)} runs ({ny} ans), {nw} workers ...", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, res in enumerate(ex.map(evaluate, tasks)):
            results.append(res)
            print(f"  [{i+1:3d}/{len(tasks)}] {res['label']:<24s} seed={res['seed']:<6d}"
                  f" LPSP={res['lpsp']:.4f}%  deg={res['deg']:.3f}  total={res['total']:.3f}"
                  f"  ELYstarts={res['ely_starts']}", flush=True)
    print(f"[bench_ultime] termine en {time.time()-t0:.0f} s", flush=True)

    cloud = os.path.join(HERE, tag + "_cloud.csv")
    with open(cloud, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["label", "seed", "lpsp_pct", "deg_keur", "eens_kwh",
                    "total_keur", "ely_starts"])
        for r in results:
            w.writerow([r["label"], r["seed"], f"{r['lpsp']:.6f}", f"{r['deg']:.6f}",
                        f"{r['eens']:.3f}", f"{r['total']:.6f}", r["ely_starts"]])

    labels = [s[0] for s in strat_list]
    stats = {}
    for lab in labels:
        rr = [r for r in results if r["label"] == lab]
        arr = {k: np.array([r[k] for r in rr]) for k in
               ("lpsp", "deg", "eens", "total", "ely_starts")}
        stats[lab] = {k: (v.mean(), v.std()) for k, v in arr.items()}
        stats[lab]["n"] = len(rr)

    base = labels[0]
    lines = [f"# bench_ultime ({tag}) : {ny} ans ; VoLL={VOLL} ; N={n_seeds} graines ; "
             f"CRN ; base={base}",
             "label ; N ; LPSP(%) ; sLPSP ; deg(kEUR) ; sdeg ; total(kEUR) ; stotal ; "
             "dtotal_vs_base(kEUR) ; ELY_starts"]
    for lab in labels:
        st = stats[lab]
        dt = st["total"][0] - stats[base]["total"][0]
        lines.append(f"{lab} ; {st['n']} ; {st['lpsp'][0]:.4f} ; {st['lpsp'][1]:.4f} ; "
                     f"{st['deg'][0]:.3f} ; {st['deg'][1]:.3f} ; "
                     f"{st['total'][0]:.3f} ; {st['total'][1]:.3f} ; "
                     f"{dt:+.3f} ; {st['ely_starts'][0]:.0f}")
    txt = os.path.join(HERE, tag + ".txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[bench_ultime] -> {txt}\n[bench_ultime] -> {cloud}")
    return stats


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")
            and (argv[argv.index(a) - 1] != "--sweep")]
    n_seeds = int(args[0]) if len(args) > 0 else 8
    ny      = int(args[1]) if len(args) > 1 else 25
    quick   = "--quick" in argv
    omni    = "--omni" in argv
    sweep   = None
    if "--sweep" in argv:
        sweep = argv[argv.index("--sweep") + 1]
    if quick:
        n_seeds, ny = 2, 1

    if sweep == "target":
        strats = [("Unifiee (test nul)", {"ENABLE": False, "NOISE_ENABLE": False,
                                          "_lol:SOC_MAX_AGED_GAIN": G_WIN})]
        for mode, tgt, dwell in SWEEP_TARGET:
            ov = {"SOC_TARGET_MODE": mode, "MIN_DWELL": dwell,
                  "_lol:SOC_MAX_AGED_GAIN": G_WIN}
            lab = f"Ult {mode}"
            if mode == "fixed":
                ov["SOC_TARGET"] = tgt
                lab += f" {tgt:.2f}"
            lab += f" d{dwell}"
            strats.append((lab, ov))
        run_all(strats, n_seeds, ny, "sweep_ultime_target")
    elif sweep == "hpre":
        strats = [("Unifiee (test nul)", {"ENABLE": False, "NOISE_ENABLE": False,
                                          "_lol:SOC_MAX_AGED_GAIN": G_WIN})]
        for h in SWEEP_HPRE:
            strats.append((f"Ult H_PRE={h}", {"H_PRE": h,
                                              "_lol:SOC_MAX_AGED_GAIN": G_WIN}))
        run_all(strats, n_seeds, ny, "sweep_ultime_hpre")
    else:
        strats = list(BENCH_STRATS) + (list(OMNI_STRATS) if omni else [])
        run_all(strats, n_seeds, ny, "bench_ultime" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main(sys.argv)
