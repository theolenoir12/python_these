# -*- coding: utf-8 -*-
"""
bench_mpc.py -- BANC D'ESSAI MPC(LP) vs strategies a regles (protocole bench_ultime).
=====================================================================================
Compare, sous le MEME bruit (CRN) et le MEME indicateur unifie VoLL=3, la
commande predictive MPC(LP) aux jalons a regles du programme Fable :

    RB2 socle                (ancrage absolu, attendu 80.102)
    RB2(SoH_all) (test nul)  (gammas + plafond, sans prevision : attendu 78.336)
    RB2(SoH_all+Pred)        (strategie ultime a regles : attendu 77.224)
    MPC (H=24)               (LP a horizon glissant, prevision bruitee backtest)
    MPC (H=48)
    (+ --omni : MPC a prevision parfaite = plafond informationnel du MPC)

Le MPC partage la MEME boucle (init_and_run_loop_forecast, H_forecast=48), la
MEME surface d'action (batterie = variable d'ajustement, ecretage et lol par
Common/get_lol) et les MEMES metriques -- seule la decision differe.

USAGE
-----
    python bench_mpc.py                      # N=8 graines, 25 ans
    python bench_mpc.py 100 25 --omni        # complet mesocentre
    python bench_mpc.py --quick              # fumee : N=2, 1 an
    python bench_mpc.py 32 25 --sweep h      # horizon MPC 6/12/24/36/48
    python bench_mpc.py 32 25 --sweep pareto # front MPC : VoLL interne 0.5..30
    python bench_mpc.py 32 25 --sweep vh2    # valeur terminale H2
    sbatch run_meso_mpc.slurm 100 25 --omni

Sorties (dans MPC/) : bench_mpc*.txt + *_cloud.csv (memes formats que
bench_ultime, directement exploitables par plot_pareto_mpc.py).
"""
import os, sys, csv, time
import importlib.util
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.abspath(os.path.join(HERE, "..", "Prédictions"))
FBP_DIR  = os.path.abspath(os.path.join(HERE, "..", "Fable_pred"))
for _p in (HERE, PRED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VOLL       = 3.0     # indicateur unifie d'EVALUATION (fige, comme bench_ultime)
MC_SEED    = 2026
H_FORECAST = 48      # fenetre passee par la boucle (>= max des MPC_H testes)
G_WIN      = 0.2     # plafond SoC vieillissant retenu pour l'ultime

ULT = os.path.join(FBP_DIR, "RB2(SoH_all+Pred)")
MPC = os.path.join(HERE, "MPC(LP)")

# (label, dossier strategie, overrides). Cles "_lol:<PARAM>" -> Common.get_lol.
BENCH_STRATS = [
    ("RB2 socle",               ULT, {"ENABLE": False, "NOISE_ENABLE": False,
                                      "GAMMA_FC": 0.0, "GAMMA_ELY": 0.0}),
    ("RB2(SoH_all) (test nul)", ULT, {"ENABLE": False, "NOISE_ENABLE": False,
                                      "_lol:SOC_MAX_AGED_GAIN": G_WIN}),
    ("RB2(SoH_all+Pred)",       ULT, {"_lol:SOC_MAX_AGED_GAIN": G_WIN}),
    ("MPC (H=24)",              MPC, {}),
    ("MPC (H=24, gel 12h)",     MPC, {"MPC_ELY_MIN_DWELL": 12}),
    ("MPC (H=48)",              MPC, {"MPC_H": 48}),
]
OMNI_STRATS = [
    ("MPC omni (H=24)", MPC, {"MPC_NOISE_ENABLE": False}),
    ("MPC omni (H=48)", MPC, {"MPC_NOISE_ENABLE": False, "MPC_H": 48}),
]

SWEEP_H      = [6, 12, 24, 36, 48]
SWEEP_PARETO = [0.5, 1.0, 3.0, 10.0, 30.0]   # MPC_VOLL interne (eval reste VoLL=3)
SWEEP_VH2    = [0.5, 1.0, 1.5, 2.0]          # MPC_V_H2 [EUR/kWh]

# --- Plan C : robustification au bruit (sur MPC H=24 bruite) -----------------
# Cible : ramener la degradation bruitee (~68 kEUR) vers le plancher omniscient
# (54.6). Leviers : durcissement du cout de commutation (sw) et cout de
# residence haut-SoC (hold). Chaque point est un test nul en cascade depuis le
# MPC nu. La reference omni (plancher) est ajoutee en tete du sweep.
SWEEP_ROBUST = [   # (label, overrides sur MPC H=24)
    ("MPC nu (ref)",        {}),
    ("sw x3",               {"MPC_SW_SCALE": 3.0}),
    ("sw x10",              {"MPC_SW_SCALE": 10.0}),
    ("sw x30",              {"MPC_SW_SCALE": 30.0}),
    ("hold 0.3",            {"MPC_BAT_HOLD_EUR": 0.3}),
    ("hold 1.0",            {"MPC_BAT_HOLD_EUR": 1.0}),
    ("sw x10 + hold 0.3",   {"MPC_SW_SCALE": 10.0, "MPC_BAT_HOLD_EUR": 0.3}),
    ("sw x10 + hold 1.0",   {"MPC_SW_SCALE": 10.0, "MPC_BAT_HOLD_EUR": 1.0}),
]

LOL_DEFAULTS = {"SOC_MAX_AGED_GAIN": 0.0, "LOL_COMBINED": False}


def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + os.path.basename(folder).replace("(", "_").replace(")", ""),
        os.path.join(folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _resolve(folder):
    """Chemin effectif du dossier strategie, ou None s'il est introuvable.

    Cherche a l'emplacement attendu, puis dans quelques racines candidates :
    robustesse aux copies PARTIELLES du depot sur le mesocentre (p.ex. un
    checkout ou Fable_pred/ n'a pas ete synchronise -> les ancrages RB2 sont
    absents). Evite qu'une strategie manquante fasse echouer TOUT le job."""
    if os.path.isfile(os.path.join(folder, "get_optimal_action_RB.py")):
        return folder
    base = os.path.basename(folder)
    for root in (HERE, FBP_DIR, PRED_DIR, os.path.abspath(os.path.join(HERE, ".."))):
        cand = os.path.join(root, base)
        if os.path.isfile(os.path.join(cand, "get_optimal_action_RB.py")):
            return cand
    return None


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
    s = _load(task["folder"])
    for k, v in task["overrides"].items():
        if not k.startswith("_lol:"):
            setattr(s, k, v)
    s.set_noise_seed(task["seed"])
    s.reset()
    t0 = time.time()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB,
                                      H_forecast=H_FORECAST, n_years=task["ny"])
    lpsp, deg, eens, ely_starts = metrics(data)
    total = deg + VOLL * eens / 1000.0
    out = dict(task)
    out.update(lpsp=lpsp, deg=deg, eens=eens, total=total, ely_starts=ely_starts,
               wall_s=time.time() - t0,
               lp_failures=int(getattr(s, "LP_FAILURES", 0)))
    return out


def _is_deterministic(ov):
    return (ov.get("NOISE_ENABLE", None) is False
            or ov.get("MPC_NOISE_ENABLE", None) is False)


def run_all(strat_list, n_seeds, ny, tag):
    # --- Validation AVANT le pool (feedback a t=0, pas apres 16 h) -----------
    # Une strategie dont le dossier est introuvable (copie partielle du depot)
    # est IGNOREE avec un avertissement bien visible, au lieu de faire echouer
    # les 402 runs. Le job produit alors les chiffres MPC + les ancrages
    # presents. Si MEME la strategie MPC manque, on s'arrete franchement.
    resolved = []
    for label, folder, overrides in strat_list:
        rf = _resolve(folder)
        if rf is None:
            print(f"[bench_mpc] !! ATTENTION : strategie '{label}' INTROUVABLE "
                  f"({folder}) -> IGNOREE.\n"
                  f"[bench_mpc]    (synchroniser le dossier parent sur le "
                  f"mesocentre pour l'inclure dans la comparaison)", flush=True)
            continue
        resolved.append((label, rf, overrides))
    if not resolved or all("MPC" not in lab for lab, _, _ in resolved):
        raise SystemExit("[bench_mpc] aucune strategie MPC disponible -- "
                         "verifier le dossier MPC(LP)/. Arret.")
    strat_list = resolved

    tasks = []
    for label, folder, overrides in strat_list:
        seeds = [MC_SEED] if _is_deterministic(overrides) \
            else [MC_SEED + i for i in range(n_seeds)]
        for sd in seeds:
            tasks.append(dict(label=label, folder=folder, overrides=overrides,
                              seed=sd, ny=ny))

    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    nw = min(nw, len(tasks))
    print(f"[bench_mpc] {len(tasks)} runs ({ny} ans), {nw} workers ...", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, res in enumerate(ex.map(evaluate, tasks)):
            results.append(res)
            xtra = f"  LPfail={res['lp_failures']}" if res["lp_failures"] else ""
            print(f"  [{i+1:3d}/{len(tasks)}] {res['label']:<24s} seed={res['seed']:<6d}"
                  f" LPSP={res['lpsp']:.4f}%  deg={res['deg']:.3f}"
                  f"  total={res['total']:.3f}  ELYstarts={res['ely_starts']}"
                  f"  ({res['wall_s']:.0f}s){xtra}", flush=True)
    print(f"[bench_mpc] termine en {time.time()-t0:.0f} s", flush=True)

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
    lines = [f"# bench_mpc ({tag}) : {ny} ans ; VoLL={VOLL} ; N={n_seeds} graines ; "
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
    print(f"\n[bench_mpc] -> {txt}\n[bench_mpc] -> {cloud}")
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

    if sweep == "h":
        strats = [("RB2(SoH_all+Pred)", ULT, {"_lol:SOC_MAX_AGED_GAIN": G_WIN})]
        for h in SWEEP_H:
            strats.append((f"MPC H={h}", MPC, {"MPC_H": h}))
        run_all(strats, n_seeds, ny, "sweep_mpc_h")
    elif sweep == "pareto":
        strats = []
        for v in SWEEP_PARETO:
            strats.append((f"MPC VoLLint={v:g}", MPC, {"MPC_VOLL": v}))
        run_all(strats, n_seeds, ny, "sweep_mpc_pareto")
    elif sweep == "vh2":
        strats = []
        for v in SWEEP_VH2:
            strats.append((f"MPC vH2={v:.2f}", MPC, {"MPC_V_H2": v}))
        run_all(strats, n_seeds, ny, "sweep_mpc_vh2")
    elif sweep == "robust":
        # Plan C : robustification (sw + hold) sur MPC H=24 bruite, + plancher
        # omniscient en reference. Base = MPC nu -> dtotal montre le GAIN.
        strats = [("MPC omni (plancher)", MPC, {"MPC_NOISE_ENABLE": False})]
        for lab, ov in SWEEP_ROBUST:
            strats.append((lab, MPC, ov))
        run_all(strats, n_seeds, ny, "sweep_mpc_robust")
    else:
        strats = list(BENCH_STRATS) + (list(OMNI_STRATS) if omni else [])
        run_all(strats, n_seeds, ny, "bench_mpc" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main(sys.argv)
