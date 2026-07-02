# -*- coding: utf-8 -*-
"""
bench_fable.py -- BANC D'ESSAI des leviers Fable (RB2(Proba), RB2(Prop)).
==========================================================================
Compare, sous le MEME bruit de prevision (common random numbers), sur le
socle cost-min (0.440/0.310) et l'indicateur unifie VoLL=3 :

    RB2 socle            (reference, levier OFF -> test nul)
    RB2(Pred) hyst       (production actuelle : bande +-1sigma + gel 12h)
    RB2(Proba)           (hysteresis sur la PROBABILITE de deficit, sans gel)
    RB2(Prop)            (modulation CONTINUE de l'ELY par Phi(net/TAU*sigma))

Reperes attendus (reopt_pred.txt, 25 ans, VoLL=3) :
    RB2 socle ................ 80.108 kEUR
    RB2(Pred) hyst (MC) ...... 79.717 kEUR   (levier actuel)
    RB2(Pred) omniscient ..... 79.027 kEUR   (borne superieure du levier)
Critere de succes : total < 79.717 (mieux que l'hysteresis actuelle) ;
ideal : se rapprocher de 79.03 en gardant des demarrages ELY ~stables.

USAGE
-----
    python bench_fable.py                  # bench nominal : N=8 graines, 25 ans
    python bench_fable.py 16 25            # N graines, n_years
    python bench_fable.py --quick          # fumee : N=2, 1 an (verifie que ca tourne)
    python bench_fable.py --omni           # + bornes omniscientes (1 run determinist)
    python bench_fable.py --sweep prop     # balayage TAU de RB2(Prop)
    python bench_fable.py --sweep proba    # balayage (P_HI,P_LO) x MIN_DWELL de RB2(Proba)
    sbatch run_meso_fable.slurm            # mesocentre (N=200)

Sorties (dans Fable/) : bench_fable*.txt (tableau) + bench_fable*_cloud.csv (brut).
Necessite GENIAL_DATA_DIR (ou layout Doctorat/Data historique) comme les autres.
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

VOLL    = 3.0        # EUR/kWh (indicateur unifie de reference these)
MC_SEED = 2026       # base des graines (common random numbers entre strategies)

# (label, dossier de la strategie, overrides de parametres module)
# Un override peut aussi cibler Common.get_lol via la cle speciale "_lol:<PARAM>"
# (ex. "_lol:SOC_MAX_AGED_GAIN"). Ces flags sont REMIS A LEUR DEFAUT a chaque
# tache (les workers du pool sont reutilises -> sinon l'etat fuirait).
BENCH_STRATS = [
    ("RB2 socle",       os.path.join(HERE, "RB2(Prop)"),        {"ENABLE": False}),
    ("RB2(Pred) hyst",  os.path.join(PRED_DIR, "RB2(Pred)"),    {}),
    ("RB2(Proba)",      os.path.join(HERE, "RB2(Proba)"),       {}),
    ("RB2(Prop)",       os.path.join(HERE, "RB2(Prop)"),        {}),
    ("RB2(PropSym)",    os.path.join(HERE, "RB2(PropSym)"),     {}),
]
OMNI_STRATS = [
    ("RB2(Pred) omni bin", os.path.join(PRED_DIR, "RB2(Pred)"), {"NOISE_ENABLE": False, "HYST_ENABLE": False}),
    ("RB2(Proba) omni",    os.path.join(HERE, "RB2(Proba)"),    {"NOISE_ENABLE": False}),
    ("RB2(Prop) omni",     os.path.join(HERE, "RB2(Prop)"),     {"NOISE_ENABLE": False}),
    ("RB2(PropSym) omni",  os.path.join(HERE, "RB2(PropSym)"),  {"NOISE_ENABLE": False}),
]

SWEEP_PROBA = [  # (P_HI, P_LO, MIN_DWELL)
    (0.55, 0.45, 0), (0.60, 0.40, 0), (0.70, 0.30, 0), (0.80, 0.20, 0),
    (0.84, 0.16, 0),                                   # == M_SIGMA=1.0 sans gel
    (0.60, 0.40, 6), (0.70, 0.30, 6), (0.84, 0.16, 12) # == prod actuelle (controle)
]
SWEEP_PROP = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]          # TAU
SWEEP_SYM  = [  # (TAU_SYM, SOC_SYM_FLOOR) ; TAU (pre-charge) reste au defaut
    (0.5, 0.40), (0.5, 0.50), (0.5, 0.65),
    (1.0, 0.40), (1.0, 0.50), (1.0, 0.65),
    (2.0, 0.50),
]
SWEEP_RHO  = [0.0, 0.5, 0.8, 0.95]                     # correlation AR(1) du bruit
# SoH_bat cross-modulation : sur le SOCLE (levier prevision OFF) -> attribution
# pure du SoH_bat, runs deterministes.
SWEEP_SOHBAT = [  # (BETA_FC_BAT, BETA_ELY_BAT)
    (0.0, 0.0),                                        # socle (controle)
    (0.0, 0.25), (0.0, 0.5), (0.0, 1.0), (0.0, 2.0),
    (0.5, 0.5),  (1.0, 1.0),
]
# Plafond SoC vieillissement-dependant (Common/get_lol), sur le socle, deterministe.
# gain g : plafond a SoH_EoL(0.7) = 0.995 - 0.3*g
SWEEP_SOCWIN = [0.0, 0.2, 0.4, 0.6, 0.8]

# Defauts des flags Common/get_lol (remis a chaque tache)
LOL_DEFAULTS = {"SOC_MAX_AGED_GAIN": 0.0, "LOL_COMBINED": False}


def _load(folder):
    """Charge get_optimal_action_RB.py du dossier comme module INDEPENDANT
    (etat module par worker, comme sens_pred_noise)."""
    name = "strat_" + os.path.basename(folder)
    for ch in "()+ ":
        name = name.replace(ch, "")
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
    """(LPSP %, deg kEUR, EENS kWh, demarrages ELY) -- memes conventions que
    sens_pred_noise.metrics (interpolation SoH_bat aux remplacements)."""
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
    on = np.abs(P_ely) > 1.0                                       # [W] seuil diag
    ely_starts = int(np.sum(on[1:] & ~on[:-1]))
    return float(lpsp), float(deg), float(eens), ely_starts


def evaluate(task):
    """Worker picklable : un run complet. task = dict(label, folder, overrides,
    seed, ny). Retourne les metriques + le cout total unifie."""
    from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
    import Common.get_lol as _gl
    # Flags Common : remis a leur DEFAUT puis surcharges par la tache (les
    # workers du pool traitent plusieurs taches -> pas d'etat residuel).
    for k, v in LOL_DEFAULTS.items():
        setattr(_gl, k, task["overrides"].get("_lol:" + k, v))
    s = _load(task["folder"])
    for k, v in task["overrides"].items():
        if not k.startswith("_lol:"):
            setattr(s, k, v)
    if hasattr(s, "set_noise_seed"):
        s.set_noise_seed(task["seed"])
    if hasattr(s, "reset"):
        s.reset()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                      n_years=task["ny"])
    lpsp, deg, eens, ely_starts = metrics(data)
    total = deg + VOLL * eens / 1000.0
    out = dict(task)
    out.update(lpsp=lpsp, deg=deg, eens=eens, total=total, ely_starts=ely_starts)
    return out


def run_all(strat_list, n_seeds, ny, tag):
    """Lance strat_list x n_seeds en parallele (common random numbers), ecrit
    <tag>.txt (stats) + <tag>_cloud.csv (brut), et affiche le tableau."""
    tasks = []
    for label, folder, overrides in strat_list:
        deterministic = overrides.get("NOISE_ENABLE", True) is False
        seeds = [MC_SEED] if deterministic else [MC_SEED + i for i in range(n_seeds)]
        for sd in seeds:
            tasks.append(dict(label=label, folder=folder, overrides=overrides,
                              seed=sd, ny=ny))

    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    nw = min(nw, len(tasks))
    print(f"[bench_fable] {len(tasks)} runs ({ny} ans), {nw} workers ...", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, res in enumerate(ex.map(evaluate, tasks)):
            results.append(res)
            print(f"  [{i+1:3d}/{len(tasks)}] {res['label']:<22s} seed={res['seed']:<6d}"
                  f" LPSP={res['lpsp']:.4f}%  deg={res['deg']:.3f}  total={res['total']:.3f}"
                  f"  ELYstarts={res['ely_starts']}", flush=True)
    print(f"[bench_fable] termine en {time.time()-t0:.0f} s", flush=True)

    # --- cloud CSV ---
    cloud = os.path.join(HERE, tag + "_cloud.csv")
    with open(cloud, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["label", "seed", "lpsp_pct", "deg_keur", "eens_kwh",
                    "total_keur", "ely_starts"])
        for r in results:
            w.writerow([r["label"], r["seed"], f"{r['lpsp']:.6f}", f"{r['deg']:.6f}",
                        f"{r['eens']:.3f}", f"{r['total']:.6f}", r["ely_starts"]])

    # --- stats + tableau ---
    labels = [s[0] for s in strat_list]
    stats = {}
    for lab in labels:
        rr = [r for r in results if r["label"] == lab]
        arr = {k: np.array([r[k] for r in rr]) for k in
               ("lpsp", "deg", "eens", "total", "ely_starts")}
        stats[lab] = {k: (v.mean(), v.std()) for k, v in arr.items()}
        stats[lab]["n"] = len(rr)

    base = labels[0]
    lines = [f"# bench_fable ({tag}) : {ny} ans ; VoLL={VOLL} ; N={n_seeds} graines ; "
             f"seeds communes (CRN) ; base={base}",
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
    print(f"\n[bench_fable] -> {txt}\n[bench_fable] -> {cloud}")
    return stats


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")
            and (argv[argv.index(a) - 1] not in ("--sweep", "--rho"))]
    n_seeds = int(args[0]) if len(args) > 0 else 8
    ny      = int(args[1]) if len(args) > 1 else 25
    quick   = "--quick" in argv
    omni    = "--omni" in argv
    sweep   = None
    if "--sweep" in argv:
        sweep = argv[argv.index("--sweep") + 1]
    rho = None
    if "--rho" in argv:
        rho = float(argv[argv.index("--rho") + 1])
    if quick:
        n_seeds, ny = 2, 1

    socle = ("RB2 socle", os.path.join(HERE, "RB2(Prop)"), {"ENABLE": False})

    if sweep == "proba":
        strats = [socle]
        for hi, lo, dwell in SWEEP_PROBA:
            strats.append((f"Proba {hi:.2f}/{lo:.2f} d{dwell}",
                           os.path.join(HERE, "RB2(Proba)"),
                           {"P_HI": hi, "P_LO": lo, "MIN_DWELL": dwell}))
        run_all(strats, n_seeds, ny, "sweep_fable_proba")
    elif sweep == "prop":
        strats = [socle]
        for tau in SWEEP_PROP:
            strats.append((f"Prop TAU={tau:.2f}", os.path.join(HERE, "RB2(Prop)"),
                           {"TAU": tau}))
        run_all(strats, n_seeds, ny, "sweep_fable_prop")
    elif sweep == "sym":
        # Reference = RB2(Prop) pur (SYM off) : la difference mesure le levier
        # symetrique SEUL. Socle inclus pour l'ancrage absolu.
        strats = [socle,
                  ("PropSym OFF (=Prop)", os.path.join(HERE, "RB2(PropSym)"),
                   {"SYM_ENABLE": False})]
        for tau_s, floor in SWEEP_SYM:
            strats.append((f"Sym t{tau_s:.1f}/f{floor:.2f}",
                           os.path.join(HERE, "RB2(PropSym)"),
                           {"TAU_SYM": tau_s, "SOC_SYM_FLOOR": floor}))
        run_all(strats, n_seeds, ny, "sweep_fable_sym")
    elif sweep == "rho":
        # Robustesse a la CORRELATION du bruit : l'iid (rho=0) est le pire-cas
        # pour le clignotement ; verifier que le classement des leviers tient.
        strats = []
        for r in SWEEP_RHO:
            strats.append((f"Pred hyst rho={r:.2f}", os.path.join(PRED_DIR, "RB2(Pred)"),
                           {"NOISE_RHO": r}))
            strats.append((f"Proba rho={r:.2f}", os.path.join(HERE, "RB2(Proba)"),
                           {"NOISE_RHO": r}))
            strats.append((f"Prop rho={r:.2f}", os.path.join(HERE, "RB2(Prop)"),
                           {"NOISE_RHO": r}))
        strats.insert(0, socle)
        run_all(strats, n_seeds, ny, "sweep_fable_rho")
    elif sweep == "soh_bat":
        # Attribution PURE du SoH_bat : cross-modulation sur le SOCLE (levier
        # prevision OFF) -> runs deterministes (1 seed).
        strats = []
        for bfc, bely in SWEEP_SOHBAT:
            strats.append((f"SoHbat bFC={bfc:.2f} bELY={bely:.2f}",
                           os.path.join(HERE, "RB2(Prop)"),
                           {"ENABLE": False, "NOISE_ENABLE": False,
                            "BETA_FC_BAT": bfc, "BETA_ELY_BAT": bely}))
        run_all(strats, n_seeds, ny, "sweep_fable_sohbat")
    elif sweep == "socwin":
        # Plafond SoC vieillissement-dependant (Common/get_lol), socle, deterministe.
        strats = []
        for g in SWEEP_SOCWIN:
            strats.append((f"SoCwin gain={g:.2f}",
                           os.path.join(HERE, "RB2(Prop)"),
                           {"ENABLE": False, "NOISE_ENABLE": False,
                            "_lol:SOC_MAX_AGED_GAIN": g}))
        run_all(strats, n_seeds, ny, "sweep_fable_socwin")
    else:
        strats = list(BENCH_STRATS) + (list(OMNI_STRATS) if omni else [])
        if rho is not None:
            strats = [(lab, fold, dict(ov, NOISE_RHO=rho) if "RB2(Pred)" in fold
                       or HERE in fold else ov) for lab, fold, ov in strats]
        run_all(strats, n_seeds, ny, "bench_fable" + ("_quick" if quick else "")
                + (f"_rho{rho:g}" if rho is not None else ""))


if __name__ == "__main__":
    main(sys.argv)
