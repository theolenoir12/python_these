# -*- coding: utf-8 -*-
"""
mc_rul_uncertainty.py -- SENSIBILITE A L'INCERTITUDE DU PRONOSTIC (ellipses Pareto).
====================================================================================
But : etayer l'argument qualitatif de la section RB2(RUL) -- la RUL est une
grandeur EXTRAPOLEE, donc bien plus incertaine que le SoH instantane -- en
MESURANT la taille des ellipses de confiance autour du point de fonctionnement
dans le plan de Pareto (LPSP %, deg kEUR/25 ans), sous propagation de l'ecart-type
REEL d'estimation (issu du MC-dropout du chapitre pronostic, cf compute_prono_sigma.py).

Modele de propagation (incertitude EPISTEMIQUE, donc correlee dans le temps) :
  - chaque tirage Monte-Carlo = UNE simulation 25 ans complete, ou le SIGNAL DE
    DECISION de la strategie est biaise par UNE realisation de l'erreur d'estimation,
    CONSTANTE sur tout le run (et non un bruit blanc qui se moyennerait a zero).
  - RB2(RUL) : la RUL_ely vue par la strategie est multipliee par (1 + eps),
    eps ~ N(0, SIGMA_RUL_REL)  [erreur RELATIVE, la RUL couvrant une large plage].
  - RB2(SoH) : le SoH_ely vu par la strategie est decale de delta,
    delta ~ N(0, SIGMA_SOH)    [erreur ABSOLUE, le SoH etant borne pres de 1].
Le biais est injecte par un WRAPPER autour de get_optimal_action_RB : les fichiers
de strategie ne sont PAS modifies.

Lecture attendue : l'ellipse RB2(RUL) (signal extrapole, SIGMA_RUL_REL grand) est
PLUS GRANDE que l'ellipse RB2(SoH) (signal instantane, SIGMA_SOH petit) -> le gain
nominal du pronostic ne survit pas a son incertitude -> argument pro-RB2(SoH).

ATTENTION : SIGMA_RUL_REL et SIGMA_SOH doivent etre renseignes avec les VRAIS ecarts-types :
    les lancer via  python compute_prono_sigma.py  (dossier Pronostic SoH) et
    reporter les valeurs ci-dessous. Les defauts sont des PLACEHOLDERS.

EXP_ELY (force du levier RUL) est parametrable : a p=0.1 (reglage retenu) le levier
est faible et l'ellipse peut etre petite ; augmenter p rend la fragilite visible.

Sorties (dans Predictions/) : mc_rul_uncertainty.{pdf,png}, .txt, _cloud.csv.
Usage :
    python mc_rul_uncertainty.py [N_SAMPLES]      # defaut N=64
    python mc_rul_uncertainty.py --replot         # relit le CSV, refait la figure
    sbatch run_meso.slurm mc_rul_uncertainty.py 200
"""
import os, sys, csv, time
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ----------------------------- PARAMETRES -----------------------------
VOLL          = 3.0      # EUR/kWh (coherent manuscrit)
MC_SEED       = 2026

# >>> A RENSEIGNER avec les vrais ecarts-types (compute_prono_sigma.py) <<<
SIGMA_RUL_REL = 0.0523   # ecart-type RELATIF d'estimation de la RUL ely (sigma_RUL/RUL)
                         # mesure : compute_prono_sigma.py (MC-dropout, PEMWE) -> 5.2 %
SIGMA_SOH     = 0.00031  # ecart-type ABSOLU d'estimation du SoH ely (unites de SoH)
                         # mesure : compute_prono_sigma.py -> 3.1e-4 (horizon ~0)

# Force du levier RUL pour RB2(RUL) (0.1 = reglage retenu ; augmenter pour
# rendre la sensibilite visible). RB2(SoH) garde son reglage interne (gamma=0.5).
EXP_ELY       = 0.1
RUL_ELY_REF   = 1000.0

STRATS = [("RB2(RUL)", "RB2(RUL)", "rul", "#1f4e79"),
          ("RB2(SoH)", "RB2(SoH)", "soh", "#d95f02")]

CLOUD_CSV = os.path.join(HERE, "mc_rul_uncertainty_cloud.csv")
STATS_TXT = os.path.join(HERE, "mc_rul_uncertainty.txt")
FIG_PDF   = os.path.join(HERE, "mc_rul_uncertainty.pdf")
FIG_PNG   = os.path.join(HERE, "mc_rul_uncertainty.png")


# ============================ CALCUL ============================
def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + folder.replace("(", "").replace(")", ""),
        os.path.join(HERE, folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
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
    deg   = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely,
                           P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(deg)


# --- wrappers d'injection du biais d'estimation (constant sur tout le run) ---
# Signature de get_optimal_action_RB (positions) :
#   0 SoC 1 P_tot_ref 2 defaillances 3 lol_tab 4 alpha_fc 5 alpha_ely 6 SoH_bat
#   7 E_h2 8 E_h2_init 9 P_fc_max 10 P_ely_max 11 RUL_fc 12 RUL_ely 13 SoH_fc 14 SoH_ely
def wrap_rul_noise(base, eps_rel):
    def f(*a):
        a = list(a)
        a[12] = max(a[12] * (1.0 + eps_rel), 0.0)      # RUL_ely biaisee (relatif)
        return base(*a)
    return f


def wrap_soh_noise(base, delta, soh_eol):
    def f(*a):
        a = list(a)
        a[14] = min(max(a[14] + delta, soh_eol), 1.0)  # SoH_ely biaise (absolu)
        return base(*a)
    return f


def evaluate(task):
    from Common.main_init_and_loop import init_and_run_loop, ELY
    s = _load(task["folder"])
    if task["kind"] == "rul":
        s.RUL_ELY_REF = RUL_ELY_REF
        s.EXP_ELY     = EXP_ELY
        if task["noisy"]:
            eps = np.random.default_rng((MC_SEED, 1, task["seed"])).normal(0.0, SIGMA_RUL_REL)
            func = wrap_rul_noise(s.get_optimal_action_RB, eps)
        else:
            func = s.get_optimal_action_RB
    else:  # soh
        if task["noisy"]:
            d = np.random.default_rng((MC_SEED, 2, task["seed"])).normal(0.0, SIGMA_SOH)
            func = wrap_soh_noise(s.get_optimal_action_RB, d, ELY["SoH_EoL"])
        else:
            func = s.get_optimal_action_RB
    data = init_and_run_loop(func)
    lpsp, deg = metrics(data)
    return dict(label=task["label"], variant="cloud" if task["noisy"] else "nominal",
                seed=task["seed"], lpsp=lpsp, deg=deg)


def compute(N):
    workers_env = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = int(workers_env) if workers_env else max(1, (os.cpu_count() or 2) - 1)

    tasks = []
    for label, folder, kind, _ in STRATS:
        for i in range(N):                                   # nuage perturbe
            tasks.append(dict(label=label, folder=folder, kind=kind, noisy=True, seed=i))
        tasks.append(dict(label=label, folder=folder, kind=kind, noisy=False, seed=-1))  # repere nominal

    print("=" * 78)
    print("SENSIBILITE INCERTITUDE PRONOSTIC : ellipses Pareto (N=%d, %d workers)" % (N, workers))
    print("  SIGMA_RUL_REL=%.3f  SIGMA_SOH=%.4f  EXP_ELY=%.2f  RUL_ELY_REF=%.0f"
          % (SIGMA_RUL_REL, SIGMA_SOH, EXP_ELY, RUL_ELY_REF), flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, tasks):
            rows.append(r)
    print("  (%d runs en %.0fs)" % (len(tasks), time.time() - t0), flush=True)

    with open(CLOUD_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "variant", "seed", "lpsp", "deg"])
        for r in rows:
            w.writerow([r["label"], r["variant"], r["seed"], "%.6f" % r["lpsp"], "%.6f" % r["deg"]])
    print("  nuage brut -> %s" % CLOUD_CSV)
    return load_cloud()


# ============================ I/O + STATS ============================
def load_cloud():
    clouds, refs = {}, {}
    with open(CLOUD_CSV, newline="") as f:
        for row in csv.DictReader(f):
            xy = (float(row["lpsp"]), float(row["deg"]))
            if row["variant"] == "cloud":
                clouds.setdefault(row["label"], []).append(xy)
            else:
                refs[row["label"]] = xy
    clouds = {k: np.array(v) for k, v in clouds.items()}
    return clouds, refs


def write_stats(clouds, refs, N):
    with open(STATS_TXT, "w") as f:
        f.write("# Sensibilite a l'incertitude du pronostic -- ellipses de Pareto\n")
        f.write("# SIGMA_RUL_REL=%.4f ; SIGMA_SOH=%.4f ; EXP_ELY=%.2f ; RUL_ELY_REF=%.0f ; N=%d\n"
                % (SIGMA_RUL_REL, SIGMA_SOH, EXP_ELY, RUL_ELY_REF, N))
        f.write("strat ; LPSP_nom ; deg_nom ; LPSP_mean ; LPSP_std ; deg_mean ; deg_std ; aire_ellipse(1sig)\n")
        for label, _, _, _ in STRATS:
            pts = clouds.get(label)
            if pts is None:
                continue
            xn, yn = refs.get(label, (float("nan"), float("nan")))
            cov = np.cov(pts[:, 0], pts[:, 1])
            area = float(np.pi * np.sqrt(max(np.linalg.det(cov), 0.0)))  # aire ellipse 1-sigma
            f.write("%s ; %.4f ; %.3f ; %.4f ; %.4f ; %.3f ; %.3f ; %.4f\n"
                    % (label, xn, yn, pts[:, 0].mean(), pts[:, 0].std(),
                       pts[:, 1].mean(), pts[:, 1].std(), area))
    print("  stats -> %s" % STATS_TXT)


# ============================ FIGURE ============================
def confidence_ellipse(x, y, ax, n_std=1.0, **kw):
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    pear = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0.0
    rx, ry = np.sqrt(1 + pear), np.sqrt(1 - pear)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry, **kw)
    sx, sy = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    tr = (transforms.Affine2D().rotate_deg(45).scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    return ax.add_patch(ell)


def make_figure(clouds, refs, N):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    handles = []
    for label, _, _, color in STRATS:
        pts = clouds.get(label)
        if pts is None:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.25, color=color, edgecolors="none")
        for ns in (1.0, 2.0):
            confidence_ellipse(pts[:, 0], pts[:, 1], ax, n_std=ns,
                               edgecolor=color, facecolor="none", lw=1.6,
                               ls="-" if ns == 1.0 else "--", alpha=0.9)
        xn, yn = refs.get(label, (None, None))
        if xn is not None:
            ax.scatter([xn], [yn], marker="*", s=180, color=color,
                       edgecolors="black", zorder=5)
        handles.append(Line2D([0], [0], color=color, lw=2, marker="*",
                              label="%s (nominal + ellipses 1/2$\\sigma$)" % label))
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€ / 25 ans]")
    ax.set_title("Dispersion du point de Pareto sous incertitude du pronostic\n"
                 "($\\sigma_{RUL}/RUL=%.0f\\%%$, $\\sigma_{SoH}=%.3f$, $N=%d$)"
                 % (SIGMA_RUL_REL * 100, SIGMA_SOH, N), fontsize=10)
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=150)
    print("  figure -> %s / %s" % (FIG_PDF, FIG_PNG))


if __name__ == "__main__":
    if "--replot" in sys.argv:
        clouds, refs = load_cloud()
        N = max((len(v) for v in clouds.values()), default=0)
    else:
        N = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 64
        clouds, refs = compute(N)
    write_stats(clouds, refs, N)
    make_figure(clouds, refs, N)
