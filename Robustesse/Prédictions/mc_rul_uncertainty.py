# -*- coding: utf-8 -*-
"""
mc_rul_uncertainty.py -- SENSIBILITE A L'INCERTITUDE DU PRONOSTIC (ellipses Pareto).
====================================================================================
Meme demarche que Analyse_sensibilite/sens_soh_estimation.py (front de Pareto +
ellipses de confiance sous propagation Monte-Carlo d'une erreur d'estimation),
appliquee ici a la RUL vue par RB2(RUL). But : etayer QUANTITATIVEMENT l'argument
de la section RB2(RUL) -- la RUL est une grandeur EXTRAPOLEE, donc bien plus
incertaine que le SoH instantane -- en mesurant la TAILLE des ellipses de
confiance autour du point de fonctionnement dans le plan de Pareto
(LPSP %, deg kEUR/25 ans), et en la comparant a celle de RB2(SoH).

MODELE DE BRUIT (piecewise-constant, rafraichi 1x/semaine, comme sens_soh_estimation)
------------------------------------------------------------------------------------
L'estimateur (LSTM + MC-dropout du chapitre pronostic) est re-interroge
periodiquement ; chaque estimation porte une erreur, CONSTANTE entre deux
rafraichissements (un bruit blanc horaire se moyennerait a tort sur la semaine).
  - RB2(RUL) : la RUL_ely vue par la strategie est multipliee par (1 + eps),
      eps ~ N(0, sigma_rel(RUL))   [erreur RELATIVE].
    *** sigma_rel DEPEND DE L'HORIZON *** : l'incertitude d'estimation de la RUL
    n'est pas la meme selon que la fin de vie est proche ou lointaine. On
    interpole sigma_rel sur la COURBE MESUREE au MC-dropout
    (Pronostic SoH/sigma_rul_vs_horizon.csv, cf compute_prono_sigma_horizons.py) :
    en U, ~5 % vers 400-440 j, jusqu'a ~12 % a 800 j. A chaque rafraichissement on
    lit la RUL_ely courante et on tire eps avec le sigma_rel correspondant.
  - RB2(SoH) : le SoH_ely vu par la strategie est decale de delta,
      delta ~ N(0, sigma_SoH)      [erreur ABSOLUE ; sigma_SoH ~ 3e-4, mesure au
      MC-dropout, quasi independante de l'horizon -> ~2 ordres de grandeur sous
      l'incertitude relative de la RUL].
Seul le SIGNAL VU PAR LE CONTROLEUR est bruite ; les vraies trajectoires (donc
LPSP et cout) subissent l'effet de la mauvaise decision -> on mesure l'impact
REEL de l'incertitude d'estimation.

Lecture attendue : l'ellipse RB2(RUL) est PLUS GRANDE que l'ellipse RB2(SoH)
-> le gain nominal du pronostic RUL ne survit pas a son incertitude -> argument
quantitatif pro-RB2(SoH).

NE MODIFIE AUCUN fichier de strategie (injection par closure autour de la fonction).

Sorties (dans Predictions/) : mc_rul_uncertainty.{pdf,png}, .txt, _cloud.csv.
Usage :
    python mc_rul_uncertainty.py [N_MC]        # defaut N=64 ; validation locale : N=8
    python mc_rul_uncertainty.py --replot      # relit le CSV, refait la figure
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
VOLL          = 3.0      # EUR/kWh (coherent manuscrit ; sert seulement aux stats de cout total)
MC_SEED0      = 2026
REFRESH_STEPS = int(24 * 7)   # rafraichissement de l'estimation = 1 semaine (Ts=3600 s)

# --- COURBE MESUREE sigma(horizon) (MC-dropout PEMWE ; sigma_rul_vs_horizon.csv) ---
# RUL croissante ; sigma_RUL RELATIF (sigma_RUL/RUL) et sigma_SoH ABSOLU associes.
# Hors de la plage mesuree, np.interp maintient la valeur du bord (clamp) : sur.
RUL_GRID_J    = [275.0,  383.0,  439.0,  556.0,  799.0]
SIG_RUL_REL   = [0.072,  0.052,  0.051,  0.066,  0.121]
SIG_SOH_ABS   = [1.4e-4, 2.7e-4, 3.5e-4, 4.3e-4, 6.8e-4]

# Reglage retenu du levier RUL (cf manuscrit). RB2(SoH) garde son reglage interne.
EXP_ELY       = 0.1
RUL_ELY_REF   = 1000.0

STRATS = [("RB2(RUL)", "RB2(RUL)", "rul", "#1f4e79"),
          ("RB2(SoH)", "RB2(SoH)", "soh", "#d95f02")]

CLOUD_CSV = os.path.join(HERE, "mc_rul_uncertainty_cloud.csv")
STATS_TXT = os.path.join(HERE, "mc_rul_uncertainty.txt")
FIG_PDF   = os.path.join(HERE, "mc_rul_uncertainty.pdf")
FIG_PNG   = os.path.join(HERE, "mc_rul_uncertainty.png")


def sigma_rul_rel(rul):
    """Ecart-type RELATIF d'estimation de la RUL a l'horizon 'rul' (interpole,
    clampe aux bords de la plage mesuree)."""
    return float(np.interp(rul, RUL_GRID_J, SIG_RUL_REL))


def sigma_soh_abs(rul):
    """Ecart-type ABSOLU d'estimation du SoH instantane (interpole sur le meme
    experiment MC-dropout ; quasi constant ~3e-4)."""
    return float(np.interp(rul, RUL_GRID_J, SIG_SOH_ABS))


# ============================ CALCUL ============================
def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + folder.replace("(", "").replace(")", ""),
        os.path.join(HERE, folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
    """LPSP [%] et cout de degradation [kEUR/25 ans], identique a
    sens_common.metrics / batch_pareto (interpolation SoH_bat aux remplacements)."""
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
    eens  = np.clip(p - r, 0, None).sum() * (LOAD['Ts'] / 3600.0)          # kWh non servis / 25 ans
    deg   = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely,
                           P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(deg), float(eens)


# --- wrappers d'injection (closure a signature nommee, style sens_soh_estimation) ---
def make_rul_noisy(base, seed):
    """Enveloppe RB2(RUL) : la RUL_ely vue par la strategie est multipliee par un
    facteur (1+eps) rafraichi chaque semaine, eps ~ N(0, sigma_rel(RUL courante))."""
    rng = np.random.default_rng(seed)
    st = {'k': 0, 'fac': 1.0}

    def strat(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
              SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
              RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if st['k'] % REFRESH_STEPS == 0:
            sig = sigma_rul_rel(RUL_ely_t)          # horizon-dependant
            st['fac'] = 1.0 + (rng.normal(0.0, sig) if sig > 0 else 0.0)
        st['k'] += 1
        rul_est = max(RUL_ely_t * st['fac'], 0.0)
        return base(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                    alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                    P_ely_max_t, RUL_fc_t, rul_est, SoH_fc_t, SoH_ely_t)
    return strat


def make_soh_noisy(base, seed, soh_eol):
    """Enveloppe RB2(SoH) : le SoH_ely vu par la strategie est decale de delta
    rafraichi chaque semaine, delta ~ N(0, sigma_SoH), borne [SoH_EoL, 1]."""
    rng = np.random.default_rng(seed)
    st = {'k': 0, 'delta': 0.0}

    def strat(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
              SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
              RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if st['k'] % REFRESH_STEPS == 0:
            sig = sigma_soh_abs(RUL_ely_t)
            st['delta'] = rng.normal(0.0, sig) if sig > 0 else 0.0
        st['k'] += 1
        soh_est = min(max(SoH_ely_t + st['delta'], soh_eol), 1.0)
        return base(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                    alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                    P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, soh_est)
    return strat


def evaluate(task):
    from Common.main_init_and_loop import init_and_run_loop, ELY
    s = _load(task["folder"])
    if task["kind"] == "rul":
        s.RUL_ELY_REF = RUL_ELY_REF
        s.EXP_ELY     = EXP_ELY
        func = (make_rul_noisy(s.get_optimal_action_RB, MC_SEED0 + task["seed"])
                if task["noisy"] else s.get_optimal_action_RB)
    else:  # soh
        func = (make_soh_noisy(s.get_optimal_action_RB, MC_SEED0 + 100000 + task["seed"],
                               ELY["SoH_EoL"])
                if task["noisy"] else s.get_optimal_action_RB)
    lpsp, deg, eens = metrics(init_and_run_loop(func))
    return dict(label=task["label"], variant="cloud" if task["noisy"] else "nominal",
                seed=task["seed"], lpsp=lpsp, deg=deg, eens=eens)


def compute(N):
    workers_env = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = int(workers_env) if workers_env else max(1, min(2, (os.cpu_count() or 2) - 1))

    tasks = []
    for label, folder, kind, _ in STRATS:
        for i in range(N):
            tasks.append(dict(label=label, folder=folder, kind=kind, noisy=True, seed=i))
        tasks.append(dict(label=label, folder=folder, kind=kind, noisy=False, seed=-1))

    print("=" * 78)
    print("SENSIBILITE INCERTITUDE PRONOSTIC : ellipses Pareto (N=%d/strat, %d workers)"
          % (N, workers))
    print("  bruit hebdo | RUL: sigma_rel(horizon) in [%.1f;%.1f]%% | SoH: sigma_abs ~%.1e"
          % (min(SIG_RUL_REL) * 100, max(SIG_RUL_REL) * 100, np.mean(SIG_SOH_ABS)), flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, tasks):
            rows.append(r)
            print("  [%s] %s seed=%3d -> LPSP %7.4f%%  deg %7.2f kEUR"
                  % (r["label"], r["variant"][:5], r["seed"], r["lpsp"], r["deg"]), flush=True)
    print("  (%d runs en %.0fs)" % (len(tasks), time.time() - t0), flush=True)

    with open(CLOUD_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "variant", "seed", "lpsp", "deg", "eens"])
        for r in rows:
            w.writerow([r["label"], r["variant"], r["seed"],
                        "%.6f" % r["lpsp"], "%.6f" % r["deg"], "%.3f" % r["eens"]])
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
        f.write("# Sensibilite a l'incertitude du pronostic -- ellipses de Pareto (RB2(RUL) vs RB2(SoH))\n")
        f.write("# bruit hebdomadaire ; RUL sigma_rel(horizon) interpole sur sigma_rul_vs_horizon.csv ;\n")
        f.write("# SoH sigma_abs interpole (idem) ; EXP_ELY=%.2f ; RUL_ELY_REF=%.0f ; N=%d/strat\n"
                % (EXP_ELY, RUL_ELY_REF, N))
        f.write("strat ; LPSP_nom ; deg_nom ; LPSP_mean ; LPSP_std ; deg_mean ; deg_std ;"
                " aire_ellipse_1sig ; grand_axe_1sig\n")
        for label, _, _, _ in STRATS:
            pts = clouds.get(label)
            if pts is None or len(pts) < 3:
                continue
            xn, yn = refs.get(label, (float("nan"), float("nan")))
            cov = np.cov(pts[:, 0], pts[:, 1])
            area = float(np.pi * np.sqrt(max(np.linalg.det(cov), 0.0)))
            eig = np.linalg.eigvalsh(cov)
            semi_major = float(np.sqrt(max(eig[-1], 0.0)))
            f.write("%s ; %.4f ; %.3f ; %.4f ; %.4f ; %.3f ; %.3f ; %.5f ; %.4f\n"
                    % (label, xn, yn, pts[:, 0].mean(), pts[:, 0].std(),
                       pts[:, 1].mean(), pts[:, 1].std(), area, semi_major))
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
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    handles = []
    for label, _, _, color in STRATS:
        pts = clouds.get(label)
        if pts is None or len(pts) < 3:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.30, color=color, edgecolors="none", zorder=2)
        for ns, lw, ls in ((1.0, 1.8, "-"), (2.0, 1.0, "--")):
            confidence_ellipse(pts[:, 0], pts[:, 1], ax, n_std=ns,
                               edgecolor=color, facecolor="none", lw=lw, ls=ls, alpha=0.9, zorder=4)
        xn, yn = refs.get(label, (None, None))
        if xn is not None:
            ax.scatter([xn], [yn], marker="*", s=280, color=color,
                       edgecolors="black", linewidth=0.6, zorder=6)
        handles.append(Line2D([0], [0], color=color, lw=2, marker="*", markeredgecolor="k",
                              label="%s (nominal + ellipses 1/2$\\sigma$)" % label))
    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Cout de degradation [k€ / 25 ans]")
    ax.set_title("Dispersion du point de Pareto sous incertitude du pronostic\n"
                 "(bruit hebdomadaire, $\\sigma_{RUL}/RUL$ horizon-dependant, $N=%d$/strat)" % N,
                 fontsize=10)
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
    print("  figure -> %s / %s" % (FIG_PDF, FIG_PNG))


if __name__ == "__main__":
    if "--replot" in sys.argv:
        clouds, refs = load_cloud()
        N = max((len(v) for v in clouds.values()), default=0)
    else:
        digs = [a for a in sys.argv[1:] if a.isdigit()]
        N = int(digs[0]) if digs else 64
        clouds, refs = compute(N)
    write_stats(clouds, refs, N)
    make_figure(clouds, refs, N)
    print("Termine.")
