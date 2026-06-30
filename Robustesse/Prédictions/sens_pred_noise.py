# -*- coding: utf-8 -*-
"""
sens_pred_noise.py -- SENSIBILITE AU BRUIT DE PREDICTION (ellipses de Pareto).
==============================================================================
But : VALIDER que la demarche de robustesse (hysteresis anti-clignotement) est
utile face au bruit de prediction. Meme esprit que la sensibilite aux seuils EoL
de Analyse_sensibilite/ (sens_eol.py) : chaque strategie porte un NUAGE Monte-Carlo
+ une ELLIPSE DE CONFIANCE dans le plan de Pareto (LPSP %, deg kEUR/25ans).

Parametre balaye = l'ECART-TYPE DU BRUIT INJECTE sur l'energie nette prevue.
On l'echantillonne dans une bande autour de la valeur backtest (sigma0=39.38 kWh)
    sigma_inject ~ Uniform([SIG_LO, SIG_HI] * sigma0)
ce qui teste une MISESTIMATION de sigma (le vrai bruit peut differer de l'estime).
POINT CLE : la BANDE d'hysteresis reste calee sur sigma0 (valeur de design figee,
strat.SIGMA_E_KWH) ; seul le bruit REELLEMENT injecte varie (strat.SIGMA_INJECT_KWH).
C'est le test honnete : on regle le filtre une fois, la realite varie.

Common random numbers : les MEMES couples (sigma_i, seed_i) sont appliques aux 4
nuages (2 strategies x {binaire, hysteresis}) -> comparaison non polluee par le MC.

Lecture attendue : l'ellipse BINAIRE (sans hysteresis) est PLUS GRANDE et placee
PLUS HAUT/DROITE (degradation par clignotement ELY) ; l'ellipse HYSTERESIS est plus
COMPACTE et placee plus bas/gauche (domine la baseline). Si oui, l'anti-clignotement
est demontre robuste au bruit de prediction.

COMPUTE vs PLOT decouples : le calcul ecrit le NUAGE BRUT (sens_pred_noise_cloud.csv)
+ les stats marginales (sens_pred_noise.txt) ; la figure est tracee a partir du
nuage. On peut donc re-tracer SANS recalculer :
    python sens_pred_noise.py --replot          # relit le CSV, refait la figure
Sorties (dans Predictions/) : sens_pred_noise.{pdf,png}, .txt, _cloud.csv.
Usage calcul : python sens_pred_noise.py [N_SAMPLES] [N_YEARS]   (defaut N=24, 25 ans)
"""
import os, sys, time, csv
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

E_REF_KWH = 273380.731444
VOLL      = 3.0
SIGMA0    = 39.38           # ecart-type backtest (kWh @18h), valeur de design
SIG_LO, SIG_HI = 0.5, 1.5   # bande de misestimation : sigma_inject in [0.5,1.5]*sigma0
MC_SEED   = 2026

# Strategies de prediction (label, dossier, couleur). RB2(SoH)=RB2(SoH+Pred) une
# fois ENABLE. Panneau du HAUT = 1er, panneau du BAS = 2e (axe Y brise).
STRATS = [("RB2(Pred)", "RB2(Pred)", "#1f4e79"),
          ("RB2(SoH+Pred)", "RB2(SoH)", "#d95f02")]
BASE_LABEL = {"RB2(Pred)": "RB2", "RB2(SoH+Pred)": "RB2(SoH)"}
TOP_LABELS = {"RB2(Pred)", "RB2"}   # le reste -> panneau du bas

VAR_STYLE = {"bin":  dict(ls="--", marker="x", name="binaire (fragile)"),
             "hyst": dict(ls="-",  marker="o", name="+ hysteresis (robuste)")}

CLOUD_CSV = os.path.join(HERE, "sens_pred_noise_cloud.csv")
STATS_TXT = os.path.join(HERE, "sens_pred_noise.txt")


# ============================ CALCUL ============================
def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + folder.replace("(", "").replace(")", "").replace("+", ""),
        os.path.join(HERE, folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
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
    cost = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely,
                          P_bat, SoC, LOAD, BAT, FC, ELY, SoH_bat) / 1000.0
    return float(lpsp), float(cost)


def evaluate(task):
    """Worker picklable. task = dict(key, folder, enable, noise, hyst,
    sigma_inject, seed, ny). Renvoie label/variant/sigma/seed/lpsp/deg."""
    from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
    s = _load(task["folder"])
    s.ENABLE       = task["enable"]
    s.NOISE_ENABLE = task["noise"]
    s.HYST_ENABLE  = task["hyst"]
    s.M_SIGMA      = 1.0
    s.MIN_DWELL    = 12
    s.SIGMA_E_KWH  = SIGMA0                     # bande hysteresis = design fige
    s.SIGMA_INJECT_KWH = task["sigma_inject"]   # bruit injecte (varie ou None)
    s.set_noise_seed(task["seed"])
    s.reset()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                      n_years=task["ny"])
    lpsp, deg = metrics(data)
    return dict(label=task["key"][0], variant=task["key"][1],
                sigma_inject=task["sigma_inject"], seed=task["seed"],
                lpsp=lpsp, deg=deg)


def compute(N, ny):
    workers_env = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = int(workers_env) if workers_env else max(1, (os.cpu_count() or 2) - 1)

    rng = np.random.default_rng(MC_SEED)
    sig_samples = [float(rng.uniform(SIG_LO, SIG_HI) * SIGMA0) for _ in range(N)]

    tasks = []
    for label, folder, _ in STRATS:                 # nuages MC : 2 strat x 2 variantes
        for variant, hyst in (("bin", False), ("hyst", True)):
            for i, sg in enumerate(sig_samples):
                tasks.append(dict(key=(label, variant), folder=folder, enable=True,
                                  noise=True, hyst=hyst, sigma_inject=sg, seed=i, ny=ny))
    for label, folder, _ in STRATS:                 # reperes deterministes
        tasks.append(dict(key=(label, "omni"), folder=folder, enable=True,
                          noise=False, hyst=False, sigma_inject=None, seed=0, ny=ny))
        tasks.append(dict(key=(BASE_LABEL[label], "baseline"), folder=folder,
                          enable=False, noise=False, hyst=False, sigma_inject=None,
                          seed=0, ny=ny))

    print("=" * 78)
    print("SENSIBILITE BRUIT DE PREDICTION : ellipses Pareto (N=%d, %d ans, %d workers)"
          % (N, ny, workers))
    print("  sigma_inject ~ U([%.1f,%.1f]*%.2f) = [%.1f, %.1f] kWh | bande hyst figee a %.2f"
          % (SIG_LO, SIG_HI, SIGMA0, SIG_LO * SIGMA0, SIG_HI * SIGMA0, SIGMA0), flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, tasks):
            rows.append(r)
    print("  (%d runs en %.0fs)" % (len(tasks), time.time() - t0), flush=True)

    with open(CLOUD_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "variant", "sigma_inject", "seed", "lpsp", "deg"])
        for r in rows:
            w.writerow([r["label"], r["variant"],
                        "" if r["sigma_inject"] is None else "%.4f" % r["sigma_inject"],
                        r["seed"], "%.6f" % r["lpsp"], "%.6f" % r["deg"]])
    print("  nuage brut -> %s" % CLOUD_CSV)
    return load_cloud()


# ============================ I/O NUAGE ============================
def load_cloud():
    """Relit sens_pred_noise_cloud.csv -> (clouds, refs).
    clouds[(label,variant)] = Nx2 array (lpsp,deg) ; refs[(label,variant)] = (lpsp,deg)."""
    clouds, refs = {}, {}
    with open(CLOUD_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["label"], row["variant"])
            xy = (float(row["lpsp"]), float(row["deg"]))
            if row["variant"] in ("bin", "hyst"):
                clouds.setdefault(key, []).append(xy)
            else:
                refs[key] = xy
    clouds = {k: np.array(v) for k, v in clouds.items()}
    return clouds, refs


def write_stats(clouds, refs, N, ny):
    with open(STATS_TXT, "w") as f:
        f.write("# Sensibilite au bruit de prediction -- ellipses de Pareto\n")
        f.write("# sigma_inject ~ U([%.2f,%.2f]*%.2f) kWh ; bande hyst figee a %.2f ; N=%s ; %s ans\n"
                % (SIG_LO, SIG_HI, SIGMA0, SIGMA0, N, ny))
        f.write("strat;variant;LPSP_mean;LPSP_std;deg_mean;deg_std;N\n")
        for label, _, _ in STRATS:
            for variant in ("bin", "hyst"):
                pts = clouds.get((label, variant))
                if pts is None:
                    continue
                f.write("%s;%s;%.4f;%.4f;%.4f;%.4f;%d\n"
                        % (label, variant, pts[:, 0].mean(), pts[:, 0].std(),
                           pts[:, 1].mean(), pts[:, 1].std(), len(pts)))
        for (label, variant), (x, y) in refs.items():
            f.write("%s;%s;%.4f;0;%.4f;0;1\n" % (label, variant, x, y))


# ============================ FIGURE ============================
def confidence_ellipse(x, y, ax, n_std=1.0, **kw):
    """Ellipse de covariance (n_std ecarts-types). Approche valeurs/vecteurs propres."""
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


def _panel_lims(vals, pad_frac=0.18, min_pad=0.05):
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * pad_frac, min_pad)
    return lo - pad, hi + pad


def make_figure(clouds, refs, ny):
    plt.rcParams.update({"font.family": "serif", "axes.labelsize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 10, "pdf.fonttype": 42})
    fig, (axT, axB) = plt.subplots(2, 1, sharex=True, figsize=(8.4, 7.2),
                                   gridspec_kw={"hspace": 0.08})
    ax_of = lambda lab: axT if lab in TOP_LABELS else axB
    yT, yB = [], []   # valeurs pour caler les ylim de chaque panneau

    # --- nuages + ellipses ---------------------------------------------------
    for label, _, col in STRATS:
        ax = ax_of(label)
        bucket = yT if label in TOP_LABELS else yB
        for variant in ("bin", "hyst"):
            pts = clouds.get((label, variant))
            if pts is None or len(pts) == 0:
                continue
            x, y = pts[:, 0], pts[:, 1]
            bucket += [y.min(), y.max()]
            st = VAR_STYLE[variant]
            ax.scatter(x, y, s=12, color=col, alpha=0.28, marker=st["marker"], zorder=3)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor="none",
                               lw=2.0, ls=st["ls"], zorder=5)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor="none",
                               lw=1.0, ls=st["ls"], alpha=0.5, zorder=5)
            ax.scatter([x.mean()], [y.mean()], s=90, color=col, marker="D",
                       edgecolor="k", linewidth=0.8, zorder=7)

    # --- reperes fixes : baselines (carre) + omniscients (etoile) ------------
    for (label, variant), (x, y) in refs.items():
        ax = ax_of(label)
        bucket = yT if label in TOP_LABELS else yB
        bucket.append(y)
        if variant == "baseline":
            ax.scatter([x], [y], s=130, color="0.25", marker="s", edgecolor="k",
                       linewidth=0.7, zorder=8)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4),
                        fontsize=12, weight="bold", color="0.2")
        else:  # omniscient
            col = dict((l, c) for l, _, c in STRATS)[label]
            ax.scatter([x], [y], s=120, color=col, marker="*", edgecolor="k",
                       linewidth=0.6, zorder=8)
            ax.annotate(label + " omni.", (x, y), textcoords="offset points",
                        xytext=(8, -3), fontsize=9, color=col)

    axT.set_ylim(*_panel_lims(yT)); axB.set_ylim(*_panel_lims(yB))

    # --- cassure d'axe (diagonales) ------------------------------------------
    axT.spines["bottom"].set_visible(False)
    axB.spines["top"].set_visible(False)
    axT.tick_params(axis="x", which="both", bottom=False)
    d = 0.012
    for ax, ys in ((axT, (0, 0)), (axB, (1, 1))):
        ax.plot([-d, +d], [ys[0] - d, ys[0] + d], transform=ax.transAxes,
                color="k", lw=1, clip_on=False)
        ax.plot([1 - d, 1 + d], [ys[1] - d, ys[1] + d], transform=ax.transAxes,
                color="k", lw=1, clip_on=False)

    for ax in (axT, axB):
        ax.grid(True, ls="--", alpha=0.5)
    axB.set_xlabel("LPSP [%]")
    fig.supylabel("Coût de dégradation [k€]", x=0.04, fontsize=15)
    axT.set_title("Robustesse au bruit de prédiction : ellipses 1σ/2σ (Pareto, %d ans)"
                  % ny, fontsize=13)

    # --- legendes ------------------------------------------------------------
    leg_col = [Line2D([0], [0], color=c, lw=3, label=l) for l, _, c in STRATS]
    leg_var = [Line2D([0], [0], color="0.3", lw=2, ls=VAR_STYLE[v]["ls"],
                      marker=VAR_STYLE[v]["marker"], label=VAR_STYLE[v]["name"])
               for v in ("bin", "hyst")]
    leg_ref = [Line2D([0], [0], color="0.25", marker="s", ls="none",
                      label="baseline (sans préd.)"),
               Line2D([0], [0], color="0.4", marker="*", ls="none",
                      label="omniscient (borne sup.)")]
    l1 = axT.legend(handles=leg_col, loc="upper left", title="Stratégie", framealpha=0.9)
    axT.add_artist(l1)
    axB.legend(handles=leg_var + leg_ref, loc="lower right", framealpha=0.9)

    # Pas de tight_layout (incompatible avec l'axe brise + supylabel) ; bbox_inches
    # ='tight' au savefig suffit a rogner proprement.
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.09, hspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "sens_pred_noise." + ext), dpi=130,
                    bbox_inches="tight")
    print("Figure -> sens_pred_noise.{pdf,png}")


def print_summary(clouds, refs):
    print("\nResume (moyenne +/- std sur le nuage) :")
    for label, _, _ in STRATS:
        for variant in ("bin", "hyst"):
            pts = clouds.get((label, variant))
            if pts is not None:
                print("  %-14s %-5s : LPSP %.4f+/-%.4f  deg %.4f+/-%.4f"
                      % (label, variant, pts[:, 0].mean(), pts[:, 0].std(),
                         pts[:, 1].mean(), pts[:, 1].std()))


def main():
    args = [a for a in sys.argv[1:] if a != "--replot"]
    replot = "--replot" in sys.argv
    N  = int(args[0]) if len(args) > 0 else 24
    ny = int(args[1]) if len(args) > 1 else 25

    if replot:
        if not os.path.exists(CLOUD_CSV):
            sys.exit("--replot : %s introuvable (lance d'abord le calcul)." % CLOUD_CSV)
        clouds, refs = load_cloud()
        print("Re-trace depuis %s (sans recalcul)." % CLOUD_CSV)
    else:
        clouds, refs = compute(N, ny)
        write_stats(clouds, refs, N, ny)

    make_figure(clouds, refs, ny)
    print_summary(clouds, refs)
    print("Donnees -> %s ; %s" % (STATS_TXT, CLOUD_CSV))


if __name__ == "__main__":
    main()
