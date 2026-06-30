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
nuages (2 strategies x {binaire, hysteresis}) -> la comparaison des ellipses n'est
pas polluee par le bruit Monte-Carlo.

Lecture attendue : l'ellipse BINAIRE (sans hysteresis) est PLUS GRANDE et placee
PLUS HAUT/DROITE (degradation par clignotement ELY, perte de fiabilite) ; l'ellipse
HYSTERESIS est plus COMPACTE et placee plus bas/gauche (domine la baseline). Si oui,
l'anti-clignotement est demontre robuste au bruit de prediction.

Sorties (dans Predictions/) : sens_pred_noise.{pdf,png} + sens_pred_noise.txt.
Usage : python sens_pred_noise.py [N_SAMPLES] [N_YEARS]   (defaut N=24, 25 ans)
"""
import os, sys, time
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from Common.Init_EMR_MG_v16_python import LOAD, BAT, FC, ELY            # noqa: E402
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast  # noqa: E402
from Common.cost_fcn_total2 import get_cost_total                       # noqa: E402

E_REF_KWH = 273380.731444
VOLL      = 3.0
SIGMA0    = 39.38           # ecart-type backtest (kWh @18h), valeur de design
SIG_LO, SIG_HI = 0.5, 1.5   # bande de misestimation : sigma_inject in [0.5,1.5]*sigma0
MC_SEED   = 2026

# Strategies de prediction (label, dossier). RB2(SoH) = RB2(SoH+Pred) une fois ENABLE.
STRATS = [("RB2(Pred)", "RB2(Pred)", "#1f4e79"),
          ("RB2(SoH+Pred)", "RB2(SoH)", "#d95f02")]
# Baselines (points fixes, sans prevision) : ENABLE=False de chaque dossier.
BASE_LABEL = {"RB2(Pred)": "RB2", "RB2(SoH+Pred)": "RB2(SoH)"}


def _load(folder):
    spec = importlib.util.spec_from_file_location(
        "strat_" + folder.replace("(", "").replace(")", "").replace("+", ""),
        os.path.join(HERE, folder, "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(data):
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
    """Worker picklable. task = dict(folder, enable, noise, hyst, sigma_inject, seed, ny)."""
    s = _load(task["folder"])
    s.ENABLE       = task["enable"]
    s.NOISE_ENABLE = task["noise"]
    s.HYST_ENABLE  = task["hyst"]
    s.M_SIGMA      = 1.0
    s.MIN_DWELL    = 12
    s.SIGMA_E_KWH  = SIGMA0                 # bande hysteresis = design fige
    s.SIGMA_INJECT_KWH = task["sigma_inject"]  # bruit injecte (varie ou None)
    s.set_noise_seed(task["seed"])
    s.reset()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                      n_years=task["ny"])
    lpsp, deg = metrics(data)
    return dict(key=task["key"], lpsp=lpsp, deg=deg)


def confidence_ellipse(x, y, ax, n_std=1.0, **kw):
    """Ellipse de covariance (n_std ecarts-types). Approche valeurs propres std."""
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


def main():
    N  = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    # Sur le mesocentre, le nombre de coeurs alloues est dans SLURM_CPUS_PER_TASK
    # (os.cpu_count() renverrait tous les coeurs physiques du noeud -> sur-souscription).
    _n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = int(_n_slurm) if _n_slurm else max(1, (os.cpu_count() or 2) - 1)

    rng = np.random.default_rng(MC_SEED)
    sig_samples = [float(rng.uniform(SIG_LO, SIG_HI) * SIGMA0) for _ in range(N)]

    # --- Construction des taches ---------------------------------------------
    tasks = []
    # nuages MC : 2 strategies x {binaire, hysteresis}, memes (sigma_i, seed_i)
    for label, folder, _ in STRATS:
        for variant, hyst in (("bin", False), ("hyst", True)):
            for i, sg in enumerate(sig_samples):
                tasks.append(dict(key=(label, variant), folder=folder, enable=True,
                                  noise=True, hyst=hyst, sigma_inject=sg, seed=i, ny=ny))
    # points de reference deterministes (1 run chacun)
    for label, folder, _ in STRATS:
        tasks.append(dict(key=(label, "omni"), folder=folder, enable=True,
                          noise=False, hyst=False, sigma_inject=None, seed=0, ny=ny))
        tasks.append(dict(key=(BASE_LABEL[label], "base"), folder=folder, enable=False,
                          noise=False, hyst=False, sigma_inject=None, seed=0, ny=ny))

    print("=" * 78)
    print("SENSIBILITE BRUIT DE PREDICTION : ellipses Pareto (N=%d, %d ans, %d workers)"
          % (N, ny, workers))
    print("  sigma_inject ~ U([%.1f,%.1f]*%.2f) = [%.1f, %.1f] kWh | bande hyst figee a %.2f"
          % (SIG_LO, SIG_HI, SIGMA0, SIG_LO * SIGMA0, SIG_HI * SIGMA0, SIGMA0))
    print("=" * 78, flush=True)
    t0 = time.time()

    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(evaluate, tasks):
            res.setdefault(r["key"], []).append((r["lpsp"], r["deg"]))
    print("  (%d runs en %.0fs)" % (len(tasks), time.time() - t0), flush=True)

    # --- Figure ---------------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "axes.labelsize": 16,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    VAR_STYLE = {"bin":  dict(ls="--", marker="x", fill=False, name="binaire (fragile)"),
                 "hyst": dict(ls="-",  marker="o", fill=True,  name="+ hysteresis (robuste)")}

    txt_lines = ["# Sensibilite au bruit de prediction -- ellipses de Pareto",
                 "# sigma_inject ~ U([%.2f,%.2f]*%.2f) kWh ; bande hyst figee a %.2f ; N=%d ; %d ans"
                 % (SIG_LO, SIG_HI, SIGMA0, SIGMA0, N, ny),
                 "strat;variant;LPSP_mean;LPSP_std;deg_mean;deg_std;N"]

    for label, folder, col in STRATS:
        for variant in ("bin", "hyst"):
            pts = np.array(res.get((label, variant), []))
            if len(pts) == 0:
                continue
            x, y = pts[:, 0], pts[:, 1]
            st = VAR_STYLE[variant]
            ax.scatter(x, y, s=14, color=col, alpha=0.30,
                       marker=st["marker"], zorder=3)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=col, facecolor='none',
                               lw=2.0, ls=st["ls"], zorder=5)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=col, facecolor='none',
                               lw=1.0, ls=st["ls"], alpha=0.5, zorder=5)
            ax.scatter([x.mean()], [y.mean()], s=90, color=col, marker='D',
                       edgecolor='k', linewidth=0.8, zorder=7)
            txt_lines.append("%s;%s;%.4f;%.4f;%.4f;%.4f;%d"
                             % (label, variant, x.mean(), x.std(), y.mean(), y.std(), len(x)))

    # --- repères fixes : baselines + omniscients ------------------------------
    seen_base = set()
    for label, folder, col in STRATS:
        blabel = BASE_LABEL[label]
        if blabel not in seen_base:
            bp = res.get((blabel, "base"))
            if bp:
                bx, by = bp[0]
                ax.scatter([bx], [by], s=120, color='0.25', marker='s', zorder=8,
                           edgecolor='k', linewidth=0.7)
                ax.annotate(blabel, (bx, by), textcoords="offset points",
                            xytext=(7, 4), fontsize=12, weight='bold', color='0.2')
                txt_lines.append("%s;baseline;%.4f;0;%.4f;0;1" % (blabel, bx, by))
            seen_base.add(blabel)
        op = res.get((label, "omni"))
        if op:
            ox, oy = op[0]
            ax.scatter([ox], [oy], s=110, color=col, marker='*', zorder=8,
                       edgecolor='k', linewidth=0.6)
            ax.annotate(label + " omni.", (ox, oy), textcoords="offset points",
                        xytext=(6, -12), fontsize=9, color=col)
            txt_lines.append("%s;omniscient;%.4f;0;%.4f;0;1" % (label, ox, oy))

    # --- legendes : couleur=strategie, style=variante -------------------------
    from matplotlib.lines import Line2D
    leg_col = [Line2D([0], [0], color=c, lw=3, label=l) for l, _, c in STRATS]
    leg_var = [Line2D([0], [0], color='0.3', lw=2, ls=VAR_STYLE[v]["ls"],
                      marker=VAR_STYLE[v]["marker"], label=VAR_STYLE[v]["name"])
               for v in ("bin", "hyst")]
    leg_ref = [Line2D([0], [0], color='0.25', marker='s', ls='none', label='baseline (sans pred.)'),
               Line2D([0], [0], color='0.4', marker='*', ls='none', label='omniscient (borne sup.)')]
    l1 = ax.legend(handles=leg_col, loc='upper left', title='Strategie', framealpha=0.9)
    ax.add_artist(l1)
    ax.legend(handles=leg_var + leg_ref, loc='lower right', framealpha=0.9)

    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Cout de degradation [k€]")
    ax.set_title("Robustesse au bruit de prediction : ellipses 1σ/2σ (Pareto, %d ans)" % ny,
                 fontsize=13)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "sens_pred_noise." + ext),
                    dpi=130, bbox_inches="tight")
    with open(os.path.join(HERE, "sens_pred_noise.txt"), "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    print("\nResume (moyenne +/- std sur le nuage) :")
    for label, _, _ in STRATS:
        for variant in ("bin", "hyst"):
            pts = np.array(res.get((label, variant), []))
            if len(pts):
                print("  %-14s %-5s : LPSP %.4f+/-%.4f  deg %.4f+/-%.4f"
                      % (label, variant, pts[:, 0].mean(), pts[:, 0].std(),
                         pts[:, 1].mean(), pts[:, 1].std()))
    print("\nFigures -> sens_pred_noise.{pdf,png} ; donnees -> sens_pred_noise.txt")


if __name__ == "__main__":
    main()
