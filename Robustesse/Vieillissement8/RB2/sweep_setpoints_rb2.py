"""
sweep_setpoints_rb2.py
======================
Test de pertinence de l'information SoH dans RB2(SoH).

Question : RB2(SoH) est-elle meilleure que RB2 *parce qu'elle exploite le SoH*,
ou seulement *parce qu'elle part de meilleurs setpoints de base* (0.440/0.320 vs
0.450/0.330 pour RB2 nu) ? Pour trancher, on balaie une grille de setpoints FIXES
de RB2 (fractions de Pmax, SANS aucune modulation SoH) autour des valeurs
specifiees, et on regarde si le point RB2(SoH) DOMINE ce nuage (=> le SoH apporte
qqch) ou s'il se pose simplement DESSUS (=> gain du seul choix de setpoints).

La grille inclut deux points cles :
  - (0.450, 0.330) = RB2 nominal
  - (0.440, 0.320) = base de RB2(SoH) MAIS figee, sans le facteur SoH_ely^0.5
    -> c'est LE comparateur du "test-nul" : meme offset de base que RB2(SoH),
       il ne reste que l'effet de la modulation dynamique par le SoH.

Chaque couple (fc_frac, ely_frac) -> 1 simulation 25 ans -> 1 point (LPSP %, cout k€).
On ajoute le point de la vraie strategie RB2(SoH) (importee de son dossier).

Sorties (dans le dossier RB2/) :
  - sweep_setpoints_rb2.txt : tous les points
  - sweep_setpoints_rb2.pdf/.png : Pareto style Pareto_2d_25y

Lancement (depuis Vieillissement8/RB2/) :
    python sweep_setpoints_rb2.py            # sweep complet + figure
    python sweep_setpoints_rb2.py --smoke    # smoke-test (2 sims, horizon court)
    python sweep_setpoints_rb2.py --replot    # refait la figure depuis le .txt
Regler N_WORKERS selon la RAM dispo (defaut = coeurs-1, plafonne conseille a 2-3).
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement8/RB2
PARENT = os.path.dirname(HERE)                               # .../Vieillissement8
sys.path.insert(0, PARENT)                                   # pour importer Common
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_total
from Common.get_lol import get_lol

# ======================= CONFIGURATION =======================
N_YEARS   = 25                       # horizon (defaut du coeur)
# Fractions de Pmax a balayer. FC non module par le SoH dans RB2(SoH) (gamma_FC=0)
# -> 2 valeurs suffisent ; ELY module (gamma_ELY=0.5) -> on encadre 0.30-0.33.
FC_FRACS  = [0.440, 0.450]
ELY_FRACS = [0.290, 0.310, 0.320, 0.330, 0.350]
# Nb de coeurs dispo : honore SLURM_CPUS_PER_TASK sur le mesocentre, sinon coeurs-1.
_N_AVAIL  = int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1))
_N_AVAIL  = max(1, _N_AVAIL)
OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb2.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb2.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb2.png")
# Points de contexte (autres EMS statiques, valeurs 25 ans figees, cf
# batch_results_summary_25y.txt) affiches en gris clair pour situer le nuage.
CONTEXT = {
    "75-25": (3.8032, 59.6765),
    "100-0": (2.4851, 66.4122),
    "RB1":   (1.2597, 80.1562),
}
# =============================================================


def make_rb2_frac(fc_frac, ely_frac):
    """Fabrique une action RB2 IDENTIQUE a l'originale (plafonds H2 inclus), mais
    avec les deux setpoints donnes en FRACTION de Pmax (fixes, sans SoH)."""
    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        P_fc_set  = fc_frac  * I.FC['P_fc_max']
        P_ely_set = ely_frac * I.ELY['P_ely_max']

        dt_h         = I.LOAD['Ts'] / 3600.0
        P_fc_h2_max  = max(E_h2_t, 0.0)             / dt_h * I.FC['eff']  * I.CONV['eta'] * 1000
        P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0) / dt_h / (I.ELY['eff'] * I.CONV['eta']) * 1000

        if P_tot_ref_t > 0:
            P_fc_avail = min(P_fc_set, P_fc_h2_max)
            if P_tot_ref_t > P_fc_avail:
                P_dc_fc_t  = P_fc_avail
                P_dc_bat_t = P_tot_ref_t - P_fc_avail
            else:
                P_dc_fc_t  = 0
                P_dc_bat_t = P_tot_ref_t
            P_dc_ely_t = 0
        if P_tot_ref_t < 0:
            P_ely_avail = min(P_ely_set, P_ely_h2_max)
            if P_tot_ref_t < -P_ely_avail:
                P_dc_ely_t = -P_ely_avail
                P_dc_bat_t = P_tot_ref_t + P_ely_avail
            else:
                P_dc_ely_t = 0
                P_dc_bat_t = P_tot_ref_t
            P_dc_fc_t = 0

        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t
        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol
    return get_optimal_action_RB


def _compute_metrics(data):
    """(LPSP %, cout k€) EXACTEMENT comme batch_pareto._compute_metrics
    (avec interpolation NaN de SoH_bat avant remplacement)."""
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc  = data["alpha_fc"][:-1]
    alpha_ely = data["alpha_ely"][:-1]
    SoH_bat   = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)),
            SoH_bat[~np.isnan(SoH_bat)])
    P_planned = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    P_real    = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    cost_keur = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
                               I.LOAD, I.BAT, I.FC, I.ELY, SoH_bat) / 1000
    return float(lpsp), float(cost_keur)


def _eval_frac(args):
    fc_frac, ely_frac, n_years = args
    data = init_and_run_loop(make_rb2_frac(fc_frac, ely_frac), n_years=n_years)
    lpsp, cost = _compute_metrics(data)
    return (fc_frac, ely_frac, lpsp, cost)


def _eval_rb2soh(n_years):
    soh_dir = os.path.join(PARENT, "RB2(SoH)")
    sys.path.insert(0, soh_dir)
    sys.modules.pop("get_optimal_action_RB", None)
    import importlib
    mod = importlib.import_module("get_optimal_action_RB")
    data = init_and_run_loop(mod.get_optimal_action_RB, n_years=n_years)
    return _compute_metrics(data)


# ----------------------------- run -----------------------------
def run_sweep(smoke=False):
    n_years = 2 if smoke else N_YEARS
    if smoke:
        combos = [(0.450, 0.330, n_years), (0.440, 0.320, n_years)]
    else:
        combos = [(fc, el, n_years) for fc in FC_FRACS for el in ELY_FRACS]
    n_workers = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB2 : {len(combos)} sims sur {n_years} ans ({n_workers} workers) ---",
          flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_eval_frac, combos), 1):
            results.append(res)
            print(f"  [{i:2d}/{len(combos)}] fc={res[0]:.3f} ely={res[1]:.3f} "
                  f"-> LPSP={res[2]:6.3f}%  cout={res[3]:8.3f} k€", flush=True)
    print("  ... RB2(SoH) (strategie reelle)", flush=True)
    soh_lpsp, soh_cost = _eval_rb2soh(n_years)
    print(f"  [SoH] RB2(SoH) -> LPSP={soh_lpsp:6.3f}%  cout={soh_cost:8.3f} k€", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Sweep setpoints RB2 (fixes, sans SoH) vs RB2(SoH) reelle - {n_years} ans\n")
        f.write(f"# P_fc_max={I.FC['P_fc_max']:.2f} W  P_ely_max={I.ELY['P_ely_max']:.2f} W\n")
        f.write("kind;fc_frac;ely_frac;LPSP(%);Cost(kEUR)\n")
        for fc, el, lpsp, cost in results:
            tag = "RB2_sweep"
            if abs(fc-0.450) < 1e-9 and abs(el-0.330) < 1e-9:
                tag = "RB2_nominal"
            elif abs(fc-0.440) < 1e-9 and abs(el-0.320) < 1e-9:
                tag = "RB2_base_of_SoH"   # meme base que RB2(SoH), SANS modulation
            f.write(f"{tag};{fc:.3f};{el:.3f};{lpsp:.4f};{cost:.4f}\n")
        f.write(f"RB2(SoH);0.440;0.320;{soh_lpsp:.4f};{soh_cost:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)
    return OUT_TXT


# ----------------------------- plot -----------------------------
def _darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


def _read_txt():
    sweep, nominal, base, soh = [], None, None, None
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            parts = line.strip().split(";")
            if len(parts) != 5:
                continue
            kind, fc, el, lpsp, cost = parts
            fc, el, lpsp, cost = float(fc), float(el), float(lpsp), float(cost)
            if kind == "RB2(SoH)":
                soh = (fc, el, lpsp, cost)
            elif kind == "RB2_nominal":
                nominal = (fc, el, lpsp, cost); sweep.append((fc, el, lpsp, cost))
            elif kind == "RB2_base_of_SoH":
                base = (fc, el, lpsp, cost); sweep.append((fc, el, lpsp, cost))
            else:
                sweep.append((fc, el, lpsp, cost))
    return sweep, nominal, base, soh


def plot():
    sweep, nominal, base, soh = _read_txt()
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 18, "axes.titlesize": 16, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
        "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
        "grid.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))

    # Contexte : autres EMS statiques (gris clair)
    for lab, (x, y) in CONTEXT.items():
        ax.scatter(x, y, color='0.7', s=45, alpha=0.8, zorder=1)
        ax.text(x + 0.03, y + 0.4, lab, fontsize=11, color='0.5',
                path_effects=LABEL_STROKE, zorder=1)

    # Nuage RB2 (setpoints fixes), relie par ligne d'iso-fc_frac pour lire la tendance
    sw = np.array([[s[2], s[3]] for s in sweep])
    fcs = sorted(set(s[0] for s in sweep))
    cmap = {fc: c for fc, c in zip(fcs, plt.cm.viridis(np.linspace(0.15, 0.75, len(fcs))))}
    for fc in fcs:
        pts = sorted([s for s in sweep if s[0] == fc], key=lambda s: s[1])
        xs = [p[2] for p in pts]; ys = [p[3] for p in pts]
        ax.plot(xs, ys, '-', color=cmap[fc], alpha=0.55, lw=1.4, zorder=2)
        ax.scatter(xs, ys, color=cmap[fc], s=42, zorder=3,
                   label=f"RB2 fixe, $f_{{FC}}$={fc:.3f}")
        for p in pts:
            ax.annotate(f"{p[1]:.3f}", (p[2], p[3]), fontsize=7, color=_darken(cmap[fc]),
                        xytext=(3, 3), textcoords="offset points", zorder=3)

    # Point "base de RB2(SoH) sans SoH" mis en evidence (comparateur test-nul)
    if base is not None:
        ax.scatter(base[2], base[3], facecolors='none', edgecolors='darkorange',
                   s=170, linewidths=2.2, zorder=4,
                   label="RB2 fixe base SoH (0.440/0.320, sans SoH)")
    # RB2 nominal
    if nominal is not None:
        ax.scatter(nominal[2], nominal[3], marker='s', color='black', s=55, zorder=5,
                   label="RB2 nominal (0.450/0.330)")

    # RB2(SoH) : la vraie strategie
    if soh is not None:
        ax.scatter(soh[2], soh[3], marker='*', color='crimson', s=300, zorder=6,
                   edgecolors='white', linewidths=0.8, label="RB2(SoH)")
        ax.text(soh[2] + 0.02, soh[3] - 0.5, "RB2(SoH)", fontsize=14, color='crimson',
                weight='bold', path_effects=LABEL_STROKE, ha='left', va='top', zorder=6)

    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.92)
    ax.set_title("Balayage des setpoints fixes de RB2 vs RB2(SoH) (25 ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
    plt.savefig(OUT_PNG, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure : {OUT_PDF}\n         {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 sims, horizon court")
    ap.add_argument("--replot", action="store_true", help="refait la figure depuis le .txt")
    args = ap.parse_args()
    if args.replot:
        plot()
    else:
        run_sweep(smoke=args.smoke)
        plot()
