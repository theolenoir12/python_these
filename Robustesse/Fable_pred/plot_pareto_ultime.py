"""Pareto 25 ans -- FABLE_PRED -- RB2 ULTIME : empilement SoH + SoH_bat + prevision.

Meme base que ../Predictions/Pareto_2d_25y_sohpred.py (baselines identiques,
socle cost-min, VoLL=3). Nomenclature these : SoH_bat=plafond SoC ;
SoH_H2=gammas setpoints (ex RB2(SoH)) ; SoH_all=les deux (ex "unifiee").
Points ajoutes (moyennes MC N=200 pour les previsionnels, cf bench_fable.txt,
bench_ultime.txt, note_rb2_ultime.txt) :
    RB2(Pred)         : pre-charge +-1sigma sur RB2 nu (impact pur
                        de la prevision)                        -> (2.5312, 58.962)
    RB2(SoH_all)      : gammas (1,2) x plafond g=0.2 (determ.)  -> (3.1051, 52.870)
    RB2(SoH_all+Pred) : l'ultime = SoH_all + pre-charge         -> (2.9421, 53.095)
      (variante cible fixe 0.90 du sweep target : (2.9221, 53.137), total 77.103)
Deux versions :
    pareto_ems_ultime.pdf/.png          : le Pareto (LPSP, deg).
    pareto_ems_ultime_isocost.pdf/.png  : + lignes d'iso-cout unifie (VoLL=3).
Ne remplace aucun fichier existant."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Analyse_sensibilite")))
try:
    import voll_common as V
    ISO_SLOPE = V.cost_lpsp_keur(1.0)
except Exception:
    ISO_SLOPE = 8.2014

# --- Points (bench_fable.txt / bench_ultime.txt, N=200) ---
PRED    = (2.5312, 58.962)   # RB2(Pred) : pre-charge sur RB2 nu, moyenne MC
SOH_ALL = (3.1051, 52.870)   # RB2(SoH_all) : gammas + plafond (deterministe)
ULTIME  = (2.9421, 53.095)   # RB2(SoH_all+Pred) cible plafond, moyenne MC


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

points = np.array([
    [10.3855, 124.1937],   # 0-100
    [20.2667, 110.7658],   # 25-75
    [8.0744, 109.0235],    # 50-50
    [3.8032, 59.6765],     # 75-25
    [2.4851, 66.4122],     # 100-0
    [2.5920, 58.8499],     # RB2  (cost-min 0.440/0.310)
    [2.9089, 54.9115],     # RB2(SoH_H2)  (0.440*SoH_fc^1 / 0.310*SoH_ely^2, ex RB2(SoH))
    [PRED[0],    PRED[1]],     # RB2(Pred) : impact pur de la prevision
    [SOH_ALL[0], SOH_ALL[1]],  # RB2(SoH_all) : gammas + plafond
    [ULTIME[0],  ULTIME[1]],   # RB2(SoH_all+Pred) : le point final
    [1.2597, 80.1562],     # RB1
    [1.3389, 140.6745],    # SoC1
    [29.4642, 109.2535],   # SoC06
    [0.0000, 0.0000]       # Ideal
])
labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH_H2)',
          'RB2(Pred)', 'RB2(SoH_all)', 'RB2(SoH_all+Pred)', 'RB1', 'SoC1',
          'SoC06', 'Ideal']

STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH_H2)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
EXTRA_COLORS = {'RB2(SoH_all)': '#1b9e77', 'RB2(Pred)': '#d95f02',
                'RB2(SoH_all+Pred)': '#c02020'}
IDEAL_COLOR = '0.3'


def color_of(label):
    if label in EXTRA_COLORS:
        return darken(EXTRA_COLORS[label], 0.9)
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def draw_isocost(axis, C_levels, xlim, ylim, label_y=None, inset=False):
    axis.set_xlim(xlim); axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 100)
    for C in C_levels:
        axis.plot(xs, C - ISO_SLOPE * xs, ls=':', color='0.6', lw=0.9, zorder=0)
        if label_y is not None:
            xl = (C - label_y) / ISO_SLOPE
            if xlim[0] < xl < xlim[1]:
                axis.text(xl, label_y, f"{C:.0f}", fontsize=7.5 if not inset else 6.5,
                          color='0.4', ha='center', va='center', zorder=1,
                          path_effects=LABEL_STROKE)


# --- Overlay OPTIONNEL du front de Pareto PD (reference optimale globale) ---
# Usage : python plot_pareto_ultime.py --dp [chemin/dp_pareto_25y_51x51_v2.npz]
# Sans argument apres --dp, cherche le npz v2 puis legacy dans
# ../Vieillissement8/DP/{results_meso,results}. Ne change RIEN par defaut.
DP_FRONT = None
if "--dp" in sys.argv:
    k = sys.argv.index("--dp")
    if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("-"):
        _dp_path = sys.argv[k + 1]
    else:
        _dp_path = None
        _dp_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "Vieillissement8", "DP"))
        for _name in ("dp_pareto_25y_51x51_v2.npz", "dp_pareto_25y_51x51.npz"):
            for _sub in ("results_meso", "results"):
                _p = os.path.join(_dp_dir, _sub, _name)
                if _dp_path is None and os.path.exists(_p):
                    _dp_path = _p
    if _dp_path and os.path.exists(_dp_path):
        _d = np.load(_dp_path)
        _nd = _d['nondominated'].astype(bool) if 'nondominated' in _d.files \
            else np.ones(len(_d['eps']), dtype=bool)
        _o = np.argsort(_d['lpsp'][_nd])
        DP_FRONT = (_d['lpsp'][_nd][_o], _d['deg_keur'][_nd][_o])
        print("front PD overlay :", _dp_path)
    else:
        print("front PD introuvable -> overlay ignore")

DP_COLOR = '#c2185b'


def build_figure(iso_cost):
    fig, ax = plt.subplots(figsize=(8, 6))
    if DP_FRONT is not None:
        ax.plot(DP_FRONT[0], DP_FRONT[1], '-', color=DP_COLOR, lw=1.6, zorder=3)
        ax.scatter(DP_FRONT[0], DP_FRONT[1], color=DP_COLOR, s=18, zorder=3)
    for i, label in enumerate(labels):
        ax.scatter(points[i, 0], points[i, 1], color=color_of(label), s=60, alpha=0.9)
    for i, label in enumerate(labels):
        col = color_of(label)
        if label in ('RB2', '75-25'):
            pass   # etiquetes dans le zoom uniquement (cluster dense)
        elif label == 'SoC06':
            ax.text(points[i, 0] - 2, points[i, 1] - 3, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE, verticalalignment='top')
        elif label in ('RB2(SoH_all)', 'RB2(Pred)', 'RB2(SoH_all+Pred)', 'RB2(SoH_H2)'):
            pass   # etiquetes uniquement dans le zoom (cluster dense)
        else:
            ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)

    zoom_labels = ['75-25', '100-0', 'RB2', 'RB2(SoH_H2)', 'RB2(Pred)',
                   'RB2(SoH_all)', 'RB2(SoH_all+Pred)', 'RB1']
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    if DP_FRONT is not None:
        axins.plot(DP_FRONT[0], DP_FRONT[1], '-', color=DP_COLOR, lw=1.6, zorder=3)
        axins.scatter(DP_FRONT[0], DP_FRONT[1], color=DP_COLOR, s=16, zorder=3)
    for i, label in enumerate(labels):
        if label in zoom_labels:
            axins.scatter(points[i, 0], points[i, 1], color=color_of(label), s=70, alpha=0.9)
    zoom_offsets = {
        'RB1':               (0.10,  0.0, 'left',   'center'),
        '100-0':             (0.10,  1.0, 'left',   'bottom'),
        'RB2':               (0.12,  0.0, 'left',   'center'),
        'RB2(Pred)':         (-0.10, 0.6, 'right',  'bottom'),
        'RB2(SoH_H2)':       (0.00,  0.6, 'center', 'bottom'),
        'RB2(SoH_all)':      (0.14,  0.0, 'left',   'center'),
        'RB2(SoH_all+Pred)': (-0.02, -1.3, 'center', 'top'),
        '75-25':             (0.10,  0.0, 'left',   'center'),
    }
    for i, label in enumerate(labels):
        if label in zoom_labels:
            dx, dy, ha, va = zoom_offsets[label]
            axins.text(points[i, 0] + dx, points[i, 1] + dy, label, fontsize=11,
                       color=color_of(label), weight='bold', path_effects=LABEL_STROKE,
                       horizontalalignment=ha, verticalalignment=va)
    axins.set_xlim(1.0, 4.5)
    axins.set_ylim(47, 84)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)

    if iso_cost:
        main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
        draw_isocost(ax, list(range(70, 141, 10)), main_xlim, main_ylim, label_y=13)
        draw_isocost(axins, [70, 75, 80, 85, 90, 95, 100], (1.0, 4.5), (47, 84))

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    stem = "pareto_ems_ultime_isocost" if iso_cost else "pareto_ems_ultime"
    outdir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(outdir, stem + ".pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(outdir, stem + ".png"), format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("saved ->", stem + ".pdf / .png")


build_figure(iso_cost=False)
build_figure(iso_cost=True)
