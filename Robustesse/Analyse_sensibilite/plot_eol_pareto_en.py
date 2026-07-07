"""
plot_eol_pareto_en.py -- Replot AUTONOME (anglais) du front de Pareto EoL.
=========================================================================
Ne relance PAS la simulation : relit les statistiques deja calculees dans
results_meso/sens_eol.txt (ou results/sens_eol.txt) et retrace la figure pour
publication (Applied Energy), labels/titre EN ANGLAIS.

Style aligne sur pareto_ems.py (serif / Computer Modern, memes tailles de
police, placement des labels lisible et non superpose).

LIMITE assumee : le .txt ne stocke que les statistiques resumees par strategie
(moyenne, ecart-type, min/max sur chaque axe), PAS le nuage de points brut ni la
covariance LPSP<->cout. On peut donc tracer :
  - le point NOMINAL de chaque strategie (le point "officiel" du front) ;
  - une ellipse 1sigma / 2sigma ALIGNEE SUR LES AXES, centree sur la moyenne MC,
    de demi-axes (n_std * sigma_LPSP, n_std * sigma_cout).
On ne peut PAS reproduire l'inclinaison (correlation) des ellipses ni le nuage de
points disperses de la figure d'origine -- cela necessiterait les tableaux MC
bruts. Pour une version 100% fidele, re-executer sens_eol.py (qui, lui, a le
nuage en memoire) apres avoir traduit ses labels.

Usage :
    ~/miniconda3/envs/simu_env/bin/python plot_eol_pareto_en.py
Sortie : results_meso/sens_eol_pareto_en.pdf (a defaut results/).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys
from matplotlib.patches import Ellipse


def darken(color, factor=0.7):
    """Assombrit une couleur en multipliant sa luminosite (HLS) par factor."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


# Halo blanc fin autour du texte (lisibilite sur fond + grille)
LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]

# --- Style identique a pareto_ems.py (serif / Computer Modern) ---
plt.rcParams.update({
    "text.usetex": False,          # Pas besoin de LaTeX externe
    "mathtext.fontset": "cm",      # Computer Modern (style LaTeX)
    "font.family": "serif",        # Police generale
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Source des donnees : on prefere results_meso/ (run mesocentre, N=200) ---
def _find_txt():
    for sub in ("results_meso", "results"):
        p = os.path.join(HERE, sub, "sens_eol.txt")
        if os.path.isfile(p):
            return p, os.path.join(HERE, sub)
    raise FileNotFoundError("sens_eol.txt introuvable dans results_meso/ ni results/")

TXT, OUTDIR = _find_txt()

# --- Ordre des strategies (identique a sens_eol.py) ---
STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']

# Code couleur par strategie (tab10), identique a sens_eol.py.
COLOR_BY_STRAT = True
POINT_COLOR = 'royalblue'
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}

# Point ideal (0,0) : pas une strategie MC -> ajoute en dur, sans ellipse.
IDEAL_POINT = (0.0, 0.0)
IDEAL_COLOR = '0.3'  # gris fonce, comme dans pareto_ems.py

# --- Placement des labels repris de pareto_ems.py ---
# Decalages en COORDONNEES DONNEES (dx, dy), + alignements, comme le 1er script.
# Defaut : (+0.5, +0.5) en haut a droite du point.
LABEL_PLACEMENT = {
    'RB2':      dict(dx=-2.7, dy=+4.0, ha='left',   va='top'),
    # 'SoC06':    dict(dx=-2.0, dy=-3.0, ha='left',   va='top'),
    'RB2(SoH)': dict(dx=+0.3, dy=-4.0, ha='center', va='top'),
    'Ideal':    dict(dx=+0.5, dy=+0.5, ha='left',   va='bottom'),
}
DEFAULT_PLACEMENT = dict(dx=+0.5, dy=+0.5, ha='left', va='bottom')


def parse_front(path):
    """Lit le bloc '## Front de Pareto' : renvoie dict strat -> stats."""
    rows = {}
    in_block = False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Front"):
                in_block = True
                continue
            if in_block:
                if s.startswith("## OAT") or s.startswith("#") and "OAT" in s:
                    break
                if not s or s.startswith("strat;") or s.startswith("#"):
                    if s.startswith("## OAT"):
                        break
                    continue
                parts = s.split(";")
                if len(parts) < 9 or parts[0] not in STRAT_ORDER:
                    continue
                try:
                    rows[parts[0]] = dict(
                        lpsp_nom=float(parts[1]), deg_nom=float(parts[2]),
                        lpsp_mean=float(parts[3]), lpsp_std=float(parts[4]),
                        deg_mean=float(parts[5]), deg_std=float(parts[6]),
                    )
                except ValueError:
                    continue
    return rows


def main():
    rows = parse_front(TXT)
    if not rows:
        raise RuntimeError("Aucune ligne 'Front de Pareto' lue dans %s" % TXT)

    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)   # nuage reproductible
    N_CLOUD = 40                     # qq points par strategie, juste pour la dispersion

    for strat in STRAT_ORDER:
        if strat not in rows:
            continue
        r = rows[strat]
        base = COLORS[strat] if COLOR_BY_STRAT else POINT_COLOR
        col = darken(base, 0.7)  # version assombrie pour points, ellipses, texte

        # petit nuage TRES leger pour figurer la dispersion (a la maniere de
        # sens_eol_pareto). Le .txt ne garde pas le nuage MC brut -> on re-tire
        # quelques points depuis la meme loi que les ellipses : normale alignee
        # sur les axes, centree (lpsp_mean, deg_mean), ecarts-types (std).
        if r['lpsp_std'] > 0 or r['deg_std'] > 0:
            cx = rng.normal(r['lpsp_mean'], r['lpsp_std'], N_CLOUD)
            cy = rng.normal(r['deg_mean'], r['deg_std'], N_CLOUD)
            ax.scatter(cx, cy, s=6, color=col, alpha=0.4, edgecolor='none', zorder=2)

        # ellipses alignees sur les axes, centrees sur la moyenne MC (1s plein, 2s tirete)
        for n_std, lw, ls, alpha in ((1.0, 1.6, '-', 0.9), (2.0, 0.9, '--', 0.5)):
            ax.add_patch(Ellipse(
                (r['lpsp_mean'], r['deg_mean']),
                width=2 * n_std * r['lpsp_std'],
                height=2 * n_std * r['deg_std'],
                edgecolor=col, facecolor='none', lw=lw, ls=ls, alpha=alpha, zorder=4))

        # point nominal (seuils EoL de base) = point officiel du front
        ax.scatter([r['lpsp_nom']], [r['deg_nom']], s=60, color=col, alpha=0.9,
                   zorder=6)

        # label, place comme dans pareto_ems.py (offset en coordonnees donnees)
        p = LABEL_PLACEMENT.get(strat, DEFAULT_PLACEMENT)
        ax.text(r['lpsp_nom'] + p['dx'], r['deg_nom'] + p['dy'], strat,
                fontsize=14, color=col, weight='bold', path_effects=LABEL_STROKE,
                horizontalalignment=p['ha'], verticalalignment=p['va'], zorder=7)

    # --- point Ideal (0,0) : reference, sans ellipse ---
    ic = darken(IDEAL_COLOR, 0.7)
    ax.scatter([IDEAL_POINT[0]], [IDEAL_POINT[1]], s=60, color=ic, alpha=0.9,
               zorder=6)
    pi = LABEL_PLACEMENT['Ideal']
    ax.text(IDEAL_POINT[0] + pi['dx'], IDEAL_POINT[1] + pi['dy'], 'Ideal',
            fontsize=14, color=ic, weight='bold', path_effects=LABEL_STROKE,
            horizontalalignment=pi['ha'], verticalalignment=pi['va'], zorder=7)

    # --- Encart de zoom sur le cluster bas-gauche (comme Pareto_2d_25y.py) ---
    # Test : on rejoue nuage + ellipses + point nominal pour qq strategies seulement.
    zoom_strats = ['75-25', '100-0', 'RB2', 'RB2(SoH)', 'RB1']
    zoom_strats = [s for s in zoom_strats if s in rows]
    if zoom_strats:
        axins = ax.inset_axes([0.4, 0.05, 0.5, 0.4])  # [x0, y0, w, h] frac. des axes

        # decalages de labels adaptes a l'echelle agrandie : (dx, dy, ha, va)
        zoom_offsets = {
            'RB1':      (0.10, 0.0,  'left',   'center'),
            '100-0':    (0.10, 1.3,  'left',   'bottom'),
            'RB2':      (-0.10, -0.8, 'right', 'top'),
            'RB2(SoH)': (0.0, -1.2,  'center', 'top'),
            '75-25':    (0.10, 0.0,  'left',   'center'),
        }

        for strat in zoom_strats:
            r = rows[strat]
            base = COLORS[strat] if COLOR_BY_STRAT else POINT_COLOR
            col = darken(base, 0.7)

            if r['lpsp_std'] > 0 or r['deg_std'] > 0:
                cx = rng.normal(r['lpsp_mean'], r['lpsp_std'], N_CLOUD)
                cy = rng.normal(r['deg_mean'], r['deg_std'], N_CLOUD)
                axins.scatter(cx, cy, s=6, color=col, alpha=0.4, edgecolor='none', zorder=2)

            for n_std, lw, ls, alpha in ((1.0, 1.6, '-', 0.9), (2.0, 0.9, '--', 0.5)):
                axins.add_patch(Ellipse(
                    (r['lpsp_mean'], r['deg_mean']),
                    width=2 * n_std * r['lpsp_std'],
                    height=2 * n_std * r['deg_std'],
                    edgecolor=col, facecolor='none', lw=lw, ls=ls, alpha=alpha, zorder=4))

            axins.scatter([r['lpsp_nom']], [r['deg_nom']], s=70, color=col, alpha=0.9,
                          zorder=6)

            dx, dy, ha, va = zoom_offsets.get(strat, (0.10, 0.0, 'left', 'center'))
            axins.text(r['lpsp_nom'] + dx, r['deg_nom'] + dy, strat, fontsize=11,
                       color=col, weight='bold', path_effects=LABEL_STROKE,
                       horizontalalignment=ha, verticalalignment=va, zorder=7)

        # bornes auto a partir des points nominaux + moyennes +/- 2 sigma
        xs, ys = [], []
        for strat in zoom_strats:
            r = rows[strat]
            xs += [r['lpsp_nom'], r['lpsp_mean'] - 2 * r['lpsp_std'],
                   r['lpsp_mean'] + 2 * r['lpsp_std']]
            ys += [r['deg_nom'], r['deg_mean'] - 2 * r['deg_std'],
                   r['deg_mean'] + 2 * r['deg_std']]
        mx = 0.10 * (max(xs) - min(xs) or 1.0)
        my = 0.10 * (max(ys) - min(ys) or 1.0)
        axins.set_xlim(min(xs) - mx, max(xs) + mx)
        axins.set_ylim(min(ys) - my, max(ys) + my)
        axins.grid(True, linestyle='--', alpha=0.5)
        axins.tick_params(labelsize=10)
        ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)

    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Degradation cost [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "sens_eol_pareto_en.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print("OK -> %s" % out)


if __name__ == "__main__":
    main()