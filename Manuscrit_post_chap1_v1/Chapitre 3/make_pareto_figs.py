# -*- coding: utf-8 -*-
"""Genere les figures de Pareto du chapitre 3 (nomenclature SoH_bat / SoH_H2 /
SoH_all), a partir du gabarit commun des figures existantes (encart zoom +
lignes d'iso-cout unifie VoLL=3).

Figures produites (PDF dans les dossiers Figures/ du chapitre + PNG de controle) :
    SoH/Figures/pareto_soh_isocost.pdf         panel + RB2(SoH_H2)
    SoH/Figures/pareto_soh_all_isocost.pdf     + RB2(SoH_bat) + RB2(SoH_all)
    RUL/Figures/pareto_rul_isocost.pdf         + RB2(RUL) (anneau creux sur RB2)
    Prediction/Figures/pareto_pred_isocost.pdf + RB2(Pred)
    Prediction/Figures/pareto_sohpred_isocost.pdf + RB2(SoH_H2+Pred)
    Prediction/Figures/pareto_ultime_isocost.pdf  + RB2(SoH_all+Pred)
    Synthese/Figures/pareto_strategies_v2.pdf  toutes + front de Pareto (PD)

Points : moyennes Monte-Carlo N=200 pour les previsionnels (bench_fable.txt,
bench_ultime.txt), runs deterministes sinon (sweep_fable_*.txt). Le front PD
est lu dans Vieillissement8/DP/results_meso/dp_pareto_25y_51x51.npz (points
non domines). Usage : python make_pareto_figs.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Robustesse", "Analyse_sensibilite"))
try:
    import voll_common as V
    ISO_SLOPE = V.cost_lpsp_keur(1.0)
except Exception:
    ISO_SLOPE = 8.2014

# ---------------- points (LPSP %, deg kEUR) ----------------
BASE = {                       # panel du chapitre 2 (inchange)
    '0-100':  (10.3855, 124.1937),
    '25-75':  (20.2667, 110.7658),
    '50-50':  (8.0744, 109.0235),
    '75-25':  (3.8032, 59.6765),
    '100-0':  (2.4851, 66.4122),
    'RB2':    (2.5920, 58.8499),
    'RB1':    (1.2597, 80.1562),
    'SoC1':   (1.3389, 140.6745),
    'SoC06':  (29.4642, 109.2535),
    'Ideal':  (0.0, 0.0),
}
# nomenclature these (labels mathtext)
L_SOHH   = 'RB2(SoH$_{\\mathrm{H_2}}$)'
L_SOHB   = 'RB2(SoH$_{\\mathrm{bat}}$)'
L_SOHA   = 'RB2(SoH$_{\\mathrm{all}}$)'
L_PRED   = 'RB2(Pred)'
L_SHP    = 'RB2(SoH$_{\\mathrm{H_2}}$+Pred)'
L_ULT    = 'RB2(SoH$_{\\mathrm{all}}$+Pred)'
L_RUL    = 'RB2(RUL)'
EXTRA = {
    L_SOHH: (2.9089, 54.9115),   # sweep_soh_attribution / sweep_fable_unified
    L_SOHB: (2.7679, 56.8420),   # sweep_fable_socwin_fine g=0.2
    L_SOHA: (3.1051, 52.8700),   # sweep_fable_unified (1,2)x0.2
    L_PRED: (2.5312, 58.9620),   # bench_fable N=200
    L_SHP:  (2.7575, 55.0750),   # bench_ultime N=200
    L_ULT:  (2.9421, 53.0950),   # bench_ultime N=200 (cible plafond)
}
EXTRA_COLORS = {
    L_SOHH: '#e441a1', L_SOHB: '#1b9e77', L_SOHA: '#d95f02',
    L_PRED: '#7d3fbf', L_SHP:  '#b8860b', L_ULT:  '#c02020',
    L_RUL:  '#404040',
}
STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', '_reserve', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
IDEAL_COLOR = '0.3'
FRONT_COL = '0.35'

LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 1.8,
    "lines.markersize": 5, "grid.alpha": 0.7, "grid.linestyle": "--",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})


def darken(color, factor=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, l * factor), s)


def color_of(label):
    if label in EXTRA_COLORS:
        return darken(EXTRA_COLORS[label], 0.9)
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def dp_front():
    p = os.path.join(REPO, "Robustesse", "Vieillissement8", "DP",
                     "results_meso", "dp_pareto_25y_51x51.npz")
    d = np.load(p)
    lpsp, deg = d["lpsp"], d["deg_keur"]
    n = len(lpsp)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and lpsp[j] <= lpsp[i] and deg[j] <= deg[i] and (
                    lpsp[j] < lpsp[i] or deg[j] < deg[i]):
                keep[i] = False
                break
    o = np.argsort(lpsp[keep])
    return lpsp[keep][o], deg[keep][o]


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


def build(outpath, extras, zoom_offsets, ring_on_rb2=False, front=False,
          zoom_ylim=(47, 84), zoom_xlim=(1.0, 4.5), iso=True):
    """extras : liste de labels EXTRA a ajouter au panel de base."""
    pts = dict(BASE)
    for lab in extras:
        pts[lab] = EXTRA[lab]
    fig, ax = plt.subplots(figsize=(8, 6))

    if front:
        fl, fd = dp_front()
        ax.plot(fl, fd, "-", color=FRONT_COL, lw=1.4, zorder=2)
        ax.scatter(fl, fd, color=FRONT_COL, s=10, zorder=2, marker="D")

    for lab, (x, y) in pts.items():
        ax.scatter(x, y, color=color_of(lab), s=60, alpha=0.9, zorder=3)
    if ring_on_rb2:
        ax.scatter(*BASE['RB2'], s=150, facecolors='none',
                   edgecolors=color_of(L_RUL), linewidths=1.6, zorder=4)
    for lab, (x, y) in pts.items():
        if lab in ('RB2', '75-25') or lab in EXTRA:
            continue           # etiquetes dans le zoom (cluster dense)
        col = color_of(lab)
        if lab == 'SoC06':
            ax.text(x - 2, y - 3, lab, fontsize=14, color=col, weight='bold',
                    path_effects=LABEL_STROKE, va='top')
        else:
            ax.text(x + 0.5, y + 0.5, lab, fontsize=14, color=col,
                    weight='bold', path_effects=LABEL_STROKE)

    zoom_labels = ['75-25', '100-0', 'RB2', 'RB1'] + list(extras)
    zoom_labels = [l for l in zoom_labels
                   if zoom_xlim[0] < pts[l][0] < zoom_xlim[1]
                   and zoom_ylim[0] < pts[l][1] < zoom_ylim[1]]
    axins = ax.inset_axes([0.45, 0.10, 0.52, 0.46])
    if front:
        axins.plot(fl, fd, "-", color=FRONT_COL, lw=1.2, zorder=2)
        axins.scatter(fl, fd, color=FRONT_COL, s=8, zorder=2, marker="D")
    for lab in zoom_labels:
        axins.scatter(*pts[lab], color=color_of(lab), s=70, alpha=0.9, zorder=3)
    if ring_on_rb2:
        axins.scatter(*BASE['RB2'], s=170, facecolors='none',
                      edgecolors=color_of(L_RUL), linewidths=1.8, zorder=4)
        axins.text(BASE['RB2'][0] + 0.10, BASE['RB2'][1] + 1.6, L_RUL,
                   fontsize=11, color=color_of(L_RUL), weight='bold',
                   path_effects=LABEL_STROKE, ha='left', va='bottom')
    base_off = {
        'RB1':   (0.10,  0.0, 'left', 'center'),
        '100-0': (0.10,  1.0, 'left', 'bottom'),
        'RB2':   (0.12, -0.5, 'left', 'top'),
        '75-25': (0.10,  0.0, 'left', 'center'),
    }
    base_off.update(zoom_offsets)
    for lab in zoom_labels:
        dx, dy, ha, va = base_off[lab]
        axins.text(pts[lab][0] + dx, pts[lab][1] + dy, lab, fontsize=11,
                   color=color_of(lab), weight='bold', path_effects=LABEL_STROKE,
                   ha=ha, va=va)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.tick_params(labelsize=10)

    if iso:
        draw_isocost(ax, list(range(70, 141, 10)), ax.get_xlim(), ax.get_ylim(),
                     label_y=13)
        draw_isocost(axins, [70, 75, 80, 85, 90, 95, 100], zoom_xlim, zoom_ylim)

    if front:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=FRONT_COL, marker="D", ls="-",
                                  markersize=4, label="front de Pareto (PD)")],
                  loc="upper right", fontsize=11, framealpha=0.9)

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.savefig(outpath.replace(".pdf", ".png"), format='png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("saved ->", os.path.relpath(outpath, HERE))


# ------------------- les 7 figures -------------------
F = os.path.join

# 1. RB2(SoH_H2) seule (section SoH, 1er levier)
build(F(HERE, "SoH", "Figures", "pareto_soh_isocost.pdf"),
      [L_SOHH],
      {'RB2':  (-0.10, 0.0, 'right', 'center'),
       L_SOHH: (0.12, 0.0, 'left', 'center')},
      zoom_ylim=(52, 84))

# 2. + SoH_bat + SoH_all (section SoH, unification)
build(F(HERE, "SoH", "Figures", "pareto_soh_all_isocost.pdf"),
      [L_SOHH, L_SOHB, L_SOHA],
      {'RB2':  (-0.10, 0.0, 'right',  'center'),
       L_SOHH: (-0.10, 0.0, 'right',  'center'),
       L_SOHB: (0.14,  0.0, 'left',   'center'),
       L_SOHA: (0.14,  0.0, 'left',   'center')})

# 3. RUL : anneau creux sur RB2, avec SoH_H2 en repere
build(F(HERE, "RUL", "Figures", "pareto_rul_isocost.pdf"),
      [L_SOHH],
      {'RB2':  (-0.10, 0.0, 'right', 'center'),
       L_SOHH: (0.12, 0.0, 'left', 'center')},
      ring_on_rb2=True, zoom_ylim=(52, 84))

# 4. RB2(Pred) (section prevision)
build(F(HERE, "Prediction", "Figures", "pareto_pred_isocost.pdf"),
      [L_PRED],
      {L_PRED: (0.05,  0.55, 'center', 'bottom'),
       'RB2':  (0.12,  0.0, 'left',  'center')},
      zoom_ylim=(52, 84))

# 5. RB2(SoH_H2+Pred) (combinaison niveau 1)
build(F(HERE, "Prediction", "Figures", "pareto_sohpred_isocost.pdf"),
      [L_SOHH, L_PRED, L_SHP],
      {L_PRED: (0.05,  0.55, 'center', 'bottom'),
       'RB2':  (0.12,  0.0, 'left',   'center'),
       L_SOHH: (0.12,  0.0, 'left',   'center'),
       L_SHP:  (-0.12, 0.0, 'right',  'center')},
      zoom_ylim=(50, 84))

# 6. RB2(SoH_all+Pred) (strategie complete)
build(F(HERE, "Prediction", "Figures", "pareto_ultime_isocost.pdf"),
      [L_PRED, L_SOHA, L_ULT],
      {L_PRED: (0.05,  0.55, 'center', 'bottom'),
       'RB2':  (0.12,  0.0, 'left',   'center'),
       L_SOHA: (0.14,  0.0, 'left',   'center'),
       L_ULT:  (-0.02, -1.5, 'center', 'top')})

# 7. synthese : tout + front PD
build(F(HERE, "Synthese", "Figures", "pareto_strategies_v2.pdf"),
      [L_SOHH, L_SOHB, L_SOHA, L_PRED, L_SHP, L_ULT],
      {L_PRED: (0.05,  0.45, 'center', 'bottom'),
       'RB2':  (0.10,  0.0, 'left',   'center'),
       '75-25': (0.07, 0.0, 'left',   'center'),
       L_SOHB: (0.09,  0.0, 'left',   'center'),
       L_SHP:  (-0.09, 0.0, 'right',  'center'),
       L_SOHH: (0.09,  0.0, 'left',   'center'),
       L_SOHA: (0.09,  0.35, 'left',  'bottom'),
       L_ULT:  (-0.01, -0.45, 'center', 'top')},
      front=True, zoom_xlim=(2.2, 4.2), zoom_ylim=(50, 62))
