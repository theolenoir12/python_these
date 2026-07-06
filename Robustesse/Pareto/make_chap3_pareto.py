# -*- coding: utf-8 -*-
"""make_chap3_pareto.py -- figures d'ESPACE DES OBJECTIFS (plans de Pareto) des
sections du chapitre 3, au STYLE de generate_pareto.py : RB2(SoH) UNIQUE, front de
Pareto (PD) MAGENTA, iso-couts SANS etiquette de valeur, taille (10, 6).
=================================================================================
CENTRALISATION + COHERENCE VISUELLE. Ce script REMPLACE l'ancien generateur qui
vivait dans l'arbre LaTeX (Chapitre 3/make_pareto_figs.py). Il reprend
EXACTEMENT le style de generate_pareto.py (meme dossier) : meme palette tab10,
front PD magenta (#c2185b) trace en ronds, lignes d'iso-cout en pointilles gris
SANS etiquette de valeur (retirees, cf. generate_pareto). La figure de la section
SoH (base + RB2(SoH)) est ainsi identique a base_soh_dp_isocost de generate_pareto.

Les figures VIVENT cote python et sont SYNCHRONISEES vers les dossiers Figures/
du manuscrit a l'execution.

Aucune simulation, aucun multiprocessing -> executable directement dans Spyder (F5).

CHIFFRES. base + RB2 + RB2(SoH) : identiques a generate_pareto (valeurs actuelles).
RB2(Pred)/RB2(RUL)/RB2(SoH+Pred) : chiffres du CHAPITRE 3 (tableaux du manuscrit) ;
ils DIFFERENT de ceux de generate_pareto (generation distincte, non melangeable)
-> on garde ceux du chapitre pour rester coherent avec ses tableaux.

Figures produites (PDF + PNG), copiees dans le manuscrit (section -> Figures/) :
  pareto_soh_isocost        SoH         RB2(SoH)                 (= base_soh_dp_isocost)
  pareto_rul_isocost        RUL         RB2(RUL) (anneau sur RB2) + RB2(SoH)
  pareto_pred_isocost       Prediction  RB2(Pred)
  pareto_sohpred_isocost    Prediction  RB2(SoH+Pred) + RB2(Pred) + RB2(SoH)
  pareto_strategies_v2      Synthese    toutes (RB2(SoH), RB2(Pred), RB2(SoH+Pred))
Chacune : panel du chapitre precedent + front PD magenta + iso-couts, encart zoom RB2.

Usage :  python make_chap3_pareto.py     (ou F5 dans Spyder)
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figures_chap3")          # copies de controle (local)
DP_NPZ = os.path.join(HERE, "data", "dp_pareto_25y_51x51_v2.npz")
ISO_SLOPE = 8.2014      # kEUR de cout LPSP par point de LPSP (VoLL=3)
DP_COLOR = "#c2185b"    # magenta soutenu pour le front PD (identique a generate_pareto)
FIGSIZE = (10, 6)

# Racine des figures du manuscrit (sync best-effort ; section -> Figures/).
LATEX_C3 = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "LaTeX", "Manuscrit_post_chap1_v1", "Chapitre 3"))

# ---------------- points (LPSP %, deg kEUR) ----------------
BASE = {                       # panel du chapitre precedent (inchange)
    '0-100':  (10.3855, 124.1937),
    '25-75':  (20.2667, 110.7658),
    '50-50':  (8.0744, 109.0235),
    '75-25':  (3.8032, 59.6765),
    '100-0':  (2.4851, 66.4122),
    'RB2':    (2.5921, 58.8431),
    'RB1':    (1.2597, 80.1562),
    'SoC1':   (1.3389, 140.6745),
    'SoC06':  (29.4642, 109.2535),
    'Ideal':  (0.0, 0.0),
}
# UN SEUL levier SoH desormais (plus de SoH_bat / SoH_all).
L_SOH  = 'RB2(SoH)'
L_PRED = 'RB2(Pred)'
L_SHP  = 'RB2(SoH+Pred)'
L_RUL  = 'RB2(RUL)'
EXTRA = {
    L_SOH:  (2.9091, 54.9057),   # RB2(SoH) : identique a generate_pareto
    L_PRED: (2.5312, 58.9620),   # RB2(Pred)      : chiffres chapitre 3 (Gen Fable)
    L_SHP:  (2.7575, 55.0750),   # RB2(SoH+Pred)  : chiffres chapitre 3
}

# --- palette IDENTIQUE a generate_pareto.py (tab10 + teintes dediees) ---
STRAT_ORDER = ['0-100', '25-75', '50-50', '75-25', '100-0',
               'RB2', 'RB2(SoH)', 'RB1', 'SoC1', 'SoC06']
COLORS = {s: c for s, c in zip(STRAT_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
EXTRA_COLORS = {L_PRED: '#000000', L_RUL: '#117733', L_SHP: '#d95f02'}
IDEAL_COLOR = '0.3'

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
        return EXTRA_COLORS[label]           # teinte dediee (non eclaircie)
    return darken(COLORS.get(label, IDEAL_COLOR), 0.7)


def dp_front():
    """Front PD (points non domines, tries par LPSP), depuis data/dp_*_v2.npz."""
    d = np.load(DP_NPZ)
    nd = d["nondominated"].astype(bool) if "nondominated" in d.files \
        else np.ones(len(d["lpsp"]), dtype=bool)
    o = np.argsort(d["lpsp"][nd])
    return d["lpsp"][nd][o], d["deg_keur"][nd][o]


def draw_isocost(axis, C_levels, xlim, ylim):
    """Lignes d'iso-cout en pointilles, SANS etiquette de valeur (cf. generate_pareto)."""
    axis.set_xlim(xlim); axis.set_ylim(ylim)
    xs = np.linspace(xlim[0], xlim[1], 100)
    for C in C_levels:
        axis.plot(xs, C - ISO_SLOPE * xs, ls=':', color='0.6', lw=0.9, zorder=0)


def build(name, sub, extras, zoom_offsets, ring_on_rb2=False, front=True,
          zoom_ylim=(47, 84), zoom_xlim=(1.0, 4.5), iso=True):
    """Construit une figure et la copie dans le manuscrit (Chapitre 3/<sub>/Figures/)."""
    pts = dict(BASE)
    for lab in extras:
        pts[lab] = EXTRA[lab]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # --- front PD (magenta, ronds ; identique a generate_pareto) ---
    if front:
        fl, fd = dp_front()
        ax.plot(fl, fd, "-", color=DP_COLOR, lw=1.6, zorder=2)
        ax.scatter(fl, fd, color=DP_COLOR, s=18, zorder=2)

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
        axins.plot(fl, fd, "-", color=DP_COLOR, lw=1.6, zorder=2)
        axins.scatter(fl, fd, color=DP_COLOR, s=16, zorder=2)
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
        draw_isocost(ax, list(range(70, 141, 10)), ax.get_xlim(), ax.get_ylim())
        draw_isocost(axins, [70, 75, 80, 85, 90, 95, 100], zoom_xlim, zoom_ylim)

    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.6)
    ax.set_xlabel("LPSP [%]", fontsize=18)
    ax.set_ylabel("Coût de dégradation [k€]", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    local = os.path.join(OUTDIR, name + ".pdf")
    plt.savefig(local, format='pdf', bbox_inches='tight')
    plt.savefig(local.replace(".pdf", ".png"), format='png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("saved ->", os.path.relpath(local, HERE))

    # --- sync manuscrit : Chapitre 3/<sub>/Figures/ ---
    dst_dir = os.path.join(LATEX_C3, sub, "Figures")
    if os.path.isdir(dst_dir):
        for ext in (".pdf", ".png"):
            shutil.copy2(local.replace(".pdf", ext),
                         os.path.join(dst_dir, name + ext))
        print("   copie manuscrit ->", os.path.join(sub, "Figures", name + ".pdf"))
    else:
        print("   (dossier manuscrit absent, sync ignoree : %s)" % dst_dir)


def main():
    # 1. SoH : RB2(SoH) seul  (= base_soh_dp_isocost de generate_pareto)
    build("pareto_soh_isocost", "SoH", [L_SOH],
          {'RB2':  (-0.10, 0.0, 'right', 'center'),
           L_SOH:  (0.12, 0.0, 'left', 'center')},
          zoom_ylim=(52, 84))

    # 2. RUL : anneau creux sur RB2, avec RB2(SoH) en repere
    build("pareto_rul_isocost", "RUL", [L_SOH],
          {'RB2':  (-0.10, 0.0, 'right', 'center'),
           L_SOH:  (0.12, 0.0, 'left', 'center')},
          ring_on_rb2=True, zoom_ylim=(52, 84))

    # 3. RB2(Pred)
    build("pareto_pred_isocost", "Prediction", [L_PRED],
          {L_PRED: (0.05,  0.55, 'center', 'bottom'),
           'RB2':  (0.12,  0.0, 'left',  'center')},
          zoom_ylim=(52, 84))

    # 4. RB2(SoH+Pred) (strategie combinee = "ultime")
    build("pareto_sohpred_isocost", "Prediction", [L_SOH, L_PRED, L_SHP],
          {L_PRED: (0.05,  0.55, 'center', 'bottom'),
           'RB2':  (0.12,  0.0, 'left',   'center'),
           L_SOH:  (0.12,  0.0, 'left',   'center'),
           L_SHP:  (-0.12, 0.0, 'right',  'center')},
          zoom_ylim=(50, 84))

    # 5. synthese : tout + front PD (zoom resserre sur l'amas RB2)
    build("pareto_strategies_v2", "Synthese", [L_SOH, L_PRED, L_SHP],
          {L_PRED:  (0.05,  0.45, 'center', 'bottom'),
           'RB2':   (0.10,  0.0, 'left',   'center'),
           '75-25': (0.07,  0.0, 'left',   'center'),
           L_SHP:   (-0.09, 0.0, 'right',  'center'),
           L_SOH:   (0.09,  0.0, 'left',   'center')},
          zoom_xlim=(2.2, 4.2), zoom_ylim=(50, 62))


if __name__ == "__main__":
    main()
