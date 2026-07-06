# -*- coding: utf-8 -*-
"""plot_defaillance_prevision.py -- figures RB1(Pred) : integration de la prevision
de profils de puissance a la ROBUSTESSE AUX DEFAILLANCES (chapitre 3).
=================================================================================
CENTRALISATION. Ces figures decrivent l'axe RB1(Pred) (dissocie du plan de Pareto
degradation). Le calcul lourd et la demarche complete vivent dans
../Defaillances/ (cf. DEMARCHE_RB1_augmentation.txt) ; ICI on RE-TRACE seulement,
a partir des valeurs FINALES deja calculees, pour tout regrouper cote Predictions
(a cote de pareto_ems_pred, pred_uncertainty_zoom, etc.). Ce script REMPLACE le
generateur qui vivait dans l'arbre LaTeX (Chapitre 3/make_defaillance_figs.py) :
il ecrit les figures ici ET met a jour la copie du manuscrit (best-effort).

Aucune simulation, aucun multiprocessing -> executable directement dans Spyder (F5).

Figures produites (PDF + PNG) dans CE dossier :
  def_gain_scenario.pdf       barres LPSP moyenne sous defaillance par scenario
                              (RB1 / RB1(Pred) omniscient / RB1(Pred) deployable)
  def_geom_sweep.pdf          heatmap du gain omniscient (b_reserve x h_pre)
  def_sigma_sensitivity.pdf   gain vs misestimation de sigma (bande figee)

Source des valeurs (../Defaillances/) :
  def_gain_scenario   <- mc_rb1_pred_ellipse.txt (nu / omni / deployable)
  def_geom_sweep      <- sweep_rb1_pred.txt      (score omniscient b_reserve x h_pre)
  def_sigma_sensitivity <- mc_rb1_pred_ellipse.txt (misestimation de sigma)

Style typographique aligne sur ../Defaillances/plot_robustesse.py (serif Computer
Modern mathtext, polices PDF embarquees, spines epurees, palette sobre).

Usage :  python plot_defaillance_prevision.py     (ou F5 dans Spyder)
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = HERE + '\RB1(Pred)'                      # figures centralisees dans Predictions/
# Copie du manuscrit (best-effort : mise a jour si le dossier existe).
LATEX_FIGS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "LaTeX", "Manuscrit_post_chap1_v1",
    "Chapitre 3", "Defaillances", "Figures"))

# ---------------- palette sobre (cf. plot_robustesse.py) ----------------
C_NU     = "#B0B0B0"      # RB1 de reference (gris)
C_OMNI   = "#9BB7D4"      # borne omnisciente (bleu doux)
C_DEPLOY = "#55A868"      # configuration deployable (vert)
C_ZERO   = "#C44E52"      # ligne zero (rouge)
CMAP     = "RdYlGn"       # gain : vert = eleve (favorable)

SCORE_NU = 1.481          # score de panne de RB1 (reference)

# ---------------- donnees embarquees (valeurs FINALES) ----------------
SCENARIOS = ["PEMFC\ntotale", "PEMFC\n50 %", "PEMWE\ntotale", "PEMWE\n50 %"]
LPSP_NU     = [2.569, 1.187, 1.424, 0.743]
LPSP_OMNI   = [2.536, 1.110, 1.393, 0.695]
LPSP_DEPLOY = [2.547, 1.144, 1.417, 0.713]

B_RESERVE = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
H_PRE     = [6, 12, 18, 24, 48, 72]
# score omniscient (lignes = b_reserve, colonnes = h_pre)
SCORE_GEOM = np.array([
    [1.481, 1.481, 1.481, 1.481, 1.481, 1.481],  # 0.75 (levier off)
    [1.485, 1.473, 1.475, 1.476, 1.487, 1.489],  # 0.80
    [1.474, 1.459, 1.453, 1.455, 1.470, 1.487],  # 0.85
    [1.484, 1.453, 1.450, 1.454, 1.485, 1.496],  # 0.90
    [1.482, 1.459, 1.438, 1.446, 1.476, 1.500],  # 0.95
    [1.490, 1.448, 1.433, 1.438, 1.480, 1.503],  # 0.99
])

SIG_RATIO = [0.50, 0.75, 1.00, 1.25, 1.50]
SIG_GAIN  = [0.001, 0.015, 0.021, 0.021, 0.005]


def set_pub_style(base=16):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.titleweight": "normal",
        "axes.labelsize": base + 1,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 3,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.0,
        "axes.grid": False,
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "figure.dpi": 100,
    })


def _despine(ax, keep=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def _save(fig, name):
    out = os.path.join(OUTDIR, name)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print("->", os.path.relpath(out, HERE))
    # mise a jour de la copie manuscrit (best-effort)
    if os.path.isdir(LATEX_FIGS):
        for ext in (".pdf", ".png"):
            shutil.copy2(out.replace(".pdf", ext),
                         os.path.join(LATEX_FIGS, name.replace(".pdf", ext)))
        print("   copie manuscrit ->", os.path.join(LATEX_FIGS, name))


# =============================================================================
def fig_gain_scenario():
    x = np.arange(len(SCENARIOS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - w, LPSP_NU, w, color=C_NU, label="RB1 (référence)",
           edgecolor="#333333")
    ax.bar(x, LPSP_OMNI, w, color=C_OMNI, label="RB1(Pred) omniscient",
           edgecolor="#333333")
    ax.bar(x + w, LPSP_DEPLOY, w, color=C_DEPLOY, label="RB1(Pred) déployable",
           edgecolor="#333333")
    for xi, v in zip(x - w, LPSP_NU):
        ax.text(xi, v + 0.03, "%.3f" % v, ha="center", va="bottom", fontsize=9,
                color="#555555")
    for xi, v in zip(x + w, LPSP_DEPLOY):
        ax.text(xi, v + 0.03, "%.3f" % v, ha="center", va="bottom", fontsize=9,
                color="#2f6b40")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("LPSP moyenne sous défaillance [%]")
    ax.set_ylim(0, max(LPSP_NU) * 1.18)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "def_gain_scenario.pdf")


# =============================================================================
def fig_geom_sweep():
    gain = SCORE_NU - SCORE_GEOM          # points de LPSP ; >0 = mieux
    nrow, ncol = gain.shape
    fig, ax = plt.subplots(figsize=(1.05 * ncol + 2.6, 0.85 * nrow + 2.2))
    vmax = np.nanmax(gain)
    im = ax.imshow(gain, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(["%d" % h for h in H_PRE])
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(["%.2f" % b for b in B_RESERVE])
    ax.set_xlabel(r"Horizon $H_{\mathrm{pre}}$ [h]")
    ax.set_ylabel(r"Largeur de réserve $b_{\mathrm{reserve}}$")

    b_opt = int(np.unravel_index(np.argmax(gain), gain.shape)[0])
    h_opt = int(np.unravel_index(np.argmax(gain), gain.shape)[1])
    for i in range(nrow):
        for j in range(ncol):
            ax.text(j, i, "%+.3f" % gain[i, j], ha="center", va="center",
                    fontsize=plt.rcParams["font.size"] * 0.72,
                    color="#1a1a1a",
                    fontweight="bold" if (i == b_opt and j == h_opt) else "normal")
    ax.add_patch(plt.Rectangle((h_opt - 0.5, b_opt - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2.8, zorder=5))
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Gain de score de panne [points de LPSP]", labelpad=12)
    cb.outline.set_linewidth(1.0)
    fig.tight_layout()
    _save(fig, "def_geom_sweep.pdf")


# =============================================================================
def fig_sigma_sensitivity():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color=C_ZERO, lw=1.6, ls="--", zorder=1)
    ax.plot(SIG_RATIO, SIG_GAIN, "-o", color=C_DEPLOY, ms=8, zorder=3,
            label="RB1(Pred) déployable")
    ax.axvline(1.0, color="0.6", lw=1.0, ls=":", zorder=1)
    for xr, g in zip(SIG_RATIO, SIG_GAIN):
        ax.text(xr, g + 0.0012, "%+.3f" % g, ha="center", va="bottom",
                fontsize=10, color="#2f6b40")
    ax.set_xlabel(r"Facteur de misestimation $\sigma_{\mathrm{injecté}}/\sigma_{\mathrm{conception}}$")
    ax.set_ylabel("Gain de score de panne [points de LPSP]")
    ax.set_ylim(-0.004, max(SIG_GAIN) * 1.35)
    ax.set_xticks(SIG_RATIO)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="lower center")
    fig.tight_layout()
    _save(fig, "def_sigma_sensitivity.pdf")


def main():
    set_pub_style()
    fig_gain_scenario()
    fig_geom_sweep()
    fig_sigma_sensitivity()


if __name__ == "__main__":
    main()
