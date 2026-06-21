"""
plot_robustesse.py -- figures (qualite publication) de l'etude de robustesse.
=======================================================================
Lit results/robustesse_results.npz (produit par run_robustesse.py) et genere des
figures pretes pour publication, en FRANCAIS, dans le meme style typographique
que l'analyse de sensibilite (cf. Analyse_sensibilite/plot_voll_summary.py) :
serif Computer Modern (mathtext), polices PDF embarquees (editables),
spines epurees, grilles legeres, palette sobre.

Figures :
  - robustesse_boxplots.pdf      : distribution de la LPSP par strategie, un
    panneau par scenario (boites triees par moyenne ; reference marche normale) ;
  - robustesse_heatmap.pdf       : LPSP moyenne (scenario x strategie) ;
  - robustesse_heatmap_delta.pdf : surcout de robustesse = LPSP(panne) - LPSP(normale) ;
  - robustesse_cdf.pdf           : fonctions de repartition de la LPSP par scenario.

Lancer :  python plot_robustesse.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import robustesse_common as rc

NPZ = os.path.join(rc.RESULTS_DIR, "robustesse_results.npz")

# --- Palette sobre (identique a l'analyse de sensibilite) --------------------
C_BOX    = "#9BB7D4"      # remplissage des boites (bleu doux)
C_BEST   = "#55A868"      # vert : strategie la plus favorable
C_MEAN   = "#1A1A1A"      # moyenne (panne)
C_NORM   = "#C44E52"      # rouge : reference marche normale
C_GREY   = "#B0B0B0"      # courbes secondaires
CMAP     = "RdYlGn_r"     # heatmaps : vert = faible (favorable), rouge = eleve

# Couleurs coherentes par strategie (memes d'une figure a l'autre)
_TAB = plt.cm.tab10(np.linspace(0, 1, 10))
STRATEGY_COLOR = {s: _TAB[i] for i, s in enumerate(rc.DEFAULT_STRATEGIES)}

# Etiquette des panneaux (a, b, c, d)
PANEL = ["(a)", "(b)", "(c)", "(d)"]


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
        "figure.titlesize": base + 4,
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


def _eval_weeks():
    return int(round(rc.EVAL_HOURS / rc.WEEK_HOURS))


def load():
    d = np.load(NPZ, allow_pickle=True)
    strategies = [str(s) for s in d["strategies"]]
    scenarios  = [str(s) for s in d["scenarios"]]
    lpsp = {sk: {st: d["lpsp|%s|%s" % (sk, st)] for st in strategies} for sk in scenarios}
    ens  = {sk: {st: d["ens|%s|%s" % (sk, st)] for st in strategies} for sk in scenarios}
    lpsp_nom = {st: d["lpsp_nom|%s" % st] for st in strategies}
    return strategies, scenarios, lpsp, ens, lpsp_nom


# =============================================================================
def fig_boxplots(strategies, scenarios, lpsp, lpsp_nom):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for idx, sk in enumerate(scenarios):
        ax = axes[idx // 2][idx % 2]
        order = sorted(strategies, key=lambda st: np.mean(lpsp[sk][st]))
        data = [lpsp[sk][st] for st in order]
        bp = ax.boxplot(
            data, vert=True, patch_artist=True, widths=0.62, showfliers=True,
            medianprops=dict(color=C_MEAN, lw=2.2),
            whiskerprops=dict(lw=1.3, color="#333333"),
            capprops=dict(lw=1.3, color="#333333"),
            boxprops=dict(lw=1.2, color="#333333"),
            flierprops=dict(marker="o", markersize=3.5, alpha=0.35,
                            markerfacecolor="0.5", markeredgecolor="none"))
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(C_BEST if i == 0 else C_BOX)
            box.set_alpha(0.92)
        xs = np.arange(1, len(order) + 1)
        ax.plot(xs, [np.mean(lpsp[sk][st]) for st in order], "D",
                color=C_MEAN, ms=6, zorder=6, label="moyenne (panne)")
        ax.plot(xs, [np.mean(lpsp_nom[st]) for st in order], "_",
                color=C_NORM, ms=15, mew=2.6, zorder=7,
                label="moyenne (marche normale)")
        ax.set_xticks(xs)
        ax.set_xticklabels([rc.STRATEGY_LABELS.get(st, st) for st in order],
                           rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel("LPSP [%]")
        ax.set_title("%s %s  —  meilleure : %s"
                     % (PANEL[idx], rc.SCENARIO_LABELS.get(sk, sk),
                        rc.STRATEGY_LABELS.get(order[0], order[0])))
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.6)
        ax.set_axisbelow(True)
        _despine(ax)
        if idx == 0:
            ax.legend(loc="upper left")
    fig.suptitle("Robustesse des EMS : distribution de la LPSP sous défaillance "
                 "(fenêtre de %d semaines, %d tirages)"
                 % (_eval_weeks(), len(next(iter(lpsp.values()))[strategies[0]])),
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(rc.RESULTS_DIR, "robustesse_boxplots.pdf")
    fig.savefig(out); plt.close(fig)
    print("->", out)


# =============================================================================
def _heatmap(M, strategies, scenarios, title, cbar_label, fname, best="min"):
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(1.15 * ncol + 3.0, 0.95 * nrow + 2.6))
    vmax = np.nanmax(M)
    im = ax.imshow(M, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([rc.STRATEGY_LABELS.get(st, st) for st in strategies],
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([rc.SCENARIO_LABELS.get(sk, sk) for sk in scenarios])

    thr = vmax * 0.6
    for i in range(nrow):
        b = int(np.argmin(M[i]) if best == "min" else np.argmax(M[i]))
        for j in range(ncol):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                    fontsize=plt.rcParams["font.size"] * 0.82,
                    color="white" if M[i, j] > thr else "#1a1a1a",
                    fontweight="bold" if j == b else "normal")
        ax.add_patch(plt.Rectangle((b - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=2.8, zorder=5))
    # Separateurs blancs facon "cellules" + suppression des bordures
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, labelpad=12)
    cb.outline.set_linewidth(1.0)
    ax.set_title(title, pad=14)
    fig.tight_layout()
    out = os.path.join(rc.RESULTS_DIR, fname)
    fig.savefig(out); plt.close(fig)
    print("->", out)


def fig_heatmap(strategies, scenarios, lpsp):
    M = np.array([[np.mean(lpsp[sk][st]) for st in strategies] for sk in scenarios])
    _heatmap(M, strategies, scenarios,
             "LPSP moyenne sous défaillance (la plus faible encadrée)",
             "LPSP moyenne [%]", "robustesse_heatmap.pdf", best="min")


def fig_heatmap_delta(strategies, scenarios, lpsp, lpsp_nom):
    M = np.array([[np.mean(lpsp[sk][st] - lpsp_nom[st]) for st in strategies]
                  for sk in scenarios])
    M = np.clip(M, 0, None)   # surcout >= 0 (le residu negatif est du bruit numerique)
    _heatmap(M, strategies, scenarios,
             "Surcoût de robustesse  =  LPSP(panne) − LPSP(marche normale)",
             "Surcoût de LPSP [points de %]", "robustesse_heatmap_delta.pdf",
             best="min")


# =============================================================================
def fig_cdf(strategies, scenarios, lpsp):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for idx, sk in enumerate(scenarios):
        ax = axes[idx // 2][idx % 2]
        best = min(strategies, key=lambda st: np.mean(lpsp[sk][st]))
        # Strategies secondaires d'abord (en gris), meilleure par-dessus
        for st in strategies:
            if st == best:
                continue
            a = np.sort(lpsp[sk][st]); y = np.arange(1, len(a) + 1) / len(a)
            ax.step(a, y, where="post", lw=1.3, alpha=0.55,
                    color=STRATEGY_COLOR[st], label=rc.STRATEGY_LABELS.get(st, st))
        a = np.sort(lpsp[sk][best]); y = np.arange(1, len(a) + 1) / len(a)
        ax.step(a, y, where="post", lw=3.2, color=C_BEST, zorder=6,
                label=rc.STRATEGY_LABELS.get(best, best) + " (meilleure)")
        ax.set_xlabel("LPSP [%]")
        ax.set_ylabel(r"$F(\mathrm{LPSP})$")
        ax.set_xlim(left=0); ax.set_ylim(0, 1.01)
        ax.set_title("%s %s" % (PANEL[idx], rc.SCENARIO_LABELS.get(sk, sk)))
        ax.grid(True, alpha=0.6)
        ax.set_axisbelow(True)
        _despine(ax)
        ax.legend(loc="lower right", ncol=2, handlelength=1.5,
                  columnspacing=1.0, labelspacing=0.3)
    fig.suptitle("Fonctions de répartition de la LPSP sous défaillance", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(rc.RESULTS_DIR, "robustesse_cdf.pdf")
    fig.savefig(out); plt.close(fig)
    print("->", out)


def main():
    if not os.path.exists(NPZ):
        raise SystemExit("Lance d'abord run_robustesse.py (resultats absents : %s)" % NPZ)
    set_pub_style()
    strategies, scenarios, lpsp, ens, lpsp_nom = load()
    fig_boxplots(strategies, scenarios, lpsp, lpsp_nom)
    fig_heatmap(strategies, scenarios, lpsp)
    fig_heatmap_delta(strategies, scenarios, lpsp, lpsp_nom)
    fig_cdf(strategies, scenarios, lpsp)


if __name__ == "__main__":
    main()
