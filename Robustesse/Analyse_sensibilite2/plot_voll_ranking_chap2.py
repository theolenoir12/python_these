"""plot_voll_ranking_chap2.py -- classement unifie VoLL pour le CHAPITRE 2 (manuscrit).
Relit results_meso/voll_summary.txt (aucune simulation), EXCLUT RB2(SoH) (reportee
au chapitre suivant), recalcule les rangs par cas + le rang moyen, et trace la
heatmap des rangs EN FRANCAIS. Sauvegarde dans le dossier figures du chapitre 2.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(HERE, "results_meso", "voll_summary.txt")
OUT = ("/home/theo/Documents/Doctorat/GENIAL/LaTeX/Manuscrit_post_chap1_v1/"
       "Chapitre 2/Sensibilite/Figures/voll_ranking.pdf")

EXCLUDE = {"RB2(SoH)"}
# En-tetes FR (ordre = colonnes du fichier)
COL_FR = {
    "Nominal": "Nominal",
    "EoL thresholds": "Seuils EoL",
    "H2 degradation thresholds": "Seuils dégr. H$_2$",
    "Replacement-cost weights": "Poids de coût",
    "Component sizing": "Dimensionnement",
}

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,
})


def parse_costs(path):
    """Lit le bloc '## Total cost per case and strategy'. -> (cases, {ems: [couts]})."""
    cases, data, in_block = [], {}, False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("## Total cost per case"):
                in_block = True
                continue
            if in_block:
                if not s:
                    continue
                if s.startswith("ems;"):
                    cases = s.split(";")[1:]
                    continue
                if s.startswith("#"):
                    break
                parts = s.split(";")
                if len(parts) < 2:
                    continue
                data[parts[0]] = [float(x) for x in parts[1:]]
    return cases, data


def main():
    cases, data = parse_costs(TXT)
    ems = [e for e in data if e not in EXCLUDE]
    M = np.array([data[e] for e in ems])              # (n_ems, n_cases) couts
    # rangs par colonne (1 = cout le plus faible)
    ranks = np.argsort(np.argsort(M, axis=0), axis=0) + 1
    mean_rank = ranks.mean(axis=1)
    order = np.argsort(mean_rank)                     # meilleur en haut
    ems = [ems[i] for i in order]
    ranks = ranks[order]
    mean_rank = mean_rank[order]

    n_ems, n_cases = ranks.shape
    full = np.column_stack([ranks, mean_rank])        # +colonne rang moyen
    col_labels = [COL_FR.get(c, c) for c in cases] + ["Rang\nmoyen"]

    fig, ax = plt.subplots(figsize=(1.05 * (n_cases + 1) + 3.2, 0.62 * n_ems + 2.0))
    im = ax.imshow(full, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=n_ems)

    # separateur visuel avant la colonne "rang moyen"
    ax.axvline(n_cases - 0.5, color="white", lw=4)

    thr = n_ems * 0.6
    for i in range(n_ems):
        best_col = ranks[i].min()
        for j in range(n_cases + 1):
            val = full[i, j]
            is_meanrank = (j == n_cases)
            txt = ("%.1f" % val) if is_meanrank else ("%d" % int(round(val)))
            bold = (not is_meanrank and int(round(val)) == 1) or (is_meanrank and i == 0)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=plt.rcParams["xtick.labelsize"] + 1,
                    color="white" if val > thr else "#1a1a1a",
                    fontweight="bold" if bold else "normal")

    ax.set_xticks(range(n_cases + 1))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_ems))
    ax.set_yticklabels(ems)
    ax.set_xticks(np.arange(-.5, n_cases + 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_ems, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Rang (1 = coût le plus faible)", labelpad=10)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close()
    print("OK ->", OUT)
    # recap console
    for e, r, m in zip(ems, ranks, mean_rank):
        print("%-7s rangs=%s  moy=%.2f" % (e, list(r), m))


if __name__ == "__main__":
    main()
