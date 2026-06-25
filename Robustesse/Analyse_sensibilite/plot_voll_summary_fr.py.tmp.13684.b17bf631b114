"""
plot_voll_summary_fr.py -- resume UNIFIE des analyses de sensibilite de l'EMS.
============================================================================
Version FRANCAISE de plot_voll_summary.py, AVEC la strategie RB2(SoH) RETIREE
(elle est exclue de EMS_ORDER et l'annexe SoH dediee n'est plus tracee).

Figures pretes pour publication, pensees pour cotoyer la figure (LaTeX/TikZ) de
workflow de la these : typographie serif Computer Modern, palette sobre, titres
en graisse normale, traits fins, PDF vectoriel avec polices editables.

N'execute AUCUNE simulation : relit seulement results_meso/*.txt et applique
l'indicateur unifie de voll_common (cout_total = cout de degradation + cout
financier de l'energie non fournie, valorisee avec la VOLL qui y est definie).

EDITION SOIGNEE POUR LA LISIBILITE EN PUBLICATION SCIENTIFIQUE :
polices agrandies, espacements ajustes, traits legerement plus epais.
MISE EN EVIDENCE : les 3 premiers rangs sont en gras, le rang 1 a un asterisque (*).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interactif pour le scripting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# On suppose voll_common accessible
try:
    import voll_common as V
    OUT = V.MESO_DIR
except ImportError:
    # Repli pour les tests si voll_common est absent
    print("Attention : voll_common introuvable. Utilisation de donnees/chemins factices.")
    class DummyV:
        MESO_DIR = "."
        EMS_ORDER = ["RuleBased", "Optimization", "Predictive", "AI-Enhanced", "Legacy_A", "Legacy_B"]
        VOLL_TIERS = [(0.01, 10), (0.05, 50), (None, 100)]
        E_REF_KWH = 1e6
        HORIZON_Y = 25
        USE_STEPWISE_LPS = False
        @staticmethod
        def total_cost_keur(lpsp, deg, clps=None): return deg + (clps if clps is not None else lpsp * 10)
        @staticmethod
        def cost_lpsp_keur(lpsp): return lpsp * 10
        @staticmethod
        def voll_eur_per_kwh(lpsp): return 50
        @staticmethod
        def build_cases():
            # Donnees factices un peu variees pour voir l'effet du classement
            ems = DummyV.EMS_ORDER
            return [
                ("Nominal", None, {e: (1.0 + i*0.1, 50.0 - i*2) for i, e in enumerate(ems)}),
                ("Charge forte", None, {e: (2.0 + i*0.2, 60.0 - i*1) for i, e in enumerate(ems)}),
                ("PV faible", None, {e: (0.5 + i*0.05, 40.0 - i*3) for i, e in enumerate(ems)})
            ]
    V = DummyV
    OUT = "."

# --- Strategies : RB2(SoH) VOLONTAIREMENT EXCLUE de cette version ---
# On filtre localement EMS_ORDER pour ne pas toucher voll_common (partage avec
# les scripts sens_*.py). Toutes les figures transversales iterent sur cette
# liste : les entrees RB2(SoH) des cas ne sont donc jamais selectionnees.
EMS_ORDER = [e for e in V.EMS_ORDER if e != "RB2(SoH)"]

# --- Jetons typographiques (Computer Modern via mathtext) ---
KEUR   = "k€"          # kilo-euros
EURKWH = "€/kWh"       # euros par kWh
SIG    = r"$\sigma$"        # sigma Computer Modern
TIMES  = r"$\times$"        # signe multiplication Computer Modern

# --- Palette sobre (tons attenues, professionnels) ---
C_DEG   = "#4C72B0"       # degradation (bleu attenue)
C_LPSP  = "#DD8452"       # cout energie non fournie / LPSP (orange attenue)
C_LINE  = "#7D5BA6"       # courbe principale (violet attenue)
C_LINE2 = "#55A868"       # courbe secondaire (vert attenue)
C_RED   = "#C44E52"       # courbe VOLL / marqueur nominal (rouge attenue)
C_BOX   = "#9BB7D4"       # remplissage uniforme des boites (bleu doux)
CMAP    = "RdYlGn_r"      # heatmap des rangs (vert = meilleur)


def set_pub_style():
    """
    Definit un style matplotlib soigne, adapte aux publications scientifiques.
    Agrandit nettement les polices et ajuste les graisses de traits.
    """
    # Taille de police de base. Tout le reste s'echelonne a partir de la.
    base_size = 24

    plt.rcParams.update({
        # --- Typographie ---
        "font.family": "serif",
        # Force Computer Modern Roman, repli sur DejaVu
        "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
        "axes.formatter.use_mathtext": True,        # indices/sigma en math CM
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,                # corrige tiret vs signe moins en CM

        # --- Backend ---
        "timezone": "UTC",

        # --- Tailles de police ---
        "font.size": base_size,                # defaut global
        "axes.titlesize": base_size + 2,       # titres de sous-figures
        "axes.titleweight": "normal",          # titres non gras pour le rendu LaTeX
        "axes.labelsize": base_size + 1,       # taille des labels X/Y
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 4,     # titre principal

        # --- Epaisseurs de traits & cadres ---
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",             # noir pur pour un fort contraste
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "lines.linewidth": 2.5,                # traits de courbe plus epais
        "patch.linewidth": 1.0,                # epaisseur des bords de barres

        # --- Grilles ---
        "axes.grid": False,                    # desactivee par defaut, activee par plot
        "grid.color": "0.85",                  # grille gris clair
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",

        # --- Legende ---
        "legend.frameon": False,               # pas de cadre autour de la legende
        "legend.loc": "best",

        # --- Enregistrement ---
        "pdf.fonttype": 42,                    # polices reelles integrees (editables)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",               # marges blanches minimales
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,                    # repli raster haute resolution
        "figure.dpi": 100,
    })


def _despine(ax, keep=("left", "bottom")):
    """Retire les cadres haut et droit, standard des graphes scientifiques."""
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


# -------------------------------------------------------------------------
def _ranks(values):
    """values : dict {ems: cout_total}. -> {ems: rang} (1 = cout le plus faible)."""
    order = sorted(values, key=lambda e: values[e])
    return {e: i + 1 for i, e in enumerate(order)}


def _rank_matrix(cases):
    """-> (R[ems, cas], rang_moyen[ems], labels) avec les EMS dans EMS_ORDER."""
    ems = EMS_ORDER
    labels = [c[0] for c in cases]
    R = np.full((len(ems), len(cases)), np.nan)
    for jc, (_, _grp, d) in enumerate(cases):
        # Chaque colonne est classee par le cout total unifie de son point
        # (moyenne Monte-Carlo) -> traitement uniforme sur tous les axes.
        # On ne classe QUE les strategies de EMS_ORDER (RB2(SoH) exclu) : les rangs
        # sont donc continus de 1 a len(EMS_ORDER), sans trou la ou etait RB2(SoH).
        rk = _ranks({e: V.total_cost_keur(*d[e]) for e in EMS_ORDER if e in d})
        for ir, e in enumerate(ems):
            if e in rk:
                R[ir, jc] = rk[e]
    return R, np.nanmean(R, axis=1), labels


# =========================================================================
def figure_ranking(cases):
    """Heatmap strategie x cas du rang du cout total ; lignes triees meilleur d'abord."""
    R, mean_rank, labels = _rank_matrix(cases)
    n = len(EMS_ORDER)
    # Tri des lignes : meilleur rang moyen en haut
    order = sorted(range(n), key=lambda i: mean_rank[i])
    Rs = R[order, :]
    mr = mean_rank[order]
    ems = [EMS_ORDER[i] for i in order]
    nrows, ncols = Rs.shape
    cmap = plt.get_cmap(CMAP)

    def txtcolor(rk):
        """Choisit la couleur du texte selon l'intensite du fond pour la lisibilite."""
        if np.isnan(rk): return "white"
        # Rangs 1,2 (vert fonce) et n-1, n (rouge fonce) -> texte blanc.
        return "white" if (rk <= 2 or rk >= n - 1) else "#1a1a1a"

    # --- Taille de figure ajustee ---
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(4.0 + 1.8 * ncols, 0.65 * nrows + 3.2),
        gridspec_kw=dict(width_ratios=[ncols, 1.2], wspace=0.10),
        constrained_layout=True)

    # 1. Heatmap principale
    im = ax.imshow(Rs, cmap=cmap, aspect="auto", vmin=1, vmax=n,
                   interpolation='nearest', rasterized=True)
    # Ticks axe X (cas de sensibilite)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")

    # Ticks axe Y (strategies EMS)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(ems)

    # Texte des cellules (les rangs)
    cell_fontsize = plt.rcParams['font.size'] * 0.9

    for ir in range(nrows):
        for jc in range(ncols):
            if not np.isnan(Rs[ir, jc]):
                rk = int(Rs[ir, jc])

                # --- MISE EN EVIDENCE ---
                text_str = "%d" % rk
                weight = 'normal'

                if rk <= 3:
                    weight = 'bold'  # les 3 premiers en gras
                    if rk == 1:
                        text_str += "*"  # le meilleur a un asterisque

                ax.text(jc, ir, text_str, ha="center", va="center",
                        fontsize=cell_fontsize, color=txtcolor(rk), fontweight=weight)

    # Grille blanche epaisse pour bien separer les cellules
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3.0)
    ax.tick_params(which="both", length=0)  # masque les traits de ticks
    for sp in ax.spines.values(): sp.set_visible(False)  # masque le cadre

    # ax.set_title("Classement des EMS par cout total\n(degradation + cout de l'energie non fournie)", pad=20)

    # 2. Colonne du rang moyen
    axm.imshow(mr.reshape(-1, 1), cmap=cmap, aspect="auto", vmin=1, vmax=n, interpolation='nearest')
    axm.set_xticks([0])
    axm.set_xticklabels(["Rang\nmoyen"])
    axm.set_yticks([])  # ticks Y masques (identiques au plot principal)

    # Texte des cellules de rang moyen
    for ir in range(nrows):
        val = mr[ir]
        # Lignes triees par rang moyen : ir=0 -> rang 1 global, etc.
        overall_rank = ir + 1

        text_str = "%.1f" % val
        weight = 'normal'

        if overall_rank <= 3:
            weight = 'bold'
            if overall_rank == 1:
                text_str += "*"

        axm.text(0, ir, text_str, ha="center", va="center",
                 fontsize=cell_fontsize, color=txtcolor(overall_rank), fontweight=weight)

    # Grille de la colonne rang moyen
    axm.set_xticks([-.5, .5], minor=True)
    axm.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    axm.grid(which="minor", color="white", linewidth=3.0)
    axm.tick_params(which="both", length=0)
    for sp in axm.spines.values(): sp.set_visible(False)

    # Barre de couleur
    cb = fig.colorbar(im, ax=axm, fraction=0.7, pad=0.15)
    cb.set_ticks(range(1, n + 1))
    cb.set_label("Rang (1 = coût le plus faible)", labelpad=15)
    cb.outline.set_linewidth(1.0)

    fig.savefig(os.path.join(OUT, "voll_ranking_fr.pdf"))
    plt.close()
    return R, mean_rank, labels


def figure_distribution(cases):
    """Boite a moustaches du cout total par strategie sur tous les cas (log), triee par mediane."""
    ems = EMS_ORDER
    # Extraction des donnees, on ignore les entrees manquantes
    data = {e: [V.total_cost_keur(*d[e])
                for _, _g, d in cases if e in d] for e in ems}
    # Tri des strategies par cout median
    order = sorted(ems, key=lambda e: np.median(data[e]) if data[e] else np.inf)
    vals = [data[e] for e in order]

    fig, ax = plt.subplots(figsize=(14, 7.5))

    # Style de boxplot soigne : traits epais, couleurs attenuees
    bp = ax.boxplot(vals, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#1a1a1a", lw=2.5),
                    whiskerprops=dict(lw=1.5, color="#333333"),
                    capprops=dict(lw=1.5, color="#333333"),
                    boxprops=dict(lw=1.5, color="#333333"),
                    flierprops=dict(marker="o", markersize=5, alpha=0.5,
                                    markerfacecolor="0.5", markeredgecolor="0.5"))

    # Remplissage des boites en bleu sobre
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BOX)
        patch.set_alpha(0.9)

    # Cas nominal : losange rouge distinct
    nominal = cases[0][2]
    for i, e in enumerate(order, 1):
        if e in nominal:
            ax.plot(i, V.total_cost_keur(*nominal[e]),
                    "D", color=C_RED, markersize=10, zorder=10, label="_nolegend_")

    # Mise en forme
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=25, ha="right")

    ax.set_ylabel("Coût total [%s]\n(échelle log)" % KEUR, labelpad=15)
    ax.set_title("Distribution du coût total sur %d cas de sensibilité" % len(cases), pad=20)

    # Grille pour la lisibilite en echelle log
    ax.grid(True, axis="y", which="both", ls="-", alpha=0.6)
    ax.set_axisbelow(True)  # grille derriere les traces
    _despine(ax)

    # Legende propre
    ax.plot([], [], "D", color=C_RED, markersize=10, label="Cas nominal")
    ax.legend(loc="upper left", frameon=True, shadow=False, fancybox=False)

    fig.savefig(os.path.join(OUT, "voll_distribution_fr.pdf"))
    plt.close()
    return order, data


def figure_decomposition(cases):
    """Barres empilees : cout de degradation vs cout de l'energie non fournie au cas nominal."""
    nominal = cases[0][2]
    # Strategies presentes dans le cas nominal
    ems = [e for e in EMS_ORDER if e in nominal]
    deg = np.array([nominal[e][1] for e in ems])
    # cout de l'energie non fournie. Par defaut (USE_STEPWISE_LPS=False) on prend
    # la valorisation agregee VoLL*EENS, coherente avec le classement ; on ne
    # reutilise la colonne 'clps' (LPS pas-a-pas) que si ce mode est reactive.
    clp = np.array([nominal[e][2] if (V.USE_STEPWISE_LPS and len(nominal[e]) > 2
                                      and nominal[e][2] is not None)
                    else V.cost_lpsp_keur(nominal[e][0]) for e in ems])
    tot = deg + clp

    # Tri des barres par cout total
    idx = np.argsort(tot)
    ems = [ems[i] for i in idx]
    deg, clp, tot = deg[idx], clp[idx], tot[idx]
    x = np.arange(len(ems))

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Barres empilees avec bords blancs pour la definition
    ax.bar(x, deg, label="Coût de dégradation", color=C_DEG, edgecolor="white", lw=1.0)
    ax.bar(x, clp, bottom=deg, label="Coût de l'énergie non fournie (VOLL)",
           color=C_LPSP, edgecolor="white", lw=1.0)

    # Valeurs totales au-dessus des barres
    val_fontsize = plt.rcParams['font.size']

    max_tot = np.max(tot) if len(tot) > 0 else 1
    for xi, t in zip(x, tot):
        ax.text(xi, t + max_tot * 0.015, "%.0f" % t, ha="center", va="bottom",
                fontsize=val_fontsize, fontweight='bold', color="#1a1a1a")

    # Mise en forme
    ax.set_xticks(x)
    ax.set_xticklabels(ems, rotation=25, ha="right")
    ax.set_ylabel("Coût [%s]" % KEUR, labelpad=15)
    ax.set_title("Décomposition du coût total : cas nominal", pad=20)

    # Legende en bas a droite, moins intrusive
    ax.legend(loc="lower right", frameon=True)

    # Grille horizontale legere
    ax.grid(True, axis="y", ls="-", alpha=0.6)
    ax.set_axisbelow(True)
    _despine(ax)

    fig.savefig(os.path.join(OUT, "voll_decomposition_fr.pdf"))
    plt.close()


def figure_voll_function():
    """La VOLL (par paliers ou constante) et le cout de l'energie non fournie vs LPSP."""
    # Ligne haute resolution pour des courbes lisses
    lp = np.linspace(0.0, 38.0, 3000)
    voll = np.array([V.voll_eur_per_kwh(x) for x in lp])
    clp  = np.array([V.cost_lpsp_keur(x) for x in lp])

    tiered = len(V.VOLL_TIERS) > 1
    # Titre clair selon que la VOLL est constante ou par paliers
    if tiered:
        ttl = "Valeur de la charge perdue (VOLL) par paliers"
    else:
        val = V.VOLL_TIERS[0][1]
        ttl = "VOLL constante : %g %s" % (val, EURKWH)

    # Figure a deux panneaux
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.0), constrained_layout=True)

    # 1. Courbe de la VOLL
    ax1.plot(lp, voll, color=C_RED, lw=3.5)  # trait epais pour le concept principal
    ax1.set_xlabel("Probabilité de défaut d'alimentation (LPSP) [%]", labelpad=10)
    ax1.set_ylabel("VOLL [%s]" % EURKWH, labelpad=10)
    ax1.set_title(ttl, pad=15)

    # Force les ticks Y sur les valeurs de paliers pour la clarte
    ytick_vals = sorted({v for _, v in V.VOLL_TIERS if v is not None})
    if ytick_vals:
        ax1.set_yticks(ytick_vals)
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    ax1.grid(True, ls="-", alpha=0.6)
    ax1.set_axisbelow(True)
    _despine(ax1)

    # Lignes verticales pointillees pour les seuils
    for thr, _ in V.VOLL_TIERS:
        if thr is not None:
            ax1.axvline(thr * 100.0, color="0.5", ls="--", lw=1.5, alpha=0.7)

    # 2. Courbe de cout resultante
    ax2.plot(lp, clp, color=C_LINE, lw=3.5)
    ax2.set_xlabel("LPSP [%]", labelpad=10)
    ax2.set_ylabel("Coût de l'énergie non fournie [%s]" % KEUR, labelpad=10)
    ax2.set_title("Coût %s VOLL(LPSP) %s énergie non fournie" % (r'$\propto$', TIMES), pad=15)

    ax2.grid(True, ls="-", alpha=0.6)
    ax2.set_axisbelow(True)
    _despine(ax2)

    # Memes seuils
    for thr, _ in V.VOLL_TIERS:
        if thr is not None:
            ax2.axvline(thr * 100.0, color="0.5", ls="--", lw=1.5, alpha=0.7)

    # Annotation informative sur les hypotheses.
    annot_text = (r"Énergie de référence $E_\mathrm{ref}=%.1f$ GWh sur %g ans"
                  % (V.E_REF_KWH / 1e6, V.HORIZON_Y))
    ax2.annotate(annot_text,
                 xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                 fontsize=plt.rcParams['font.size'] * 0.85, color="0.3")

    fig.savefig(os.path.join(OUT, "voll_voll_function_fr.pdf"))
    plt.close()


def write_recap(cases, mean_rank, labels, data):
    """Ecrit un recapitulatif texte (UTF-8)."""
    path = os.path.join(OUT, "voll_summary_fr.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Résumé unifié -- cout_total = coût de dégradation + coût de l'énergie non fournie\n")
            f.write("# VOLL [€/kWh] : " +
                    " ; ".join("<%g%% -> %g" % ((t * 100.0) if t else 999, v)
                               for t, v in V.VOLL_TIERS) + "\n")
            f.write("# E_ref = %.3f MWh (énergie nette planifiée, %g ans) ; "
                    "energie_non_fournie = LPSP%%/100 * E_ref\n\n" % (V.E_REF_KWH/1000.0, V.HORIZON_Y))

            f.write("## Classement global (rang moyen sur %d colonnes de sensibilité)\n"
                    % len(labels))
            f.write("rang_moyen;ems;cout_total_median_keur;min;max\n")

            # Indices tries selon le rang moyen
            ems_list = EMS_ORDER
            sorted_idx = sorted(range(len(ems_list)), key=lambda k: mean_rank[k])

            for i in sorted_idx:
                e = ems_list[i]
                vals = data[e]
                if vals:
                    f.write("%.2f;%s;%.2f;%.2f;%.2f\n"
                            % (mean_rank[i], e, np.median(vals), min(vals), max(vals)))
                else:
                    f.write("%.2f;%s;NA;NA;NA\n" % (mean_rank[i], e))

            # Tableau de couts : uniquement les vrais cas de cout -> entete derive
            # de `cases`, pas de `labels`.
            f.write("\n## Coût total par cas et stratégie [k€]\n")
            f.write("ems;" + ";".join(c[0] for c in cases) + "\n")
            for e in ems_list:
                row = [e]
                for _, _g, d in cases:
                    if e in d:
                        row.append("%.2f" % V.total_cost_keur(*d[e]))
                    else:
                        row.append("NA")
                f.write(";".join(row) + "\n")
        return path
    except IOError as e:
        print(f"Erreur d'ecriture du recapitulatif : {e}")
        return None


def main():
    print("=== Résumé unifié (coût de dégradation + coût de l'énergie non fournie) ===", flush=True)

    # 1. Applique le style soigne globalement
    set_pub_style()

    # 2. Chargement des donnees (V.build_cases() gere l'IO/parsing)
    cases = V.build_cases()
    if not cases:
        print("Erreur : aucun cas chargé.")
        return
    # Traduction des libelles de cas (build_cases vient du module partage voll_common,
    # en anglais : on renomme localement sans toucher au module).
    LABELS_FR = {
        "Nominal": "Nominal",
        "EoL thresholds": "Seuils de fin de vie",
        "H2 degradation thresholds": "Seuils de dégradation H2",
        "Replacement-cost weights": "Coûts de remplacement",
        "Component sizing": "Dimensionnement",
        "Calendar aging": "Vieillissement calendaire",
    }
    cases = [(LABELS_FR.get(lbl, lbl), grp, d) for lbl, grp, d in cases]
    print("    %d cas : %s" % (len(cases), ", ".join(c[0] for c in cases)), flush=True)

    # 3. Generation des figures
    print("Génération des figures...", end="", flush=True)
    R, mean_rank, labels = figure_ranking(cases)
    order, data = figure_distribution(cases)
    figure_decomposition(cases)
    figure_voll_function()
    print(" Terminé.")

    # 4. Ecriture du resume texte
    recap = write_recap(cases, mean_rank, labels, data)

    # 5. Sortie console
    print("\n" + "=" * 64)
    print("CLASSEMENT GLOBAL (rang moyen du coût total, 1 = meilleur)")
    print("-" * 64)
    ems_list = EMS_ORDER
    sorted_idx = sorted(range(len(ems_list)), key=lambda k: mean_rank[k])
    for i in sorted_idx:
        e = ems_list[i]
        med_cost = np.median(data[e]) if data[e] else np.nan
        print("  rang_moyen %5.2f  %-12s  (coût total médian %.1f k€)"
              % (mean_rank[i], e, med_cost))
    print("=" * 64)
    print("Figures (PDF) -> %s" % OUT)
    if recap:
        print("Recap (Txt)   -> %s" % recap)


if __name__ == "__main__":
    # S'assure que le dossier de sortie existe
    if not os.path.exists(OUT) and OUT != ".":
        os.makedirs(OUT, exist_ok=True)
    main()
