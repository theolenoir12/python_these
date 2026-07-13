"""
rank_ems_unified.py
===================
Post-traitement : lit batch_results_summary_25y.txt (produit par batch_pareto.py,
10 strategies EMS sur 25 ans, LPSP % + deg kEUR via le MEME estimateur ledger) et
applique le COUT UNIFIE de Robustesse/Analyse_sensibilite/voll_common.py :

    total [kEUR] = deg [kEUR] + VoLL * energie_non_fournie
                 = voll_common.total_cost_keur(LPSP, deg)     (VoLL=3 EUR/kWh)

Classe toutes les strategies par cout unifie croissant, identifie le minimiseur
GLOBAL et le minimiseur PARMI LES RB, et trace un nuage (LPSP, deg) avec droites
d'iso-cout unifie, RB2 mis en avant.

Aucune simulation : purement post-traitement (relance batch_pareto.py pour
regenerer batch_results_summary_25y.txt si besoin).

Lancement (python = anaconda) :
    python rank_ems_unified.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Analyse_sensibilite")))
import voll_common as V

IN_TXT  = os.path.join(HERE, "batch_results_summary_25y.txt")
OUT_TXT = os.path.join(HERE, "rank_ems_unified_25y.txt")
OUT_PDF = os.path.join(HERE, "rank_ems_unified_25y.pdf")
OUT_PNG = os.path.join(HERE, "rank_ems_unified_25y.png")

# Familles a base de regles (pour identifier "la meilleure des RB").
RB_LABELS = {"RB1", "RB2", "RB2(SoH)"}


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("Label") or s.startswith("#"):
                continue
            p = s.split(";")
            if len(p) != 3:
                continue
            label = p[0]
            try:
                lpsp, deg = float(p[1]), float(p[2])
            except ValueError:
                continue
            rows.append(dict(label=label, lpsp=lpsp, deg=deg))
    return rows


def main():
    if not os.path.exists(IN_TXT):
        sys.exit(f"Introuvable : {IN_TXT}\n-> lance d'abord batch_pareto.py (sortie _25y).")
    rows = read_rows(IN_TXT)
    # 'Ideal' (0,0) est un repere, pas une strategie : on l'exclut du classement.
    rows = [r for r in rows if r["label"].lower() != "ideal"]
    for r in rows:
        r["clps"]  = V.cost_lpsp_keur(r["lpsp"])
        r["total"] = V.total_cost_keur(r["lpsp"], r["deg"])
        r["is_rb"] = r["label"] in RB_LABELS

    rows_sorted = sorted(rows, key=lambda r: r["total"])
    rbs = [r for r in rows_sorted if r["is_rb"]]
    best_all = rows_sorted[0]
    best_rb  = min(rbs, key=lambda r: r["total"]) if rbs else None

    # --- txt ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Cout unifie = deg + VoLL*(LPSP/100)*E_REF/1000  (voll_common.py)\n")
        f.write(f"# VoLL={V.VOLL_TIERS}  E_REF_KWH={V.E_REF_KWH:.3f}  "
                f"-> {V.cost_lpsp_keur(1.0):.4f} kEUR par point de LPSP  (25 ans)\n")
        f.write("rang;label;RB;LPSP(%);deg(kEUR);coutLPSP(kEUR);total(kEUR)\n")
        for i, r in enumerate(rows_sorted, 1):
            f.write(f"{i};{r['label']};{int(r['is_rb'])};{r['lpsp']:.4f};"
                    f"{r['deg']:.4f};{r['clps']:.4f};{r['total']:.4f}\n")
        if best_rb is not None:
            f.write(f"\n# MEILLEURE RB : {best_rb['label']} -> total={best_rb['total']:.4f} kEUR\n")
            for r in rbs:
                f.write(f"#   RB {r['label']:<9} total={r['total']:.4f} kEUR "
                        f"(LPSP {r['lpsp']:.3f}%  deg {r['deg']:.2f})\n")
        f.write(f"# MIN global    : {best_all['label']} total={best_all['total']:.4f} kEUR\n")

    # --- console ---
    print(f"VoLL={V.VOLL_TIERS}  ({V.cost_lpsp_keur(1.0):.4f} kEUR / point LPSP, 25 ans)\n")
    print("rang  label        RB   LPSP%    deg     +LPSP    = total kEUR")
    for i, r in enumerate(rows_sorted, 1):
        tag = "RB" if r["is_rb"] else "  "
        star = "  <== meilleure RB" if r is best_rb else ("  <-- min global" if r is best_all else "")
        print(f"{i:>3}  {r['label']:<11} {tag}  {r['lpsp']:6.3f}  {r['deg']:6.2f}  "
              f"{r['clps']:6.2f}   {r['total']:7.2f}{star}")

    # --- figure : (LPSP, deg), iso-cout unifie, RB en avant ---
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 16, "axes.titlesize": 14, "legend.fontsize": 11,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    slope = V.cost_lpsp_keur(1.0)
    lps = np.array([r["lpsp"] for r in rows]); dgs = np.array([r["deg"] for r in rows])
    xs = np.linspace(max(0, lps.min() - 0.3), lps.max() + 0.3, 50)
    tot = np.array([r["total"] for r in rows])
    for C in np.linspace(tot.min(), tot.max(), 8):
        ax.plot(xs, C - slope * xs, ls=":", color="0.8", lw=0.8, zorder=0)

    # Fenetre du zoom (zone competitive) : sert au filtrage des labels.
    ZX = (1.45, 3.25); ZY = (60.0, 82.0)
    def in_zoom(r):
        return ZX[0] <= r["lpsp"] <= ZX[1] and ZY[0] <= r["deg"] <= ZY[1]

    def draw_point(axx, r, big):
        is_best_rb = (r is best_rb)
        if r["is_rb"]:
            axx.scatter(r["lpsp"], r["deg"],
                        marker="*" if is_best_rb else "o",
                        s=(380 if big else 340) if is_best_rb else (95 if big else 90),
                        color="crimson" if is_best_rb else "darkorange",
                        edgecolors="white", linewidths=0.8, zorder=5)
        else:
            axx.scatter(r["lpsp"], r["deg"], color="0.55", s=60 if big else 55, zorder=3)

    def label_point(axx, r, fs):
        axx.annotate(f"{r['label']} ({r['total']:.1f})", (r["lpsp"], r["deg"]),
                     fontsize=fs, color="black", xytext=(5, 4),
                     textcoords="offset points", zorder=6, path_effects=LABEL_STROKE)

    # Axe principal : tous les marqueurs ; labels UNIQUEMENT hors zone de zoom
    # (les points de la zone competitive sont annotes dans l'inset).
    for r in rows:
        draw_point(ax, r, big=False)
        if not in_zoom(r):
            label_point(ax, r, fs=9)

    # --- Inset zoome sur la zone competitive (RB2 / RB1 / 100-0 / RB2(SoH)) ---
    axins = ax.inset_axes([0.40, 0.44, 0.57, 0.52])
    axins.set_zorder(20)                       # au-dessus des labels de l'axe principal
    axins.patch.set_facecolor("white"); axins.patch.set_alpha(1.0)
    for C in np.linspace(tot.min(), tot.max(), 20):
        axins.plot(xs, C - slope * xs, ls=":", color="0.8", lw=0.8, zorder=0)
    for r in rows:
        if not in_zoom(r):
            continue
        draw_point(axins, r, big=True)
        label_point(axins, r, fs=10)
    axins.set_xlim(*ZX); axins.set_ylim(*ZY)
    axins.grid(True, linestyle="--", alpha=0.4)
    axins.tick_params(labelsize=10)
    axins.set_title("zone compétitive", fontsize=10)
    ax.indicate_inset_zoom(axins, edgecolor="0.4")

    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Coût unifié des stratégies EMS (25 ans, VoLL=3 €/kWh)\n"
                 "étoile = meilleure stratégie à base de règles")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEcrit : {OUT_TXT}\n        {OUT_PDF}\n        {OUT_PNG}")


if __name__ == "__main__":
    main()
