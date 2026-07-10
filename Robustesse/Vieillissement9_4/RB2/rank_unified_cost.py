"""
rank_unified_cost.py
====================
Post-traitement du sweep RB2 : pour chaque combinaison (setpoints fixes) ET pour
RB2(SoH), calcule le COUT UNIFIE (degradation + cout financier du LPSP) avec la
VoLL EXACTEMENT definie dans Analyse_sensibilite/voll_common.py (VoLL=3 EUR/kWh
constante, E_REF=273380.73 kWh sur 25 ans). Classe, identifie le minimiseur, et
trace une figure focalisee : nuage RB2 fixe (gris) + le meilleur setpoint fixe
(min cout unifie) + RB2(SoH), annotes de leur cout unifie.

Aucune simulation : lit sweep_setpoints_rb2.txt.
Lancement (depuis Vieillissement8/RB2/) : python rank_unified_cost.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE   = os.path.dirname(os.path.abspath(__file__))
# voll_common vit dans Robustesse/Analyse_sensibilite/
SENS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Analyse_sensibilite"))
sys.path.insert(0, SENS_DIR)
import voll_common as V   # source de verite pour la VoLL

IN_TXT  = os.path.join(HERE, "sweep_setpoints_rb2.txt")
OUT_TXT = os.path.join(HERE, "sweep_setpoints_rb2_unified.txt")
OUT_PDF = os.path.join(HERE, "sweep_setpoints_rb2_unified.pdf")
OUT_PNG = os.path.join(HERE, "sweep_setpoints_rb2_unified.png")


def read_rows():
    rows = []
    with open(IN_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 5:
                continue
            kind, fc, el, lpsp, cost = p
            rows.append(dict(kind=kind, fc=float(fc), el=float(el),
                             lpsp=float(lpsp), deg=float(cost)))
    return rows


def main():
    rows = read_rows()
    for r in rows:
        # clps=None -> valorisation agregee VoLL*EENS (USE_STEPWISE_LPS=False)
        r["total"] = V.total_cost_keur(r["lpsp"], r["deg"])
        r["clpsp"] = V.cost_lpsp_keur(r["lpsp"])

    # tri par cout unifie croissant
    rows_sorted = sorted(rows, key=lambda r: r["total"])

    # minimiseur PARMI LES SETPOINTS FIXES de RB2 (hors RB2(SoH))
    fixed = [r for r in rows if r["kind"] != "RB2(SoH)"]
    best_fixed = min(fixed, key=lambda r: r["total"])
    soh = next(r for r in rows if r["kind"] == "RB2(SoH)")
    # minimiseur GLOBAL (toutes combinaisons, RB2(SoH) comprise)
    best_all = rows_sorted[0]

    # --- ecriture txt ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Cout unifie = deg + VoLL*(LPSP/100)*E_REF/1000  (voll_common.py)\n")
        f.write(f"# VoLL={V.VOLL_TIERS}  E_REF_KWH={V.E_REF_KWH:.3f}  "
                f"-> {V.cost_lpsp_keur(1.0):.4f} kEUR par point de LPSP\n")
        f.write("rang;kind;fc_frac;ely_frac;LPSP(%);deg(kEUR);coutLPSP(kEUR);total(kEUR)\n")
        for i, r in enumerate(rows_sorted, 1):
            f.write(f"{i};{r['kind']};{r['fc']:.3f};{r['el']:.3f};{r['lpsp']:.4f};"
                    f"{r['deg']:.4f};{r['clpsp']:.4f};{r['total']:.4f}\n")
        f.write(f"\n# MIN parmi setpoints FIXES : fc={best_fixed['fc']:.3f} "
                f"ely={best_fixed['el']:.3f} -> total={best_fixed['total']:.4f} kEUR\n")
        f.write(f"# RB2(SoH)                  : total={soh['total']:.4f} kEUR\n")
        f.write(f"# MIN global                : {best_all['kind']} "
                f"(fc={best_all['fc']:.3f} ely={best_all['el']:.3f}) "
                f"total={best_all['total']:.4f} kEUR\n")

    # --- console ---
    print(f"VoLL={V.VOLL_TIERS}  ({V.cost_lpsp_keur(1.0):.4f} kEUR / point LPSP)\n")
    print("rang  kind             fc     ely    LPSP%    deg     +LPSP    = total kEUR")
    for i, r in enumerate(rows_sorted, 1):
        star = "  <--" if r is best_all else ""
        print(f"{i:>3}  {r['kind']:<16} {r['fc']:.3f}  {r['el']:.3f}  "
              f"{r['lpsp']:6.3f}  {r['deg']:6.2f}  {r['clpsp']:6.2f}   "
              f"{r['total']:7.3f}{star}")
    print(f"\nMIN setpoints fixes : fc={best_fixed['fc']:.3f} ely={best_fixed['el']:.3f}"
          f"  -> {best_fixed['total']:.3f} kEUR")
    print(f"RB2(SoH)            : -> {soh['total']:.3f} kEUR "
          f"(ecart {soh['total']-best_fixed['total']:+.3f} kEUR)")

    # --- figure ---
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 18, "axes.titlesize": 15, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))

    # nuage des setpoints fixes (gris), non retenus
    for r in fixed:
        if r is best_fixed:
            continue
        ax.scatter(r["lpsp"], r["deg"], color="0.72", s=45, zorder=2)
        ax.annotate(f"{r['fc']:.2f}/{r['el']:.2f}", (r["lpsp"], r["deg"]),
                    fontsize=7, color="0.5", xytext=(3, 3),
                    textcoords="offset points", zorder=2)

    # meilleur setpoint fixe (min cout unifie)
    ax.scatter(best_fixed["lpsp"], best_fixed["deg"], marker="o", color="darkorange",
               s=150, zorder=5, edgecolors="white", linewidths=0.8,
               label=(f"RB2 fixe optimal ({best_fixed['fc']:.3f}/{best_fixed['el']:.3f})"
                      f"  {best_fixed['total']:.1f} k€"))
    ax.annotate(f"{best_fixed['total']:.1f} k€", (best_fixed["lpsp"], best_fixed["deg"]),
                fontsize=11, color="darkorange", weight="bold", path_effects=LABEL_STROKE,
                xytext=(6, -12), textcoords="offset points", zorder=5)

    # RB2(SoH)
    ax.scatter(soh["lpsp"], soh["deg"], marker="*", color="crimson", s=300, zorder=6,
               edgecolors="white", linewidths=0.8,
               label=f"RB2(SoH)  {soh['total']:.1f} k€")
    ax.annotate(f"{soh['total']:.1f} k€", (soh["lpsp"], soh["deg"]),
                fontsize=11, color="crimson", weight="bold", path_effects=LABEL_STROKE,
                xytext=(6, 6), textcoords="offset points", zorder=6)

    # droites d'iso-cout unifie passant par les 2 points (pente = -VoLL*E_REF/1e5)
    slope = -V.cost_lpsp_keur(1.0)   # d(deg)/d(LPSP) a cout total constant
    xs = np.array([min(r["lpsp"] for r in rows) - 0.05,
                   max(r["lpsp"] for r in rows) + 0.05])
    for r, c in [(best_fixed, "darkorange"), (soh, "crimson")]:
        ax.plot(xs, r["deg"] + slope * (xs - r["lpsp"]), ls=":", color=c, lw=1.0,
                alpha=0.6, zorder=1)

    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_title("Coût unifié (deg + VoLL·LPSP) : meilleur setpoint fixe vs RB2(SoH)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEcrit : {OUT_TXT}\n        {OUT_PDF}\n        {OUT_PNG}")


if __name__ == "__main__":
    main()
