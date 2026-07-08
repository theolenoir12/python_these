"""
rank_rb2soh_unified.py
======================
Post-traitement du sweep RB2(SoH) (sweep_setpoints_rb2soh.txt) : cout UNIFIE
(degradation + VoLL*LPSP) via Analyse_sensibilite/voll_common.py, classement, et
figure "amplitude des gains" = cout unifie en fonction de gamma_ELY (une courbe
par base f_ELY), avec la ligne de reference du MEILLEUR RB2 FIXE (lu depuis
sweep_setpoints_rb2.txt) : tout point RB2(SoH) SOUS cette ligne = gain net reel
de l'exploitation du SoH ; au-dessus = le SoH n'apporte rien de plus qu'un
setpoint fixe bien regle.

Aucune simulation. Lancement (depuis Vieillissement8/RB2/) :
    python rank_rb2soh_unified.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE     = os.path.dirname(os.path.abspath(__file__))
SENS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Analyse_sensibilite"))
sys.path.insert(0, SENS_DIR)
import voll_common as V

SOH_TXT   = os.path.join(HERE, "sweep_setpoints_rb2soh.txt")
# La reference "meilleur RB2 fixe" est produite dans le dossier RB2/ (a cote).
FIXED_TXT = os.path.abspath(os.path.join(HERE, "..", "RB2", "sweep_setpoints_rb2.txt"))
OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb2soh_unified.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb2soh_unified.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb2soh_unified.png")


def read_soh():
    rows = []
    with open(SOH_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 7:
                continue
            r = dict(kind=p[0], fc=float(p[1]), gfc=float(p[2]), el=float(p[3]),
                     gel=float(p[4]), lpsp=float(p[5]), deg=float(p[6]))
            r["total"] = V.total_cost_keur(r["lpsp"], r["deg"])
            rows.append(r)
    return rows


def read_fixed_refs():
    """Depuis sweep_setpoints_rb2.txt : (meilleur RB2 fixe [min cout unifie],
    RB2 nominal nu [0.450/0.330]). Chacun None si absent."""
    best = None
    nominal = None
    if not os.path.exists(FIXED_TXT):
        return None, None
    with open(FIXED_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 5 or p[0] == "RB2(SoH)":
                continue
            lpsp, deg = float(p[3]), float(p[4])
            tot = V.total_cost_keur(lpsp, deg)
            cand = dict(fc=float(p[1]), el=float(p[2]), lpsp=lpsp, deg=deg, total=tot)
            if best is None or tot < best["total"]:
                best = cand
            if p[0] == "RB2_nominal":
                nominal = cand
    return best, nominal


def main():
    rows = read_soh()
    rows_sorted = sorted(rows, key=lambda r: r["total"])
    best_soh = rows_sorted[0]
    nominal = next((r for r in rows if r["kind"] == "RB2(SoH)_nominal"), None)
    fixed_best, rb2_nominal = read_fixed_refs()

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Cout unifie = deg + VoLL*(LPSP/100)*E_REF/1000  (voll_common.py)\n")
        f.write(f"# VoLL={V.VOLL_TIERS}  {V.cost_lpsp_keur(1.0):.4f} kEUR / point LPSP\n")
        f.write("rang;kind;f_ELY;gamma_ELY;LPSP(%);deg(kEUR);total(kEUR)\n")
        for i, r in enumerate(rows_sorted, 1):
            f.write(f"{i};{r['kind']};{r['el']:.3f};{r['gel']:.3f};{r['lpsp']:.4f};"
                    f"{r['deg']:.4f};{r['total']:.4f}\n")
        f.write(f"\n# MIN RB2(SoH) : f_ELY={best_soh['el']:.3f} gamma={best_soh['gel']:.3f}"
                f" -> {best_soh['total']:.4f} kEUR\n")
        if nominal:
            f.write(f"# RB2(SoH) nominal (0.320^0.5) : {nominal['total']:.4f} kEUR\n")
        if rb2_nominal:
            f.write(f"# ref RB2 NOMINAL nu : {rb2_nominal['fc']:.3f}/{rb2_nominal['el']:.3f}"
                    f" -> {rb2_nominal['total']:.4f} kEUR "
                    f"(gain MIN RB2(SoH) : {best_soh['total']-rb2_nominal['total']:+.4f})\n")
        if fixed_best:
            f.write(f"# ref MEILLEUR RB2 FIXE : {fixed_best['fc']:.3f}/{fixed_best['el']:.3f}"
                    f" -> {fixed_best['total']:.4f} kEUR\n")
            f.write(f"# gain MIN RB2(SoH) vs meilleur fixe : "
                    f"{best_soh['total']-fixed_best['total']:+.4f} kEUR\n")

    print(f"VoLL={V.VOLL_TIERS}  ({V.cost_lpsp_keur(1.0):.4f} kEUR / point LPSP)\n")
    print("rang  kind               f_ELY  gamma   LPSP%    deg     total kEUR")
    for i, r in enumerate(rows_sorted, 1):
        print(f"{i:>3}  {r['kind']:<18} {r['el']:.3f}  {r['gel']:.2f}   "
              f"{r['lpsp']:6.3f}  {r['deg']:6.2f}   {r['total']:7.3f}")
    print(f"\nMIN RB2(SoH) : f_ELY={best_soh['el']:.3f} gamma={best_soh['gel']:.2f}"
          f" -> {best_soh['total']:.3f} kEUR")
    if nominal:
        print(f"RB2(SoH) nominal (0.320^0.5) -> {nominal['total']:.3f} kEUR")
    if rb2_nominal:
        print(f"RB2 nominal nu ({rb2_nominal['fc']:.3f}/{rb2_nominal['el']:.3f}) "
              f"-> {rb2_nominal['total']:.3f} kEUR "
              f"(gain MIN RB2(SoH) : {best_soh['total']-rb2_nominal['total']:+.3f})")
    if fixed_best:
        print(f"meilleur RB2 FIXE ({fixed_best['fc']:.3f}/{fixed_best['el']:.3f}) "
              f"-> {fixed_best['total']:.3f} kEUR")
        print(f"=> gain MIN RB2(SoH) vs meilleur fixe : "
              f"{best_soh['total']-fixed_best['total']:+.3f} kEUR")

    # --- figure : cout unifie vs gamma_ELY, une courbe par base ---
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 17, "axes.titlesize": 14, "legend.fontsize": 11,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    bases = sorted(set(r["el"] for r in rows))
    cmap = {b: c for b, c in zip(bases, plt.cm.viridis(np.linspace(0.1, 0.8, len(bases))))}
    for b in bases:
        pts = sorted([r for r in rows if r["el"] == b], key=lambda r: r["gel"])
        gs = [p["gel"] for p in pts]; ts = [p["total"] for p in pts]
        ax.plot(gs, ts, "-o", color=cmap[b], ms=6, label=f"$f_{{ELY}}$={b:.3f}")

    if rb2_nominal:
        ax.axhline(rb2_nominal["total"], color="0.35", ls=":", lw=1.8,
                   label=(f"RB2 nominal nu ({rb2_nominal['fc']:.2f}/{rb2_nominal['el']:.2f})"
                          f" = {rb2_nominal['total']:.1f} k€"))
    if fixed_best:
        ax.axhline(fixed_best["total"], color="darkorange", ls="--", lw=1.8,
                   label=(f"meilleur RB2 fixe ({fixed_best['fc']:.2f}/{fixed_best['el']:.2f})"
                          f" = {fixed_best['total']:.1f} k€"))
    if nominal:
        ax.scatter([nominal["gel"]], [nominal["total"]], marker="*", color="crimson",
                   s=280, zorder=6, edgecolors="white", linewidths=0.8,
                   label=f"RB2(SoH) nominal = {nominal['total']:.1f} k€")

    ax.set_xlabel(r"exposant $\gamma_{ELY}$  (0 = setpoint fixe)")
    ax.set_ylabel("Coût unifié [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.92)
    ax.set_title("RB2(SoH) : coût unifié vs intensité de la modulation SoH")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEcrit : {OUT_TXT}\n        {OUT_PDF}\n        {OUT_PNG}")


if __name__ == "__main__":
    main()
