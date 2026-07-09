# -*- coding: utf-8 -*-
"""
analyse_mpc2.py -- Depouillement du plan B (jobs mesocentre 213286 + 213287).
==============================================================================
Lit MPC2/bench_mpc.txt (comparatif complet N=100 avec ancrages RB2 + points
omniscients) et MPC2/sweep_mpc_pareto.txt (front MPC, VoLL interne), reconstruit
l'EENS de chaque strategie (total = deg + VoLL*EENS/1000), calcule les VoLL de
CROISEMENT MPC/regles, et produit la figure a deux panneaux :
  - gauche : espace des objectifs (LPSP vs deg) + front DP en fond ;
  - droite : cout unifie vs VoLL d'evaluation + point de croisement.

Aucune dependance aux *_cloud.csv (non necessaires) : tout est reconstruit des
tableaux de synthese. Lancement : python analyse_mpc2.py
Sorties : pareto_mpc2.pdf / .png dans MPC2/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VOLL = 3.0     # VoLL d'evaluation de reference (fige, comme tout le comparatif)

# Front DP v2 (borne offline). Adapter le chemin si besoin.
DP_FRONT = os.path.join(ROOT, "Vieillissement8", "DP", "results_meso",
                        "dp_pareto_25y_51x51_v2.txt")


def parse_bench(path):
    """{label: dict(N, lpsp, slpsp, deg, sdeg, total, stotal, ely)} depuis un .txt."""
    rows = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("label"):
                continue
            p = [x.strip() for x in line.split(";")]
            if len(p) < 10:
                continue
            rows[p[0]] = dict(N=int(p[1]), lpsp=float(p[2]), slpsp=float(p[3]),
                              deg=float(p[4]), sdeg=float(p[5]), total=float(p[6]),
                              stotal=float(p[7]), ely=float(p[9]))
    for r in rows.values():                      # EENS reconstruit [kWh]
        r["eens"] = (r["total"] - r["deg"]) / VOLL * 1000.0
    return rows


def dp_front(path):
    pts = []
    try:
        with open(path) as f:
            for line in f:
                tok = [t for t in line.split() if t != "*"]
                if len(tok) < 5:
                    continue
                try:
                    v = [float(t) for t in tok[:5]]
                except ValueError:
                    continue
                pts.append((v[1], v[2]))         # LPSP%, deg_kE
    except OSError:
        return None
    a = np.array(pts)
    return a[np.argsort(a[:, 0])] if len(a) else None


def crossover(bench, a, b):
    """VoLL* ou total_a(V) = total_b(V) ; None si pentes egales."""
    da, ea = bench[a]["deg"], bench[a]["eens"] / 1000.0
    db, eb = bench[b]["deg"], bench[b]["eens"] / 1000.0
    return None if abs(ea - eb) < 1e-9 else (db - da) / (ea - eb)


def main():
    bench = parse_bench(os.path.join(HERE, "bench_mpc.txt"))

    print("=== Reconstruction EENS ===")
    for lab, r in bench.items():
        print(f"{lab:<26s} LPSP={r['lpsp']:.2f}%  deg={r['deg']:.2f}  "
              f"EENS={r['eens']:.0f} kWh  tot@3={r['total']:.2f}  ELY={r['ely']:.0f}")

    print("\n=== VoLL de croisement MPC / regles ===")
    for mpc in ("MPC (H=24)", "MPC (H=48)"):
        for rb in ("RB2 socle", "RB2(SoH_all) (test nul)", "RB2(SoH_all+Pred)"):
            if mpc in bench and rb in bench:
                V = crossover(bench, mpc, rb)
                print(f"  {mpc} croise {rb:<26s} a VoLL = {V:.2f} EUR/kWh")

    if "MPC omni (H=48)" in bench:
        g = bench["MPC (H=48)"]["total"] - bench["MPC omni (H=48)"]["total"]
        print(f"\n=== Cout de l'imperfection de prevision (H=48) : "
              f"{g:.1f} kEUR (bruite {bench['MPC (H=48)']['total']:.1f} "
              f"vs omni {bench['MPC omni (H=48)']['total']:.1f}) ===")

    # --- Overlay du gagnant du plan C (sweep robust : sw x3) ------------------
    win_key = None
    rob_path = os.path.join(HERE, "sweep_mpc_robust.txt")
    if os.path.exists(rob_path):
        rob = parse_bench(rob_path)
        cand = [v for k, v in rob.items() if k.startswith("sw x3")]
        if cand:
            win_key = "MPC sw x3 (plan C)"
            bench[win_key] = cand[0]
            for rb in ("RB2(SoH_all+Pred)",):
                if rb in bench:
                    V = crossover(bench, win_key, rb)
                    print(f"\n=== Plan C : {win_key} total@3 = {cand[0]['total']:.2f} "
                          f"kEUR ; croise {rb} a VoLL = {V:.2f} EUR/kWh ===")

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
        "axes.labelsize": 15, "legend.fontsize": 9.5, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "grid.alpha": 0.6, "grid.linestyle": "--",
        "grid.linewidth": 0.6})
    disp = {"RB2(SoH_all) (test nul)": "RB2(SoH_all)"}
    dp = dp_front(DP_FRONT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    groups = {
        "RB2 socle": ("tab:blue", "o"), "RB2(SoH_all) (test nul)": ("tab:cyan", "o"),
        "RB2(SoH_all+Pred)": ("tab:green", "o"),
        "MPC (H=24)": ("tab:red", "s"), "MPC (H=48)": ("tab:orange", "s"),
        "MPC (H=24, gel 12h)": ("0.6", "x"),
        "MPC omni (H=24)": ("tab:purple", "D"), "MPC omni (H=48)": ("magenta", "D")}
    if win_key:
        groups[win_key] = ("crimson", "*")
    if dp is not None:
        ax1.plot(dp[:, 0], dp[:, 1], "-", color="0.4", lw=1.6, zorder=1,
                 label="Front DP (borne offline)")
    for lab, (c, m) in groups.items():
        if lab not in bench:
            continue
        r = bench[lab]
        ax1.errorbar(r["lpsp"], r["deg"], xerr=r["slpsp"], yerr=r["sdeg"],
                     fmt=m, ms=8, color=c, ecolor=c, capsize=3, zorder=3,
                     label=disp.get(lab, lab))
    ax1.set_xlabel("LPSP [%]"); ax1.set_ylabel(u"Coût de dégradation [k€]")
    ax1.set_title("Espace des objectifs (25 ans, VoLL=3)", fontsize=13)
    ax1.grid(True); ax1.legend(loc="upper right")

    Vg = np.linspace(0, 8, 200)
    lines = {"RB2 socle": ("tab:blue", "-"), "RB2(SoH_all) (test nul)": ("tab:cyan", "-"),
             "RB2(SoH_all+Pred)": ("tab:green", "-"), "MPC (H=24)": ("tab:red", "-"),
             "MPC (H=48)": ("tab:orange", "--"), "MPC omni (H=48)": ("magenta", ":")}
    if win_key:
        lines[win_key] = ("crimson", "-.")
    for lab, (c, ls) in lines.items():
        if lab not in bench:
            continue
        r = bench[lab]
        ax2.plot(Vg, r["deg"] + Vg * r["eens"] / 1000.0, ls, color=c, lw=1.9,
                 label=disp.get(lab, lab))
    if "MPC (H=24)" in bench and "RB2(SoH_all+Pred)" in bench:
        Vx = crossover(bench, "MPC (H=24)", "RB2(SoH_all+Pred)")
        Tx = bench["MPC (H=24)"]["deg"] + Vx * bench["MPC (H=24)"]["eens"] / 1000.0
        ax2.plot(Vx, Tx, "k*", ms=15, zorder=5)
        ax2.annotate(f"croisement\nMPC / RB ultime\nVoLL={Vx:.2f}", (Vx, Tx),
                     xytext=(Vx + 0.7, Tx + 10), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="k", lw=1))
    ax2.axvline(VOLL, color="0.7", lw=1, ls=":")
    ax2.set_xlabel(u"VoLL d'évaluation [€/kWh]")
    ax2.set_ylabel(u"Coût unifié total [k€]")
    ax2.set_title(u"Sensibilité au VoLL et point de croisement", fontsize=13)
    ax2.grid(True); ax2.legend(loc="upper left"); ax2.set_xlim(0, 8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(HERE, f"pareto_mpc2.{ext}"),
                    dpi=160, bbox_inches="tight")
    print("\n-> pareto_mpc2.pdf / .png")


if __name__ == "__main__":
    main()
