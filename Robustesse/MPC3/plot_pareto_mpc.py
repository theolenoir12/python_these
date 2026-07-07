# -*- coding: utf-8 -*-
"""
plot_pareto_mpc.py -- Figure Pareto (LPSP vs cout de degradation) du bench MPC.
================================================================================
Lit bench_mpc_cloud.csv (et, s'ils existent, sweep_mpc_pareto_cloud.csv et le
front DP v2 du mesocentre) et trace le nuage strategie par strategie :
moyenne +- 1 sigma sur les graines, conventions graphiques des figures Pareto
du manuscrit (serif, mathtext cm).

    python plot_pareto_mpc.py [bench_mpc_cloud.csv]

Sortie : pareto_mpc.pdf (+ .png) dans MPC/.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bench_mpc_cloud.csv")
SWEEP_PARETO = os.path.join(HERE, "sweep_mpc_pareto_cloud.csv")
DP_FRONT = os.path.join(HERE, "..", "Vieillissement8", "DP", "results_meso",
                        "dp_pareto_25y_51x51_v2.txt")

plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.labelsize": 18, "legend.fontsize": 12,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "grid.alpha": 0.7, "grid.linestyle": "--", "grid.linewidth": 0.6,
})


def read_cloud(path):
    """{label: (lpsp[], deg[])} depuis un *_cloud.csv du banc."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            out.setdefault(row["label"], ([], []))
            out[row["label"]][0].append(float(row["lpsp_pct"]))
            out[row["label"]][1].append(float(row["deg_keur"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}


def read_dp_front(path):
    """Points (LPSP%, deg kEUR) du front DP v2 (lignes numeriques du txt)."""
    pts = []
    try:
        with open(path) as f:
            for line in f:
                tok = [t for t in line.split() if t != "*"]
                if len(tok) < 5:
                    continue
                try:
                    vals = [float(t) for t in tok[:5]]
                except ValueError:
                    continue
                pts.append((vals[1], vals[2]))          # LPSP%, deg_kE
    except OSError:
        pass
    return np.array(pts) if pts else None


clouds = read_cloud(CLOUD)
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

dp = read_dp_front(DP_FRONT)
if dp is not None:
    dp = dp[np.argsort(dp[:, 0])]
    ax.plot(dp[:, 0], dp[:, 1], "-", color="0.45", lw=1.5, zorder=1,
            label="Front DP v2 (borne offline)")

if os.path.exists(SWEEP_PARETO):
    sw = read_cloud(SWEEP_PARETO)
    pts = np.array([[v[0].mean(), v[1].mean()] for v in sw.values()])
    pts = pts[np.argsort(pts[:, 0])]
    ax.plot(pts[:, 0], pts[:, 1], ":", color="tab:red", lw=1.5, zorder=2,
            label="Front MPC (balayage VoLL interne)")

for i, (lab, (lp, dg)) in enumerate(clouds.items()):
    col = colors[i % 10]
    ax.errorbar(lp.mean(), dg.mean(), xerr=lp.std(), yerr=dg.std(),
                fmt="o", ms=7, color=col, ecolor=col, elinewidth=1.2,
                capsize=3, zorder=3, label=f"{lab} (N={len(lp)})")

ax.set_xlabel("LPSP [%]")
ax.set_ylabel("Degradation cost [k€]")
ax.grid(True)
ax.legend(loc="best", framealpha=0.9)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(HERE, f"pareto_mpc.{ext}"),
                bbox_inches="tight", dpi=200)
print("-> pareto_mpc.pdf / .png")
