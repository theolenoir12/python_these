"""Recree une figure similaire a batt_deg_dod_cumul.PNG.

Trace la degradation cumulee delta_1C (uAh) en fonction du SoC (%) a partir
des donnees de Robustesse/Vieillissement8/Common/Cumulative_degradation_bat.txt.

Le fichier de donnees a une ligne par point au format  <x>,<y>
ou '.' est le separateur decimal et ',' le separateur de colonnes
(x = SoC en %, y = degradation cumulee en uAh).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "Robustesse" / "Vieillissement8" / "Common" / "Cumulative_degradation_bat.txt"
OUT = HERE / "batt_deg_dod_cumul.pdf"

NAVY = "#1a1a6e"
LIGHT = "#7a86e0"

# Police Latin Modern (= lmodern du manuscrit), fine et non grasse, fournie localement.
_LM = HERE / "fonts" / "lmroman10-regular.otf"
if _LM.exists():
    font_manager.fontManager.addfont(str(_LM))

plt.rcParams.update({
    'font.family': 'serif',           # serif type Computer Modern, comme lmodern du manuscrit
    # 'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',         # symboles (delta, mu) en Computer Modern
    'font.weight': 'normal',          # pas de gras
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'figure.titleweight': 'normal',
    'font.size': 26,
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.linewidth': 2,
})

def load_data(path):
    """Lit les colonnes (x, y) en gerant ',' comme separateur de colonnes."""
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Format : <x>,<y>  (separateur de colonnes = ',', decimal = '.')
        x_str, y_str = line.split(",")
        xs.append(float(x_str))
        ys.append(float(y_str))
    return np.array(xs), np.array(ys)


def main():
    x, y = load_data(DATA)

    # Noeuds s_0 ... s_10 : SoC = 0, 10, ..., 100 (degradation interpolee).
    soc_nodes = np.arange(0, 101, 10)
    deg_nodes = np.interp(soc_nodes, x, y)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Courbe complete + marqueurs diamant aux noeuds.
    ax.plot(x, y, "-", color=NAVY, lw=1.5, zorder=2)
    ax.plot(soc_nodes, deg_nodes, "D", color=NAVY, ms=8, zorder=3)

    # --- Axes, grille, libelles ---
    ax.set_xlabel("SoC [%]", fontsize=16)
    ax.set_ylabel(r"$\delta_{1C}$ Dégradation cumulée [$\mu$Ah]", fontsize=16)


    ax.set_xlim(-7, 105)
    ax.set_ylim(-20, 400)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 401, 100))
    ax.grid(True, ls="--", color="0.6", lw=0.8)
    ax.tick_params(labelsize=13)
 

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Figure enregistree : {OUT}")


if __name__ == "__main__":
    main()
    plt.show()
