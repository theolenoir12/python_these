import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from Init_EMR_MG_v16_python import *

# -------------------------------------------------------------------
# 1. Configuration du style (Identique à votre charte graphique)
# -------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 26,     
    'axes.titlesize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.linewidth': 2,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
})

text_style = [PathEffects.withStroke(linewidth=4, foreground='white')]


def voltage_ely(alpha_ely,i_ely) : 
    voltage = ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely / ELY['n_parallel'] 
              + A * ELY['T'] * np.log((i_ely / S / ELY['n_parallel'] + j_in) / j_0)
              + B * ELY['T'] * np.log(1 - i_ely / S / ELY['n_parallel'] / j_L / (1 - alpha_ely)))
    return voltage     

# -------------------------------------------------------------------
# 3. Simulation des données (Vieillissement via alpha_ely)
# -------------------------------------------------------------------
# On fait varier le courant de 0 A jusqu'à la limite de transport de matière
# Pour alpha_ely = 0.35, le courant max sûr avant divergence est d'environ 80 A
i_ely_range = np.linspace(0.1, 220, 200)

# Génération des états de vieillissement (de 0 = Neuf à 0.35 = Vieilli)
n_curves = 15
alpha_levels = np.linspace(0, 0.35, n_curves)

fig, ax = plt.subplots(figsize=(10, 6))

# Palette de couleur progressive identique (Plasma tronqué)
cmap_aging = LinearSegmentedColormap.from_list(
    'aging_pemely', plt.cm.plasma(np.linspace(0, 0.85, 256)))

# -------------------------------------------------------------------
# 4. Tracé des courbes de polarisation
# -------------------------------------------------------------------
for idx, alpha in enumerate(alpha_levels):
    v_cell = voltage_ely(alpha, i_ely_range * (1 - alpha))
    # Calcul de la couleur proportionnelle au niveau de dégradation
    color = cmap_aging(idx / (n_curves - 1))
    ax.plot(i_ely_range * (1 - alpha), v_cell, color=color, linewidth=2.5, alpha=0.85, zorder=2)

# -------------------------------------------------------------------
# 5. Mise en forme et limites des axes
# -------------------------------------------------------------------
# ax.set_xlim(0, 80)
# ax.set_ylim(0.4, 1.1)

ax.set_xlabel('Courant de PEMWE [A]', fontweight='bold', fontsize=28)
ax.set_ylabel('Tension de PEMWE [V]', fontweight='bold', fontsize=28)

# Flèche indicative de la dégradation de la tension
arrow = FancyArrowPatch((50, 0.85), (50, 0.65),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2, linestyle='--')

# -------------------------------------------------------------------
# 6. Légende : Colorbar en insert (Coin supérieur droit, zone libre)
# -------------------------------------------------------------------
cax = ax.inset_axes([0.1, 0.5, 0.03, 0.35])

cb = ColorbarBase(cax, cmap=cmap_aging, orientation='vertical')
cb.set_ticks([0, 1])
cb.set_ticklabels(['Début de vie', 'Fin de vie'])
cb.ax.tick_params(labelsize=26)

# Inversion pour correspondre à la baisse de performance (Violet en haut, Jaune en bas)
cax.invert_yaxis()
cb.outline.set_linewidth(1.5)
cb.ax.yaxis.set_tick_params(length=4, width=1.5)

# -------------------------------------------------------------------
# 7. Export final vectoriel (.pdf & .svg)
# -------------------------------------------------------------------
plt.tight_layout()
plt.savefig("vieillissement_pemely.pdf", bbox_inches='tight')
plt.savefig("vieillissement_pemely.svg", bbox_inches='tight')
plt.show()