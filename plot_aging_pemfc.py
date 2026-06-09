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

# -------------------------------------------------------------------
# 2. Définition du modèle et des paramètres de la PEMFC
# -------------------------------------------------------------------
# # Constantes physiques et géométriques réalistes pour une cellule standard
# S = 100          # Surface active [cm²]
# j_in = 0.002     # Courant de fuite interne [A/cm²]
# j_0 = 0.0005     # Densité de courant d'échange [A/cm²]
# j_L = 1.5        # Densité de courant limite [A/cm²]
# A = 1.1e-4       # Coefficient de pertes d'activation [V/K]
# B = 2.2e-4       # Coefficient de pertes de transport de matière [V/K]

# FC = {
#     'n_series': 1,       # Nombre de cellules en série
#     'n_parallel': 1,     # Nombre de branches en parallèle
#     'E_0': 1.2,          # Tension thermodynamique à vide [V]
#     'R': 0.001,          # Résistance ohmique interne [Ohm]
#     'T': 353.15          # Température de fonctionnement [K] (80 °C)
# }

def voltage_fc(alpha_fc, i_fc): 
    voltage = FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc / FC['n_parallel'] 
              - A * FC['T'] * np.log((i_fc / S / FC['n_parallel'] + j_in) / j_0)
              - B * FC['T'] * np.log(1 - i_fc / S / FC['n_parallel'] / j_L / (1 - alpha_fc)))
    return voltage

# -------------------------------------------------------------------
# 3. Simulation des données (Vieillissement via alpha_fc)
# -------------------------------------------------------------------
# On fait varier le courant de 0 A jusqu'à la limite de transport de matière
# Pour alpha_fc = 0.35, le courant max sûr avant divergence est d'environ 80 A
i_fc_range = np.linspace(0.1, 220, 200)

# Génération des états de vieillissement (de 0 = Neuf à 0.35 = Vieilli)
n_curves = 15
alpha_levels = np.linspace(0, 0.35, n_curves)

fig, ax = plt.subplots(figsize=(10, 6))

# Palette de couleur progressive identique (Plasma tronqué)
cmap_aging = LinearSegmentedColormap.from_list(
    'aging_pemfc', plt.cm.plasma(np.linspace(0, 0.85, 256)))

# -------------------------------------------------------------------
# 4. Tracé des courbes de polarisation
# -------------------------------------------------------------------
for idx, alpha in enumerate(alpha_levels):
    v_cell = voltage_fc(alpha, i_fc_range * (1 - alpha))
    # Calcul de la couleur proportionnelle au niveau de dégradation
    color = cmap_aging(idx / (n_curves - 1))
    ax.plot(i_fc_range * (1 - alpha), v_cell, color=color, linewidth=2.5, alpha=0.85, zorder=2)

# -------------------------------------------------------------------
# 5. Mise en forme et limites des axes
# -------------------------------------------------------------------
# ax.set_xlim(0, 80)
# ax.set_ylim(0.4, 1.1)

ax.set_xlabel('Courant de PEMFC [A]', fontweight='bold', fontsize=28)
ax.set_ylabel('Tension de PEMFC [V]', fontweight='bold', fontsize=28)

# Flèche indicative de la dégradation de la tension
arrow = FancyArrowPatch((50, 0.85), (50, 0.65),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2, linestyle='--')

# -------------------------------------------------------------------
# 6. Légende : Colorbar en insert (Coin supérieur droit, zone libre)
# -------------------------------------------------------------------
cax = ax.inset_axes([0.1, 0.15, 0.03, 0.35])

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
plt.savefig("vieillissement_pemfc.pdf", bbox_inches='tight')
plt.savefig("vieillissement_pemfc.svg", bbox_inches='tight')
plt.show()