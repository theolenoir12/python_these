import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap

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

# Effet de contour blanc pour assurer une lisibilité maximale sur l'écran
text_style = [PathEffects.withStroke(linewidth=4, foreground='white')]

# -------------------------------------------------------------------
# 2. Simulation des données (Vieillissement / Perte de capacité)
# -------------------------------------------------------------------
cycles = np.linspace(0, 1200, 400)

# Modèle non-linéaire réaliste : dégradation lente puis accélération en fin de vie
# Atteint précisément le seuil critique des 80% autour du cycle 1200
capacity = 100 - 0.005 * cycles - 0.0000095 * (cycles**2)

fig, ax = plt.subplots(figsize=(10, 6))

# Définition de la palette de couleur progressive (Plasma tronqué comme votre code)
cmap_aging = LinearSegmentedColormap.from_list(
    'aging_capacity', plt.cm.plasma(np.linspace(0, 0.85, 256)))

# Tracé sous forme de scatter dense pour appliquer le gradient de couleur sur la courbe
sc = ax.scatter(cycles, capacity, c=cycles, cmap=cmap_aging, 
                s=15, alpha=0.8, edgecolors='none', zorder=2)

# -------------------------------------------------------------------
# 3. Mise en forme et limites des axes
# -------------------------------------------------------------------
ax.set_xlim(0, 1200)
ax.set_ylim(80, 100)

ax.set_xlabel('Nombre de cycles de charge/décharge', fontweight='bold',fontsize=28)
ax.set_ylabel('Capacité [%]', fontweight='bold',fontsize=28)

# Ligne horizontale indiquant le seuil critique standard de fin de vie (EOL)
ax.axhline(80, color='black', linewidth=1.5, linestyle='--')

# -------------------------------------------------------------------
# 4. Légende : Colorbar en insert (Coin supérieur droit, zone libre)
# -------------------------------------------------------------------
cax = ax.inset_axes([0.1, 0.25, 0.03, 0.35])

cb = ColorbarBase(cax, cmap=cmap_aging, orientation='vertical')
cb.set_ticks([0, 1])
cb.set_ticklabels(['Début de vie', 'Fin de vie'])
cb.ax.tick_params(labelsize=26)

# Inversion pour avoir le "Début de vie" (violet) en haut et "Fin de vie" en bas
cax.invert_yaxis()
cb.outline.set_linewidth(1.5)
cb.ax.yaxis.set_tick_params(length=4, width=1.5)


# -------------------------------------------------------------------
# 6. Export final vectoriel (.pdf)
# -------------------------------------------------------------------
plt.tight_layout()
plt.savefig("vieillissement_batterie.pdf", bbox_inches='tight')
plt.savefig("vieillissement_batterie.svg", bbox_inches='tight')
plt.show()