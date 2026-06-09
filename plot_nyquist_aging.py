import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects

# -------------------------------------------------------------------
# 1. Configuration du style
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

# Effet de contour blanc pour la lisibilité
text_style = [PathEffects.withStroke(linewidth=4, foreground='white')]

# -------------------------------------------------------------------
# 2. Simulation des données (Modèle complet)
# -------------------------------------------------------------------
f = np.logspace(6, -2, 300)
w = 2 * np.pi * f
num_cycles = 12
colors = plt.cm.plasma(np.linspace(0, 0.85, num_cycles))
# colors = plt.cm.gray(np.linspace(0, 0.7, num_cycles))

fig, ax = plt.subplots(figsize=(15, 7))

for i in range(num_cycles):
    prog = i / (num_cycles - 1)
    L = 1.8e-7
    R_ohm = 0.035 + 0.012 * prog
    
    # SEI (Cercle HF)
    R_sei = 0.012 + 0.004 * prog
    C_sei = 4e-5
    Z_sei = R_sei / (1 + (1j * w * R_sei * C_sei)**0.92)
    
    # Transfert de charge
    R_ct = 0.02 + 0.13 * prog * (1 + 0.5 * prog)
    C_dl = 0.015
    Z_ct = R_ct / (1 + (1j * w * R_ct * C_dl)**0.78)
    
    # Warburg (Diffusion)
    Aw = 0.005 + 0.015 * prog
    Z_W = Aw / np.sqrt(w) * (1 - 1j)
    
    Z = 1j * w * L + R_ohm + Z_sei + Z_ct + Z_W
    
    ax.plot(Z.real, -Z.imag, marker='o', linestyle='None', 
            color=colors[i], markersize=3.5, alpha=0.6)

# -------------------------------------------------------------------
# 3. Mise en forme
# -------------------------------------------------------------------
ax.set_xlim(0.025, 0.27)
ax.set_ylim(-0.015, 0.08) # Limite demandée

ax.set_xlabel(r'Re(Z) [$\Omega$]', fontweight='bold')
ax.set_ylabel(r'-Im(Z) [$\Omega$]', fontweight='bold')
ax.axhline(0, color='black', linewidth=1.5, linestyle='--')

# -------------------------------------------------------------------
# 4. Légende colorbar dans le coin haut gauche
# -------------------------------------------------------------------
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap

cax = ax.inset_axes([0.03, 0.62, 0.04, 0.35])

cmap_legend = LinearSegmentedColormap.from_list(
    'aging', plt.cm.plasma(np.linspace(0, 0.85, 256)))

# cmap_legend = LinearSegmentedColormap.from_list(
#     'aging', plt.cm.gray(np.linspace(0, 0.7, 256))
# )


cb = ColorbarBase(cax, cmap=cmap_legend, orientation='vertical')
cb.set_ticks([0, 1])
cb.set_ticklabels(['Début de vie', 'Fin de vie'])
cb.ax.tick_params(labelsize=22)

cax.invert_yaxis()
cb.outline.set_linewidth(1.5)
cb.ax.yaxis.set_tick_params(length=4, width=1.5)

# -------------------------------------------------------------------
# 4. Annotations Ultra-Lisibles
# -------------------------------------------------------------------
arrow_opt = dict(mutation_scale=30, facecolor='white', edgecolor='black', linewidth=2.5, zorder=10)

# Annotation 1 : Résistance Ohmique
arrow1 = FancyArrowPatch((0.045, -0.003), (0.07, -0.003), **arrow_opt)
ax.add_patch(arrow1)
t1 = ax.text(0.057, -0.0055, 'Augmentation de\nla résistance interne', 
             color='black', fontsize=20, ha='center', va='top', 
             fontweight='bold', zorder=11)
t1.set_path_effects(text_style)

# Annotation 2 : Vieillissement (Cycle Ageing)
arrow2 = FancyArrowPatch((0.13, 0.0035), (0.22, 0.009), **arrow_opt)
ax.add_patch(arrow2)

# Texte "Increase in charge transfer..."
t3 = ax.text(0.175, 0.0048, 'Augmentation de la\nrésistance de transfert de charge', 
             color='black', fontsize=20, ha='center', va='top', rotation=3.5, 
             fontweight='bold', zorder=11)
t3.set_path_effects(text_style)


# -------------------------------------------------------------------
# 5. Export final
# -------------------------------------------------------------------
plt.tight_layout()
plt.savefig("nyquist_aging.pdf", bbox_inches='tight')
plt.show()