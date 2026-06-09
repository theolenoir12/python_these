import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Configuration du style scientifique ---
plt.rcParams.update({
    "font.family": "serif",      # Style de police classique pour les revues
    "font.size": 13,             # +2pt par rapport à la version précédente (11)
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300            # Haute résolution pour l'impression
})

# --- Données et positions (rehaussées vers le haut) ---
# Format : (Label, Start, End, Y_center, Color)
processes = [
    ("Electric double-layer\ncharging", 1e-6, 1e-2, 8.8, '#4A90E2'),
    ("Membrane\nhumidification", 2*1e0, 1e4, 8.8, '#4A90E2'),
    
    ("Charge transfer fuel cell\nreactions", 1e-4, 2*1e0, 7.2, '#50C878'),
    ("Liquid water\ntransport", 3*1e0, 2e3, 7.2, '#50C878'),
    
    ("Gas diffusion processes", 3e-3, 2e1, 5.6, '#F5A623'),
    ("Changes in catalytic\nproperties/poisoning", 1e1, 3e4, 5.6, '#F5A623'),
    
    ("Temperature\neffects", 2e1, 3e4, 4.0, '#D0021B'),
    ("Degradation and\naging effects", 2e4, 1e8, 4.0, '#7B61FF'),
]

time_regions = [
    ("Microseconds", 1e-6, 1e-3, '#ECF0F1'),
    ("Milliseconds", 1e-3, 1e0, '#D5DBDB'),
    ("Seconds", 1e0, 60, '#ECF0F1'),
    ("Minutes", 60, 3600, '#D5DBDB'),
    ("Hours", 3600, 86400, '#ECF0F1'),
    ("Days", 86400, 2.6e6, '#D5DBDB'),
    ("Months", 2.6e6, 1e8, '#ECF0F1'),
]

# --- Création de la figure ---
fig, ax = plt.subplots(figsize=(12, 6))

# 1. Configuration des axes
ax.set_xscale('log')
ax.set_xlim(1e-6, 1e8)
ax.set_ylim(0.5, 10)  # On commence à 0.5 pour laisser de la place en haut
ax.set_xlabel('Time (s)', fontweight='bold', labelpad=10)

# 2. Dessin des blocs de processus
for label, start, end, y, color in processes:
    # Rectangle avec une légère transparence et bordure grise
    rect = patches.Rectangle((start, y-0.6), end-start, 1.2, 
                             linewidth=0.8, edgecolor='#333333', 
                             facecolor=color, alpha=0.3, zorder=3)
    ax.add_patch(rect)
    
    # Texte centré (moyenne géométrique pour l'échelle log)
    text_x = start * np.sqrt(end/start)
    ax.text(text_x, y, label, ha='center', va='center', 
            linespacing=1.2, fontweight='medium')

# 3. Barre temporelle en bas (style propre)
for name, start, end, color in time_regions:
    rect = patches.Rectangle((start, 0.8), end-start, 0.8, 
                             facecolor=color, edgecolor='#BDC3C7', zorder=2)
    ax.add_patch(rect)
    
    text_x = start * np.sqrt(end/start)
    ax.text(text_x, 1.2, name, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#2C3E50')

# 4. Esthétique de la grille et des bordures
ax.grid(True, which="both", ls="--", lw=0.5, color='#BDC3C7', alpha=0.6, zorder=1)
ax.set_yticks([]) # On cache les graduations Y car elles ne sont pas physiques
ax.spines['left'].set_visible(False) # On enlève la bordure gauche pour aérer

# On force l'affichage des ticks majeurs
ax.tick_params(axis='x', which='major', length=7, width=1.2)
ax.tick_params(axis='x', which='minor', length=4, width=0.8)

plt.tight_layout()

# Sauvegarde optionnelle en format vectoriel (PDF/EPS) pour la publication
# plt.savefig("fuel_cell_timescales.pdf", bbox_inches='tight')

plt.show()