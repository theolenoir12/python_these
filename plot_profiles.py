import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
import os

# Constants
kB = 1.38065e-23        # (J/K) Boltzmann's constant
F = 96485               # (C/mol) Faraday's constant
R = 8.314               # (J/mol/K) Gas constant
q = 1.6e-19             # (C) Electronic charge

# Load data
sidelec_PV    = []
sidelec_conso = []

# with open('sidelec_roche_plate_csv.csv', 'r') as csvfile:
with open('sidelec_roche_plate_csv2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';') # change contents to floats
    for row in reader: # each row is a list
        sidelec_PV.append(float(row[1]))
        sidelec_conso.append(float(row[2]))

plt.rcParams.update({
    "text.usetex": False,          # Pas besoin de LaTeX externe
    "mathtext.fontset": "cm",      # Computer Modern (style LaTeX)
    "font.family": "serif",        # Police générale
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6
})

sidelec_conso = np.array(sidelec_conso*20)
sidelec_PV    = np.array(sidelec_PV*20)

T = 365*24*3600  # horizon de temps
temps = np.arange(0, T - 24*3600, 3600)
n     = len(temps)


# --- Figure à  deux sous-graphiques ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3), dpi=300, sharex=True)

# --- Conso en haut ---
ax1.plot(temps/3600/24, sidelec_conso[0:n]/1000, color='black', linewidth=1)
ax1.set_ylabel(r"Power [kW]", fontsize=16)
ax1.set_title("Load profile", fontsize=16)
ax1.set_ylim(sidelec_conso.min() * 0.9/1000, sidelec_conso.max() * 1.03/1000)
ax1.tick_params(labelsize=16)
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# --- PV en bas ---
ax2.plot(temps/3600/24, sidelec_PV[0:n]/1000, color='black', linewidth=1)
ax2.set_xlabel(r"Days", fontsize=16)
ax2.set_ylabel(r"Power [kW]", fontsize=16)
ax2.set_title("PV production profile", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# plt.rc('font', weight='bold')
# plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

sidelec_conso = sidelec_conso[0:n]
sidelec_PV    = sidelec_PV[0:n]
# ===============================
# Paramètres du zoom
# ===============================
t0, t1 = 125, 132  # jours
time_days = temps / 3600 / 24
mask = (time_days >= t0) & (time_days <= t1)

# Calcul préalable des limites Y pour les zones de zoom
ymin_conso = sidelec_conso[mask].min()/1000
ymax_conso = sidelec_conso[mask].max()/1000
ymin_pv = sidelec_PV[mask].min()/1000
ymax_pv = sidelec_PV[mask].max()/1000

# ===============================
# Fonction utilitaire pour dessiner les connexions personnalisées
# ===============================
def connecter_zoom_au_rectangle(ax_parent, axi, zoom_xlim, zoom_ylim, rect_coords):
    t0_z, t1_z = zoom_xlim
    y0_z, y1_z = zoom_ylim
    rx, ry, rw, rh = rect_coords 

    # 1. FOND BLANC : sur l'axe parent (zorder 3)
    rect_fill = patches.Rectangle(
        (rx, ry), rw, rh, transform=ax_parent.transAxes,
        facecolor='white', edgecolor='none', zorder=3
    )
    ax_parent.add_patch(rect_fill)

    # 2. BORDURE ROUGE : ajoutée à l'ENCART (axi) pour être au-dessus de tout
    # On utilise quand même le transform du parent pour le placement
    rect_border = patches.Rectangle(
        (rx, ry), rw, rh, transform=ax_parent.transAxes,
        facecolor='none', edgecolor='red', linewidth=2, 
        zorder=100, clip_on=False  # clip_on=False permet de dessiner hors des limites de l'encart
    )
    axi.add_patch(rect_border)

    # 3. Cadre de sélection sur le plot principal
    rect_sel = patches.Rectangle(
        (t0_z, y0_z), (t1_z - t0_z), (y1_z - y0_z),
        linewidth=2, edgecolor='red', facecolor='none', zorder=5
    )
    ax_parent.add_patch(rect_sel)

    # 4. Lignes de connexion TL et BR
    con_kw = dict(axesA=ax_parent, axesB=ax_parent, color="red", lw=1.5, zorder=5)
    ax_parent.add_artist(ConnectionPatch(
        xyA=(t0_z, y1_z), coordsA=ax_parent.transData,
        xyB=(rx, ry + rh), coordsB=ax_parent.transAxes, **con_kw
    ))
    ax_parent.add_artist(ConnectionPatch(
        xyA=(t1_z, y0_z), coordsA=ax_parent.transData,
        xyB=(rx + rw, ry), coordsB=ax_parent.transAxes, **con_kw
    ))
# ===============================
# Création des Encarts et application du style
# ===============================
# Vos coordonnées de rectangle blanc validées
rect_coords_final1 = (0.48, 0.41, 0.486, 0.5)
rect_coords_final2 = (0.47, 0.41, 0.496, 0.5)


# --- Encart Conso ---
axins = inset_axes(ax1, width="45%", height="40%", loc="upper right", borderpad=1)
axins.plot(time_days[mask], sidelec_conso[mask]/1000, color='black', linewidth=1)
axins.set_xlim(t0, t1)
axins.set_ylim(ymin_conso, ymax_conso)

# Appel de la fonction magique pour ax1
connecter_zoom_au_rectangle(ax1, axins, (t0, t1), (ymin_conso, ymax_conso), rect_coords_final1)

# --- Encart PV ---
axins_pv = inset_axes(ax2, width="45%", height="40%", loc='upper right', borderpad=1)
axins_pv.plot(time_days[mask], sidelec_PV[mask]/1000, color='black', linewidth=1)
axins_pv.set_xlim(t0, t1)
axins_pv.set_ylim(ymin_pv, ymax_pv)

# Appel de la fonction magique pour ax2
connecter_zoom_au_rectangle(ax2, axins_pv, (t0, t1), (ymin_pv, ymax_pv), rect_coords_final2)

# --- Style commun des encarts ---
for axi in [axins, axins_pv]:
    axi.set_facecolor('white')
    axi.set_zorder(4)
    
    # Grille sur l'encart
    axi.grid(True, linestyle='--', alpha=0.6, linewidth=0.5, zorder=0)

    # Ticks : Plus petits et en rouge
    axi.tick_params(axis='both', which='major', labelsize=10, colors='black')
        
    plt.draw() # Nécessaire pour générer les labels avant de les modifier
    xticks = axi.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)
        
    # for label in axi.get_xticklabels() + axi.get_yticklabels():
    #     label.set_fontweight('bold')
    #     label.set_zorder(5)

    axi.set_xlabel("")
    axi.set_ylabel("")
    
    # Bordures de l'encart (Spines)
    # for spine in axi.spines.values():
    #     spine.set_edgecolor('red')
    #     spine.set_linewidth(1.5)
    #     spine.set_zorder(5)

# --- Mise en page et sauvegarde ---
plt.tight_layout(pad=0.4)
plt.savefig("sidelec_conso_PV_encart_propre.pdf", bbox_inches='tight', pad_inches=0.02)
plt.show()