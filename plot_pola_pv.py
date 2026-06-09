import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration pour publication scientifique
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',          # Police serif classique pour les articles
    'font.size': 12,                 # Taille de police lisible
    'axes.linewidth': 1.5,           # Épaisseur du cadre
    'xtick.direction': 'in',         # Ticks vers l'intérieur (typique en physique)
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.top': True,               # Ticks en haut et à droite
    'ytick.right': True,
    'mathtext.fontset': 'cm'         # Police mathématique type LaTeX
})

# ==========================================
# 2. Génération des données simulées
# ==========================================
G_levels = [1000, 800, 600, 400, 200]  # Irradiance en W/m^2
V = np.linspace(0, 22.5, 1000)         # Axe des tensions

# Paramètres de référence du panneau
I_sc_ref = 5.0  # Courant de court-circuit à 1000 W/m²
V_oc_ref = 22.0 # Tension en circuit ouvert à 1000 W/m²
B = 0.5         # Paramètre de courbure (facteur d'idéalité ajusté)

donnees = {}

for G in G_levels:
    # Le courant est proportionnel à l'irradiance
    I_sc = I_sc_ref * (G / 1000)
    # La tension diminue de façon logarithmique avec l'irradiance
    V_oc = V_oc_ref + 1.5 * np.log(G / 1000) if G > 0 else 0
    
    # Modèle analytique simplifié du courant
    I = I_sc * (1 - np.exp(B * (V - V_oc)))
    I[I < 0] = 0  # Pas de courant négatif dans ce quadrant
    
    P = I * V     # Puissance
    
    # Recherche du Point de Puissance Maximale (MPP)
    idx_mpp = np.argmax(P)
    
    donnees[G] = {
        'V': V,
        'I': I,
        'P': P,
        'mpp': (V[idx_mpp], I[idx_mpp], P[idx_mpp])
    }

# ==========================================
# 3. Création de la figure
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'hspace': 0.3})

# --- Graphique du haut : Courant-Tension (I-V) ---
for G in G_levels:
    d = donnees[G]
    ax1.plot(d['V'], d['I'], color='black', linewidth=2)
    
    # Point MPP
    ax1.plot(d['mpp'][0], d['mpp'][1], 'ko', markersize=6)
    
    # Labels d'irradiance (placés manuellement pour éviter les superpositions)
    y_offset = 0.3
    if G == 1000: y_offset = 0.0
    if G == 800: y_offset = 0.1
    if G == 600: y_offset = 0.2
    if G == 400: y_offset = 0.3
    if G == 200: y_offset = 0.35

    ax1.text(8, d['mpp'][1] - y_offset, f'{G} W/m$^2$', fontsize=14, ha='center')

# Annotation du MPP sur la courbe 1000 W/m² (I-V)
mpp_1000_v, mpp_1000_i, _ = donnees[1000]['mpp']
ax1.annotate('MPP', xy=(mpp_1000_v, mpp_1000_i), xytext=(20.5, 4.8),
             arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.05),
             fontsize=14, ha='left', va='center')

ax1.set_xlim(0, 22.5)
ax1.set_ylim(0, 5.5)
ax1.set_xlabel(r'$V_{PC}$ [V]', fontsize=14)
ax1.set_ylabel(r'$I_{PV}$ [A]', fontsize=14)
ax1.set_yticks(np.arange(0, 6, 1))

# --- Graphique du bas : Puissance-Tension (P-V) ---
for G in G_levels:
    d = donnees[G]
    ax2.plot(d['V'], d['P'], color='black', linewidth=2)
    
    # Point MPP
    ax2.plot(d['mpp'][0], d['mpp'][2], 'ko', markersize=6)

# Labels d'irradiance inclinés (comme sur l'image d'origine)
ax2.text(9, 43, '1000 W/m$^2$', fontsize=14, rotation=22, ha='center')
ax2.text(12.5, 12.7, '200 W/m$^2$', fontsize=14, rotation=6, ha='center')

# Annotation du MPP sur la courbe 1000 W/m² (P-V)
mpp_1000_p = donnees[1000]['mpp'][2]
ax2.annotate('MPP', xy=(mpp_1000_v, mpp_1000_p), xytext=(15, 83),
             arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.05),
             fontsize=14, ha='right', va='center')

ax2.set_xlim(0, 22.5)
ax2.set_ylim(0, 90)
ax2.set_xlabel(r'$V_{PV}$ [V]', fontsize=14)
ax2.set_ylabel(r'$P_{PV}$ [W]', fontsize=14)
ax2.set_yticks(np.arange(0, 86, 17)) # Ticks spécifiques à l'image : 0, 17, 34...

# ==========================================
# 4. Finalisation et sauvegarde
# ==========================================
# Ajustement des marges pour ne rien couper
plt.tight_layout()

# Optionnel : Sauvegarder en haute résolution pour publication (format vectoriel PDF ou EPS)
plt.savefig('pola-pv_publication.pdf', format='pdf', bbox_inches='tight', dpi=300)

plt.show()