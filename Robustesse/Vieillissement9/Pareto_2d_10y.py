import numpy as np
import matplotlib.pyplot as plt

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

# Données bleues (points d'origine)
points = np.array([
    [6.4580, 139.0737],   # 0-100
    [24.2427, 117.3058],  # 25-75
    [10.6781, 91.7129],   # 50-50
    [4.6191, 42.0234],    # 75-25
    [4.5041, 42.2478],    # 100-0
    [1.4011, 29.4214],    # RB2
    [1.5942, 24.1781],    # RB2(SoH)
    [1.6869, 61.9176],    # RB1
    [0.7930, 147.7755],   # SoC1
    [34.6110, 116.8908],  # SoC06
   # [1.8027, 24.7250],    # RB2(RUL)
    [0.0000, 0.0000]      # Ideal
])


labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 
          'RB2(SoH)', 'RB1', 'SoC1', 'SoC06','Ideal']

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))

# Affichage des points bleus
ax.scatter(points[:, 0], points[:, 1], color='royalblue', s=60, alpha=0.8)

# Ajout des labels avec conditions spécifiques
for i, label in enumerate(labels):
    if label == 'RB2(RUL)':
        ax.text(points[i, 0] - 0.2, points[i, 1] - 0.4, label, 
                fontsize=14, color='black', verticalalignment='top')
    
    # elif label == 'RB2':
    #     ax.text(points[i, 0] - 1.5, points[i, 1] + 2.5, label, 
    #             fontsize=14, color='black', verticalalignment='top')  
        
    elif label == 'SoC06':
        ax.text(points[i, 0] - 3, points[i, 1] + 3.5, label, 
                fontsize=14, color='black', verticalalignment='top')         
    elif label == 'SoC1':
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.3, label, 
                fontsize=14, color='black', verticalalignment='top')  
    elif label == '75-25':
        ax.text(points[i, 0] +0.5, points[i, 1] +0.4, label, 
                fontsize=14, color='black', verticalalignment='top')    
        
    elif label == 'RB2(SoH)':
        # MODIFICATION ICI : on soustrait au lieu d'ajouter pour descendre le label
        # Ajuste le -0.8 selon tes préférences de distance
        ax.text(points[i, 0]+0.5, points[i, 1] - 0.5, label, 
                fontsize=14, color='black', verticalalignment='top', horizontalalignment='center')  
    
    else:
        # Placement standard pour les autres
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, 
                fontsize=14, color='black')

# Style de l'axe
ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Degradation [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("pareto_ems.pdf", format='pdf', bbox_inches='tight')
plt.show()