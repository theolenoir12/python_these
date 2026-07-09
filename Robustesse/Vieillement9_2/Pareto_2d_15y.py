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
    [10.4291, 72.2274],   # 0-100
    [20.2484, 64.1032],  # 25-75
    [8.0822, 63.1622],  # 50-50
    [3.8271, 32.4428],    # 75-25
    [2.4565, 38.6825],    # 100-0
    [2.6666, 34.6169],    # RB2
    [2.5100, 32.4441],    # RB2(SoH)
    [1.2658, 46.0521],    # RB1
    [1.3529, 82.3372],   # SoC1
    [29.3615, 63.6004],  # SoC06
    # [1.8278, 40.5302],    # RB2(RUL)
    # [16.6714, 176.2612],  # f0.25
    # [21.2558, 195.8383],  # f0.5
    # [33.7325, 197.9101],  # f0.75
    # [29.2080, 124.6891],  # f0.1
    # [24.3900, 28.4552],   # f0.01
    # [28.8937, 50.0139],   # f0.05
    # [23.1259, 26.3755],   # f0.001
    [0.0000, 0.0000]      # Ideal
])


labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 'RB2(SoH)',
           'RB1', 'SoC1', 'SoC06','Ideal']

# labels = ['0-100', '25-75', '50-50', '75-25', '100-0', 'RB2', 
#           'RB2(SoH)', 'RB1', 'SoC1', 'SoC06','RB2(RUL)','f0.25','f0.1','f0.01','f0.001','Ideal']

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))

# Affichage des points bleus
ax.scatter(points[:, 0], points[:, 1], color='royalblue', s=60, alpha=0.8)

# # Ajout des labels avec conditions spécifiques
for i, label in enumerate(labels):
#     if label == 'RB2(RUL)':
#         ax.text(points[i, 0] - 0.2, points[i, 1] - 0.4, label, 
#                 fontsize=14, color='black', verticalalignment='top')
    
#     elif label == 'RB2':
#         ax.text(points[i, 0] - 1.5, points[i, 1] + 11.5, label, 
#                 fontsize=14, color='black', verticalalignment='top')  
        
#     elif label == 'SoC06':
#         ax.text(points[i, 0] - 3, points[i, 1] - 5.5, label, 
#                 fontsize=14, color='black', verticalalignment='top')         
#     elif label == 'SoC1':
#         ax.text(points[i, 0] + 0.5, points[i, 1] + 0.3, label, 
#                 fontsize=14, color='black', verticalalignment='top')  
#     elif label == '75-25':
#         ax.text(points[i, 0] +0.5, points[i, 1] -0.4, label, 
#                 fontsize=14, color='black', verticalalignment='top')    
        
#     elif label == 'RB2(SoH)':
#         # MODIFICATION ICI : on soustrait au lieu d'ajouter pour descendre le label
#         # Ajuste le -0.8 selon tes préférences de distance
#         ax.text(points[i, 0]+0.5, points[i, 1] - 4, label, 
#                 fontsize=14, color='black', verticalalignment='top', horizontalalignment='center')  
    
#     else:
#         # Placement standard pour les autres
        ax.text(points[i, 0] + 0.5, points[i, 1] + 0.5, label, 
                fontsize=14, color='black')

# Style de l'axe
ax.set_xlabel("LPSP [%]", fontsize=18)
ax.set_ylabel("Degradation [k€]", fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("pareto_ems.pdf", format='pdf', bbox_inches='tight')
plt.show()