import numpy as np
import matplotlib.pyplot as plt

# Données bleues (points d'origine)
points = np.array([
    [4.08, 600],
    [6.37, 538],
    [14.6, 520],
    [8.5, 548],
    [24,477],
    [11.60,339],
    [17.14,331],
    [0, 667],
    [29.47, 461],
    [7.8,   474],
    [5.06,  503],
    [8.32,  443],
    [41.88, 497],
    [4.79,  465],
    [21.80, 468],
    [2.76,  472],
    [3.11,  444],
    [17.31, 177],
    [0,     194],
    [0,     0]
])
labels = ['MPCo_x05_h12','MPCo_x05_h4','MPC_x05_h8','MPC_x05_h4','RL1','ECMS_x_0_5','ECMS_x_0','ECMS_x_1','25-75','50-50','75-25','Écon_bat','Écon_H2','Full_bat','Full_h2',
          'SOC0.6','RB1', 'DP_x_0', 'DP_x_0_5', 'Ideal']

points_ECMS = np.array([
    [11.60,339],
    [17.14,331],
    [0    ,667]
])

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))

# Affichage des points bleus
ax.scatter(points[:, 0], points[:, 1], color='royalblue', s=60, alpha=0.8)


# Ajout des labels bleus
for i, label in enumerate(labels):
    ax.text(points[i, 0] + 0.05, points[i, 1] + 5, label,
            fontsize=10, color='black')


# Style de l'axe
ax.set_title("Visualisation 2D des points", fontsize=14)
ax.set_xlabel("Désalignement", fontsize=12)
ax.set_ylabel("Dégradations", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
