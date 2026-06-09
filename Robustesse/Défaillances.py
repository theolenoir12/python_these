import matplotlib.pyplot as plt
import numpy as np

# Valeurs 1D
values = [95.38, 108.9, 99.27, 99.27, 99.27, 105.06, 99.27, 99.97, 107.67, 128.84,]
labels = ['RL1','ECMS_x1','25-75','50-50','75-25','Écon_bat','Full-bat', 'RB1',
          'SOC0.6','Full_h2']

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 2.5))

# Affichage des points sur un axe horizontal
y = np.zeros_like(values)
ax.scatter(values, y, color='darkgreen', s=60, alpha=0.8)

# Ajout des labels au-dessus des points
for i, label in enumerate(labels):
    if i%6 == 0:
        ax.text(values[i], 0.045, label, ha='center', fontsize=9)
    elif i%6 == 1 : 
        ax.text(values[i], -0.05, label, ha='center', fontsize=9)
    elif i%6 == 2 :
        ax.text(values[i], 0.03, label, ha='center', fontsize=9)
    elif i%6 == 3 :
        ax.text(values[i], 0.015, label, ha='center', fontsize=9)
    elif i%6 == 4 :
        ax.text(values[i], -0.015, label, ha='center', fontsize=9)
    else : 
        ax.text(values[i], -0.03, label, ha='center', fontsize=9)

# Mise en forme
ax.set_yticks([])  # Supprimer l'axe Y
ax.set_xlabel("Défaillance", fontsize=12)
ax.set_title("Distribution des performances moyen-terme des EMS", fontsize=13)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# Encadrer un peu plus l’axe
ax.set_xlim(min(values) - 5, max(values) + 5)

plt.tight_layout()
plt.show()
