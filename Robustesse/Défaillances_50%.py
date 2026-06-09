import matplotlib.pyplot as plt
import numpy as np

# Valeurs 1D
values = [17.84    ,30.87  ,8.59   ,3.28   ,14.83     ,32.85    ,13.52     ,23.18    ,8.81    ,7.64 ,43.96]
labels = ['ECMS_x1','25-75','50-50','75-25','Econ_bat','Econ_h2','Full_bat','Full_h2','SOC0.6','RB1','RL1']

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
ax.set_xlabel("Défaillance 50%", fontsize=12)
ax.set_title("Distribution des performances moyen-terme des EMS", fontsize=13)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# Encadrer un peu plus l’axe
ax.set_xlim(min(values) - 5, max(values) + 5)

plt.tight_layout()
plt.show()
