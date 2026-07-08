import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration du style scientifique ---
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.7,
    "grid.linestyle": "--"
})
    

def create_plot(file_path, output_name):
    # Chargement des données
    data = pd.read_csv(file_path, header=None)
    
    # Calcul des axes
    # x = (Power / P_max) * 100
    x = (data[0] / data[0].max()) * 100
    y = data[1]

    # Création de la figure
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Plot : Trait bleu unique, sans légende
    ax.plot(x, y, color='#0072BD', linestyle='-')

    # Labels en anglais
    ax.set_xlabel('Power / $P_{\mathrm{max}}$ [%]')
    ax.set_ylabel('Efficiency [%]')
    
    # Nettoyage et limites
    ax.grid(True)
    ax.set_xlim(0, 100)
    
    # Ajustement automatique des marges
    plt.tight_layout()

    # Sauvegarde en format vectoriel (PDF) et PNG
    plt.savefig(f'{output_name}.pdf')
    plt.show()

# --- Génération des deux figures ---
create_plot('ELY_efficiency_LU_table_power.csv', 'efficiency_ely')
create_plot('FC_efficiency_LU_table_power.csv', 'efficiency_fc')

print("Figures generated successfully: efficiency_ely.pdf and efficiency_fc.pdf")