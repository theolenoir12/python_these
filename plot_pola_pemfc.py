import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# --- Configuration du style Publication Scientifique ---
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

# --- Paramètres Physiques ---
FC = {
    'R': 0.001,                # (Ohm) Kong et al.
    'eff': 0.50,
    'n_series': 10,            # Number of cells in series
    'n_parallel': 1,           # Number of stacks in parallel
    'T': 273 + 60,             # (K) FC temperature
    'E_0': 1.23,               # (V) 57.5/53 from Bressel et al.
    'CAPEX':2500,              # €/kW
    'CAPEX_stack': 0.3*2500,   #€/kW on prend 30% du CAPEX total (qui inclut BoP etc.)
    'SoH_EoL' : 0.9,           # 20% de perte en tension
    'j_0': 1e-3,               # [A/cm2] courant d'échange  
    'j_L': 1.2                 # [A/cm2] courant limite 
}

j_in = 0.0051             # (A/cm2) Internal current (Suyao model)
A    =  0.6e-4             # Tafel's constant
B    = -1.5e-4             # Concentration drop constant
S    = 220                 # (cm2) Electrode surface (Kong et al. + Bressel)

def calculate_vst(j_val):
    term_act = A * FC['T'] * np.log((j_val + j_in) / FC['j_0'])
    term_ohm = FC['R'] * (j_val * S)
    ratio_conc = 1 - (j_val / FC['j_L'])
    ratio_conc = np.maximum(ratio_conc, 1e-10)
    term_conc = - B * FC['T'] * np.log(ratio_conc)
    return FC['n_series'] * (FC['E_0'] - term_ohm - term_act + term_conc)

# Génération des données
j_range = np.linspace(0.0001, FC['j_L'] - 0.01, 500)
v_stack = calculate_vst(j_range)

# --- Création de la figure (Format MATLAB [5 5 20 11] -> ~7.8 x 4.3 inches) ---
fig, ax = plt.subplots(figsize=(10, 6))

# Tracé de la courbe (Noir profond, très épais)
ax.plot(j_range, v_stack, color='black', linewidth=4, zorder=5)

# --- Zones de Pertes (Couleurs plus saturées et contrastées) ---
# Activation
ax.axvspan(0, 0.18, color='#FF9999', alpha=0.4, lw=0, label='Activation')
# Ohmique
ax.axvspan(0.18, 0.9, color='#FFFF99', alpha=0.4, lw=0, label='Ohmic')
# Concentration
ax.axvspan(0.9, FC['j_L'], color='#99CCFF', alpha=0.4, lw=0, label='Concentration')

# --- Ajout des Textes avec Halo et LaTeX ---
def add_impact_text(x, y, text, color):
    # On garde fontsize et fontweight ici, pas besoin de LaTeX
    txt = ax.text(x, y, text, color=color, fontsize=14, fontweight='bold', 
                  ha='center', va='center', zorder=6)
    txt.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

# Utilise simplement \n dans une chaîne de caractères normale
add_impact_text(0.095, 10, "Pertes\nd'activation", '#990000')
add_impact_text(0.5, 8, "Pertes ohmiques", '#998800')
add_impact_text(1.05, 4.5, "Pertes par\n concentration", '#004488')

# --- Mise en forme des axes ---
# Utilisation de la syntaxe LaTeX bold demandée
ax.set_xlabel(r'$\mathbf{Densité \ de\ courant\ \mathit{j}\ [A/cm^2]}$', fontsize=22, labelpad=10)
ax.set_ylabel(r'$\mathbf{Tension\ \mathit{V_{st}}\ [V]}$', fontsize=22, labelpad=10)

ax.tick_params(axis='both', which='major', labelsize=18)

# Grille (plus marquée pour la lecture de données)
ax.grid(True, which='both', linestyle=':', linewidth=1, alpha=0.7, zorder=0)

# Limites et suppression des bordures superflues
ax.set_xlim(0, FC['j_L'])
ax.set_ylim(0, 15) # Zoomé sur la zone utile pour plus de clarté
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# --- Sauvegarde en PDF haute qualité ---
plt.savefig('PEMFC_Polarization_Curve.pdf', format='pdf', bbox_inches='tight')
plt.show()