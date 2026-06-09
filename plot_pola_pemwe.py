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

j_in = 0.0051             # (A/cm2) Internal current (Suyao model)
A    =  0.6e-4             # Tafel's constant
B    = -1.5e-4             # Concentration drop constant
S    = 220                 # (cm2) Electrode surface (Kong et al. + Bressel)

ELY = {
    'R': 0.001,              # (Ohm) Kong et al.
    'eff': 0.65,
    'n_series': 10,
    'n_parallel': 1,
    'T': 273 + 60,
    'E_0': 1.23,
    'CAPEX':2500,            #€ / kW
    'CAPEX_stack': 563,      #€/kW
    'SoH_EoL' : 0.9,         # 20% de perte de tension
    'j_0': 1e-4,
    'j_L': 10/3         # [A/cm2] cohérent avec le 1 A/cm2 = 30% Pmax
}

def calculate_vst_we(j_val):
    """Calcule la tension du stack électrolyseur (V = E0 + Pertes)"""
    # i/np*S correspond à j_val
    term_act = A * ELY['T'] * np.log((j_val + j_in) / ELY['j_0'])
    term_ohm = ELY['R'] * (j_val * S)  # Simplifié selon votre eq
    
    # Pour l'électrolyseur, la perte de concentration fait MONTER la tension.
    # ln(1 - j/jL) est négatif, donc on soustrait le terme pour augmenter V.
    ratio_conc = 1 - (j_val / ELY['j_L'])
    ratio_conc = np.maximum(ratio_conc, 1e-10)
    term_conc = - B * ELY['T'] * np.log(ratio_conc)
    
    # Equation PEMWE : V = n_s * (E0 + Ohmique + Activation + Concentration)
    # Note : - term_conc car ln(1-x) est négatif
    return ELY['n_series'] * (ELY['E_0'] + term_ohm + term_act - term_conc)

# Génération des données
j_range = np.linspace(0.0001, ELY['j_L'] - 0.05, 500)
v_stack = calculate_vst_we(j_range)

# --- Création de la figure ---
fig, ax = plt.subplots(figsize=(10, 6))

# Tracé de la courbe (Noir profond, très épais)
ax.plot(j_range, v_stack, color='black', linewidth=4, zorder=5)

# --- Zones de Pertes (Adaptées à la forme PEMWE) ---
# Activation (Début de courbe, montée rapide)
ax.axvspan(0, 0.6, color='#FF9999', alpha=0.4, lw=0)
# Ohmique (Zone linéaire centrale)
ax.axvspan(0.6, 2.6, color='#FFFF99', alpha=0.4, lw=0)
# Concentration (Fin de courbe, montée asymptotique)
ax.axvspan(2.6, ELY['j_L'], color='#99CCFF', alpha=0.4, lw=0)

# --- Ajout des Textes avec Halo ---
def add_impact_text(x, y, text, color):
    txt = ax.text(x, y, text, color=color, fontsize=14, fontweight='bold', 
                  ha='center', va='center', zorder=6)
    txt.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

# Positionnement adapté à une courbe ascendante
add_impact_text(0.3, 16, "Surtension \n d'activation", '#990000')
add_impact_text(1.5, 20, "Surtension ohmique", '#998800')
add_impact_text(3.0, 24.5, "Surtension de \nconcentration", '#004488')

# --- Mise en forme des axes ---
ax.set_xlabel(r'$\mathbf{Densité\ de\ courant\ \mathit{|j|}\ [A/cm^2]}$', fontsize=22, labelpad=10)
ax.set_ylabel(r'$\mathbf{Tension\ \mathit{V_{st}}\ [V]}$', fontsize=22, labelpad=10)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True, which='both', linestyle=':', linewidth=1, alpha=0.7, zorder=0)

# Limites (ajustées pour ns=12 et E0=1.48)
ax.set_xlim(0, ELY['j_L'])
ax.set_ylim(12, 26) 

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# --- Sauvegarde ---
plt.savefig('PEMWE_Polarization_Curve.pdf', format='pdf', bbox_inches='tight')
plt.show()