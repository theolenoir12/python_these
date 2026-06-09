import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm

# --- 1. Configuration Style Scientifique ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 22,
    "axes.labelsize": 28,            
    "axes.titlesize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.linewidth": 2.5,           
    "mathtext.fontset": "cm",        
})

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def simulate_ica(v, age_factor):
    health = 1 - (0.35 * age_factor)
    shift = 0.018 * age_factor
    y =  gaussian(v, 130 * health, 3.28 + shift, 0.012)
    y += gaussian(v, 120 * health, 3.32 + shift, 0.012)
    y += gaussian(v, 780 * health, 3.36 + shift, 0.010)
    y += gaussian(v, 560 * (health**1.4), 3.395 + shift, 0.015)
    return y + 10

# --- 2. Données ---
v = np.linspace(3.2, 3.6, 1000)
num_curves = 10 
cmap = cm.jet

fig, ax = plt.subplots(figsize=(15, 8))

for i in range(num_curves):
    prog = i / (num_curves - 1)
    ax.plot(v, simulate_ica(v, prog), color=cmap(prog), lw=3.5, alpha=0.85)

##Color bar
cax = ax.inset_axes([0.55, 0.88, 0.4, 0.04]) 

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])

# On lie la colorbar à cet axe spécifique 'cax'
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

# Formatage
cbar.set_ticks([0.05, 0.95])
cbar.set_ticklabels(['New', 'Aged'], fontweight='bold', fontsize=22)
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(length=0)

# --- Ligne de sauvegarde ultra-robuste ---

# --- 4. Mise en forme finale ---
ax.set_xlabel(r"Voltage [V]", fontweight='bold', labelpad=10)
ax.set_ylabel(r"d$Q$/d$V$ [Ah$\cdot$V$^{-1}$]", fontweight='bold', labelpad=10)

ax.set_xlim(3.2, 3.55)
ax.set_ylim(0, 900)

# Ticks académiques
ax.tick_params(which='major', direction='in', length=10, width=2.5, top=True, right=True)
ax.tick_params(which='minor', direction='in', length=5, width=1.5, top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.tight_layout()
plt.draw()
plt.savefig('ic_curve.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()