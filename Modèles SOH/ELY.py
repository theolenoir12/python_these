from Init_EMR_MG_v16_python import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def voltage(alpha_ely,i_ely) : 
    voltage = ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely / ELY['n_parallel'] 
              + A * ELY['T'] * np.log((i_ely / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
              + B * ELY['T'] * np.log(1 - i_ely / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))
    
    return voltage

N = 10000
M = 5

alpha_ely     = np.linspace(0, 0.226330713, M)
P_ely_max     = np.zeros(M)
i_ely_at_Pmax = np.zeros(M)


# ---- Calculs ----
u_ely_all = []
P_ely_all = []
i_ely_all = [np.linspace(0, ELY['n_parallel']*0.999*S*ELY['j_L']*(1-alpha_ely[k]), N) for k in range(M)] 

for i in range(len(alpha_ely)):
    u_ely = []
    P_ely = []
    i_valid = []
    i_ely = i_ely_all[i]

    for j in range(len(i_ely)):
        
        val = voltage(alpha_ely[i], i_ely[j])
        if not (np.isnan(val) or np.isinf(val)):  # garder uniquement valeurs valides
            u_ely.append(val)
            P_ely.append(val * i_ely[j])
            i_valid.append(i_ely[j])

    # conversion en array numpy
    u_ely = np.array(u_ely)
    P_ely = np.array(P_ely)
    i_valid = np.array(i_valid)

    u_ely_all.append(u_ely)
    P_ely_all.append(P_ely)
    i_ely_all.append(i_valid)

    # max sur les valeurs valides seulement
    idx_max = np.argmax(P_ely)
    P_ely_max[i] = P_ely[idx_max]
    i_ely_at_Pmax[i] = i_valid[idx_max]

i_ely_nom = 0.75 * i_ely_at_Pmax[0]  # courant de référence fixé au BoL


# ---- Figure 1 : P_ely(i_ely) pour chaque alpha ----
plt.figure(figsize=(8,6))
for i, a in enumerate(alpha_ely):
    plt.plot(i_ely_all[i], P_ely_all[i], label=f"α={a:.3f}")
plt.xlabel("i_ely [A]")
plt.ylabel("P_ely [W]")
plt.axvline(x=i_ely_nom,color='k',linestyle=':',lw=1.5)
plt.annotate(rf'$i_{{nom,ref}}$', xy=(i_ely_nom+5, 45),fontsize=24)
plt.title("Puissance en fonction du courant")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("P_ely_vs_i_ely.png", dpi=300)

# ---- Figure 2 : u_ely(i_ely) pour chaque alpha ----
plt.figure(figsize=(8,6))
for i, a in enumerate(alpha_ely):
    plt.plot(i_ely_all[i], u_ely_all[i], label=f"α={a:.3f}")
plt.xlabel("i_ely [A]")
plt.ylabel("u_ely [V]")
plt.axvline(x=i_ely_nom,color='k',linestyle=':',lw=1.5)
plt.annotate(rf'$i_{{nom,ref}}$', xy=(i_ely_nom+5, 45),fontsize=24)
plt.title("Tension en fonction du courant")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("u_ely_vs_i_ely.png", dpi=300)

# ---- Figure 3 : P_ely_max(alpha_ely) ----
plt.figure(figsize=(8,6))
plt.plot(alpha_ely, P_ely_max, "o-", label="Simulation")

# Régression linéaire
reg = LinearRegression().fit(alpha_ely.reshape(-1,1), P_ely_max)
slope = reg.coef_[0]
intercept = reg.intercept_

# Courbe ajustée
P_ely_fit = reg.predict(alpha_ely.reshape(-1,1))
plt.plot(alpha_ely, P_ely_fit, "--", label=f"Fit: P_max = {slope:.2f}*α + {intercept:.2f}")

plt.xlabel("α_ely")
plt.ylabel("P_ely_max [W]")
plt.title("Puissance max en fonction de α")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("P_ely_max_vs_alpha.png", dpi=300)

# ---- Figure 4 : i_ely_at_Pmax(alpha_ely) ----
plt.figure(figsize=(8,6))
plt.plot(alpha_ely, i_ely_at_Pmax, "o-", label="Simulation")

reg_I = LinearRegression().fit(alpha_ely.reshape(-1,1), i_ely_at_Pmax)
slope_I, intercept_I = reg_I.coef_[0], reg_I.intercept_
i_ely_fit = reg_I.predict(alpha_ely.reshape(-1,1))

plt.plot(alpha_ely, i_ely_fit, "--", label=f"Fit: i* = {slope_I:.2f}*α + {intercept_I:.2f}")
plt.xlabel("α_ely")
plt.ylabel("i_ely* [A]")
plt.title("Courant de puissance max en fonction de α")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("i_ely_opt_vs_alpha.png", dpi=300)

# ---- Impression de l'équation ----
print(f"Équation de la droite : P_ely_max = {slope:.4f} * alpha_ely + {intercept:.4f} [W]")
print(f"Équation de la droite (i_ely*): i_ely* = {slope_I:.4f} * alpha_ely + {intercept_I:.4f} [A]")

# ---- SoH basé sur la tension au courant nominal (75% de i* à BoL) ----
# Pour l'électrolyseur, la tension MONTE avec le vieillissement :
# SoH = V(i_nom, alpha=0) / V(i_nom, alpha)
i_ely_nom = 0.75 * i_ely_at_Pmax[0]  # courant de référence fixé au BoL
M = 1000
alpha_ely     = np.linspace(0, 0.226330713, M)

SoH_ely = np.array([
    voltage(0.0, i_ely_nom) / voltage(alpha_ely[k], i_ely_nom)
    for k in range(M)
])

# ---- Alpha EoL par résolution non-linéaire (brentq) ----
from scipy.optimize import brentq

####################################################################################################################
SoH_EoL_ely = 0.85
####################################################################################################################

V_bol_ely   = voltage(0.0, i_ely_nom)

def residual_ely(alpha):
    return V_bol_ely / voltage(alpha, i_ely_nom) - SoH_EoL_ely

# Pour l'ELY : tension monte avec alpha, donc SoH décroît → résiduel change de signe
alpha_max_search = 0.248679
if residual_ely(alpha_max_search) > 0:
    print("Attention : SoH_EoL non atteint dans [0, 1[. Augmenter la plage d'alpha.")
else:
    alpha_EoL_ely = brentq(residual_ely, 0.0, alpha_max_search, xtol=1e-10, rtol=1e-10)
    print(f"Alpha EoL (SoH = {SoH_EoL_ely}) : alpha_ely_EoL = {alpha_EoL_ely:.6f}")
    print(f"Vérification : SoH(alpha_EoL) = {V_bol_ely / voltage(alpha_EoL_ely, i_ely_nom):.6f}")

#########################################################################################################

from scipy.optimize import curve_fit

SoH_center = 0.85  # point de centrage

def model_poly3(x, a, b, c, d):
    s = x - SoH_center
    return a * s**3 + b * s**2 + c * s + d

def residual_ely(alpha, SoH):
    return V_bol_ely / voltage(alpha, i_ely_nom) - SoH

SoH_ely   = np.linspace(SoH_EoL_ely, 1, M)
P_ely_max = np.zeros(M)
for i in range(M):
    alpha_ely = brentq(residual_ely, 0.0, 0.248679, args=(SoH_ely[i],), xtol=1e-10, rtol=1e-10)
    i_ely_max = (-732.6 * alpha_ely + 732.6)
    P_ely_max[i] = i_ely_max * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely) * i_ely_max / ELY['n_parallel'] 
                                               + A * ELY['T'] * np.log((i_ely_max / S / ELY['n_parallel'] + j_in) / ELY['j_0'])
                                               + B * ELY['T'] * np.log(1 - i_ely_max / S / ELY['n_parallel'] / ELY['j_L'] / (1 - alpha_ely)))

# Ajustement sur la plage cible [SoH_EoL_ely, 1] uniquement
mask_fit = (SoH_ely >= SoH_EoL_ely) & (SoH_ely <= 1.0)
popt, _ = curve_fit(model_poly3, SoH_ely[mask_fit], P_ely_max[mask_fit],
                    p0=[1, 1, 1, P_ely_max[mask_fit].mean()],
                    maxfev=10000)

soh_fine    = np.linspace(SoH_EoL_ely, 1.0, M)
p_max_fit   = model_poly3(soh_fine, *popt)

# ---- Figure 6 : P_ely_max vs SoH ----
plt.figure(figsize=(8, 6))
plt.plot(SoH_ely, P_ely_max, "s", label="Simulation", color='tab:red', markersize=4)
plt.plot(soh_fine, p_max_fit, "--", color='black', label="Fit")
plt.xlabel(r"$SoH_{FC}$")
plt.ylabel(r"$P_{ely,max}$ [W]")
plt.title("Puissance maximale vs SoH (Modèle Polynôme degré 3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Pmax_vs_SoH_perfect_fit_ely.png", dpi=300)

# ---- Print de l'équation ----
print(f"\nÉquation optimisée (s = SoH - {SoH_center}) :")
print(f"P_ely_max = {popt[0]:.4f}*SoH³ + {popt[1]:.4f}*SoH² + {popt[2]:.4f}*SoH + {popt[3]:.4f}")

# ---- Performance sur [SoH_EoL_ely, 1] ----
y_pred    = model_poly3(SoH_ely[mask_fit], *popt)
residuals = P_ely_max[mask_fit] - y_pred
rmse      = np.sqrt(np.mean(residuals**2))
r2        = 1 - np.sum(residuals**2) / np.sum((P_ely_max[mask_fit] - P_ely_max[mask_fit].mean())**2)
print(f"\n--- Performance du Fit sur la plage [SoH_EoL_ely, 1] ---")
print(f"R²   : {r2:.8f}")
print(f"RMSE : {rmse:.4f} W")
print(f"Erreur max : {np.max(np.abs(residuals)):.4f} W")

plt.show()

np.savetxt('resultats_ely.csv', 
           np.column_stack((SoH_ely, P_ely_max)), 
           delimiter=';', 
           header='SoH_ely;P_ely_max', 
           comments='')

# ==============================================================================
# ---- Figure 7 : Rendement η(P_ely) pour chaque alpha, via tension thermoneutre ----
# ==============================================================================
#
# Fondement physique (électrolyseur — logique INVERSE de la pile) :
#   La tension thermoneutre V_th = ΔH / (n·F) = 285800 / (2 × 96485) ≈ 1.481 V/cellule
#   est ici la tension MINIMALE théorique pour décomposer H2O (base enthalpique, LHV eau liq.)
#   La tension réelle V_cell > V_th à cause des surtensions (activation, ohmique, diffusion).
#   Le rendement η = V_th / V_cell mesure la fraction d'énergie électrique
#   effectivement stockée dans les liaisons chimiques du H2. Le reste (V_cell - V_th) / V_cell
#   part en chaleur (effet Joule + irréversibilités).
#   Cohérence avec la production H2 :
#       P_H2_produit = η × P_ely   =>   plus V_cell est élevée, moins on produit de H2 par Watt
#   Le vieillissement augmente V_cell → η baisse → même puissance électrique donne moins de H2.

V_th = 285800 / (2 * 96485)  # Tension thermoneutre par cellule [V] ≈ 1.481 V

N_eta = 10000
M_eta = 5
alpha_eta = np.linspace(0, 0.226330713, M_eta)

plt.figure(figsize=(8, 6))

for k, a in enumerate(alpha_eta):
    i_range = np.linspace(1e-3, ELY['n_parallel'] * 732.6 * (1 - a), N_eta)

    eta_list = []
    P_list   = []

    for i_val in i_range:
        V_stack = voltage(a, i_val)
        if np.isnan(V_stack) or np.isinf(V_stack) or V_stack <= 0:
            continue
        V_cell = V_stack / ELY['n_series']
        if V_cell <= 0:
            continue
        # Rendement thermodynamique (LHV) — inversé par rapport à la FC
        eta    = V_th / V_cell
        P_elec = V_stack * i_val
        eta_list.append(eta)
        P_list.append(P_elec)

    eta_arr  = np.array(eta_list)
    P_arr    = np.array(P_list)
    sort_idx = np.argsort(P_arr)
    plt.plot(P_arr[sort_idx] / 1000, eta_arr[sort_idx] * 100, label=f"α={a:.3f}")

# Marquer la puissance nominale BoL
V_nom_BoL   = voltage(0.0, i_ely_nom)
P_nom_BoL   = V_nom_BoL * i_ely_nom
eta_nom_BoL = V_th / (V_nom_BoL / ELY['n_series']) * 100
plt.axvline(x=P_nom_BoL / 1000, color='k', linestyle=':', lw=1.5)
plt.annotate(
    rf"$P_{{nom,BoL}}$" + f"\nη={eta_nom_BoL:.1f}%",
    xy=(P_nom_BoL / 1000, eta_nom_BoL),
    xytext=(P_nom_BoL / 1000 + 0.05, eta_nom_BoL + 1),
    fontsize=13,
    arrowprops=dict(arrowstyle="->", lw=1.2)
)

plt.xlabel(r"$P_{ely}$ [kW]")
plt.ylabel(r"$\eta_{ely}$ [%]  (base LHV)")
plt.title("Rendement de l'électrolyseur\n"
          + r"$\eta = V_{th} \,/\, V_{cell}$,  $V_{th} = \Delta H / nF \approx 1.481$ V")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("eta_ely_vs_P_ely.png", dpi=300)

