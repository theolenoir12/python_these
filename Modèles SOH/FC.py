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

def voltage(alpha_fc,i_fc) : 
    voltage = FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc / FC['n_parallel'] 
              - A * FC['T'] * np.log((i_fc / S / FC['n_parallel'] + j_in) / FC['j_0'])
              - B * FC['T'] * np.log(1 - i_fc / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))
    
    return voltage

N = 10000
M = 3

# alpha_fc     = np.linspace(0, 0.32981253, M)
alpha_fc     = np.linspace(0, 0.25, M)
P_fc_max     = np.zeros(M)
i_fc_at_Pmax = np.zeros(M)


# ---- Calculs ----
u_fc_all = []
P_fc_all = []
i_fc_all = [np.linspace(0, FC['n_parallel']*0.999*FC['j_L']*S*(1-alpha_fc[k]), N) for k in range(M)] 

for i in range(len(alpha_fc)):
    u_fc = []
    P_fc = []
    i_valid = []
    i_fc = i_fc_all[i]

    for j in range(len(i_fc)):
        
        val = voltage(alpha_fc[i], i_fc[j])
        if not (np.isnan(val) or np.isinf(val)):  # garder uniquement valeurs valides
            u_fc.append(val)
            P_fc.append(val * i_fc[j])
            i_valid.append(i_fc[j])

    # conversion en array numpy
    u_fc = np.array(u_fc)
    P_fc = np.array(P_fc)
    i_valid = np.array(i_valid)

    u_fc_all.append(u_fc)
    P_fc_all.append(P_fc)
    i_fc_all.append(i_valid)

    # max sur les valeurs valides seulement
    idx_max = np.argmax(P_fc)
    P_fc_max[i] = P_fc[idx_max]
    i_fc_at_Pmax[i] = i_valid[idx_max]

i_fc_nom = 0.75 * i_fc_at_Pmax[0]  # courant de référence fixé au BoL
labels   = ['BoL','MoL','EoL']
# ---- Figure 1 : P_fc(i_fc) pour chaque alpha ----
plt.figure(figsize=(10,6))
[plt.plot(i_fc_all[i],P_fc_all[i],label=labels[i]) for i in range(3)]
plt.xlabel(r'$i_{FC}\ [A]$',fontsize=20);plt.ylabel(r'$P_{FC}\ [W]$',fontsize=20)
plt.axvline(i_fc_nom,color='k',ls=':')
# POINTS ET TEXTES ARBITRAIRES À AJUSTER MANUELLEMENT (ICI BoL=180, EoL=155)
plt.plot(i_fc_nom, 1855, 'o', color='tab:blue') # Point BoL
plt.plot(i_fc_nom, 1615, 'o', color='tab:green') # Point EoL
plt.annotate(r'$P_{FC}(i_{nom,ref}, t=0)$',(110,1855),fontsize=18)
plt.annotate(r'$P_{FC}(i_{nom,ref}, t=t_{EoL})$',(155,1430),fontsize=18)
plt.annotate(rf'$i_{{nom,ref}}$', xy=(i_fc_nom+5, 45),fontsize=20) 
plt.legend();plt.grid(True, ls="--", alpha=0.5);plt.tight_layout()
plt.savefig("P_fc_vs_i_fc.pdf") # Commenter pour affichage direct
plt.show()

# ---- Figure 2 : u_fc(i_fc) pour chaque alpha ----
plt.figure(figsize=(8,6))
for i, a in enumerate(alpha_fc):
    plt.plot(i_fc_all[i], u_fc_all[i], label=f"α={a:.3f}")
plt.xlabel(r'$i_{FC}\ [A]$',fontsize=24)
plt.ylabel(r'$u_{FC}\ [V]$',fontsize=24)
plt.axvline(x=i_fc_nom,color='k',linestyle=':',lw=1.5)
plt.annotate(rf'$i_{{nom,ref}}$', xy=(i_fc_nom+5, 45),fontsize=24)
plt.title("Tension en fonction du courant")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("u_fc_vs_i_fc.png", dpi=300)

# ---- Figure 3 : P_fc_max(alpha_fc) ----
plt.figure(figsize=(8,6))
plt.plot(alpha_fc, P_fc_max, "o-", label="Simulation")

# Régression linéaire
reg = LinearRegression().fit(alpha_fc.reshape(-1,1), P_fc_max)
slope = reg.coef_[0]
intercept = reg.intercept_

# Courbe ajustée
P_fc_fit = reg.predict(alpha_fc.reshape(-1,1))
plt.plot(alpha_fc, P_fc_fit, "--", label=f"Fit: P_max = {slope:.2f}*α + {intercept:.2f}")

plt.xlabel("α_fc")
plt.ylabel("P_fc_max [W]")
plt.title("Puissance max en fonction de α")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("P_fc_max_vs_alpha.png", dpi=300)

# ---- Figure 4 : i_fc_at_Pmax(alpha_fc) ----
plt.figure(figsize=(8,6))
plt.plot(alpha_fc, i_fc_at_Pmax, "o-", label="Simulation")

reg_I = LinearRegression().fit(alpha_fc.reshape(-1,1), i_fc_at_Pmax)
slope_I, intercept_I = reg_I.coef_[0], reg_I.intercept_
i_fc_fit = reg_I.predict(alpha_fc.reshape(-1,1))

plt.plot(alpha_fc, i_fc_fit, "--", label=f"Fit: i* = {slope_I:.2f}*α + {intercept_I:.2f}")
plt.xlabel("α_fc")
plt.ylabel("i_fc* [A]")
plt.title("Courant de puissance max en fonction de α")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("i_fc_opt_vs_alpha.png", dpi=300)

# ---- Impression de l'équation ----
print(f"Équation de la droite : P_fc_max = {slope:.4f} * alpha_fc + {intercept:.4f} [W]")
print(f"Équation de la droite (i_fc*): i_fc* = {slope_I:.4f} * alpha_fc + {intercept_I:.4f} [A]")

# ---- SoH basé sur la tension au courant nominal (75% de i* à BoL) ----
i_fc_nom = 0.75 * i_fc_at_Pmax[0]  # courant de référence fixé au BoL
M = 1000
alpha_fc     = np.linspace(0, 0.22223223, M)

SoH_fc = np.array([
    voltage(alpha_fc[k], i_fc_nom) / voltage(0.0, i_fc_nom)
    for k in range(M)
])

# ---- Alpha EoL par résolution non-linéaire (brentq) ----
from scipy.optimize import brentq

####################################################################################################################
SoH_EoL_fc = 0.85
####################################################################################################################

V_bol_fc    = voltage(0.0, i_fc_nom)

def residual_fc(alpha):
    return voltage(alpha, i_fc_nom) / V_bol_fc - SoH_EoL_fc

# Recherche d'un encadrement valide : on cherche alpha_max tel que SoH < SoH_EoL
alpha_max_search = 0.274215  # borne haute (alpha < 1 par construction)
if residual_fc(alpha_max_search) > 0:
    print("Attention : SoH_EoL non atteint dans [0, 1[. Augmenter la plage d'alpha.")
else:
    alpha_EoL_fc = brentq(residual_fc, 0.0, alpha_max_search, xtol=1e-10, rtol=1e-10)
    print(f"\nAlpha en fin de vie :")
    print(f"Alpha EoL (SoH = {SoH_EoL_fc}) : alpha_fc_EoL = {alpha_EoL_fc:.6f}")
    print(f"Vérification : SoH(alpha_EoL) = {voltage(alpha_EoL_fc, i_fc_nom) / V_bol_fc:.6f}")

#########################################################################################################
from scipy.optimize import curve_fit

SoH_center = 0.85  # point de centrage

def model_poly3(x, a, b, c, d):
    s = x - SoH_center
    return a * s**3 + b * s**2 + c * s + d

def residual_fc(alpha, SoH):
    return voltage(alpha, i_fc_nom) / V_bol_fc - SoH

SoH_fc   = np.linspace(SoH_EoL_fc, 1, M)
P_fc_max = np.zeros(M)
for i in range(M):
    alpha_fc = brentq(residual_fc, 0.0, 0.274215, args=(SoH_fc[i],), xtol=1e-10, rtol=1e-10)
    i_fc_max    = (-234.8032 * alpha_fc + 238.8252)
    P_fc_max[i] = i_fc_max * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc) * i_fc_max / FC['n_parallel']
                                            - A * FC['T'] * np.log((i_fc_max / S / FC['n_parallel'] + j_in) / FC['j_0'])
                                            - B * FC['T'] * np.log(1 - i_fc_max / S / FC['n_parallel'] / FC['j_L'] / (1 - alpha_fc)))

# Ajustement sur la plage cible [0.8, 1] uniquement
mask_fit = (SoH_fc >= SoH_EoL_fc) & (SoH_fc <= 1.0)
popt, _ = curve_fit(model_poly3, SoH_fc[mask_fit], P_fc_max[mask_fit],
                    p0=[1, 1, 1, P_fc_max[mask_fit].mean()],
                    maxfev=10000)

soh_fine    = np.linspace(SoH_EoL_fc, 1.0, M)
p_max_fit   = model_poly3(soh_fine, *popt)

# ---- Figure 6 : P_fc_max vs SoH ----
plt.figure(figsize=(8, 6))
plt.plot(SoH_fc, P_fc_max, "s", label="Simulation", color='tab:red', markersize=4)
plt.plot(soh_fine, p_max_fit, "--", color='black', label="Fit")
plt.xlabel(r"$SoH_{FC}$")
plt.ylabel(r"$P_{fc,max}$ [W]")
plt.title("Puissance maximale vs SoH (Modèle Polynôme degré 3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Pmax_vs_SoH_perfect_fit_fc.png", dpi=300)

# ---- Print de l'équation ----
print(f"\nÉquation optimisée (s = SoH - {SoH_center}) :")
print(f"P_fc_max = {popt[0]:.4f}*SoH³ + {popt[1]:.4f}*SoH² + {popt[2]:.4f}*SoH + {popt[3]:.4f}")

# ---- Performance sur [0.8, 1] ----
y_pred    = model_poly3(SoH_fc[mask_fit], *popt)
residuals = P_fc_max[mask_fit] - y_pred
rmse      = np.sqrt(np.mean(residuals**2))
r2        = 1 - np.sum(residuals**2) / np.sum((P_fc_max[mask_fit] - P_fc_max[mask_fit].mean())**2)
print(f"\n--- Performance du Fit sur la plage [0.8, 1] ---")
print(f"R²   : {r2:.8f}")
print(f"RMSE : {rmse:.4f} W")
print(f"Erreur max : {np.max(np.abs(residuals)):.4f} W")

plt.show()

np.savetxt('resultats_fc.csv', 
           np.column_stack((SoH_fc, P_fc_max)), 
           delimiter=';', 
           header='SoH_fc;P_fc_max', 
           comments='')

# ==============================================================================
# # ---- Figure 7 : Rendement η(P_fc) pour chaque alpha, via tension thermoneutre ----
# # ==============================================================================
# #
# # Fondement physique :
# #   La tension thermoneutre V_th = ΔH / (n·F) = 285800 / (2 × 96485) ≈ 1.481 V/cellule
# #   représente l'enthalpie totale de réaction H2 + ½O2 → H2O(liq) ramenée en volts.
# #   Le rendement η = V_cellule / V_th mesure la fraction de l'enthalpie du H2
# #   effectivement convertie en électricité. La part (1 - η) est dissipée en chaleur.
# #   Ce rendement est cohérent avec le calcul de consommation H2 :
# #       P_H2 = P_elec / η   =>   pas de double-comptage du vieillissement
# #   car η est calculé depuis le même modèle voltage(alpha, i) que les courbes de polarisation.

# V_th = 285800 / (2 * 96485)  # Tension thermoneutre par cellule [V] ≈ 1.481 V

# N_eta = 10000
# M_eta = 5
# alpha_eta = np.linspace(0, 0.32981253, M_eta)

# plt.figure(figsize=(8, 6))

# for k, a in enumerate(alpha_eta):
#     # Plage de courant valide pour cet alpha (même borne que figures 1 & 2)
#     i_range = np.linspace(1e-3, FC['n_parallel'] * 219.9 * (1 - a), N_eta)

#     eta_list = []
#     P_list   = []

#     for i_val in i_range:
#         V_stack = voltage(a, i_val)
#         if np.isnan(V_stack) or np.isinf(V_stack) or V_stack <= 0:
#             continue
#         # Tension moyenne par cellule
#         V_cell = V_stack / FC['n_series']
#         # Rendement thermodynamique (LHV)
#         eta = V_cell / V_th
#         # Puissance électrique totale
#         if i_val != 1e-3 :
#             if V_stack * i_val < P_elec :
#                 break
#             else : 
#                 P_elec = V_stack * i_val
#         else :
#             P_elec = V_stack * i_val
#         eta_list.append(eta)
#         P_list.append(P_elec)

#     eta_arr = np.array(eta_list)
#     P_arr   = np.array(P_list)

#     # Trier par puissance croissante pour un tracé propre
#     sort_idx = np.argsort(P_arr)
#     plt.plot(P_arr / 1000, eta_arr * 100, label=f"α={a:.3f}")

# # Marquer la puissance nominale BoL (i_fc_nom calculé plus haut)
# V_nom_BoL   = voltage(0.0, i_fc_nom)
# P_nom_BoL   = V_nom_BoL * i_fc_nom
# eta_nom_BoL = (V_nom_BoL / FC['n_series']) / V_th * 100
# plt.annotate(
#     rf"$P_{{nom,BoL}}$" + f"\nη={eta_nom_BoL:.1f}%",
#     xy=(P_nom_BoL / 1000, eta_nom_BoL),
#     xytext=(P_nom_BoL / 1000 + 0.05, eta_nom_BoL + 1),
#     fontsize=13,
#     arrowprops=dict(arrowstyle="->", lw=1.2)
# )

# plt.xlabel(r"$P_{fc}$ [kW]")
# plt.ylabel(r"$\eta_{fc}$ [%]  (base LHV)")
# plt.title("Rendement de la pile à combustible\n"
#           + r"$\eta = V_{cell} \,/\, V_{th}$,  $V_{th} = \Delta H / nF \approx 1.481$ V")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.savefig("eta_fc_vs_P_fc.png", dpi=300)


plt.show()