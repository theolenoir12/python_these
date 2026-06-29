# -*- coding: utf-8 -*-
"""
DIAGNOSTIC : pourquoi les intervalles MC Dropout sont-ils serrés pour les
profils de puissance (alors qu'ils paraissent plus larges/réalistes pour le SoH) ?

Trois analyses sur le modèle de consommation entraîné (pv_profils_mcdropout.py) :

 1. Largeur de bande vs horizon (accumulation) : la bande grandit-elle avec le
    nombre de pas récursifs, ou reste-t-elle bornée (signal périodique amorti) ?

 2. Épistémique vs erreur totale : la bande MC (incertitude du modèle) est-elle
    petite devant l'erreur réelle (RMSE) ? Un grand écart = bruit ALÉATOIRE non
    capté par le MC Dropout (qui ne mesure que l'incertitude ÉPISTÉMIQUE).

 3. Sensibilité au taux de dropout : on transfère les poids dans des modèles à
    dropout croissant (0.05 -> 0.4) et on mesure largeur de bande + couverture.
    Montre que 0.05 (placé avant le Dense seulement) injecte très peu de variance.
"""

import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'resultats_mc_dropout')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_diagnostic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROFIL_COL = 2          # consommation (bruitée dans csv2)
WS         = 96
UNITS      = 80
T          = 500
CONF       = 0.95
LONG_H     = 480        # horizon long pour tester l'accumulation (20 jours)
SHORT_H    = 96         # horizon de travail (4 jours)
DROPOUTS   = [0.05, 0.1, 0.2, 0.4]

plt.rcParams.update({"font.family": "serif", "font.size": 13,
                     "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": '--'})

def build_model(ws, units, dr):
    inp = Input(shape=(ws, 1)); x = LSTM(units)(inp); x = Dropout(dr)(x); out = Dense(1)(x)
    m = tf.keras.Model(inp, out); m.compile(optimizer='adam', loss='mse'); return m

def mc_predict(model, seed_norm, n_steps, T):
    windows = np.tile(seed_norm.flatten(), (T, 1))
    samples = np.zeros((T, n_steps), dtype=np.float32)
    for t in range(n_steps):
        p = model(windows[:, :, np.newaxis], training=True).numpy().flatten()
        samples[:, t] = p
        windows = np.roll(windows, -1, axis=1); windows[:, -1] = p
    return samples

# ── Données + modèle ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep=';', header=None)
data = df.iloc[:, PROFIL_COL].values.reshape(-1, 1).astype(float)
n = len(data)
train_end, val_end = int(n * 0.70), int(n * 0.85)

scaler = MinMaxScaler(feature_range=(0, 1)); scaler.fit(data[:train_end])
val_norm = scaler.transform(data[train_end:val_end])
test_real = data[val_end:].flatten()
seed = val_norm[-WS:]

base_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'modele_mc_dropout_consommation.keras'))
base_weights = base_model.get_weights()
amplitude = float(np.ptp(data[:val_end]))   # amplitude crête-à-crête du signal
print(f"Signal consommation : amplitude crête-à-crête ≈ {amplitude:.0f} Wh")

# ── 1) Largeur de bande vs horizon (dropout nominal 0.05) ────────────────────
samples_long = mc_predict(base_model, seed, LONG_H, T)
samples_long = np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in samples_long])
std_lead = samples_long.std(axis=0)                       # std MC par pas (Wh)
print("\n[1] Largeur de bande (std MC) vs horizon :")
for h in [1, 24, 48, 96, 240, 480]:
    if h <= LONG_H:
        print(f"    h={h:>3} : std MC = {std_lead[h-1]:6.1f} Wh "
              f"({100*std_lead[h-1]/amplitude:.2f} % de l'amplitude)")

# ── 2) Épistémique (bande MC) vs erreur totale (RMSE) à 96 h ─────────────────
samples_96 = samples_long[:, :SHORT_H]
mean_96 = samples_96.mean(axis=0)
std_96  = samples_96.std(axis=0)
test_96 = test_real[:SHORT_H]
rmse_96 = float(np.sqrt(mean_squared_error(test_96, mean_96)))
epis_96 = float(std_96.mean())
ci_lo = np.quantile(samples_96, (1-CONF)/2, axis=0)
ci_hi = np.quantile(samples_96, 1-(1-CONF)/2, axis=0)
cov_96 = float(np.mean((test_96 >= ci_lo) & (test_96 <= ci_hi)))
print(f"\n[2] À 96 h : std MC moyenne (épistémique) = {epis_96:.1f} Wh | "
      f"RMSE (erreur totale) = {rmse_96:.1f} Wh | ratio = {epis_96/rmse_96:.2f}")
print(f"    Couverture IC{int(CONF*100)}% = {cov_96*100:.0f}%  "
      f"(<<{int(CONF*100)}% => sous-dispersion : bruit aléatoire non capté)")

# ── 3) Sensibilité au taux de dropout (transfert de poids) ───────────────────
print("\n[3] Sensibilité au taux de dropout (mêmes poids, dropout d'inférence variable) :")
sens = []
for dr in DROPOUTS:
    m = build_model(WS, UNITS, dr); m.set_weights(base_weights)
    s = mc_predict(m, seed, SHORT_H, T)
    s = np.array([scaler.inverse_transform(r.reshape(-1, 1)).flatten() for r in s])
    mean_d = s.mean(axis=0)
    std_d = s.std(axis=0).mean()
    lo = np.quantile(s, (1-CONF)/2, axis=0); hi = np.quantile(s, 1-(1-CONF)/2, axis=0)
    cov = np.mean((test_96 >= lo) & (test_96 <= hi))
    rmse_d = np.sqrt(mean_squared_error(test_96, mean_d))
    sens.append((dr, std_d, cov*100, rmse_d))
    print(f"    dropout={dr:.2f} : std MC = {std_d:6.1f} Wh | couverture = {cov*100:4.0f}% | RMSE = {rmse_d:6.1f} Wh")

# ── Figure de synthèse ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) std vs horizon
axes[0].plot(np.arange(1, LONG_H+1), std_lead, color='#2980B9', lw=2)
axes[0].axvline(SHORT_H, color='#E74C3C', ls=':', lw=1.5, label='Horizon 4 j')
axes[0].set_title("(1) Largeur de bande MC vs horizon")
axes[0].set_xlabel("Pas de prévision [h]"); axes[0].set_ylabel("Écart-type MC [Wh]")
axes[0].legend()

# (b) épistémique vs total à 96h
axes[1].fill_between(np.arange(SHORT_H), ci_lo, ci_hi, color='#3498DB', alpha=0.3,
                     label='Bande MC (épistémique)')
axes[1].plot(np.arange(SHORT_H), test_96, color='#1A1A2E', lw=1.5, label='Réel')
axes[1].plot(np.arange(SHORT_H), mean_96, color='#E74C3C', lw=1.8, ls='--', label='Moyenne MC')
axes[1].set_title(f"(2) Bande MC ({epis_96:.0f} Wh) << RMSE ({rmse_96:.0f} Wh)\nCouverture {cov_96*100:.0f}%")
axes[1].set_xlabel("Pas [h]"); axes[1].set_ylabel("Consommation [Wh]"); axes[1].legend(fontsize=10)

# (c) sensibilité dropout
drs = [s[0] for s in sens]; stds = [s[1] for s in sens]; covs = [s[2] for s in sens]
ax2 = axes[2]; ax2b = ax2.twinx()
ax2.plot(drs, stds, 'o-', color='#2980B9', lw=2, label='std MC')
ax2b.plot(drs, covs, 's--', color='#27AE60', lw=2, label='couverture')
ax2.set_title("(3) Sensibilité au dropout d'inférence")
ax2.set_xlabel("Taux de dropout"); ax2.set_ylabel("Écart-type MC [Wh]", color='#2980B9')
ax2b.set_ylabel("Couverture IC95% [%]", color='#27AE60')
ax2b.axhline(95, color='#27AE60', ls=':', lw=1)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'diagnostic_intervalles_mc.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"\nFigure sauvegardée : {out}")
