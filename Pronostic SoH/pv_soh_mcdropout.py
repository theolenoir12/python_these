# -*- coding: utf-8 -*-
"""
Étape 2 : Prédiction récursive LSTM + Monte-Carlo Dropout
Méthode : Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- Même architecture et loss (MSE) que l'étape 1
- Dropout actif à l'inférence (training=True) pour approximation variationnelle
- Incertitude = variance empirique sur T passages stochastiques
"""

import os, random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# ── Reproductibilité ───────────────────────────────────────────────────────────
SEED = 41 #43 pour ely
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ── Hyperparamètres (issus de l'étape 1) ──────────────────────────────────────
HP = {
    'learning_rate': 1e-3,
    'window_size':   100,
    'batch_size':    64,
    'lstm_units':    150,
    'dropout_rate':  0.05,
    'epochs':        1000,
}
T = 500          # Nombre de passages MC Dropout
CONF = 0.95      # Niveau de confiance de l'intervalle
OUTPUT_DIR = 'resultats_mc_dropout'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Données ────────────────────────────────────────────────────────────────────

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# --- Remplacer la section "Données" par ceci ---
print("Chargement des données...")
df = pd.read_csv('SoH_ely.csv', sep=';', header=None)
datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]
n = len(datapro)

# Définition des indices
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

# Découpage RAW
train_raw = datapro[:train_end]
val_raw   = datapro[train_end:val_end]
test_raw  = datapro[val_end:].flatten()

# Scaler : ON FIT UNIQUEMENT SUR LE TRAIN
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[0.90], [1.00]])) #on rescale en connaissance de cause sur la plage de SoH
train_norm = scaler.fit_transform(train_raw)
val_norm   = scaler.transform(val_raw)

# Préparation des séquences
ws = HP['window_size']
X_train, y_train = create_sequences(train_norm, ws)

# Pour la validation, on a besoin des 'ws' derniers points du train pour prédire le début du val
val_context = np.vstack((train_norm[-ws:], val_norm))
X_val, y_val = create_sequences(val_context, ws)

print(f"Total: {n} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(test_raw)}")


# ── Modèle (identique à l'étape 1, loss MSE) ──────────────────────────────────
def build_model(ws, units, dr, lr):
    """
    Architecture identique à l'étape 1.
    Le Dropout avec training=True à l'inférence constitue
    l'approximation variationnelle de Gal & Ghahramani (2016).
    """
    inputs  = Input(shape=(ws, 1))
    x       = LSTM(units)(inputs)
    x       = Dropout(dr)(x)
    outputs = Dense(1)(x)
    model   = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse'
    )
    return model

# ── Entraînement ───────────────────────────────────────────────────────────────
tf.keras.backend.clear_session()
model = build_model(ws, HP['lstm_units'], HP['dropout_rate'], HP['learning_rate'])

callbacks = [
    # tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', # On surveille la validation !
    #     factor=0.8, 
    #     patience=50, 
    #     min_lr=1e-6, 
    #     verbose=1
    # ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # CRUCIAL
        patience=50, 
        restore_best_weights=True, # On revient au "meilleur" modèle
        verbose=1
    ),
]

print(f"=== Entraînement avec Validation (max {HP['epochs']} epochs) ===")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), # On ajoute le set de validation ici
    epochs=HP['epochs'],
    batch_size=HP['batch_size'],
    callbacks=callbacks,
    verbose=0,
)

model.save(os.path.join(OUTPUT_DIR, 'modele_mc_dropout_scaler2.keras'))

print(f"Terminé en {(time.time()-t0)/60:.1f} min — "
      f"{len(history.history['loss'])} epochs effectives")

# ── MC Dropout — T passages stochastiques ─────────────────────────────────────
def mc_dropout_predict(model, seed_window, n_steps, T):
    """
    Réalise T prédictions récursives avec Dropout actif (training=True).
    Chaque trajectoire est indépendante : le réseau s'auto-alimente
    avec sa propre prédiction à chaque pas.

    Retourne
    --------
    samples : ndarray (T, n_steps) — trajectoires normalisées
    """
    # Dupliquer la même seed pour les T trajectoires
    windows = np.tile(seed_window.flatten(), (T, 1))  # (T, ws)
    samples = np.zeros((T, n_steps), dtype=np.float32)

    for t in range(n_steps):
        x_batch = windows[:, :, np.newaxis]            # (T, ws, 1)
        # training=True : le Dropout reste actif → chaque ligne donne
        # une prédiction différente (approximation variationnelle)
        preds = model(x_batch, training=True).numpy().flatten()  # (T,)
        samples[:, t] = preds
        # Auto-alimentation : chaque trajectoire utilise sa propre pred
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds

    return samples


print(f"\n=== MC Dropout : T={T} trajectoires ===")
# La seed est maintenant la fin du set de validation
seed = val_norm[-ws:] 
t0 = time.time()
samples = mc_dropout_predict(model, seed, len(test_raw), T)
print(f"Terminé en {time.time()-t0:.1f} s")

# ── Dénormalisation et statistiques ───────────────────────────────────────────
# Dénormaliser chaque trajectoire
samples_real = np.zeros_like(samples)
for i in range(T):
    samples_real[i] = scaler.inverse_transform(
        samples[i].reshape(-1, 1)
    ).flatten()

# Statistiques sur l'ensemble des trajectoires
mean_pred = np.mean(samples_real, axis=0)   # Prédiction centrale
std_pred  = np.std(samples_real,  axis=0)   # Incertitude épistémique

# Intervalle de confiance (par quantiles — plus robuste que ±z*σ)
alpha    = 1 - CONF
ci_lower = np.quantile(samples_real, alpha / 2,     axis=0)
ci_upper = np.quantile(samples_real, 1 - alpha / 2, axis=0)

rmse = float(np.sqrt(mean_squared_error(test_raw, mean_pred)))
mae  = float(mean_absolute_error(test_raw, mean_pred))
# Taux de couverture : % de vraies valeurs dans l'IC
coverage = float(np.mean((test_raw >= ci_lower) & (test_raw <= ci_upper)))
print(f"\nTest RMSE    : {rmse:.5f}")
print(f"Test MAE     : {mae:.5f}")
print(f"Couverture IC {int(CONF*100)}% : {coverage*100:.1f}%  (cible : {int(CONF*100)}%)")

# ── Visualisation ─────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                      'axes.titlesize': 11, 'axes.labelsize': 10})

C = dict(train_val='#2980B9', test='#1A1A2E', mean='#E74C3C',
          ci='#F1948A', traj='#AED6F1', loss='#2ECC71')

x_tv   = np.arange(val_end )
x_test = np.arange(val_end, n)
train_val_raw = datapro[:val_end]
test_raw      = datapro[val_end:] 

fig = plt.figure(figsize=(16, 16))
gs  = gridspec.GridSpec(3, 1, figure=fig,
                        height_ratios=[1.6, 1.6, 0.8], hspace=0.45)

# Panneau 1 : Vue d'ensemble
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_tv,   train_val_raw, color=C['train_val'], lw=1.2,
          label='Train + Val')
ax1.plot(x_test, test_raw,      color=C['test'],      lw=1.5,
          label='Réel (Test)')
ax1.plot(x_test, mean_pred,     color=C['mean'],      lw=1.8,
          linestyle='--', label='Prédiction moyenne MC')
ax1.fill_between(x_test, ci_lower, ci_upper,
                  color=C['ci'], alpha=0.4,
                  label=f'IC {int(CONF*100)}% (quantiles)')
ax1.axvline(val_end, color='#7F8C8D', linestyle=':', lw=1.5)
ax1.set_title("MC Dropout (Gal & Ghahramani) — Vue d'ensemble",
              fontweight='bold')
ax1.set_ylabel("SoH")
ax1.legend(loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim([0, n])

# Panneau 2 : Zoom test + quelques trajectoires individuelles
ax2 = fig.add_subplot(gs[1])
# Afficher 30 trajectoires pour donner une idée de la dispersion
for i in range(min(30, T)):
    ax2.plot(x_test, samples_real[i], color=C['traj'],
              lw=0.5, alpha=0.3)
ax2.plot(x_test, test_raw,  color=C['test'], lw=1.8,
          label='Valeurs réelles', zorder=5)
ax2.plot(x_test, mean_pred, color=C['mean'], lw=2.0,
          linestyle='--', label=f'Moyenne MC (RMSE={rmse:.4f})', zorder=6)
ax2.fill_between(x_test, ci_lower, ci_upper,
                  color=C['ci'], alpha=0.35,
                  label=f'IC {int(CONF*100)}% | Couverture={coverage*100:.0f}%')
ax2.set_title("Zoom Test — Trajectoires MC et Incertitude Épistémique",
              fontweight='bold')
ax2.set_xlabel("Temps (jours)")
ax2.set_ylabel("SoH")
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.4)

# Panneau 3 : Loss + tableau
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                          width_ratios=[2.2, 1])
ax3 = fig.add_subplot(gs_bot[0])
ax3.plot(history.history['loss'], color=C['loss'], lw=1.5,
          label='Train Loss (MSE)')
ax3.set_yscale('log')
ax3.set_title("Convergence (MSE)", fontweight='bold')
ax3.set_xlabel("Époques")
ax3.set_ylabel("MSE Loss (log)")
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.4)

ax4 = fig.add_subplot(gs_bot[1])
ax4.axis('off')
table_data = [
    ['Paramètre / Métrique', 'Valeur'],
    ['Window Size',          str(HP['window_size'])],
    ['LSTM Units',           str(HP['lstm_units'])],
    ['Dropout Rate',         str(HP['dropout_rate'])],
    ['MC Samples (T)',       str(T)],
    ['', ''],
    ['Test RMSE',            f"{rmse:.5f}"],
    ['Test MAE',             f"{mae:.5f}"],
    [f'Couverture IC {int(CONF*100)}%', f"{coverage*100:.1f}%"],
]
tbl = ax4.table(cellText=table_data, loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.4)
for col in range(2):
    tbl[(0, col)].set_facecolor('#2C3E50')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')

fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'mc_dropout_final_sinusoide.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure sauvegardée : {out_path}")
plt.show()