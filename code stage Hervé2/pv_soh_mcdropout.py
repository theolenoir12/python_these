# -*- coding: utf-8 -*-
"""
Phase de TEST LSTM avec Monte-Carlo Dropout — version optimisée.
- Entraînement sur 650 epochs fixes avec ModelCheckpoint (pas d'EarlyStopping)
- MC Dropout vectorisé : n_steps appels TF au lieu de MC_SAMPLES × n_steps
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import time

def load_for_mc_dropout(path): #pour recharger un nouveau modèle
    model = tf.keras.models.load_model(path)
    # On force le dropout à l'inférence
    return model

# =============================================================================
# CONFIGURATION
# =============================================================================
HP = {
    'learning_rate' : 0.001,
    'window_size'   : 100,
    'batch_size'    : 64,
    'lstm_units'    : 100,
    'epochs'        : 650,
}

MC_SAMPLES        = 1000
DROPOUT_RATE      = 0.01
recurrent_dropout = 0.05
CONFIDENCE_LVLS   = [95]
OUTPUT_DIR        = 'resultats_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION
# =============================================================================
df       = pd.read_csv('SoH_ely.csv', sep=';', header=None)
datapro  = df.iloc[:, 0].values.reshape(-1, 1)[::24]
n        = len(datapro)

train_val_end = int(n * 0.85)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(datapro[:train_val_end])
data_normalized = scaler.transform(datapro)

train_val_raw = datapro[:train_val_end]
test_raw      = datapro[train_val_end:]
ws            = HP['window_size']

print(f"Total : {n} | Train+Val : {len(train_val_raw)} | Test : {len(test_raw)}")

# =============================================================================
# 2. SÉQUENCES
# =============================================================================
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

train_val_slice  = data_normalized[:train_val_end]
X_train, y_train = create_sequences(train_val_slice, ws)
print(f"Séquences : X={X_train.shape}, y={y_train.shape}")

# =============================================================================
# 3. MODÈLE
# =============================================================================
def build_model(window_size, lstm_units, learning_rate, dropout_rate, recurrent_dropout):
    inputs = tf.keras.Input(shape=(window_size, 1))
    x      = LSTM(lstm_units, recurrent_dropout=recurrent_dropout)(inputs)
    x      = Dropout(dropout_rate)(x)   # ← DOIT être > 0 pour MC Dropout
    # Optionnel : un 2ᵉ dropout avant Dense pour plus de variance
    # x    = Dropout(dropout_rate / 2)(x)
    output = Dense(1)(x)
    model  = tf.keras.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

# =============================================================================
# 4. ENTRAÎNEMENT — 650 epochs fixes, checkpoint sur la meilleure loss
# =============================================================================
tf.keras.backend.clear_session()
model = build_model(ws, HP['lstm_units'], HP['learning_rate'], DROPOUT_RATE, recurrent_dropout)
model.summary()


print(f"\n=== Entraînement : {HP['epochs']} epochs (configuration finale) ===")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    epochs=HP['epochs'],
    batch_size=HP['batch_size'],
    verbose=2
)
print(f"Entraînement terminé en {(time.time()-t0)/60:.1f} min")


# =============================================================================
# 5. MC DROPOUT VECTORISÉ
#    À chaque pas t, on propage MC_SAMPLES fenêtres en un seul appel batch.
#    windows : (MC_SAMPLES, ws)  — toutes identiques au départ, puis divergent.
# =============================================================================
def predict_mc_recursive_fast(model, seed_window, n_steps, n_samples):
    """
    Prédiction récursive vectorisée.
    - seed_window : array (ws,) normalisé
    - Retourne    : array (n_samples, n_steps) dénormalisé
    """
    # Initialisation : n_samples copies identiques de la fenêtre de départ
    windows = np.tile(seed_window.flatten(), (n_samples, 1))  # (S, ws)

    all_preds = np.zeros((n_samples, n_steps), dtype=np.float32)

    for t in range(n_steps):
        x_batch = windows[:, :, np.newaxis]            # (S, ws, 1)
        # Un seul appel TF pour les S trajectoires simultanément
        preds_t = model(x_batch, training=True).numpy().flatten()  # (S,)
        all_preds[:, t] = preds_t
        # Glissement de fenêtre : on retire le plus ancien, on ajoute la prédiction
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds_t

    return all_preds   # encore normalisé

seed = train_val_slice[-ws:]

print(f"\n=== MC Dropout vectorisé : {MC_SAMPLES} trajectoires × "
      f"{len(test_raw)} pas ===")
t0 = time.time()
traj_norm = predict_mc_recursive_fast(model, seed, len(test_raw), MC_SAMPLES)
print(f"MC Dropout terminé en {time.time()-t0:.1f} s")

# Dénormalisation
traj_real = np.array([
    scaler.inverse_transform(traj_norm[i].reshape(-1, 1)).flatten()
    for i in range(MC_SAMPLES)
])   # (MC_SAMPLES, n_test)

mean_pred   = np.mean(traj_real,   axis=0)
median_pred = np.median(traj_real, axis=0)
std_pred    = np.std(traj_real,    axis=0)

intervals = {}
for cl in CONFIDENCE_LVLS:
    lo = np.percentile(traj_real, (100 - cl) / 2,      axis=0)
    hi = np.percentile(traj_real, 100 - (100 - cl) / 2, axis=0)
    intervals[cl] = (lo, hi)

test_true = test_raw.flatten()
rmse_test = float(np.sqrt(np.mean((mean_pred - test_true)**2)))
mae_test  = float(mean_absolute_error(test_true, mean_pred))
print(f"Test RMSE : {rmse_test:.6f} | MAE : {mae_test:.6f}")

# =============================================================================
# 6. FIGURE
# =============================================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
})

fig = plt.figure(figsize=(16, 18))
gs_main = gridspec.GridSpec(3, 1, figure=fig,
                             height_ratios=[1.6, 1.6, 0.8], hspace=0.42)

C = dict(train_val='#2980B9', test='#1A1A2E', mean='#E74C3C',
         median='#8E44AD', ci95='#F1948A', ci90='#F5B7B1',
         std='#F39C12', loss='#2ECC71')

x_tv   = np.arange(train_val_end)
x_test = np.arange(train_val_end, n)

# ── Panneau 1 : Vue d'ensemble ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs_main[0])
ax1.plot(x_tv,   train_val_raw, color=C['train_val'], lw=1.1,
         label='Train + Validation (85%)')
ax1.plot(x_test, test_true,     color=C['test'],      lw=1.3,
         label='Test réel (15%)')
ax1.plot(x_test, mean_pred,     color=C['mean'],      lw=1.8,
         linestyle='--', label='Prédiction (moyenne MC)')
ax1.fill_between(x_test, *intervals[95], color=C['ci95'], alpha=0.45, label='IC 95%')
ax1.axvline(train_val_end, color='#7F8C8D', linestyle=':', lw=1.4, label='Début test')
ax1.set_title("Vue d'ensemble — Prédiction récursive MC Dropout (LSTM)",
              fontweight='bold')
ax1.set_xlabel("Temps (jours)")
ax1.set_ylabel("State of Health (SoH)")
ax1.legend(loc='lower left', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.set_xlim([0, n - 1])

# ── Panneau 2 : Zoom test ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs_main[1])
ax2.plot(x_test, test_true,     color=C['test'],   lw=1.5, label='Valeurs réelles')
ax2.plot(x_test, mean_pred,     color=C['mean'],   lw=1.8, linestyle='--',
         label=f'Moyenne MC (RMSE={rmse_test:.4f})')
ax2.plot(x_test, median_pred,   color=C['median'], lw=1.2, linestyle=':',
         label='Médiane MC')
ax2.fill_between(x_test, *intervals[95], color=C['ci95'], alpha=0.40, label='IC 95%')

ax2b = ax2.twinx()
ax2b.plot(x_test, std_pred, color=C['std'], lw=0.9, linestyle='-.', alpha=0.75,
          label='Écart-type MC')
ax2b.set_ylabel("Écart-type", color=C['std'], fontsize=9)
ax2b.tick_params(axis='y', labelcolor=C['std'])
ax2b.set_ylim(bottom=0)

lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labs1 + labs2, loc='lower left', framealpha=0.9)
ax2.set_title(
    f"Zoom zone de test — {MC_SAMPLES} passes MC | "
    f"RMSE={rmse_test:.4f}  MAE={mae_test:.4f}", fontweight='bold')
ax2.set_xlabel("Temps (jours)")
ax2.set_ylabel("State of Health (SoH)")
ax2.grid(True, linestyle='--', alpha=0.45)

# ── Panneau 3 : Loss + tableau ───────────────────────────────────────────────
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2],
                                          width_ratios=[2.2, 1])
ax3 = fig.add_subplot(gs_bot[0])
ax3.semilogy(history.history['loss'], color=C['loss'], lw=1.2, label='Loss MSE')


ax3.set_title("Courbe de loss — entraînement complet (log Y)", fontweight='bold')
ax3.set_xlabel("Époques"); ax3.set_ylabel("MSE (normalisé)")
ax3.legend(); ax3.grid(True, which='both', linestyle='--', alpha=0.4)

ax4 = fig.add_subplot(gs_bot[1])
ax4.axis('off')
table_data = [
    ['Paramètre / Métrique',   'Valeur'],
    ['learning_rate',          str(HP['learning_rate'])],
    ['window_size',            str(HP['window_size'])],
    ['batch_size',             str(HP['batch_size'])],
    ['lstm_units',             str(HP['lstm_units'])],
    ['Dropout rate',           str(DROPOUT_RATE)],
    ['Passes MC Dropout',      str(MC_SAMPLES)],
    ['',                       ''],
    ['Test RMSE (moy. MC)',    f"{rmse_test:.5f}"],
    ['Test MAE  (moy. MC)',    f"{mae_test:.5f}"],
    ['Données train+val',      str(train_val_end)],
    ['Données test',           str(len(test_raw))],
]
tbl = ax4.table(cellText=table_data, loc='center', cellLoc='left')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.2); tbl.scale(1.1, 1.28)
for col in range(2):
    tbl[(0, col)].set_facecolor('#2C3E50')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')
    for hr in [9, 10]:
        tbl[(hr, col)].set_facecolor('#FDEBD0')
        tbl[(hr, col)].set_text_props(fontweight='bold')

fig.suptitle(
    "Prédiction récursive LSTM — Monte-Carlo Dropout\n"
    "Train+Val 85% / Test 15% — Entraînement complet sans arrêt anticipé",
    fontsize=13, fontweight='bold', y=0.998
)

plt.show()

# Définition du suffixe
suffix = f"dr_{DROPOUT_RATE}_rec_{recurrent_dropout}"

# Sauvegarde figure
out_path = os.path.join(OUTPUT_DIR, f'test_mc_dropout_{suffix}.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Figure sauvegardée : {out_path}")

# Sauvegarde modèle
model.save(os.path.join(OUTPUT_DIR, f'model_{suffix}.keras'))
