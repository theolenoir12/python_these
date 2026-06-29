# -*- coding: utf-8 -*-
"""
Prédiction "one shot" déterministe (SANS MC Dropout) des profils de puissance.

But : vérifier d'abord que le modèle LSTM produit des courbes cohérentes avec
la réalité, avant d'introduire l'incertitude (MC Dropout).
    - architecture config_12 d'origine : Dropout = 0
    - une seule prédiction récursive (le réseau s'auto-alimente)
    - figures : vue d'ensemble + zoom test, pour production PV et consommation
"""

import os, random, time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Reproductibilité ─────────────────────────────────────────────────────────
SEED = 41
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')

HP = {
    'learning_rate': 1e-3,
    'window_size':   96,
    'batch_size':    64,
    'lstm_units':    80,
    'dropout_rate':  0.0,    # one shot déterministe -> pas de dropout (config_12)
    'epochs':        160,
}
HORIZON = None   # None = tout le set de test

PROFILS = {
    'production':   {'col': 1, 'titre': 'Production PV',  'ylabel': 'Production [Wh]'},
    'consommation': {'col': 2, 'titre': 'Consommation',   'ylabel': 'Consommation [Wh]'},
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_oneshot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Outils ───────────────────────────────────────────────────────────────────
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_model(ws, units, dr, lr):
    inputs  = Input(shape=(ws, 1))
    x       = LSTM(units)(inputs)
    x       = Dropout(dr)(x)
    outputs = Dense(1)(x)
    model   = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def predict_recursive(model, seed_window, n_steps):
    curr = seed_window.copy().reshape(1, -1, 1)
    preds = []
    for _ in range(n_steps):
        p = model.predict(curr, verbose=0)
        preds.append(p[0, 0])
        curr = np.concatenate([curr[:, 1:, :], p.reshape(1, 1, 1)], axis=1)
    return np.array(preds).reshape(-1, 1)

# ── Boucle par profil ────────────────────────────────────────────────────────
for profil, cfg in PROFILS.items():
    print(f"\n{'#'*70}\n# PROFIL : {cfg['titre']} (colonne {cfg['col']})\n{'#'*70}")

    df = pd.read_csv(DATA_PATH, sep=';', header=None)
    datapro = df.iloc[:, cfg['col']].values.reshape(-1, 1).astype(float)
    n = len(datapro)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(datapro[:train_end])
    val_norm   = scaler.transform(datapro[train_end:val_end])

    ws = HP['window_size']
    X_train, y_train = create_sequences(train_norm, ws)
    val_context = np.vstack((train_norm[-ws:], val_norm))
    X_val, y_val = create_sequences(val_context, ws)

    test_full = datapro[val_end:].flatten()
    n_steps = len(test_full) if HORIZON is None else min(HORIZON, len(test_full))
    test_raw = test_full[:n_steps]

    print(f"Total: {n} | Train seq: {len(X_train)} | Val seq: {len(X_val)} | Test: {len(test_full)} | Horizon: {n_steps}")

    tf.keras.backend.clear_session()
    model = build_model(ws, HP['lstm_units'], HP['dropout_rate'], HP['learning_rate'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                          restore_best_weights=True, verbose=1)
    print(f"=== Entraînement (max {HP['epochs']} epochs) ===")
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=HP['epochs'], batch_size=HP['batch_size'],
                        callbacks=[es], verbose=0)
    print(f"Terminé en {(time.time()-t0)/60:.1f} min — {len(history.history['loss'])} epochs")

    # Prédiction récursive unique (seed = fin de la validation)
    seed = val_norm[-ws:]
    pred_norm = predict_recursive(model, seed, n_steps)
    mean_pred = scaler.inverse_transform(pred_norm).flatten()

    rmse = float(np.sqrt(mean_squared_error(test_raw, mean_pred)))
    mae  = float(mean_absolute_error(test_raw, mean_pred))
    print(f"Test RMSE : {rmse:.2f} Wh | Test MAE : {mae:.2f} Wh")

    # ── Figure : vue d'ensemble + zoom ───────────────────────────────────────
    x_tv   = np.arange(val_end)
    x_test = np.arange(val_end, val_end + n_steps)

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_tv, datapro[:val_end], color='#2980B9', lw=1.0, label='Train + Val')
    ax1.plot(x_test, test_raw,  color='#1A1A2E', lw=1.2, label='Réel (Test)')
    ax1.plot(x_test, mean_pred, color='#E74C3C', lw=1.6, ls='--', label='Prédiction récursive')
    ax1.axvline(val_end, color='#7F8C8D', ls=':', lw=1.5)
    ax1.set_title(f"One shot déterministe — {cfg['titre']} — Vue d'ensemble", fontweight='bold')
    ax1.set_ylabel(cfg['ylabel']); ax1.legend(loc='upper left')
    ax1.grid(True, ls='--', alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x_test, test_raw,  color='#1A1A2E', lw=1.6, label='Réel (Test)')
    ax2.plot(x_test, mean_pred, color='#E74C3C', lw=1.8, ls='--',
             label=f'Prédiction (RMSE={rmse:.0f} Wh, MAE={mae:.0f} Wh)')
    ax2.set_title("Zoom Test", fontweight='bold')
    ax2.set_xlabel("Temps [h]"); ax2.set_ylabel(cfg['ylabel'])
    ax2.legend(loc='upper right'); ax2.grid(True, ls='--', alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'oneshot_{profil}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")

print("\nTerminé pour tous les profils.")
