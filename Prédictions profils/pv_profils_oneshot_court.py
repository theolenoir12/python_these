# -*- coding: utf-8 -*-
"""
Prédiction "one shot" déterministe à HORIZON COURT (sans MC Dropout).

La prévision récursive n'a de sens que sur un horizon court pour des profils
oscillatoires. On entraîne le modèle (config_12, dropout=0), on le sauvegarde,
puis on trace la prévision sur quelques horizons courts (96 h, 168 h) pour la
production PV et la consommation.

Le modèle est rechargé s'il existe déjà -> itération rapide sur les horizons
sans ré-entraînement.
"""

import os, random, time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
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
    'learning_rate': 1e-3, 'window_size': 96, 'batch_size': 64,
    'lstm_units': 80, 'dropout_rate': 0.0, 'epochs': 160,
}
HORIZONS = [96, 168]          # 4 jours, 1 semaine
CONTEXT  = 168                # heures d'historique affichées avant la prévision

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
        X.append(data[i:i + window_size]); y.append(data[i + window_size])
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
    curr, preds = seed_window.copy().reshape(1, -1, 1), []
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
    train_end, val_end = int(n * 0.70), int(n * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(datapro[:train_end])
    val_norm   = scaler.transform(datapro[train_end:val_end])

    ws = HP['window_size']
    model_path = os.path.join(OUTPUT_DIR, f'modele_oneshot_{profil}.keras')

    if os.path.exists(model_path):
        print(f"Modèle existant rechargé : {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        X_train, y_train = create_sequences(train_norm, ws)
        val_context = np.vstack((train_norm[-ws:], val_norm))
        X_val, y_val = create_sequences(val_context, ws)
        tf.keras.backend.clear_session()
        model = build_model(ws, HP['lstm_units'], HP['dropout_rate'], HP['learning_rate'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                              restore_best_weights=True, verbose=1)
        print(f"=== Entraînement (max {HP['epochs']} epochs) ===")
        t0 = time.time()
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=HP['epochs'], batch_size=HP['batch_size'],
                  callbacks=[es], verbose=0)
        model.save(model_path)
        print(f"Entraîné en {(time.time()-t0)/60:.1f} min, sauvegardé.")

    test_full = datapro[val_end:].flatten()
    seed = val_norm[-ws:]
    max_h = max(HORIZONS)
    pred_full = scaler.inverse_transform(predict_recursive(model, seed, max_h)).flatten()

    # Figure : un sous-graphe par horizon
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(14, 5 * len(HORIZONS)))
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, H in zip(axes, HORIZONS):
        test_h = test_full[:H]
        pred_h = pred_full[:H]
        rmse = float(np.sqrt(mean_squared_error(test_h, pred_h)))
        mae  = float(mean_absolute_error(test_h, pred_h))
        print(f"  Horizon {H:>4} h -> RMSE {rmse:8.1f} Wh | MAE {mae:8.1f} Wh")

        x_ctx  = np.arange(val_end - CONTEXT, val_end)
        x_hor  = np.arange(val_end, val_end + H)
        ax.plot(x_ctx, datapro[val_end - CONTEXT:val_end].flatten(),
                color='#2980B9', lw=1.4, label='Historique')
        ax.plot(x_hor, test_h,  color='#1A1A2E', lw=1.8, label='Réel')
        ax.plot(x_hor, pred_h,  color='#E74C3C', lw=2.0, ls='--', label='Prédiction')
        ax.axvline(val_end, color='#7F8C8D', ls=':', lw=1.5)
        ax.set_title(f"{cfg['titre']} — horizon {H} h ({H//24} j) — "
                     f"RMSE={rmse:.0f} Wh, MAE={mae:.0f} Wh", fontweight='bold')
        ax.set_xlabel("Temps [h]"); ax.set_ylabel(cfg['ylabel'])
        ax.legend(loc='upper right'); ax.grid(True, ls='--', alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'oneshot_court_{profil}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")

print("\nTerminé pour tous les profils.")
