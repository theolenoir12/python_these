# -*- coding: utf-8 -*-
"""
Étape 1 : Recherche d'hyperparamètres LSTM pour la prédiction de profils de
puissance (production PV et consommation), avec validation RÉCURSIVE.

Même méthode que le pronostic de SoH (cf. dossier "Pronostic SoH",
pv_soh_EVT.py) appliquée ici aux profils horaires de puissance :
    - architecture  Input(ws,1) -> LSTM(units) -> Dropout(dr) -> Dense(1)
    - loss MSE, optimiseur Adam, EarlyStopping sur val_loss
    - sélection des hyperparamètres qui minimisent la RMSE de validation
      obtenue par prédiction RÉCURSIVE (et non sur un simple pas-à-pas).

La grille est centrée autour de la "config_12" du stage (lr=1e-3, ws=96,
bs=64, units=80, ep=160), qui minimisait déjà la loss de validation dans
hyperparametres_vs_performances_production.xlsx. On ajoute une variation de
dropout car le MC Dropout (étape 2) exige dropout > 0.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')

# Profils à traiter : nom -> (colonne du CSV, libellé)
PROFILS = {
    'production':   {'col': 1, 'titre': 'Production PV'},
    'consommation': {'col': 2, 'titre': 'Consommation'},
}

# Grille d'hyperparamètres (centrée sur config_12 du stage)
HYPERPARAMS_GRID = {
    'learning_rate': [1e-3],
    'window_size':   [96],
    'batch_size':    [64],
    'lstm_units':    [80, 100],
    'dropout_rate':  [0.0, 0.05, 0.1],   # 0 = référence ; >0 nécessaire au MC Dropout
    'epochs':        [160],
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_recherche_hp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. OUTILS
# =============================================================================
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_model(ws, units, dr, lr):
    model = tf.keras.Sequential([
        Input(shape=(ws, 1)),
        LSTM(units),
        Dropout(dr),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def predict_recursive(model, seed_window, n_steps):
    """Prédiction récursive déterministe : le réseau s'auto-alimente."""
    curr = seed_window.copy().reshape(1, -1, 1)
    preds = []
    for _ in range(n_steps):
        p = model.predict(curr, verbose=0)
        preds.append(p[0, 0])
        curr = np.concatenate([curr[:, 1:, :], p.reshape(1, 1, 1)], axis=1)
    return np.array(preds).reshape(-1, 1)

# =============================================================================
# 3. BOUCLE DE RECHERCHE (par profil)
# =============================================================================
keys   = list(HYPERPARAMS_GRID.keys())
combos = list(itertools.product(*[HYPERPARAMS_GRID[k] for k in keys]))

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

    results_list = []
    best_rmse, best_cfg = np.inf, None

    for idx, values in enumerate(combos, start=1):
        hp = dict(zip(keys, values))
        run_id = f"{profil}_ws{hp['window_size']}_u{hp['lstm_units']}_dr{hp['dropout_rate']}"
        print(f"\n>>> {run_id} ({idx}/{len(combos)})")

        ws = hp['window_size']
        X_t, y_t = create_sequences(train_norm, ws)
        # Contexte = fin du train + val pour pouvoir prédire dès le début du val
        val_context = np.vstack((train_norm[-ws:], val_norm))
        X_v, y_v = create_sequences(val_context, ws)

        model = build_model(ws, hp['lstm_units'], hp['dropout_rate'], hp['learning_rate'])
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        history = model.fit(X_t, y_t, validation_data=(X_v, y_v),
                            epochs=hp['epochs'], batch_size=hp['batch_size'],
                            callbacks=[es], verbose=0)

        # Validation RÉCURSIVE
        seed = train_norm[-ws:]
        y_v_pred = scaler.inverse_transform(predict_recursive(model, seed, len(y_v)))
        y_v_true = scaler.inverse_transform(y_v.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_v_true, y_v_pred))
        mae  = mean_absolute_error(y_v_true, y_v_pred)

        # Figure récapitulative du run
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax_loss = fig.add_subplot(gs[0, 0])
        ax_loss.plot(history.history['loss'], label='Train', color='#2C3E50')
        ax_loss.plot(history.history['val_loss'], label='Val', color='#E67E22')
        ax_loss.set_title(f"Loss - {run_id}"); ax_loss.set_yscale('log')
        ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

        ax_pred = fig.add_subplot(gs[0, 1])
        ax_pred.plot(y_v_true, label='Réel (Val)', color='black', lw=1.2)
        ax_pred.plot(y_v_pred, label='Prédit (récursif)', color='crimson', ls='--')
        ax_pred.set_title(f"Validation récursive - RMSE: {rmse:.1f} Wh")
        ax_pred.legend(); ax_pred.grid(True, alpha=0.3)

        ax_txt = fig.add_subplot(gs[1, :]); ax_txt.axis('off')
        txt = "\n".join(f"{k}: {v}" for k, v in hp.items())
        ax_txt.text(0.5, 0.5, f"{txt}\n\nVal RMSE (rec): {rmse:.2f} Wh\nVal MAE (rec): {mae:.2f} Wh",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{run_id}.png"), dpi=120)
        plt.close(fig)

        res = hp.copy()
        res.update({'profil': profil, 'run_id': run_id,
                    'val_rmse_rec': rmse, 'val_mae_rec': mae,
                    'epochs_reached': len(history.history['loss'])})
        results_list.append(res)

        if rmse < best_rmse:
            best_rmse, best_cfg = rmse, hp
            model.save(os.path.join(OUTPUT_DIR, f'best_model_{profil}.keras'))

    df_results = pd.DataFrame(results_list)
    excel_path = os.path.join(OUTPUT_DIR, f'resultats_hp_{profil}.xlsx')
    df_results.to_excel(excel_path, index=False)

    print(f"\n{'='*60}")
    print(f"{cfg['titre']} — meilleure Val RMSE (rec) : {best_rmse:.2f} Wh")
    print(f"Config retenue : {best_cfg}")
    print(f"Résultats : {excel_path}")
    print('='*60)
