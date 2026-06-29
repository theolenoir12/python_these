# -*- coding: utf-8 -*-
"""
Étape 1 : Recherche d'hyperparamètres LSTM (incluant le Dropout)
- Sauvegarde des courbes de Loss pour chaque run
- Export des résultats vers un fichier Excel (.xlsx)
- Validation basée sur la performance récursive (SoH)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# =============================================================================
# 1. CONFIGURATION ET GRILLE
# =============================================================================
HYPERPARAMS_GRID = {
    'learning_rate': [1e-3],
    'window_size':   [100],
    'batch_size':    [64],
    'lstm_units':    [100, 150],
    'dropout_rate':  [0.05, 0.1],  # On force des valeurs > 0 pour le futur MC Dropout
    'epochs':        [1000],
}

OUTPUT_DIR = 'hyperparam_search_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
print("Chargement des données...")
df = pd.read_csv('SoH_ely.csv', sep=';', header=None)
datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]

n = len(datapro)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

# Scaler ajusté sur train uniquement
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_norm = scaler.fit_transform(datapro[:train_end])
data_val_norm   = scaler.transform(datapro[train_end:val_end])

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# =============================================================================
# 3. MODÈLE ET PRÉDICTION RÉCURSIVE
# =============================================================================
def build_model(ws, units, dr, lr):
    model = tf.keras.Sequential([
        Input(shape=(ws, 1)),
        LSTM(units),
        Dropout(dr),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def predict_recursive(model, seed_window, n_steps):
    curr_window = seed_window.copy().reshape(1, -1, 1)
    preds = []
    for _ in range(n_steps):
        p = model.predict(curr_window, verbose=0)
        preds.append(p[0, 0])
        # Mise à jour de la fenêtre : on enlève le 1er, on ajoute la prédiction
        curr_window = np.concatenate([curr_window[:, 1:, :], p.reshape(1, 1, 1)], axis=1)
    return np.array(preds).reshape(-1, 1)

# =============================================================================
# 4. BOUCLE DE RECHERCHE
# =============================================================================
keys = list(HYPERPARAMS_GRID.keys())
combos = list(itertools.product(*[HYPERPARAMS_GRID[k] for k in keys]))
results_list = []

best_rmse = np.inf
best_cfg = None

for idx, values in enumerate(combos, start=1):
    hp = dict(zip(keys, values))
    run_id = f"run_{idx}_ws{hp['window_size']}_u{hp['lstm_units']}_dr{hp['dropout_rate']}"
    print(f"\n>>> {run_id} ({idx}/{len(combos)})")

    # Séquences
    X_t, y_t = create_sequences(data_train_norm, hp['window_size'])
    # Pour la val, on inclut la fin du train pour le contexte
    val_context = np.vstack((data_train_norm[-hp['window_size']:], data_val_norm))
    X_v, y_v = create_sequences(val_context, hp['window_size'])

    # Entraînement
    model = build_model(hp['window_size'], hp['lstm_units'], hp['dropout_rate'], hp['learning_rate'])
    es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), 
                        epochs=hp['epochs'], batch_size=hp['batch_size'], 
                        callbacks=[es], verbose=0)

    # Inférence récursive sur Validation
    seed = data_train_norm[-hp['window_size']:]
    y_v_pred_norm = predict_recursive(model, seed, len(y_v))
    
    # Dénormalisation
    y_v_pred = scaler.inverse_transform(y_v_pred_norm)
    y_v_true = scaler.inverse_transform(y_v.reshape(-1, 1))
    
    curr_rmse = np.sqrt(mean_squared_error(y_v_true, y_v_pred))
    curr_mae  = mean_absolute_error(y_v_true, y_v_pred)

    # --- VISUALISATION (Même syntaxe que ton code précédent) ---
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Plot 1 : Courbe de Loss
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(history.history['loss'], label='Train Loss', color='#2C3E50')
    ax_loss.plot(history.history['val_loss'], label='Val Loss', color='#E67E22')
    ax_loss.set_title(f"Loss Evolution - {run_id}")
    ax_loss.set_yscale('log')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot 2 : Prédiction Récursive sur Validation
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.plot(y_v_true, label='Réel (Val)', color='black', linewidth=1.5)
    ax_pred.plot(y_v_pred, label='Prédit (Rec)', color='crimson', linestyle='--')
    ax_pred.set_title(f"Recursive Val - RMSE: {curr_rmse:.5f}")
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)

    # Plot 3 : Zoom sur la fin ou erreurs (optionnel, ici texte des params)
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis('off')
    txt_params = "\n".join([f"{k}: {v}" for k, v in hp.items()])
    ax_txt.text(0.5, 0.5, f"Params:\n{txt_params}\n\nFinal RMSE: {curr_rmse:.6f}", 
                ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{run_id}.png"), dpi=120)
    plt.close(fig)

    # Log des résultats
    res = hp.copy()
    res.update({
        'run_id': run_id,
        'rmse_val': curr_rmse,
        'mae_val': curr_mae,
        'epochs_reached': len(history.history['loss'])
    })
    results_list.append(res)

    if curr_rmse < best_rmse:
        best_rmse = curr_rmse
        best_cfg = hp
        model.save(os.path.join(OUTPUT_DIR, 'best_model_step1.keras'))

# =============================================================================
# 5. EXPORT EXCEL ET RÉSUMÉ
# =============================================================================
df_results = pd.DataFrame(results_list)
excel_path = os.path.join(OUTPUT_DIR, 'resultats_hyperparametres.xlsx')
df_results.to_excel(excel_path, index=False)

print("\n" + "="*50)
print(f"RECHERCHE TERMINÉE")
print(f"Meilleur RMSE : {best_rmse:.6f}")
print(f"Fichier Excel sauvegardé : {excel_path}")
print("="*50)