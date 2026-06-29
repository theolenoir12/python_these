# -*- coding: utf-8 -*-
"""
Étape 2 : Prédiction récursive LSTM + Monte-Carlo Dropout pour les profils de
puissance (production PV et consommation).

Méthode identique au pronostic de SoH (Pronostic SoH/pv_soh_mcdropout.py) :
Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation".
    - architecture identique à l'étape 1, loss MSE
    - Dropout actif à l'inférence (training=True) -> approximation variationnelle
    - T trajectoires récursives stochastiques
    - prédiction moyenne + intervalle de confiance (quantiles)

Hyperparamètres : config_12 du stage (lr=1e-3, ws=96, bs=64, units=80, ep=160),
qui minimise la loss de validation dans hyperparametres_vs_performances_production.xlsx.
Le dropout, nul dans config_12, est fixé à 0.05 (comme le pipeline SoH) car le
MC Dropout exige dropout > 0.
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
    'dropout_rate':  0.05,   # config_12 = 0 ; >0 imposé par le MC Dropout
    'epochs':        160,
}
T    = 500       # nombre de passages MC Dropout
CONF = 0.95      # niveau de confiance de l'intervalle

# Horizon de prévision récursif, STRICTEMENT 96 h (4 jours) : seul horizon qui
# a du sens pour ces profils oscillatoires (cf. dérive sur horizon long).
HORIZON = 96

# Horizons intermédiaires (h) pour la quantification de l'énergie au bus.
HORIZONS_ENERGIE = [48, 96]   # 2 jours, 4 jours

PROFILS = {
    'production':   {'col': 1, 'titre': 'Production PV',  'ylabel': 'Production [Wh]'},
    'consommation': {'col': 2, 'titre': 'Consommation',   'ylabel': 'Consommation [Wh]'},
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'resultats_mc_dropout')
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

def mc_dropout_predict(model, seed_window, n_steps, T):
    """T prédictions récursives indépendantes, Dropout actif (training=True)."""
    windows = np.tile(seed_window.flatten(), (T, 1))      # (T, ws)
    samples = np.zeros((T, n_steps), dtype=np.float32)
    for t in range(n_steps):
        x_batch = windows[:, :, np.newaxis]               # (T, ws, 1)
        preds = model(x_batch, training=True).numpy().flatten()
        samples[:, t] = preds
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds
    return samples

# ── Boucle par profil ────────────────────────────────────────────────────────
for profil, cfg in PROFILS.items():
    print(f"\n{'#'*70}\n# PROFIL : {cfg['titre']} (colonne {cfg['col']})\n{'#'*70}")

    df = pd.read_csv(DATA_PATH, sep=';', header=None)
    datapro = df.iloc[:, cfg['col']].values.reshape(-1, 1).astype(float)
    n = len(datapro)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_raw = datapro[:train_end]
    val_raw   = datapro[train_end:val_end]
    test_full = datapro[val_end:].flatten()

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(train_raw)
    val_norm   = scaler.transform(val_raw)

    ws = HP['window_size']
    X_train, y_train = create_sequences(train_norm, ws)
    val_context = np.vstack((train_norm[-ws:], val_norm))
    X_val, y_val = create_sequences(val_context, ws)

    print(f"Total: {n} | Train seq: {len(X_train)} | Val seq: {len(X_val)} | Test: {len(test_full)}")

    # Entraînement du modèle final
    tf.keras.backend.clear_session()
    model = build_model(ws, HP['lstm_units'], HP['dropout_rate'], HP['learning_rate'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                          restore_best_weights=True, verbose=1)
    print(f"=== Entraînement (max {HP['epochs']} epochs) ===")
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=HP['epochs'], batch_size=HP['batch_size'],
                        callbacks=[es], verbose=0)
    model.save(os.path.join(OUTPUT_DIR, f'modele_mc_dropout_{profil}.keras'))
    print(f"Terminé en {(time.time()-t0)/60:.1f} min — "
          f"{len(history.history['loss'])} epochs effectives")

    # Horizon de prévision
    n_steps = len(test_full) if HORIZON is None else min(HORIZON, len(test_full))
    test_raw = test_full[:n_steps]

    # MC Dropout (seed = fin du set de validation)
    print(f"=== MC Dropout : T={T} trajectoires sur {n_steps} pas ===")
    t0 = time.time()
    seed = val_norm[-ws:]
    samples = mc_dropout_predict(model, seed, n_steps, T)
    print(f"Terminé en {time.time()-t0:.1f} s")

    # Dénormalisation + statistiques
    samples_real = np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten()
                             for s in samples])
    mean_pred = samples_real.mean(axis=0)
    alpha = 1 - CONF
    ci_lower = np.quantile(samples_real, alpha / 2,     axis=0)
    ci_upper = np.quantile(samples_real, 1 - alpha / 2, axis=0)

    rmse = float(np.sqrt(mean_squared_error(test_raw, mean_pred)))
    mae  = float(mean_absolute_error(test_raw, mean_pred))
    coverage = float(np.mean((test_raw >= ci_lower) & (test_raw <= ci_upper)))
    print(f"Test RMSE : {rmse:.2f} Wh | Test MAE : {mae:.2f} Wh | "
          f"Couverture IC {int(CONF*100)}% : {coverage*100:.1f}%")

    # ── Énergie cumulée au bus sur l'horizon (réel vs prédit + IC MC) ────────
    # Les valeurs étant des Wh par pas horaire, l'énergie cumulée = somme.
    energie = {}
    print("  Énergie cumulée au bus :")
    for h in HORIZONS_ENERGIE:
        if n_steps < h:
            continue
        e_real    = float(test_raw[:h].sum())
        e_pred    = float(mean_pred[:h].sum())
        e_samples = samples_real[:, :h].sum(axis=1)        # distribution MC
        e_lo = float(np.quantile(e_samples, alpha / 2))
        e_hi = float(np.quantile(e_samples, 1 - alpha / 2))
        err   = e_pred - e_real
        err_pct = 100 * err / e_real if e_real != 0 else np.nan
        energie[h] = dict(e_real=e_real, e_pred=e_pred, e_lo=e_lo, e_hi=e_hi,
                          err=err, err_pct=err_pct, e_std=float(e_samples.std()))
        print(f"    {h:>3} h ({h//24} j) : réel={e_real/1000:8.1f} kWh | "
              f"prédit={e_pred/1000:8.1f} kWh | écart={err/1000:+7.1f} kWh "
              f"({err_pct:+.1f}%) | IC95%=[{e_lo/1000:.1f}, {e_hi/1000:.1f}] kWh")

    # ── Figure diagnostique ─────────────────────────────────────────────────
    C = dict(train_val='#2980B9', test='#1A1A2E', mean='#E74C3C',
             ci='#F1948A', traj='#AED6F1', loss='#2ECC71')
    CONTEXT = 336   # 2 semaines d'historique affichées avant la prévision
    x_ctx  = np.arange(val_end - CONTEXT, val_end)
    x_test = np.arange(val_end, val_end + n_steps)

    fig = plt.figure(figsize=(16, 16))
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.6, 1.6, 0.8], hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_ctx, datapro[val_end - CONTEXT:val_end].flatten(),
             color=C['train_val'], lw=1.2, label='Historique (Val)')
    ax1.plot(x_test, test_raw,  color=C['test'], lw=1.2, label='Réel (Test)')
    ax1.plot(x_test, mean_pred, color=C['mean'], lw=1.6, ls='--', label='Prédiction moyenne MC')
    ax1.fill_between(x_test, ci_lower, ci_upper, color=C['ci'], alpha=0.4,
                     label=f'IC {int(CONF*100)}%')
    ax1.axvline(val_end, color='#7F8C8D', ls=':', lw=1.5)
    ax1.set_title(f"MC Dropout — {cfg['titre']} — Vue d'ensemble", fontweight='bold')
    ax1.set_ylabel(cfg['ylabel']); ax1.legend(loc='upper left')
    ax1.grid(True, ls='--', alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    for i in range(min(30, T)):
        ax2.plot(x_test, samples_real[i], color=C['traj'], lw=0.5, alpha=0.3)
    ax2.plot(x_test, test_raw,  color=C['test'], lw=1.6, label='Réel', zorder=5)
    ax2.plot(x_test, mean_pred, color=C['mean'], lw=1.8, ls='--',
             label=f'Moyenne MC (RMSE={rmse:.0f} Wh)', zorder=6)
    ax2.fill_between(x_test, ci_lower, ci_upper, color=C['ci'], alpha=0.35,
                     label=f'IC {int(CONF*100)}% | Couverture={coverage*100:.0f}%')
    ax2.set_title("Zoom Test — trajectoires MC et incertitude", fontweight='bold')
    ax2.set_xlabel("Temps [h]"); ax2.set_ylabel(cfg['ylabel'])
    ax2.legend(loc='upper right'); ax2.grid(True, ls='--', alpha=0.4)

    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[2.2, 1])
    ax3 = fig.add_subplot(gs_bot[0])
    ax3.plot(history.history['loss'], color=C['loss'], lw=1.5, label='Train Loss (MSE)')
    ax3.plot(history.history['val_loss'], color='#E67E22', lw=1.2, label='Val Loss (MSE)')
    ax3.set_yscale('log'); ax3.set_title("Convergence (MSE)", fontweight='bold')
    ax3.set_xlabel("Époques"); ax3.set_ylabel("MSE (log)")
    ax3.legend(); ax3.grid(True, ls='--', alpha=0.4)

    ax4 = fig.add_subplot(gs_bot[1]); ax4.axis('off')
    table_data = [
        ['Paramètre / Métrique', 'Valeur'],
        ['Window Size',   str(HP['window_size'])],
        ['LSTM Units',    str(HP['lstm_units'])],
        ['Dropout Rate',  str(HP['dropout_rate'])],
        ['MC Samples (T)', str(T)],
        ['Horizon (h)',   str(n_steps)],
        ['', ''],
        ['Test RMSE (Wh)', f"{rmse:.1f}"],
        ['Test MAE (Wh)',  f"{mae:.1f}"],
        [f'Couverture IC {int(CONF*100)}%', f"{coverage*100:.1f}%"],
    ]
    for h in HORIZONS_ENERGIE:
        if h in energie:
            e = energie[h]
            table_data.append([f'Énergie {h//24}j réel (kWh)',  f"{e['e_real']/1000:.1f}"])
            table_data.append([f'Énergie {h//24}j écart (kWh)', f"{e['err']/1000:+.1f} ({e['err_pct']:+.1f}%)"])
    tbl = ax4.table(cellText=table_data, loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.4)
    for col in range(2):
        tbl[(0, col)].set_facecolor('#2C3E50')
        tbl[(0, col)].set_text_props(color='white', fontweight='bold')

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'mc_dropout_{profil}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")

print("\nTerminé pour tous les profils.")
