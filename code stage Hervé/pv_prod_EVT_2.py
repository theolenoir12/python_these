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
    'window_size':   [96],
    'batch_size':    [64],
    'lstm_units':    [80],
    'dropout_rate':  [0],  # On force des valeurs > 0 pour le futur MC Dropout
    'epochs':        [160],
}

OUTPUT_DIR = 'figures_conso'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
print("Chargement des données...")
df = pd.read_csv('sidelec_roche_plate_csv2.csv', sep=';', header=None)
datapro = df.iloc[:, 2].values.reshape(-1, 1)[0:8760] #1 pour prod et 2 pour conso

n = len(datapro)
train_end = int(n * 0.70)
val_end   = train_end + 4 * 24

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
    #es = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=False)
    
    history = model.fit(X_t, y_t, validation_data=(X_v, y_v), 
                        epochs=hp['epochs'], batch_size=hp['batch_size'], 
                        verbose=2)

    # Inférence récursive sur Validation
    seed = data_train_norm[-hp['window_size']:]
    y_v_pred_norm = predict_recursive(model, seed, len(y_v))
    
    # Dénormalisation
    y_v_pred = scaler.inverse_transform(y_v_pred_norm)
    y_v_true = scaler.inverse_transform(y_v.reshape(-1, 1))
    
    curr_rmse = np.sqrt(mean_squared_error(y_v_true, y_v_pred))
    curr_mae  = mean_absolute_error(y_v_true, y_v_pred)
    
    # ─────────────────────────────────────────────────────────────────
    # PUBLICATION-READY FIGURES  (v4)
    # ─────────────────────────────────────────────────────────────────
    import matplotlib.patches as patches
    from matplotlib.patches import ConnectionPatch
    
    STYLE = {
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize":    14,
        "axes.titlesize":    16,
        "xtick.labelsize":   14,
        "ytick.labelsize":   14,
        "legend.fontsize":   14,
        "axes.linewidth":    1.0,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "axes.grid":         True,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.5,
        "figure.dpi":        300,
    }
    COLORS = {
        "train": "#2166AC",
        "val":   "#D6604D",
        "pred":  "#1A9641",
        "true":  "#252525",
    }
    
    UNIT_SCALE = 1e-3   # W → kW
    TIME_SCALE = 1 / 24 # steps → days
    
    # ── In-sample (train) one-step-ahead predictions ──────────────────
    y_t_pred_norm = model.predict(X_t, verbose=0)
    y_t_pred    = scaler.inverse_transform(y_t_pred_norm) * UNIT_SCALE
    y_t_true_kw = scaler.inverse_transform(y_t.reshape(-1, 1)) * UNIT_SCALE
    y_v_pred_kw = y_v_pred * UNIT_SCALE   # already inverse-transformed W, → kW
    y_v_true_kw = y_v_true * UNIT_SCALE
    
    train_mse  = mean_squared_error(y_t_true_kw, y_t_pred)
    # curr_rmse is in W (raw), convert for display
    curr_rmse_kw = curr_rmse * UNIT_SCALE
    
    # ── Loss histories ─────────────────────────────────────────────────
    # Train axis : MSE in kW²  (scale² because MSE is quadratic)
    scale_factor_kw = (scaler.data_max_[0] - scaler.data_min_[0]) * UNIT_SCALE
    mse_train_hist  = np.array(history.history["loss"])  * scale_factor_kw**2
    
    # Val axis   : RMSE in kW  (sqrt of MSE, then rescale)
    rmse_val_hist   = np.sqrt(np.array(history.history["val_loss"])) * scale_factor_kw
    
    best_epoch    = int(np.argmin(rmse_val_hist)) + 1
    best_val_rmse = rmse_val_hist[best_epoch - 1]
    epochs_arr    = np.arange(1, len(mse_train_hist) + 1)
    
    # ── FIGURE 1 : Learning curves (RMSE in kW) + best epoch ──────────
    with plt.rc_context(STYLE):
        fig1, ax = plt.subplots(figsize=(7.0, 4.5))

        epochs_arr = np.arange(1, len(mse_train_hist) + 1)
    
        ax.semilogy(epochs_arr, mse_train_hist,
                    color=COLORS["train"], linewidth=1.4, label="Training MSE")
    
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss function (MSE [kW²])")
        ax.set_title("Learning curve — LSTM model")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x:.3g}"
        ))
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
        plt.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig1_loss.pdf"), bbox_inches="tight")
        fig1.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig1_loss.png"), bbox_inches="tight", dpi=300)
        #plt.close(fig1)
    
    # ── FIGURE 2 : Training set — full view + white-backed inset zoom ──
    ZOOM = 96

    with plt.rc_context(STYLE):
        fig2, ax_main = plt.subplots(figsize=(7.0, 4.5))
    
        t_days = np.arange(len(y_t_true_kw)) * TIME_SCALE
    
        # ── Graphique principal ────────────────────────────────────────────────
        ax_main.plot(t_days, y_t_true_kw.flatten(), color=COLORS["true"], linewidth=0.9, label="Observed")
        ax_main.plot(t_days, y_t_pred.flatten(), color=COLORS["pred"], linewidth=0.9, linestyle="--", label="One-step-ahead prediction")
        ax_main.set_xlabel("Time [d]")
        ax_main.set_ylabel("Power [kW]")
        ax_main.set_title("Training set — observed vs. one-step-ahead prediction")
        ax_main.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper right")
        ax_main.text(0.02, 0.9, f"MSE = {train_mse:.3f} kW²", transform=ax_main.transAxes, fontsize=13,
                     verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6"))
    
        # ── Calcul des limites du Zoom ────────────────────────────────────────
        zoom_start = len(y_t_true_kw) // 3
        zoom_end   = zoom_start + ZOOM
        t0_z, t1_z = t_days[zoom_start], t_days[zoom_end - 1]
        
        y_tr_z = y_t_true_kw.flatten()[zoom_start:zoom_end]
        y_pr_z = y_t_pred.flatten()[zoom_start:zoom_end]
        y0_z, y1_z = min(y_tr_z.min(), y_pr_z.min()) * 0.97, max(y_tr_z.max(), y_pr_z.max()) * 1.03
    
        # ── Inset position (axes-fraction coords) ─────────────────────
        ins_x, ins_y, ins_w, ins_h = 0.50, 0.30, 0.46, 0.44
    
        # Marges pour que le rectangle blanc déborde sous les textes (titre, numéros)
        pad_gauche, pad_droite, pad_bas, pad_haut = 0.045, 0.0, 0.06, 0.00
        bg_x = ins_x - pad_gauche
        bg_y = ins_y - pad_bas
        bg_w = ins_w + pad_gauche + pad_droite
        bg_h = ins_h + pad_bas + pad_haut
    
        # 1. Grand rectangle blanc dessiné en arrière-plan de l'encart
        rect_white = patches.Rectangle(
            (bg_x, bg_y), bg_w, bg_h,
            transform=ax_main.transAxes,
            facecolor="white", edgecolor="crimson", linewidth=1.5, zorder=3, clip_on=False
        )
        ax_main.add_patch(rect_white)
    
        # 2. Inset axes placé sur le rectangle blanc
        ax_inset = ax_main.inset_axes([ins_x, ins_y, ins_w, ins_h], zorder=4)
        ax_inset.set_facecolor("white")
        ax_inset.plot(t_days[zoom_start:zoom_end],
                      y_t_true_kw.flatten()[zoom_start:zoom_end],
                      color=COLORS["true"], linewidth=1.2)
        ax_inset.plot(t_days[zoom_start:zoom_end],
                      y_t_pred.flatten()[zoom_start:zoom_end],
                      color=COLORS["pred"], linewidth=1.2, linestyle="--")
        ax_inset.set_xlim(t0_z, t1_z)
        ax_inset.set_ylim(y0_z, y1_z)
        # ax_inset.set_yticks(np.arange(0,26,5))
        ax_inset.tick_params(labelsize=12)
        ax_inset.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        for spine in ax_inset.spines.values():
            spine.set_edgecolor("0.3")
            spine.set_linewidth(1.2)
    
        # 3. Cadre de sélection rouge sur le plot principal
        rect_sel = patches.Rectangle(
            (t0_z, y0_z), t1_z - t0_z, y1_z - y0_z,
            linewidth=1.5, edgecolor="crimson", facecolor="none", zorder=5
        )
        ax_main.add_patch(rect_sel)
    
        # 4. Lignes de connexion alignées sur le grand rectangle blanc
        con_kw = dict(color="crimson", linewidth=1.2, zorder=5)
        ax_main.annotate("",
            xy=(bg_x, bg_y + bg_h), xycoords=ax_main.transAxes,
            xytext=(t0_z, y1_z),       textcoords=ax_main.transData,
            arrowprops=dict(arrowstyle="-", **con_kw))
        ax_main.annotate("",
            xy=(bg_x + bg_w, bg_y), xycoords=ax_main.transAxes,
            xytext=(t1_z, y0_z),       textcoords=ax_main.transData,
            arrowprops=dict(arrowstyle="-", **con_kw))
    
        plt.tight_layout()
        fig2.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig2_train.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig2_train.png"), bbox_inches="tight", dpi=300)
        #plt.close(fig2)
    
    # ── FIGURE 3 : Validation set — recursive prediction ──────────────
    with plt.rc_context(STYLE):
        fig3, ax = plt.subplots(figsize=(7.0, 4.5))
    
        v_days = np.arange(len(y_v_true_kw)) * TIME_SCALE + t_days[-1]
    
        ax.plot(v_days, y_v_true_kw.flatten(),
                color=COLORS["true"], linewidth=1.0, label="Observed")
        ax.plot(v_days, y_v_pred_kw.flatten(),
                color=COLORS["val"], linewidth=1.0, linestyle="--",
                label="Recursive multi-step prediction")
        ax.set_xlabel("Time [d]")
        ax.set_ylabel("Power [kW]")
        ax.set_title("Validation set — observed vs. recursive multi-step prediction")
        ax.set_xticks(np.arange(int(min(v_days))+1,int(max(v_days))+1,1))
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper right")
        ax.text(0.02, 0.04,
                f"RMSE = {curr_rmse_kw:.3f} kW",
                transform=ax.transAxes, fontsize=13,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.6", alpha=1.0))
        plt.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig3_val.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(OUTPUT_DIR, f"{run_id}_fig3_val.png"), bbox_inches="tight", dpi=300)
        #plt.close(fig3)
        plt.show()

    # ─────────────────────────────────────────────────────────────────

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