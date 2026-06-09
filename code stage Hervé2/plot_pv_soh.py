# -*- coding: utf-8 -*-
"""
Phase de TEST LSTM avec Monte-Carlo Dropout — version optimisée.
- Entraînement sur 650 epochs fixes avec ModelCheckpoint (pas d'EarlyStopping)
- MC Dropout vectorisé : n_steps appels TF au lieu de MC_SAMPLES × n_steps
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib as mpl

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
import matplotlib.ticker as ticker

def load_for_mc_dropout(path): #pour recharger un nouveau modèle
    model = tf.keras.models.load_model(path)
    # On force le dropout à l'inférence
    return model

model = load_for_mc_dropout('resultats_test/model_dr_0.01_rec_0.05.keras')

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
recurrent_dropout = 0.2
CONFIDENCE_LVLS   = [95,90]
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

# ─────────────────────────────────────────────────────────────────────────────
# STYLE GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':         'serif',
    'font.serif':          ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':    'cm',
 
    # Tailles augmentées pour lisibilité dans un PDF de revue
    'font.size':           11,
    'axes.titlesize':      11,
    'axes.labelsize':      11,
    'xtick.labelsize':     9.5,
    'ytick.labelsize':     9.5,
    'legend.fontsize':     9.5,
    'figure.titlesize':    12,
 
    'lines.linewidth':     1.4,
    'patch.linewidth':     0.5,
    'axes.linewidth':      0.7,
    'xtick.major.width':   0.7,
    'ytick.major.width':   0.7,
    'xtick.minor.width':   0.4,
    'ytick.minor.width':   0.4,
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'xtick.minor.size':    2.5,
    'ytick.minor.size':    2.5,
 
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
 
    'axes.grid':           True,
    'grid.linewidth':      0.4,
    'grid.alpha':          0.35,
    'grid.linestyle':      '--',
    'grid.color':          '#888888',
 
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.08,
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
    'legend.framealpha':   0.93,
    'legend.edgecolor':    '#AAAAAA',
    'legend.fancybox':     False,
})
 
# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BLUE    = '#1F4E79'   # train + validation
ORANGE  = '#C05000'   # test réel  — trait épais, contraste fort
CRIMSON = '#A01010'   # prédiction moyenne MC
PURPLE  = '#4B3F72'   # médiane MC
SAGE    = '#2D6A4F'   # écart-type axe secondaire
CI95_FC = '#BDD7EE'   # IC 95 % — légèrement plus saturé qu'avant
CI90_FC = '#7EB8DA'   # IC 90 % — nettement plus foncé pour se distinguer
SPLIT_C = '#444444'
 
W_COL   = 7.2         # largeur figure simple colonne (inches)
 
x_tv   = np.arange(train_val_end)
x_test = np.arange(train_val_end, n)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — VUE D'ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(W_COL, 4.0))
 
# ── zones de fond (plus subtiles)
ax1.axvspan(0,             train_val_end - 1, color='#EBF3FB', alpha=0.60, lw=0)
ax1.axvspan(train_val_end - 1, n - 1,        color='#FDF0E8', alpha=0.70, lw=0)
 
# ── IC 95 % uniquement en vue d'ensemble (sinon trop chargé)
ax1.fill_between(x_test, *intervals[95],
                 color=CI95_FC, alpha=0.75, lw=0, zorder=2, label='IC 95 %')
 
# ── séries
ax1.plot(x_tv,   train_val_raw,
         color=BLUE,    lw=1.4, zorder=3,
         label='Training + validation (85 %)')
ax1.plot(x_test, test_true,
         color=ORANGE,  lw=1.8, zorder=5,      # plus épais : visible malgré échelle
         label='Test — valeurs réelles (15 %)')
ax1.plot(x_test, mean_pred,
         color=CRIMSON, lw=1.6, ls='--', zorder=4, dashes=(5, 2.5),
         label=r'Prédiction MC (moyenne, $N_\mathrm{MC}=$'f'{MC_SAMPLES})')
 
# ── ligne de séparation
ax1.axvline(train_val_end, color=SPLIT_C, ls=':', lw=1.0, zorder=6)
 
# ── étiquettes de région dans le corps du graphe (pas au-dessus du titre)
y_label = ax1.get_ylim()[0] if False else None   # calculé après autoscale
# on les place après avoir fixé les limites
ax1.set_xlim([0, n - 1])
ax1.set_xlabel('Temps (jours)', labelpad=5)
ax1.set_ylabel('State of Health (SoH)', labelpad=6)
 
fig1.canvas.draw()   # force autoscale pour récupérer ylim
ylo, yhi = ax1.get_ylim()
ymid = ylo + (yhi - ylo) * 0.96   # proche du haut du graphe, à l'intérieur
 
ax1.text(train_val_end * 0.50, ymid,
         'Training + validation',
         ha='center', va='top', fontsize=9,
         color='#1F4E79', style='italic',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))
ax1.text(train_val_end + (n - train_val_end) * 0.50, ymid,
         'Test',
         ha='center', va='top', fontsize=9,
         color='#7A3010', style='italic',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))
 
ax1.set_title(
    'Prédiction récursive LSTM — Monte-Carlo Dropout',
    loc='left', pad=6, fontweight='normal'
)
ax1.legend(loc='lower left',
           borderpad=0.6, labelspacing=0.35, handlelength=2.4,
           handletextpad=0.5)
 
fig1.tight_layout()
fig1.savefig('resultats_test/fig1_overview.pdf')
print('fig1_overview.pdf  ✓')
 
 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ZOOM ZONE DE TEST
# ═════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(W_COL, 4.0))
 
# ── IC (95 % fond, 90 % devant — écart visible entre les deux teintes)
ax2.fill_between(x_test, *intervals[95],
                 color=CI95_FC, alpha=0.70, lw=0, zorder=1, label='IC 95 %')
ax2.fill_between(x_test, *intervals[90],
                 color=CI90_FC, alpha=0.80, lw=0, zorder=2, label='IC 90 %')
 
# ── séries principales
ax2.plot(x_test, test_true,
         color=ORANGE,  lw=2.0, zorder=5, label='Valeurs réelles')
ax2.plot(x_test, mean_pred,
         color=CRIMSON, lw=1.6, ls='--', zorder=6, dashes=(5, 2.5),
         label=f'Moyenne MC  (RMSE = {rmse_test:.4f})')
ax2.plot(x_test, median_pred,
         color=PURPLE,  lw=1.0, ls=':', zorder=4, label='Médiane MC')
 
# ── axe secondaire : écart-type MC — trait plein fin, couleur distincte
ax2b = ax2.twinx()
ax2b.plot(x_test, std_pred,
          color=SAGE, lw=1.2, ls=(0, (3, 1, 1, 1)), alpha=0.85, zorder=3,
          label=r'Écart-type MC $\hat{\sigma}$')
ax2b.set_ylabel(r'Écart-type MC $\hat{\sigma}$',
                color=SAGE, fontsize=10, labelpad=6)
ax2b.tick_params(axis='y', labelcolor=SAGE, labelsize=9,
                 direction='in', width=0.6)
ax2b.set_ylim(bottom=0)
ax2b.spines['right'].set_edgecolor(SAGE)
ax2b.spines['right'].set_linewidth(0.7)
ax2b.yaxis.set_minor_locator(ticker.AutoMinorLocator())
 
# ── encart métriques — police fixe, fond blanc opaque
ax2.text(0.017, 0.06,
         f'RMSE = {rmse_test:.5f}\nMAE  \u2009= {mae_test:.5f}',
         transform=ax2.transAxes,
         fontsize=9, va='bottom', family='monospace',
         bbox=dict(boxstyle='round,pad=0.35', fc='white',
                   ec='#888888', lw=0.6, alpha=0.96))
 
# ── légende sur une seule colonne, à droite de l'encart métriques
lines_a, labs_a = ax2.get_legend_handles_labels()
lines_b, labs_b = ax2b.get_legend_handles_labels()
ax2.legend(lines_a + lines_b, labs_a + labs_b,
           ncol=1, loc='upper right',
           borderpad=0.6, labelspacing=0.38, handlelength=2.4,
           handletextpad=0.5)
 
ax2.set_xlim([x_test[0], x_test[-1]])
ax2.set_xlabel('Temps (jours)', labelpad=5)
ax2.set_ylabel('State of Health (SoH)', labelpad=6)
ax2.set_title(
    f'Zoom — zone de test  |  {MC_SAMPLES} passes MC',
    loc='left', pad=6, fontweight='normal'
)
 
fig2.tight_layout()
fig2.savefig('resultats_test/fig2_zoom.pdf')
print('fig2_zoom.pdf  ✓')
 
plt.show()
 
