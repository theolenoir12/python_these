# -*- coding: utf-8 -*-
"""
Figures propres (qualité publication) à HORIZON 2 JOURS (48 h), par profil.

Deux figures par profil, à partir des modèles MC Dropout entraînés :
    - <profil>_prediction.pdf : contexte + prévision moyenne vs réel (vue prévision)
    - <profil>_mcdropout.pdf  : zoom 48 h avec trajectoires MC + bande IC 95 %

Graine fixée -> figures reproductibles.
"""

import os, random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ── Reproductibilité ─────────────────────────────────────────────────────────
SEED = 41
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'sidelec_roche_plate_csv2.csv')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'resultats_mc_dropout')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures_export')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WS      = 96
T       = 500
CONF    = 0.95
HORIZON = 48     # 2 jours stricts
CONTEXT = 168    # 1 semaine d'historique affichée sur la figure prédiction
N_TRAJ  = 25     # trajectoires MC affichées sur la figure d'incertitude

PROFILS = {
    'production':   {'col': 1, 'titre': 'Production photovoltaïque', 'ylabel': 'Production [Wh]'},
    'consommation': {'col': 2, 'titre': 'Consommation',             'ylabel': 'Consommation [Wh]'},
}

# ── Style serif / publication ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "font.size": 16, "axes.titlesize": 24, "axes.labelsize": 22,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18,
    "figure.dpi": 300, "savefig.dpi": 600,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": '--',
})

def run_mc_inference(model, seed_norm, n_steps, T):
    windows = np.tile(seed_norm.flatten(), (T, 1))
    samples = np.zeros((T, n_steps), dtype=np.float32)
    for t in range(n_steps):
        preds = model(windows[:, :, np.newaxis], training=True).numpy().flatten()
        samples[:, t] = preds
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds
    return samples

# ── Boucle par profil ────────────────────────────────────────────────────────
for profil, cfg in PROFILS.items():
    model_path = os.path.join(MODEL_DIR, f'modele_mc_dropout_{profil}.keras')
    if not os.path.exists(model_path):
        print(f"Modèle manquant pour {profil}. Lance d'abord pv_profils_mcdropout.py.")
        continue

    print(f"Traitement de {cfg['titre']}...")
    model = tf.keras.models.load_model(model_path)

    df = pd.read_csv(DATA_PATH, sep=';', header=None)
    datapro = df.iloc[:, cfg['col']].values.reshape(-1, 1).astype(float)
    n = len(datapro)
    train_end, val_end = int(n * 0.70), int(n * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1)); scaler.fit(datapro[:train_end])
    train_val_raw = datapro[:val_end].flatten()
    test_full     = datapro[val_end:].flatten()
    val_norm      = scaler.transform(datapro[train_end:val_end])

    n_steps = min(HORIZON, len(test_full))
    test_raw = test_full[:n_steps]

    seed = val_norm[-WS:]
    samples_norm = run_mc_inference(model, seed, n_steps, T)
    samples_real = np.array([scaler.inverse_transform(s.reshape(-1, 1)).flatten()
                             for s in samples_norm])

    mean_pred = samples_real.mean(axis=0)
    ci_lower  = np.quantile(samples_real, (1 - CONF) / 2,     axis=0)
    ci_upper  = np.quantile(samples_real, 1 - (1 - CONF) / 2, axis=0)
    rmse      = np.sqrt(mean_squared_error(test_raw, mean_pred))
    coverage  = np.mean((test_raw >= ci_lower) & (test_raw <= ci_upper)) * 100

    x_test = np.arange(val_end, val_end + n_steps)
    x_ctx  = np.arange(val_end - CONTEXT, val_end)

    # ── FIGURE 1 : PRÉDICTION (contexte + prévision 2 j) ─────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(x_ctx, train_val_raw[val_end - CONTEXT:val_end], color='#2C3E50', lw=2,
             label='Données historiques')
    ax1.plot(x_test, test_raw,  color='#E67E22', lw=2.5, label='Données de test')
    ax1.plot(x_test, mean_pred, color='#2980B9', lw=2.8, ls='--', label='Prédiction')
    ax1.axvline(val_end, color='#7F8C8D', ls=':', lw=1.5)
    ax1.set_title(cfg['titre'])
    ax1.set_xlabel("Temps [h]"); ax1.set_ylabel(cfg['ylabel'])
    ax1.legend(loc='upper left', frameon=True)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, f"{profil}_prediction.pdf"))
    plt.close(fig1)

    # ── FIGURE 2 : MC DROPOUT (zoom 2 j + trajectoires + bande) ──────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for i in range(min(N_TRAJ, T)):
        ax2.plot(x_test, samples_real[i], color='#AED6F1', lw=0.6, alpha=0.4,
                 zorder=1, label='Trajectoires MC' if i == 0 else None)
    ax2.fill_between(x_test, ci_lower, ci_upper, color='#3498DB', alpha=0.3, zorder=2,
                     label=f'Intervalle de confiance [{int(CONF*100)}%]')
    ax2.plot(x_test, test_raw,  color='#E67E22', lw=3, zorder=4, label='Réel')
    ax2.plot(x_test, mean_pred, color='#2C3E50', lw=3, ls='--', zorder=5, label='Prédiction moyenne')
    ax2.set_title(f"{cfg['titre']} — RMSE {rmse:.0f} Wh, couv. {coverage:.0f}%")
    ax2.set_xlabel("Temps [h]"); ax2.set_ylabel(cfg['ylabel'])
    ax2.legend(loc='upper right')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, f"{profil}_mcdropout.pdf"))
    plt.close(fig2)

    print(f"  RMSE {rmse:.0f} Wh | couverture {coverage:.0f}%")

print(f"\nTerminé. Figures exportées dans : {OUTPUT_DIR}")
