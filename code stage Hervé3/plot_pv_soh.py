import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── CONFIGURATION DES DOSSIERS ──────────────────────────────────────────────
OUTPUT_DIR = "figures_export"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ── CONFIGURATION GRAPHIQUE (OPTIMISÉE POUR LATEX / LMODERN) ────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False, # Mettre à True si vous avez une distribution LaTeX installée
    "font.size": 16,          # Augmenté pour la lisibilité
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 19,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": '--'
})

# ── CONFIGURATION DES COMPOSANTS ───────────────────────────────────────────
# Mapping : Fichier -> (Titre, Label Y)
DATA_CONFIG = {
    # 'SoH_sinus.csv':{'titre': 'Sinusoïde', 'ylabel':''},
    # 'SoH_bat.csv': {'titre': 'Batterie', 'ylabel': 'SoH'},
    # 'SoH_fc.csv':  {'titre': 'PEMFC',    'ylabel': ''},
    'SoH_ely.csv': {'titre': 'PEMWE',    'ylabel': ''}
}

MODEL_PATH       = 'resultats_mc_dropout/modele_mc_dropout.keras'
MODEL_PATH_SINUS = 'resultats_mc_dropout/modele_mc_dropout_sinusoide.keras'
WS = 100
# WS = 149 #pour sinusoïde
T = 500
CONF = 0.95

# Chargement du modèle
model = tf.keras.models.load_model(MODEL_PATH)
model_sin = tf.keras.models.load_model(MODEL_PATH_SINUS)

# ── FONCTION DE PRÉDICTION MC DROPOUT ──────────────────────────────────────
def run_mc_inference(model, seed_norm, n_steps, T):
    windows = np.tile(seed_norm.flatten(), (T, 1))
    samples = np.zeros((T, n_steps), dtype=np.float32)
    for t in range(n_steps):
        x_batch = windows[:, :, np.newaxis]
        preds = model(x_batch, training=True).numpy().flatten()
        samples[:, t] = preds
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds
    return samples

# ── BOUCLE DE GÉNÉRATION ────────────────────────────────────────────────────
for filename, config in DATA_CONFIG.items():
    if not os.path.exists(filename):
        print(f"Fichier {filename} manquant.")
        continue

    print(f"Traitement de {config['titre']}...")
    
    # Préparation des données
    df = pd.read_csv(filename, sep=';', header=None)
    datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]
    n = len(datapro)
    train_end, val_end = int(n * 0.70), int(n * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(datapro[:train_end])
    
    train_val_raw = datapro[:val_end].flatten()
    test_raw = datapro[val_end:].flatten()
    val_norm = scaler.transform(datapro[train_end:val_end])

    # Inférence
    seed = val_norm[-WS:]
    if filename == 'SoH_sinus.csv' :
        samples_norm = run_mc_inference(model_sin, seed, len(test_raw), T)
    else:
        samples_norm = run_mc_inference(model, seed, len(test_raw), T)
    samples_real = np.array([scaler.inverse_transform(s.reshape(-1,1)).flatten() for s in samples_norm])
    
    mean_pred = np.mean(samples_real, axis=0)
    ci_lower = np.quantile(samples_real, (1-CONF)/2, axis=0)
    ci_upper = np.quantile(samples_real, 1-(1-CONF)/2, axis=0)
    rmse = np.sqrt(mean_squared_error(test_raw, mean_pred))
    coverage = np.mean((test_raw >= ci_lower) & (test_raw <= ci_upper)) * 100

    x_all = np.arange(n)
    x_test = np.arange(val_end, n)

    # --- FIGURE 1 : TRAJECTOIRE COMPLÈTE ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(x_all[:val_end], train_val_raw, color='#2C3E50', lw=2, label='Données historiques')
    ax1.plot(x_test, test_raw, color='#E67E22', lw=2.3, label='Données de test')
    ax1.plot(x_test, mean_pred, color='#2980B9', lw=2.8, linestyle='--', label='Prédiction')
    ax1.fill_between(x_test, ci_lower, ci_upper, color='#2980B9', alpha=0.2, label=f'Intervalle de confiance [{int(CONF*100)}%]')

    ax1.set_title(config['titre'])
    ax1.set_xlabel("Temps [j]")
    ax1.set_ylabel(config['ylabel'])
    ax1.legend(loc='lower left', frameon=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, f"{config['titre']}_trajectoire.pdf"))
    plt.close(fig1)

    # --- FIGURE 2 : ZOOM TEST ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for i in range(3):
        ax2.plot(x_test, samples_real[i], color='#3498DB', lw=0.5, alpha=0.3)
    
    ax2.plot(x_test, test_raw, color='#E67E22', lw=3, label='Réel')
    ax2.plot(x_test, mean_pred, color='#2C3E50', lw=3, linestyle='--', label='Prédiction')
    ax2.fill_between(x_test, ci_lower, ci_upper, color='#3498DB', alpha=0.3, label='Intervalle de confiance [95%]')

    # # Cartouche de texte en français
    # stats_text = (f"RMSE : {rmse:.4f}")
    # props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey')
    # ax2.text(0.05, 0.15, stats_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    ax2.set_title(f"RMSE : {rmse:.4f}")
    ax2.set_xlabel("Temps [j]")
    ax2.set_ylabel(config['ylabel'])
    ax2.legend(loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, f"{config['titre']}_zoom.pdf"))
    plt.close(fig2)

print(f"\nTerminé. Les figures ont été exportées dans le dossier : {OUTPUT_DIR}")