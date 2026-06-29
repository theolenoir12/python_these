import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ── CONFIGURATION DES DOSSIERS ──────────────────────────────────────────────
OUTPUT_DIR = "figures_export_rul"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ── CONFIGURATION GRAPHIQUE ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3
})

# ── PARAMÈTRES ─────────────────────────────────────────────────────────────
FILENAME = 'SoH_ely_2.csv'
MODEL_PATH = 'resultats_mc_dropout/modele_mc_dropout_scaler2.keras'
WS = 100        # Window Size
T = 500         # Nombre de simulations Monte-Carlo
CONF = 0.95     # Intervalle de confiance
THRESHOLD = 0.90 # Seuil de fin de vie (SoH_EoL)
MAX_STEPS = 1500 

# Chargement du modèle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle non trouvé : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# ── FONCTION DE PRÉDICTION JUSQU'AU SEUIL ──────────────────────────────────
def run_mc_until_threshold(model, seed_norm, scaler, threshold, T, conf, max_steps):
    windows = np.tile(seed_norm.flatten(), (T, 1))
    all_samples_norm = []
    step = 0
    reached_eol = False
    alpha = (1 - conf) / 2
    
    print("Calcul de la trajectoire de fin de vie...")
    
    while not reached_eol and step < max_steps:
        x_batch = windows[:, :, np.newaxis]
        preds_norm = model(x_batch, training=True).numpy().flatten()
        all_samples_norm.append(preds_norm)
        
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1] = preds_norm
        
        # On vérifie si la borne supérieure (scénario optimiste) a atteint le seuil
        preds_real = scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
        ci_upper = np.quantile(preds_real, 1 - alpha)
        
        if ci_upper <= threshold:
            reached_eol = True
        
        step += 1
    return np.array(all_samples_norm).T

# ── TRAITEMENT ET VISUALISATION ─────────────────────────────────────────────
if not os.path.exists(FILENAME):
    print(f"Fichier {FILENAME} introuvable.")
else:
    # 1. Données et Scaling
    df = pd.read_csv(FILENAME, sep=';', header=None)
    datapro = df.iloc[:, 0].values.reshape(-1, 1)[::24]
    n_history = len(datapro)
    
    val_end = int(n_history * 0.85) # t_now
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(np.array([[0.9], [1.00]]))
    scaler.fit(datapro[:int(n_history)])
    
    history_raw = datapro[:val_end].flatten()
    seed_norm = scaler.transform(datapro[val_end-WS:val_end])

    # 2. Inférence
    samples_norm = run_mc_until_threshold(model, seed_norm, scaler, THRESHOLD, T, CONF, MAX_STEPS)
    samples_real = np.array([scaler.inverse_transform(s.reshape(-1,1)).flatten() for s in samples_norm])
    
    mean_pred = np.mean(samples_real, axis=0)
    ci_lower = np.quantile(samples_real, (1-CONF)/2, axis=0)
    ci_upper = np.quantile(samples_real, 1-(1-CONF)/2, axis=0)
    
    # 3. Calcul du RUL (basé sur la moyenne)
    n_pred = samples_real.shape[1]
    x_history = np.arange(val_end)
    x_pred = np.arange(val_end, val_end + n_pred)
    
    idx_eol = np.where(mean_pred <= THRESHOLD)[0]
    rul_val = idx_eol[0] if len(idx_eol) > 0 else n_pred
    t_eol = val_end + rul_val

    # 4. Génération de la figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historique et Prédiction
    ax.plot(x_history, history_raw, color='#2C3E50', lw=2, label='Données historiques')
    ax.plot(x_pred, mean_pred, color='#2980B9', lw=2.8, linestyle='--', label='Prédiction')
    ax.fill_between(x_pred, ci_lower, ci_upper, color='#2980B9', alpha=0.15, label='Intervalle de confiance [95%]')
    
    # Seuil de fin de vie
    ax.axhline(y=THRESHOLD, color='#C0392B', linestyle='--', lw=1.7, label=f'Seuil EoL')

    # --- REPRÉSENTATION DU RUL (Double flèche) ---
    # On place la flèche légèrement au-dessus du seuil pour la visibilité
    y_arrow = THRESHOLD + 0.045 
    
    # Double flèche entre t_now (val_end) et t_eol
    ax.annotate('', xy=(val_end, y_arrow), xytext=(t_eol, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    
    # Texte RUL centré sur la flèche
    ax.text((val_end + t_eol)/2, y_arrow + 0.0, f"RUL($t$)", 
            ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Repères verticaux pour t_now et t_eol
    ax.axvline(x=val_end, color='gray', linestyle=':', alpha=1)
    ax.axvline(x=t_eol, color='gray', linestyle=':', alpha=1)
    
    # Annotations d'axes spécifiques
    ax.text(val_end+20, THRESHOLD - 0.04, '$t$', ha='center', fontsize=20)
    ax.text(t_eol+50, THRESHOLD - 0.04, '$t_{EoL}$', ha='center', fontsize=20)

    ax.scatter(1530, THRESHOLD, color='red', s=50, zorder=5)
    ax.scatter(1630, THRESHOLD, color='red', s=50, zorder=5)
    
    # Ajouter l'annotation juste au-dessus (ou à côté)
    ax.text(1550-70, THRESHOLD - 0.01, r'$t_{EoL-min}$', 
            fontsize=20, ha='center', va='bottom', fontweight='bold')
    ax.text(1550+70, THRESHOLD+ 0.0075, r'$t_{EoL-max}$', 
            fontsize=20, ha='center', va='bottom', fontweight='bold')
    
    # Mise en forme finale
    ax.set_title("PEMWE", pad=0)
    ax.set_xlabel("Temps [j]")
    ax.set_ylabel("SoH")
    ax.set_ylim(THRESHOLD - 0.05, 1.02)
    ax.legend(loc='center left')
    
    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "representation_rul_ely.pdf")
    fig.savefig(save_path)
    plt.show()

    print(f"RUL calculé : {rul_val} unités de temps.")
    print(f"Graphique sauvegardé : {save_path}")