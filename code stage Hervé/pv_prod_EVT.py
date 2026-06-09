import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# 1. Charger la deuxième colonne du CSV
df= pd.read_csv('sidelec_roche_plate.csv', sep=';', header=None)
datapro = df.iloc[:,1].values.reshape(-1, 1)

# 2. Normaliser les données (essentiel pour les LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(datapro)


# 3.creation de la structure de la séquence
def create_sequences(datapro, window_size):
    X, y = [], []
    for i in range(len(datapro)- window_size):
        X.append(datapro[i:i+window_size])
        y.append(datapro[i+window_size])
    return np.array(X), np.array(y)

# On définit une fenêtre de 72 points de données
window_size = 72
X, y = create_sequences(data_normalized, window_size)


# 4.Calcul des indices de coupure
n = len(data_normalized)
train_end = int(n * 0.7)        # 70% pour l'entraînement
val_end   = train_end + 4 * 24       # 15% pour la validation (de 70% à 85%)
test_end  = val_end   + 4 * 24

# Création des sets (données normalisées)
train_data = data_normalized[:train_end]
val_data = data_normalized[train_end:val_end]
test_data = data_normalized[val_end:test_end]

print(f"Total: {n} | Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# On applique la fenêtre temporelle (ex: 24h) à chaque groupe
X_train, y_train = create_sequences(train_data, window_size)
X_val, y_val = create_sequences(val_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

# Redimensionnement pour le LSTM [échantillons, pas de temps, 1]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Nouvelle forme de X_train (prêt pour LSTM): {X_train.shape}")

# 1. Définir le modèle LSTM avec Tensorflow

model = tf.keras.Sequential([
    # On définit l'entrée ici
    Input(shape=(window_size, 1)), 
    
    # Le LSTM n'a plus besoin de input_shape
    LSTM(50), 
    
    Dense(1)
])

#compiler le modèle

model.compile(optimizer='adam', loss='mse') 

# 3. Afficher un résumé du modèle
model.summary()


#Entrainement et enregistrement de l'historique
# On ajoute 'validation_data' pour que Keras calcule la perte sur le test à chaque étape
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val), # Utilisation du groupe de Validation
    verbose=2
)

#sauvegarder le modèle

model.save('modele_lstm.keras')

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('modele_lstm.keras')

# Évaluation sur des données jamais vues du tout
test_loss = model.evaluate(X_test, y_test)
print(f"Erreur finale sur le groupe de Test : {test_loss}")

# Prédiction sur le jeu test
predictions = model.predict(X_test)


# Inverser la normalisation pour les données réelles et prédites
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_original = scaler.inverse_transform(predictions)


# ─────────────────────────────────────────────
# Global rcParams — journal style
# ─────────────────────────────────────────────
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "grid.alpha": 0.5, "grid.color": "#aaaaaa",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "savefig.dpi": 600, "savefig.bbox": "tight", "mathtext.fontset": "stix",
})
C_TRAIN, C_VAL, C_ACTUAL, C_PRED = "#1a6faf", "#d94801", "#1a6faf", "#d94801"

# ─────────────────────────────────────────────
# Figure 1 — Data split overview
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.plot(range(0, train_end),       data_normalized[:train_end],  color=C_TRAIN, lw=0.8, label="Training set (70%)")
ax.plot(range(train_end, val_end), data_normalized[train_end:val_end], color=C_VAL, lw=0.8, label="Validation set (15%)")
ax.plot(range(val_end, n),         data_normalized[val_end:],    color="#2ca02c", lw=0.8, label="Test set (15%)")
ax.axvline(train_end, color="#555", lw=0.7, linestyle=":")
ax.axvline(val_end,   color="#555", lw=0.7, linestyle=":")
ax.set_xlabel("Time step (h)")
ax.set_ylabel("Normalized power output")
ax.set_title("Dataset Partitioning for LSTM Training")
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc").get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig("fig1_data_split.png")

# ─────────────────────────────────────────────
# Figure 2 — Training & Validation Loss
# ─────────────────────────────────────────────
epochs = range(1, len(history.history['loss']) + 1)
best_ep = int(np.argmin(history.history['val_loss'])) + 1
best_val = history.history['val_loss'][best_ep - 1]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.semilogy(epochs, history.history['loss'],     color=C_TRAIN, label="Training MSE")
ax.semilogy(epochs, history.history['val_loss'], color=C_VAL, linestyle="--", label="Validation MSE")
ax.axvline(best_ep, color=C_VAL, lw=0.7, linestyle=":", alpha=0.7)
ax.scatter([best_ep], [best_val], color=C_VAL, s=30, zorder=5)
ax.annotate(f"Best epoch: {best_ep}", xy=(best_ep, best_val),
            xytext=(best_ep + len(history.history['loss'])*0.07, best_val * 1.8),
            fontsize=7, color=C_VAL, arrowprops=dict(arrowstyle="-", color=C_VAL, lw=0.7))
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_title("Training and Validation Loss")
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc").get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig("fig2_loss_curves.png")

# ─────────────────────────────────────────────
# Figure 3 — Test set: Actual vs Predicted
# ─────────────────────────────────────────────
rmse = np.sqrt(np.mean((y_test_original - predictions_original) ** 2))

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(y_test_original,    color=C_ACTUAL, lw=1.0, alpha=0.9, label="Measured")
ax.plot(predictions_original, color=C_PRED, lw=1.0, linestyle="--", alpha=0.85, label="LSTM forecast")
ax.text(0.02, 0.97, f"RMSE = {rmse:.2f} Wh", transform=ax.transAxes,
        fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", lw=0.6))
ax.set_xlabel("Time step (h)")
ax.set_ylabel("PV Power Output (Wh)")
ax.set_title("LSTM Forecast vs. Measured Data — Test Set")
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc").get_frame().set_linewidth(0.5)
fig.tight_layout()
fig.savefig("fig3_test_comparison.png")

plt.show()