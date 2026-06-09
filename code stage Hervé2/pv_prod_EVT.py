# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:50:14 2026

@author: herve
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:41:09 2026

@author: herve
"""
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
val_end = int(n * 0.85)         # 15% pour la validation (de 70% à 85%)


# Création des sets (données normalisées)
train_data = data_normalized[:train_end]
val_data = data_normalized[train_end:val_end]
test_data = data_normalized[val_end:]

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
    verbose=1
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


# 2. Création du graphique

plt.figure(figsize=(12, 5))
plt.plot(range(0, train_end), train_data, label='Entraînement (70%)', color='blue')
plt.plot(range(train_end, val_end), val_data, label='Validation (15%)', color='orange')
plt.plot(range(val_end, n), test_data, label='Test (15%)', color='green')
plt.title("Répartition des données pour le LSTM")
plt.legend()
plt.show()



plt.figure(figsize=(12, 6))

# Tracer les deux courbes de production et consommation

plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.title('courbes de Loss')
plt.xlabel('epochs')
plt.ylabel('valeurs perte')
plt.legend()
plt.grid(True)
plt.show()


# Création figure réalité VS Test
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='production réelle', color='blue')
plt.plot(predictions_original, label='Prédiction LSTM', color='red', linestyle='--')
plt.title('Comparaison sur le jeu de Test')
plt.xlabel('Temps (heure)')
plt.ylabel('production (wh)')
plt.legend()
plt.grid(True)

# Afficher le graphique
plt.show()
plt.savefig('comparaison_production_prediction.png', dpi=300, bbox_inches='tight')


