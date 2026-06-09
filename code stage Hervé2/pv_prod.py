# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:41:09 2026

@author: herve
"""

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

# 4. Diviser en ensembles d'entraînement et de test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


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
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

#sauvegarder le modèle

model.save('modele_lstm.keras')

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('modele_lstm.keras')

# Faire des prédictions
predictions = model.predict(X_test)


# Inverser la normalisation pour les données réelles et prédites
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_original = scaler.inverse_transform(predictions)


# 2. Création du graphique
plt.figure(figsize=(12, 6))

# Tracer les deux courbes de production et consommation

plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.title('courbes de Loss')
plt.xlabel('epochs = 100')
plt.ylabel('valeurs perte')
plt.legend()
plt.show()

# Créer une figure
plt.figure(figsize=(12, 6))

# Tracer la courbe réelle
plt.plot(y_test_original, label='production réelle', color='blue')

# Tracer la courbe de prédiction
plt.plot(predictions_original, label='Prédiction LSTM', color='red', linestyle='--')

# Ajouter des labels et une légende
plt.title('Comparaison : production réelle vs Prédiction LSTM')
plt.xlabel('Temps (heure) avec epochs = 100')
plt.ylabel('production (wh)')
plt.legend()
plt.grid(True)

# Afficher le graphique
plt.show()
plt.savefig('comparaison_production_prediction.png', dpi=300, bbox_inches='tight')

