# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:47:35 2026

@author: herve
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:14:32 2026

@author: herve
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Charger la deuxième colonne du CSV
df= pd.read_csv('sidelec_roche_plate_csv_bruit.csv', sep=';', header=None)
dataconso = df.iloc[:,2].values.reshape(-1, 1)

# 2. Normaliser les données (essentiel pour les LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(dataconso)


# 3.creation de la structure de la séquence
def create_sequences(data_normalized, window_size):
    X, y = [], []
    for i in range(len(data_normalized) - window_size):
        X.append(data_normalized[i:i+window_size])
        y.append(data_normalized[i+window_size])
    return np.array(X), np.array(y)

# On définit une fenêtre de 96 points de données
window_size = 96
X, y = create_sequences(data_normalized, window_size)

# 3. Reshape pour LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4.Calcul des indices de coupure
n = len(data_normalized)
train_end = int(n * 0.85)        # 70% pour l'entraînement


#découpage des séquences X et y
X_train,y_train  = X[:train_end], y[:train_end]
X_test, y_test   = X[train_end:train_end + 96], y[train_end:train_end+96]

# découpage des données (données normalisées)
train_data = data_normalized[:train_end]
test_data = data_normalized[train_end:train_end + 96]

print(f"Total: {n} | Train: {len(train_data)} | Test: {len(test_data)}")



# Redimensionnement pour le LSTM [échantillons, pas de temps, 1]
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)


# 1. Définir le modèle LSTM avec Tensorflow
#LSTM units
nbr_neurones = 80

model = tf.keras.Sequential([
    # On définit l'entrée ici
    Input(shape=(window_size, 1)), 
    
    # première couche LSTM, return_sequences=True lors de l'ajout d'une seconde couche
    LSTM(nbr_neurones, return_sequences=False), 
     
    Dropout(0),
    
    # Couche de sortie
    Dense(1)
])
#compiler le modèle
lr = 0.001
opt = Adam(learning_rate=lr)
model.compile(optimizer=opt, loss='mse')
# 3. Afficher un résumé du modèle
model.summary()

# Demarrer chronomètre d'entrainement
start_time_train = time.time()

#arrête l'entraînement si le modèle ne progresse plus et remet automatiquement les poids de la meilleure époque dans le modèle.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,             # Arrête si pas d'amélioration pendant 10 époques
    restore_best_weights=True # Recharge automatiquement les meilleurs paramètres
)

#Entrainement et enregistrement de l'historique
epoques = 160
history = model.fit(
    X_train, y_train,
    epochs= epoques ,
    batch_size= 2**(int(np.log2(np.sqrt(train_end)))),
    #validation_data=(X_test, y_test), # Utilisation du groupe de Test
    callbacks=[early_stop],
    verbose=1,
   
)



# Arrêter le chronomètre d'entrainement
end_time_train = time.time()

# Calculer et afficher la durée d'entrainement
duration_train = end_time_train - start_time_train

# Conversion en minutes et secondes
minutes = int(duration_train // 60)
seconds = int(duration_train % 60)

# Affichage en minutes (et secondes si besoin)
if minutes > 0:
    print(f"Durée totale d'entraînement : {minutes} min {seconds} sec")
else:
    print(f"Durée totale d'entraînement : {seconds:.2f} sec")


# Démarrer le chronomètre de prédiction
start_time_predic = time.time()

#prédiction récursive sur les données de validation
def predict_recursive(model, start_sequence, steps):
    predictions = []
    current_sequence = start_sequence.reshape(window_size, 1).copy()
    
    for _ in range(steps):
        # Prédiction du point suivant
        res = model.predict(current_sequence.reshape(1, window_size, 1), verbose=0)
        next_pred = res[0][0]
        predictions.append(next_pred)
        
        # Mise à jour : on enlève le premier, on ajoute la prédiction à la fin
        current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
    return np.array(predictions).reshape(-1, 1)

# --- 2. Préparation des données pour le test ---
# On commence la prédiction JUSTE après le set d'entraînement'
# La séquence de départ est donc la fin de train_data
start_seq = train_data[-window_size:]
steps_to_predict = len(y_test)

# Calcul des prédictions sur le set de TEST
predictions_scaled = predict_recursive(model, start_seq, steps_to_predict)
predictions_inv = scaler.inverse_transform(predictions_scaled)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Arrêter le chronomètre de prédiction
end_time_predic = time.time()

# Calculer et afficher la durée de prédiction
duration_predic = end_time_predic - start_time_predic


print(f"Durée totale de prédiction : {duration_predic:.4f} secondes")

# On inverse aussi les données réelles pour l'affichage
train_inv = scaler.inverse_transform(train_data)
test_inv = scaler.inverse_transform(test_data)


# 1. Prédire sur les données d'entraînement (pour voir si le modèle a bien appris)
train_predictions_scaled = model.predict(X_train)

# 2. Inverser la normalisation pour revenir aux Watts-heure (Wh)
# On compare y_train (réel) et train_predictions (prédit)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
train_predictions_inv = scaler.inverse_transform(train_predictions_scaled)



#--------------- calcul du pas d'échantillonage-----------------


# 6. Charger le CSV avec les timestamps 
#df = pd.read_csv('sidelec_roche_plate_csv_bruit.csv', sep=';', header=None)

# Convertir la première colonne (0) en datetime
#df[0] = pd.to_datetime(df[0])

# Calculer la différence entre deux timestamps consécutifs
#time_diff_days = (df[0].iloc[1] - df[0].iloc[0]).total_seconds() / 60 # en minutes

# Convertir en heure
#time_diff_hours = time_diff_days / 60 

#print(f"Pas d'échantillonnage : {time_diff_hours:.0f} heure")

# ----------------- Calcul des horodatages pour train et val -------------------

#df[0] = pd.to_datetime(df[0])


# 3. Créer la colonne 'timestamp' en ajoutant les heures (24 données/jour)
#df['timestamp'] = df[0] + pd.to_timedelta(df.index % 24, unit='h')

train_timestamps = df['timestamp'].iloc[:train_end]
val_timestamps = df['timestamp'].iloc[train_end:train_end + 96]

train_start = train_timestamps.iloc[0].strftime('%d/%m/%Y %H:%M')
train_end_str = train_timestamps.iloc[-1].strftime('%d/%m/%Y %H:%M')
val_start = val_timestamps.iloc[0].strftime('%d/%m/%Y %H:%M')
val_end_time = val_timestamps.iloc[-1].strftime('%d/%m/%Y %H:%M')



#--------------- legende Hyperparamètres---------------------

# 1. Trouver l'index de la meilleure époque (celle où la val_loss est minimale)
best_epoch_idx = np.argmin(history.history['loss'])
                           
# 1. Récupération des pertes correspondant à cette époque
final_train_mse = history.history['loss'][best_epoch_idx]


# Calcul des métriques sur le set de TEST
mae_test = mean_absolute_error(y_test_inv, predictions_inv)
rmse_test = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))

print(f"Performance sur le TEST SET -> MAE: {mae_test:.2f} Wh | RMSE: {rmse_test:.2f} Wh")

# En utilisant vos variables de prédiction récursive
mse_recursive = mean_squared_error(y_test_inv, predictions_inv)
print(f"MSE Récursif sur 96h : {mse_recursive:.2f} Wh²")




# Si vous avez calculé la MAE sur les données inversées précédemment :
# final_mae = erreur_watts 

# 2. Construction de la chaîne de caractères
params_results_text = (
    f"--- Configuration ---\n"
   # f"Pas d'échantillonnage : {time_diff_hours:.0f} heure\n"
    f"Learning Rate:{lr}\n"
    f"Window Size: {window_size}\n"
    f"Batch Size: 64\n"
    f"LSTM Units: {nbr_neurones}\n"
    f"époques : {epoques}\n\n" 
    f"--- Performances ---\n"
    f"MSE (Train): {final_train_mse:.6f} Wh²\n"
    f"MSE (Test):{mse_recursive:.6f} Wh²\n"
    f"MAE (Test):{mae_test:.2f}\n"
    f"RMSE(Test):{rmse_test:.2f}Wh\n"
    f"Temps d'entraînement : {minutes} min {seconds} sec\n"
    f"Temps de prédiction : {duration_predic:.4f} sec\n"
)

# 3.  graphique 1 : Courbe de Loss
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
#ax.plot(history.history['val_loss'], label='Val Loss (MSE)', color='red')

# On place le bloc en bas à droite (loc='lower right') pour ne pas gêner la courbe qui descend
ax.text(0.98, 0.2, params_results_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        multialignment='center',    # <--- CENTRE LE TEXTE À L'INTÉRIEUR DU BLOC
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_title("Évolution des performances en fonction des Hyperparamètres")
ax.set_xlabel("Époques")
ax.set_ylabel("Mean Squared Error (Wh²)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.show()

#visualisation-----------------------------------------------------------

 
# --- Indices pour l'alignement ---
train_idx = range(0, train_end)
test_idx = range(train_end, train_end + 96)

# --- Inversion des données ---
train_actual = scaler.inverse_transform(data_normalized[:train_end])
test_actual = scaler.inverse_transform(data_normalized[train_end:train_end + 96])

# graphique 2: comparaison globale des données

plt.figure(figsize=(15, 7))

# 1. Courbe d'entraînement (Passé)
plt.plot(train_idx, train_actual, label='Historique (Entraînement)', color='blue', alpha=0.5)

# 2. Courbe réelle de Test (Futur réel)
plt.plot(test_idx, test_actual, label='Réalité (Test)', color='green', linewidth=2)

# 3. Ta prédiction récursive (Futur prédit)
plt.plot(test_idx, predictions_inv, label='Prédiction Récursive (96h)', color='red', linestyle='--')

# Ligne de séparation
plt.axvline(x=train_end, color='black', linestyle='-', label='Début de prédiction')

plt.title("Comparaison : Comparaison globale des données")
plt.xlabel("Temps (secondes)")
plt.ylabel("Energie de consommation (Wh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#graphique 3 Zoom sur le Test
plt.figure(figsize=(12, 6))

plt.plot(test_idx, test_actual, label='Données Réelles', color='green', marker='o', markersize=3)
plt.plot(test_idx, predictions_inv, label='Prédiction Récursive', color='red', linestyle='--', marker='x', markersize=3)
plt.axvline(x=train_end, color='black', linestyle='-', label='Début de prédiction')

plt.title(f"Données réelles vs Test sur les 96h premières heures\n correspondant aux Hyperparamètres de la Configuration 12\nMAE: {mae_test:.2f} Wh | RMSE: {rmse_test:.2f} Wh")
plt.xlabel("Temps de prédiction (secondes)")
plt.ylabel("Energie de consommation (Wh)")
plt.legend()
plt.grid(True)
plt.show()
#---------------------------------------
# 1. Prédire sur les données d'entraînement
train_predictions_scaled = model.predict(X_train)

# 2. Inverser la normalisation pour revenir aux unités réelles (Wh)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
train_predictions_inv = scaler.inverse_transform(train_predictions_scaled)

# --- Visualisation ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Graphique 1 : Vue globale sur tout le set d'entraînement
# 3. Calcul de l'erreur spécifique à l'entraînement
mae_train = mean_absolute_error(y_train_inv, train_predictions_inv)
mse_train = mean_squared_error(y_train_inv, train_predictions_inv)
print(f"Erreur moyenne sur l'entraînement (MAE Train) : {mae_train:.2f} Wh")


ax1.plot(y_train_inv, label='Réalité (Train)', color='green', alpha=0.6)
ax1.plot(train_predictions_inv, label='Prédiction (Train Fit)', color='blue', linestyle='--', alpha=0.8)
ax1.set_title(f"Comparaison globale : Réalité vs Apprentissage (Set d'Entraînement)\n MSE(Train):{final_train_mse:.6f}")
ax1.set_ylabel("Energie de consommation (Wh)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphique 2 : Zoom sur les 500 premiers points pour voir la précision du "fit"
zoom_range = 500 
ax2.plot(y_train_inv[:zoom_range], label='Réalité (Zoom)', color='green', marker='o', markersize=2)
ax2.plot(train_predictions_inv[:zoom_range], label='Prédiction (Zoom)', color='blue', linestyle='--')
ax2.set_title(f"Focus sur les {zoom_range} premières heures d'entraînement")
ax2.set_xlabel("Temps (Secondes)")
ax2.set_ylabel("Energie de consommation (Wh)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




