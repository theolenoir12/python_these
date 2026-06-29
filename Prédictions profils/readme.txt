PRÉDICTION DE PROFILS DE PUISSANCE (production PV et consommation)
==================================================================

Même méthode que le dossier "Pronostic SoH", appliquée aux profils horaires
de puissance. Réseau LSTM + prédiction récursive + Monte-Carlo Dropout
(Gal & Ghahramani 2016) pour quantifier l'incertitude.

Données
-------
Fichier : ../sidelec_roche_plate_csv2.csv  (CSV ';', sans en-tête)
    colonne 0 = temps (s)
    colonne 1 = production PV (Wh)
    colonne 2 = consommation / charge BRUITÉE (Wh)  <- profil réaliste
~17520 points horaires (~2 ans). Découpage 70% train / 15% val / 15% test.
(L'ancien sidelec_roche_plate_csv.csv, 1 an, conso non bruitée, reste utilisable.)

Architecture et hyperparamètres
-------------------------------
Input(ws,1) -> LSTM(units) -> Dropout(dr) -> Dense(1), loss MSE, Adam.
Issus de "config_12" (min loss de validation, code stage Hervé2) :
    learning_rate = 1e-3
    window_size   = 96
    batch_size    = 64
    lstm_units    = 80
    epochs        = 160
    dropout_rate  = 0.05   (config_12 = 0 ; >0 imposé par le MC Dropout)

Scripts (à exécuter dans l'ordre)
---------------------------------
1. pv_profils_EVT.py
   Recherche d'hyperparamètres avec validation RÉCURSIVE (optionnel : sert à
   confirmer/affiner config_12). Sorties -> resultats_recherche_hp/

2. pv_profils_mcdropout.py
   Entraîne le modèle final pour chaque profil, applique le MC Dropout
   (T=500 trajectoires), calcule RMSE/MAE/couverture et sauvegarde le modèle
   + une figure diagnostique. Sorties -> resultats_mc_dropout/

3. plot_pv_profils.py
   Recharge les modèles et exporte les figures propres 2 j (PDF serif, graine
   fixée) : <profil>_prediction.pdf (prévision vs réel) et <profil>_mcdropout.pdf
   (trajectoires MC + bande IC 95%). Sorties -> figures_export/

4. pv_profils_backtest_net.py
   Backtest multi-origines : distribution empirique de l'écart d'ÉNERGIE NETTE
   (conso - PV) à 2 j -> bruit d'estimation pour l'EMS (sigma, biais, quantiles).
   Sorties -> resultats_backtest/ (figure + CSV). C'est l'outil retenu pour
   quantifier l'incertitude (approche empirique "C").

(diag_mc_intervalles.py : diagnostic expliquant pourquoi les bandes MC Dropout
 sont serrées — épistémique seul, signal périodique borné, sensibilité dropout.)

Horizon
-------
Horizon de travail = 2 jours (48 h) STRICT partout. La prédiction récursive sur
un signal oscillatoire ne dérive pas tant que l'horizon reste court.

Incertitude (approche retenue : C, empirique)
---------------------------------------------
Le MC Dropout ne capte que l'incertitude ÉPISTÉMIQUE et sous-estime l'erreur
réelle (dominée par l'aléatoire : météo PV, bruit conso). Pour l'EMS, on utilise
la distribution empirique du backtest sur l'énergie nette 2 j :
    biais ~ 0 | sigma ~ 80 kWh | intervalle 90% ~ [-143, +89] kWh (~27% relatif).
"""
