RB2(SoH+RUL+Pred)
=================

Combinaison multiplicative des facteurs SoH et RUL sur les deux setpoints H2,
puis inhibition eventuelle du setpoint ELY par la couche de prediction. Aucun
plafond de strategie n'est ajoute. Mettre tous les exposants a zero et desactiver
la prediction redonne exactement RB2.

Optimisation : python ../optimize_rb2_augmentations.py --layer all

Optimum V10 moyen sur trois graines : SoH (gamma_fc=0.25, gamma_ely=0), RUL
inactif (deux gammas nuls), prediction (24 h, SoC 0.99, bande 1.5 sigma, aucun
maintien minimal). Cout unifie 75.2836 kEUR, LPSP 0.7560 %, degradation
63.4071 kEUR. Le cumul retient donc SoH+Pred et rejette la modulation RUL.
