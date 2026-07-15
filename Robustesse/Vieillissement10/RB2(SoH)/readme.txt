RB2(SoH)
========

Les deux seules variables de commande restent les setpoints de puissance FC et
ELY. Aucune regle de plafond ni seuil d'activation supplementaire n'est utilise.

La loi SoH retenue est :

  usure_i = clip((1 - SoH_i) / (1 - SoH_EoL,i), 0, 1)
  P_i,set = c_i P_i,nom (1 - strength_i * usure_i^shape_i)

Cette ecriture remplace SoH^gamma, mal conditionnee lorsque SoH_EoL=0.90 :
strength donne directement la baisse relative maximale du setpoint a l'EoL et
shape indique si elle est progressive (1) ou concentree en fin de vie (>1).

Les simulations sur 25 ans montrent un vrai front LPSP/degradation. Avec
VoLL=3 EUR/kWh, le minimum du cout unifie reste cependant le cas nul RB2 : la
penalite d'EENS domine le gain de vieillissement.

Le point actif est donc un choix Pareto explicite, et non un optimum du cout
unifie : il minimise la degradation parmi les configurations de la grille dont
la LPSP reste <= 1.10 %. Parametres : strength_FC=strength_ELY=0.25 et
shape_FC=shape_ELY=1. Il donne LPSP=1.09760 %, degradation=62.20838 kEUR,
contre 0.77502 % et 63.37461 kEUR pour RB2.

Commandes :

  python ../optimize_rb2_augmentations.py --layer soh_normalized
  python ../analyze_rb2soh_tradeoff.py --lpsp-cap 1.10

Resultats : ../DIAGNOSTIC_RB2SOH_TRADEOFF.txt et ../RB2SoH_tradeoff.png.
