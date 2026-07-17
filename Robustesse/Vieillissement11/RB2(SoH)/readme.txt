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

Avec le plancher batterie psi=1 sous 1C, RB2 donne sur 25 ans LPSP=0.77502 %,
degradation=63.37461 kEUR et cout unifie=75.55008 kEUR. Le meilleur point
RB2(SoH) teste utilise strength_FC=0.025, strength_ELY=0 et shape=4 :
LPSP=0.75920 %, degradation=63.39517 kEUR et cout unifie=75.32207 kEUR.
Le gain unifie est limite a 0.30 %.

Le point de degradation minimale sous l'iso-cout RB2 utilise
strength_FC=strength_ELY=0.025 et shape=1 : LPSP=0.77199 %, degradation=
63.35631 kEUR et cout unifie=75.48411 kEUR. La baisse de degradation n'est que
de 0.029 %. Une baisse proche de 1 % n'apparait qu'en acceptant environ +4.6 %
de cout unifie. Le front existe donc, mais le SoH seul reste peu concluant.

Le point actif ci-dessus reste le minimum du cout unifie de la grille ciblee.

Commandes :

  python ../optimize_rb2_augmentations.py --layer soh_validated25
  python ../optimize_rb2_augmentations.py --layer soh_validated_shapes25

Resultats : ../Optimization_results_psi1/.
