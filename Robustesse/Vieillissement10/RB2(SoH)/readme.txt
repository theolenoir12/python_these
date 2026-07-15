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

Apres correction du facteur C-rate batterie conforme a Maheshwari et al., RB2
donne sur 25 ans LPSP=0.68124 %, degradation=24.69699 kEUR et cout unifie=
35.39924 kEUR. Le meilleur point RB2(SoH) teste utilise strength_FC=0.025,
strength_ELY=0 et shape=2 : LPSP=0.66738 %, degradation=24.63466 kEUR et
cout unifie=35.11909 kEUR. Le gain unifie est de 0.79 %, donc reproductible
mais inferieur au seuil de plusieurs pourcents retenu pour une amelioration
substantielle.

Le petit compromis Pareto le plus lisible utilise strength_FC=strength_ELY=
0.05 et shape=1 : LPSP=0.69768 %, degradation=24.39285 kEUR et cout unifie=
35.35326 kEUR. Il diminue la degradation de 1.23 % tout en restant legerement
sous l'iso-cout RB2. Ce point sert aux figures de compromis, tandis que le
point actif ci-dessus reste le minimum du cout unifie de la grille ciblee.

Commandes :

  python ../optimize_rb2_augmentations.py --layer soh_validated25
  python ../optimize_rb2_augmentations.py --layer soh_validated_shapes25

Resultats : ../Optimization_results_validated/.
