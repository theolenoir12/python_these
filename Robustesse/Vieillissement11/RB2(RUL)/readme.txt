RB2(RUL)
========

Les deux seules variables de commande restent les setpoints de puissance FC et
ELY. Ils sont multiplies par min(RUL/RUL_ref, 1)^gamma. gamma=0 redonne RB2
exactement. Avant que l'estimateur dispose d'assez d'historique, RUL=inf et le
facteur vaut 1 : aucune attenuation artificielle au debut d'une vie.

Optimisation : python ../optimize_rb2_augmentations.py --layer rul

Resultat V10 sur 25 ans : le cas nul gamma_fc=gamma_ely=0 est l'optimum de la
grille. Toute modulation RUL testee degrade le cout unifie. La strategie active
est donc exactement RB2 ; ce resultat nul est conserve, pas masque.
