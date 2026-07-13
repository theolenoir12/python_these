"""RB2 V10: consignes economiques et secours de faisabilite aux bornes de SoC.

Les parametres ont ete selectionnes sur le cout unifie (LPSP + degradation)
par un balayage de 25 ans. Aucun SoH composant ne pilote la decision.
"""

from rb2_policy import make_rb2_policy

FC_BASE = 0.31
ELY_BASE = 0.22
FC_EMERGENCY = 0.90
ELY_EMERGENCY = 0.225

get_optimal_action_RB = make_rb2_policy(
    FC_BASE,
    ELY_BASE,
    FC_EMERGENCY,
    ELY_EMERGENCY,
)
