"""RB2 augmentee par le RUL estime des composants H2."""

from Common.rb2_policy import make_augmented_rb2_policy


# Optimum V10 : le cas nul est meilleur que toutes les modulations RUL testees.
# Le dossier reste actif comme couche/test nul et pour les futures variantes de
# pronostic, mais n'impose pas un derating que les resultats ne justifient pas.
PARAMETERS = {
    "fc_setpoint": 0.59,
    "ely_setpoint": 0.49,
    "rul_ref_fc_days": 3000.0,
    "rul_ref_ely_days": 3000.0,
    "rul_gamma_fc": 0.0,
    "rul_gamma_ely": 0.0,
}

get_optimal_action_RB = make_augmented_rb2_policy(**PARAMETERS)
