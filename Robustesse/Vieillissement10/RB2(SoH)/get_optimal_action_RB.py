"""RB2 augmentee par le SoH des deux convertisseurs H2."""

from Common.rb2_policy import make_augmented_rb2_policy


# Minimum du cout unifie de la verification ciblee post-validation batterie
# (25 ans, VoLL=3 EUR/kWh). La modulation reste un setpoint H2 pur.
PARAMETERS = {
    "fc_setpoint": 0.59,
    "ely_setpoint": 0.49,
    "soh_mode": "normalized_wear",
    "soh_strength_fc": 0.025,
    "soh_strength_ely": 0.0,
    "soh_shape_fc": 2.0,
    "soh_shape_ely": 2.0,
}

get_optimal_action_RB = make_augmented_rb2_policy(**PARAMETERS)
