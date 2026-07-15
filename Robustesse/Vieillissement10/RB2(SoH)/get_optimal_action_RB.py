"""RB2 augmentee par le SoH des deux convertisseurs H2."""

from Common.rb2_policy import make_augmented_rb2_policy


# Point du front minimisant la degradation sous LPSP <= 1.10 % sur 25 ans.
# Le minimum du cout unifie avec VoLL=3 EUR/kWh reste le cas nul RB2.
PARAMETERS = {
    "fc_setpoint": 0.59,
    "ely_setpoint": 0.49,
    "soh_mode": "normalized_wear",
    "soh_strength_fc": 0.25,
    "soh_strength_ely": 0.25,
    "soh_shape_fc": 1.0,
    "soh_shape_ely": 1.0,
}

get_optimal_action_RB = make_augmented_rb2_policy(**PARAMETERS)
