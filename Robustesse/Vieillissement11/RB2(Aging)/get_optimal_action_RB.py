"""Configuration de reference RB2(Aging) V11, sans prevision."""

from Common.rb2_aging_policy import make_aging_rb2_policy


PARAMETERS = {
    "fc_setpoint": 0.59,
    "ely_setpoint": 0.49,
    "fc_hold_setpoint": 0.59,
    "ely_hold_setpoint": 0.46,
    "fc_min_on_h": 0.0,
    "ely_min_on_h": 1.0,
    "fc_reversible_trigger_uv": float("inf"),
    "ely_reversible_trigger_uv": float("inf"),
    "fc_recovery_h": 0.0,
    "ely_recovery_h": 0.0,
    "permanent_strength_fc": 0.0,
    "permanent_strength_ely": 0.0,
}

get_optimal_action_RB = make_aging_rb2_policy(**PARAMETERS)
