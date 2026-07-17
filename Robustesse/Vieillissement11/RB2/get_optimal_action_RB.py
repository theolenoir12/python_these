"""RB2 V10 : deux consignes fixes de puissance H2, sans plafond de secours."""

from rb2_policy import make_rb2_policy

FC_SETPOINT = 0.59
ELY_SETPOINT = 0.49

get_optimal_action_RB = make_rb2_policy(FC_SETPOINT, ELY_SETPOINT)
