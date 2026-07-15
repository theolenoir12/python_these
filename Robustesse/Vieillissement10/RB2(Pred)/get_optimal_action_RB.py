"""RB2 augmentee par une prediction du profil de puissance nette."""

from Common.rb2_policy import make_augmented_rb2_policy


# Optimum robuste V10 sur trois graines, avec l'incertitude du backtest agregee.
PARAMETERS = {
    "fc_setpoint": 0.59,
    "ely_setpoint": 0.49,
    "forecast_enabled": True,
    "forecast_horizon_h": 24.0,
    "forecast_soc_target": 0.99,
    "forecast_noise_enabled": True,
    "forecast_bias_kwh": -2.32,
    "forecast_sigma_kwh": 39.38,
    "forecast_noise_rho": 0.0,
    "forecast_hysteresis_sigma": 1.5,
    "forecast_min_dwell_h": 0.0,
    "forecast_seed": 0,
}

get_optimal_action_RB = make_augmented_rb2_policy(**PARAMETERS)
