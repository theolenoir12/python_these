"""Briques communes aux politiques RB2, toutes exprimees par deux setpoints H2.

Les augmentations SoH, RUL et prediction ne modifient pas l'arbitrage RB2 :
elles calculent uniquement les consignes PEMFC et PEMWE transmises a
``dispatch_rb2_setpoints``. Les seules saturations restantes sont physiques
(stock H2, puissance instantanee et defaillances via ``get_lol``).
"""

from __future__ import annotations

import math

import numpy as np

from . import Init_EMR_MG_v16_python as I
from .get_lol import get_lol


def dispatch_rb2_setpoints(
    p_fc_set, p_ely_set, SoC_t, P_tot_ref_t, defaillances, lol_tab,
    alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init,
    P_fc_max_t, P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
):
    """Applique le dispatch RB2 historique a deux consignes en watts."""
    del lol_tab, alpha_fc_t, alpha_ely_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t

    p_fc_set = max(float(p_fc_set), 0.0)
    p_ely_set = max(float(p_ely_set), 0.0)
    dt_h = I.LOAD["Ts"] / 3600.0

    # Contraintes physiques communes, et non parametres supplementaires de RB2.
    p_fc_h2_max = (
        max(E_h2_t, 0.0) / dt_h * I.FC["eff"] * I.CONV["eta"] * 1000.0
    )
    p_ely_h2_max = (
        max(E_h2_init - E_h2_t, 0.0) / dt_h
        / (I.ELY["eff"] * I.CONV["eta"]) * 1000.0
    )

    p_fc = 0.0
    p_ely = 0.0
    if P_tot_ref_t > 0.0:
        p_fc_available = min(p_fc_set, p_fc_h2_max)
        if P_tot_ref_t > p_fc_available:
            p_fc = p_fc_available
            p_bat = P_tot_ref_t - p_fc_available
        else:
            p_bat = P_tot_ref_t
    elif P_tot_ref_t < 0.0:
        p_ely_available = min(p_ely_set, p_ely_h2_max)
        if P_tot_ref_t < -p_ely_available:
            p_ely = -p_ely_available
            p_bat = P_tot_ref_t + p_ely_available
        else:
            p_bat = P_tot_ref_t
    else:
        p_bat = 0.0

    if "FC" in defaillances and P_tot_ref_t > 0.0:
        p_fc = 0.0
        p_bat = P_tot_ref_t
    if "ELY" in defaillances and P_tot_ref_t < 0.0:
        p_ely = 0.0
        p_bat = P_tot_ref_t

    return get_lol(
        SoC_t, (p_bat, p_fc, p_ely), P_tot_ref_t, defaillances,
        E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
    )


def make_rb2_policy(fc_setpoint, ely_setpoint):
    """Construit la RB2 statique, parametree par deux fractions nominales."""
    fc_setpoint = float(fc_setpoint)
    ely_setpoint = float(ely_setpoint)

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
    ):
        return dispatch_rb2_setpoints(
            fc_setpoint * I.FC["P_fc_max"],
            ely_setpoint * I.ELY["P_ely_max"],
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
            alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
            P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )

    rule.rb2_parameters = {
        "fc_setpoint": fc_setpoint,
        "ely_setpoint": ely_setpoint,
        "layers": (),
    }
    return rule


def _bounded_power_factor(value, reference, exponent):
    """Loi (value/reference)^exponent bornee a [0, 1], avec cas nul exact."""
    if exponent == 0.0:
        return 1.0
    value = float(value)
    if not math.isfinite(value):
        return 1.0
    return min(max(value, 0.0) / float(reference), 1.0) ** float(exponent)


def _soh_factor(soh, exponent):
    if exponent == 0.0:
        return 1.0
    return min(max(float(soh), 0.0), 1.0) ** float(exponent)


def _normalized_wear_factor(soh, soh_eol, strength, shape):
    """Facteur 1-strength*x^shape, avec x=vie SoH consommee normalisee.

    Contrairement a ``SoH**gamma``, x parcourt exactement [0, 1] entre l'etat
    neuf et l'EoL. ``shape > 1`` concentre donc la baisse du setpoint en fin de
    vie, tandis que ``strength`` fixe directement la baisse maximale a l'EoL.
    """
    strength = float(strength)
    shape = float(shape)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("soh_strength doit appartenir a [0, 1]")
    if shape <= 0.0:
        raise ValueError("soh_shape doit etre strictement positif")
    if strength == 0.0:
        return 1.0
    wear = (1.0 - float(soh)) / (1.0 - float(soh_eol))
    wear = min(max(wear, 0.0), 1.0)
    return 1.0 - strength * wear ** shape


class _ForecastGate:
    """Etat de l'inhibition ELY declenchee par l'energie nette prevue."""

    def __init__(
        self, enabled, horizon_h, soc_target, noise_enabled, bias_kwh,
        sigma_kwh, noise_rho, hysteresis_sigma, min_dwell_h, seed,
    ):
        self.enabled = bool(enabled)
        self.horizon_steps = max(
            1, int(round(float(horizon_h) / (I.LOAD["Ts"] / 3600.0)))
        )
        self.soc_target = float(soc_target)
        self.noise_enabled = bool(noise_enabled)
        self.bias_kwh = float(bias_kwh)
        self.sigma_kwh = max(float(sigma_kwh), 0.0)
        self.noise_rho = float(noise_rho)
        self.hysteresis_sigma = max(float(hysteresis_sigma), 0.0)
        self.min_dwell_steps = max(
            0, int(round(float(min_dwell_h) / (I.LOAD["Ts"] / 3600.0)))
        )
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.state_on = False
        self.dwell = 0
        self.eps = 0.0

    def set_noise_seed(self, seed):
        self.seed = int(seed)
        self.reset()

    def __call__(self, future_net_w, soc):
        if not self.enabled or future_net_w is None or float(soc) >= self.soc_target:
            return False
        future = np.asarray(future_net_w, dtype=float)[:self.horizon_steps]
        if future.size == 0:
            return False

        net_wh = float(future.sum()) * I.LOAD["Ts"] / 3600.0
        if self.noise_enabled:
            xi = float(self.rng.standard_normal())
            rho = min(max(self.noise_rho, 0.0), 0.999999)
            self.eps = rho * self.eps + math.sqrt(1.0 - rho * rho) * xi
            net_wh += (self.bias_kwh + self.sigma_kwh * self.eps) * 1000.0

        threshold_wh = self.hysteresis_sigma * self.sigma_kwh * 1000.0
        if threshold_wh == 0.0 and self.min_dwell_steps == 0:
            return net_wh > 0.0
        if self.dwell > 0:
            self.dwell -= 1
        elif not self.state_on and net_wh > threshold_wh:
            self.state_on = True
            self.dwell = self.min_dwell_steps
        elif self.state_on and net_wh < -threshold_wh:
            self.state_on = False
            self.dwell = self.min_dwell_steps
        return self.state_on


def make_augmented_rb2_policy(
    fc_setpoint=0.59, ely_setpoint=0.49,
    soh_gamma_fc=0.0, soh_gamma_ely=0.0,
    soh_mode="power",
    soh_strength_fc=0.0, soh_strength_ely=0.0,
    soh_shape_fc=1.0, soh_shape_ely=1.0,
    rul_ref_fc_days=1.0, rul_ref_ely_days=1.0,
    rul_gamma_fc=0.0, rul_gamma_ely=0.0,
    forecast_enabled=False, forecast_horizon_h=18.0,
    forecast_soc_target=0.99, forecast_noise_enabled=False,
    forecast_bias_kwh=-2.32, forecast_sigma_kwh=39.38,
    forecast_noise_rho=0.0, forecast_hysteresis_sigma=0.0,
    forecast_min_dwell_h=0.0, forecast_seed=0,
):
    """Construit RB2(SoH/RUL/Pred) sans ajouter d'autre variable de commande.

    En mode historique ``power``, les deux consignes sont :

      P_FC  = c_FC  P_FC,nom  SoH_FC^gS  min(RUL_FC/Rref_FC,1)^gR
      P_ELY = c_ELY P_ELY,nom SoH_ELY^gS min(RUL_ELY/Rref_ELY,1)^gR

    En mode ``normalized_wear``, le facteur SoH vaut
    1 - strength * ((1 - SoH)/(1 - SoH_EoL))^shape. La baisse maximale a
    l'EoL est ainsi directement controlee par ``strength``.

    La couche de prediction ne cree pas un plafond : elle inhibe la consigne
    ELY (setpoint nul) lorsque le deficit net cumule prevu justifie de conserver
    le surplus present dans la batterie.
    """
    if soh_mode not in ("power", "normalized_wear"):
        raise ValueError("soh_mode inconnu : %s" % soh_mode)
    params = {key: value for key, value in locals().items()}
    gate = _ForecastGate(
        forecast_enabled, forecast_horizon_h, forecast_soc_target,
        forecast_noise_enabled, forecast_bias_kwh, forecast_sigma_kwh,
        forecast_noise_rho, forecast_hysteresis_sigma,
        forecast_min_dwell_h, forecast_seed,
    )

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        P_tot_ref_future=None,
    ):
        if soh_mode == "normalized_wear":
            fc_soh_factor = _normalized_wear_factor(
                SoH_fc_t, I.FC["SoH_EoL"], soh_strength_fc, soh_shape_fc
            )
            ely_soh_factor = _normalized_wear_factor(
                SoH_ely_t, I.ELY["SoH_EoL"], soh_strength_ely, soh_shape_ely
            )
        else:
            fc_soh_factor = _soh_factor(SoH_fc_t, soh_gamma_fc)
            ely_soh_factor = _soh_factor(SoH_ely_t, soh_gamma_ely)
        fc_factor = fc_soh_factor * _bounded_power_factor(
            RUL_fc_t, rul_ref_fc_days, rul_gamma_fc
        )
        ely_factor = ely_soh_factor * _bounded_power_factor(
            RUL_ely_t, rul_ref_ely_days, rul_gamma_ely
        )
        p_fc_set = float(fc_setpoint) * I.FC["P_fc_max"] * fc_factor
        p_ely_set = float(ely_setpoint) * I.ELY["P_ely_max"] * ely_factor
        if gate(P_tot_ref_future, SoC_t):
            p_ely_set = 0.0

        return dispatch_rb2_setpoints(
            p_fc_set, p_ely_set, SoC_t, P_tot_ref_t, defaillances, lol_tab,
            alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init,
            P_fc_max_t, P_ely_max_t, RUL_fc_t, RUL_ely_t,
            SoH_fc_t, SoH_ely_t,
        )

    rule.reset = gate.reset
    rule.set_noise_seed = gate.set_noise_seed
    rule.forecast_horizon_steps = gate.horizon_steps if forecast_enabled else 0
    rule.rb2_parameters = params
    return rule
