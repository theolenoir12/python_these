"""RB2 V11 informee par les etats de vieillissement, sans prevision.

La politique conserve l'arbitrage batterie/H2 de RB2. Elle ne modifie que les
deux consignes H2 a partir d'informations disponibles au temps courant : usure
permanente, perte reversible et historique marche/arret de la politique.
"""

from __future__ import annotations

import math

from . import Init_EMR_MG_v16_python as I
from .rb2_policy import dispatch_rb2_setpoints


def _steps(hours):
    return max(0, int(round(float(hours) / (I.LOAD["Ts"] / 3600.0))))


def _wear_factor(soh, soh_eol, strength, shape):
    wear = (1.0 - float(soh)) / (1.0 - float(soh_eol))
    wear = min(max(wear, 0.0), 1.0)
    return 1.0 - float(strength) * wear ** float(shape)


def make_aging_rb2_policy(
    fc_setpoint=0.59, ely_setpoint=0.49,
    fc_hold_setpoint=0.50, ely_hold_setpoint=0.42,
    fc_min_on_h=4.0, ely_min_on_h=2.0,
    fc_min_off_h=0.0, ely_min_off_h=0.0,
    fc_reversible_trigger_uv=12_000.0,
    ely_reversible_trigger_uv=12_000.0,
    fc_recovery_h=2.0, ely_recovery_h=1.0,
    fc_recovery_soc_min=0.80, ely_recovery_soc_max=0.70,
    permanent_strength_fc=0.02, permanent_strength_ely=0.02,
    permanent_shape=3.0,
):
    """Construit une RB2(Aging) a deux consignes variables.

    ``hold_setpoint`` abaisse temporairement le seuil apres un demarrage afin
    d'eviter un nouvel arret lors d'un creux de charge modere. Un stock de perte
    reversible eleve peut au contraire imposer un repos court, seulement quand
    le SoC rend ce repos energetiquement prudent.
    """
    params = {key: value for key, value in locals().items()}
    min_on_fc, min_on_ely = _steps(fc_min_on_h), _steps(ely_min_on_h)
    min_off_fc, min_off_ely = _steps(fc_min_off_h), _steps(ely_min_off_h)
    recovery_fc, recovery_ely = _steps(fc_recovery_h), _steps(ely_recovery_h)
    memory = {}

    def reset():
        memory.clear()
        memory.update(
            fc_on=False, ely_on=False, fc_on_left=0, ely_on_left=0,
            fc_off_left=0, ely_off_left=0,
            fc_rest_left=0, ely_rest_left=0,
        )

    reset()

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        aging_context=None,
    ):
        context = aging_context or {}
        fc_age = context.get("fc", {})
        ely_age = context.get("ely", {})

        fc_factor = _wear_factor(
            fc_age.get("soh_permanent", SoH_fc_t), I.FC["SoH_EoL"],
            permanent_strength_fc, permanent_shape,
        )
        ely_factor = _wear_factor(
            ely_age.get("soh_permanent", SoH_ely_t), I.ELY["SoH_EoL"],
            permanent_strength_ely, permanent_shape,
        )
        fc_fraction = fc_setpoint
        ely_fraction = ely_setpoint
        if memory["fc_on"] and memory["fc_on_left"] > 0:
            fc_fraction = min(fc_fraction, fc_hold_setpoint)
        if memory["ely_on"] and memory["ely_on_left"] > 0:
            ely_fraction = min(ely_fraction, ely_hold_setpoint)
        if memory["fc_off_left"] > 0:
            memory["fc_off_left"] -= 1
            fc_fraction = 0.0
        if memory["ely_off_left"] > 0:
            memory["ely_off_left"] -= 1
            ely_fraction = 0.0

        if memory["fc_rest_left"] > 0:
            memory["fc_rest_left"] -= 1
            fc_fraction = 0.0
        elif (
            P_tot_ref_t > 0.0 and float(SoC_t) >= fc_recovery_soc_min
            and fc_age.get("reversible_uv", 0.0) >= fc_reversible_trigger_uv
        ):
            memory["fc_rest_left"] = max(recovery_fc - 1, 0)
            fc_fraction = 0.0

        if memory["ely_rest_left"] > 0:
            memory["ely_rest_left"] -= 1
            ely_fraction = 0.0
        elif (
            P_tot_ref_t < 0.0 and float(SoC_t) <= ely_recovery_soc_max
            and ely_age.get("reversible_uv", 0.0) >= ely_reversible_trigger_uv
        ):
            memory["ely_rest_left"] = max(recovery_ely - 1, 0)
            ely_fraction = 0.0

        action, lol = dispatch_rb2_setpoints(
            fc_fraction * I.FC["P_fc_max"] * fc_factor,
            ely_fraction * I.ELY["P_ely_max"] * ely_factor,
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
            alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
            P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )

        fc_on = abs(float(action[1])) > 1e-9
        ely_on = abs(float(action[2])) > 1e-9
        if memory["fc_on"] and not fc_on:
            memory["fc_off_left"] = min_off_fc
        if memory["ely_on"] and not ely_on:
            memory["ely_off_left"] = min_off_ely
        if fc_on and not memory["fc_on"]:
            memory["fc_on_left"] = min_on_fc
        elif memory["fc_on_left"] > 0:
            memory["fc_on_left"] -= 1
        if ely_on and not memory["ely_on"]:
            memory["ely_on_left"] = min_on_ely
        elif memory["ely_on_left"] > 0:
            memory["ely_on_left"] -= 1
        memory["fc_on"], memory["ely_on"] = fc_on, ely_on
        return action, lol

    rule.reset = reset
    rule.rb2_parameters = params
    rule.uses_aging_context = True
    return rule
