"""RB2(SoH) V11 : meme dispatch que RB2, deux setpoints fonctions des SoH.

La politique est sans memoire et n'ajoute aucune branche de dispatch. Le test
d'attribution est exact : avec SoH_bat=SoH_fc=SoH_ely=1, les deux consignes
valent celles du parent RB2 et l'action est identique bit a bit.
"""

from __future__ import annotations

from . import Init_EMR_MG_v16_python as I
from .rb2_policy import dispatch_rb2_setpoints


def _wear(soh, eol, shape):
    value = (1.0 - float(soh)) / (1.0 - float(eol))
    return min(max(value, 0.0), 1.0) ** float(shape)


def make_rb2_soh_policy_v11(
    fc_setpoint=0.59,
    ely_setpoint=0.49,
    fc_self=0.0,
    fc_from_bat=0.0,
    fc_from_ely=0.0,
    ely_self=0.0,
    ely_from_bat=0.0,
    ely_from_fc=0.0,
    shape=1.0,
    soh_source="permanent",
    fc_min=0.30,
    fc_max=0.80,
    ely_min=0.25,
    ely_max=0.75,
):
    """Construit une RB2 dont seules les deux consignes dependent des SoH.

    Les six coefficients sont des variations absolues maximales de fraction de
    Pmax entre BoL et EoL de la variable source. Exemple : ``fc_from_bat=0.05``
    ajoute au plus 0.05 au setpoint FC lorsque la batterie atteint son EoL.
    """
    params = {key: value for key, value in locals().items()}
    if float(shape) <= 0.0:
        raise ValueError("shape doit etre strictement positif")
    if soh_source not in ("permanent", "operando"):
        raise ValueError("soh_source doit valoir 'permanent' ou 'operando'")

    def rule(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        aging_context=None,
    ):
        if soh_source == "operando" and aging_context is not None:
            SoH_fc_t = aging_context["fc"]["soh_operando"]
            SoH_ely_t = aging_context["ely"]["soh_operando"]
        w_bat = _wear(SoH_bat_t, I.BAT["SoH_EoL"], shape)
        w_fc = _wear(SoH_fc_t, I.FC["SoH_EoL"], shape)
        w_ely = _wear(SoH_ely_t, I.ELY["SoH_EoL"], shape)

        fc_fraction = (
            float(fc_setpoint) + float(fc_self) * w_fc
            + float(fc_from_bat) * w_bat + float(fc_from_ely) * w_ely
        )
        ely_fraction = (
            float(ely_setpoint) + float(ely_self) * w_ely
            + float(ely_from_bat) * w_bat + float(ely_from_fc) * w_fc
        )
        fc_fraction = min(max(fc_fraction, float(fc_min)), float(fc_max))
        ely_fraction = min(max(ely_fraction, float(ely_min)), float(ely_max))

        return dispatch_rb2_setpoints(
            fc_fraction * I.FC["P_fc_max"],
            ely_fraction * I.ELY["P_ely_max"],
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
            alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
            P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        )

    rule.rb2_parameters = params
    rule.soh_attribution_control = {
        "all_soh_one_equals_parent": True,
        "parent_fc_setpoint": float(fc_setpoint),
        "parent_ely_setpoint": float(ely_setpoint),
    }
    return rule
