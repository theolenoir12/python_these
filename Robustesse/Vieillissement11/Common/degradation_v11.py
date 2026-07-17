"""Modeles PEMFC/PEMWE V11 avec separation permanent/recuperable.

Le cout et le remplacement portent sur la perte permanente. La perte
reversible et le conditionnement fini restent des etats de performance, sans
etre extrapoles comme destruction de capital sur 25 ans. Cette convention est
celle du DOE : la vie a +10 % est mesuree apres le break-in/conditioning.
"""

from __future__ import annotations

import math

from . import Init_EMR_MG_v16_python as I
from .electrochemistry import (
    ELY_VOLTAGE_REFERENCE, FC_VOLTAGE_REFERENCE,
    ely_current_density, fc_current_density,
)

MODEL_ID = "v11-doe-rakousky-mccay-colombo-2026-07-16"


ELY_V11 = {
    "steady_2_uvph": 4.8,
    # Rakousky ne donne que les ancres 1 A/cm2 (0) et 2 A/cm2. Une
    # interpolation convexe explicite est retenue et prolongee sans rupture;
    # aucune acceleration arbitraire de 100 uV/h n'est ajoutee au-dessus de 2.
    "stress_exponent": 2.0,
    "breakin_q_uvph": 85.8150585195458,
    "breakin_tau_h": 300.0,
    "reversible_2_uvph": 164.56432769422844,
    "recovery_0_per_h": 0.810789083835173,
    "recovery_1_per_h": 0.002139477822280927,
    # Terme infere pour reproduire le protocole E de Rakousky, non identifiable
    # separement de la duree de recuperation; il doit etre teste a zero.
    "start_uv": 11.7,
    "idle_uvph": 1.5,
}

FC_V11 = {
    "irr_steady_uvph": 1.2,
    "irr_dynamic_uvph": 4.8,
    "rev_steady_uvph": 52.0,
    "rev_dynamic_uvph": 22.0,
    "j_ref": 0.5,
    # Colombo Table 4 decrit la sensibilite de tension aux points de mesure du
    # meme cycle, pas le dommage cause par chacun de ces points. Le dommage
    # McCay est donc applique par heure ON selon le regime, sans pseudo-loi en j;
    # la sensibilite au courant reste dans la polarisation electrochimique.
    "current_exponent": 0.0,
    "steadiness_tau_h": 6.0,
    "change_scale_a_cm2": 0.08,
    "recovery_rest_per_h": 2.0,
    "recovery_operating_per_h": 0.002,
    # Ordre de grandeur historique V10, non identifie par McCay/Colombo;
    # la variante sans penalite de demarrage est obligatoire en sensibilite.
    "start_uv": 20.0,
    "idle_uvph": 3.0,
}


def new_ely_state():
    return {
        "irreversible_uv": 0.0, "breakin_uv": 0.0,
        "reversible_uv": 0.0, "start_uv": 0.0, "idle_uv": 0.0,
        "clock_h": 0.0, "starts": 0, "previous_on": False,
    }


def new_fc_state():
    return {
        "irreversible_uv": 0.0, "reversible_uv": 0.0,
        "start_uv": 0.0, "idle_uv": 0.0, "steadiness": 1.0,
        "previous_j": 0.0, "starts": 0, "previous_on": False,
    }


def _recoverable_step(value, generation_uvph, recovery_per_h, dt_h):
    if recovery_per_h > 1e-12:
        equilibrium = generation_uvph / recovery_per_h
        return equilibrium + (value - equilibrium) * math.exp(-recovery_per_h * dt_h)
    return value + generation_uvph * dt_h


def _ely_rates(j, clock_h):
    p = ELY_V11
    j = max(float(j), 0.0)
    stress = max(j - 1.0, 0.0)
    x = min(stress, 1.0)
    irreversible = p["steady_2_uvph"] * stress ** p["stress_exponent"]
    breakin = p["breakin_q_uvph"] * x * math.exp(-clock_h / p["breakin_tau_h"])
    reversible = p["reversible_2_uvph"] * x
    if j <= 1.0:
        recovery = p["recovery_0_per_h"] + (p["recovery_1_per_h"] - p["recovery_0_per_h"]) * j
    elif j <= 2.0:
        recovery = p["recovery_1_per_h"] * (2.0 - j)
    else:
        recovery = 0.0
    return irreversible, breakin, reversible, max(recovery, 0.0)


def advance_ely_density(state, j, previous_j, dt_h):
    state = dict(state)
    j, previous_j = max(float(j), 0.0), max(float(previous_j), 0.0)
    irr, breakin, rev, recovery = _ely_rates(j, state["clock_h"])
    state["irreversible_uv"] += irr * dt_h
    state["breakin_uv"] += breakin * dt_h
    state["reversible_uv"] = _recoverable_step(state["reversible_uv"], rev, recovery, dt_h)
    on = j > 1e-9
    if on and previous_j <= 1e-9:
        state["start_uv"] += ELY_V11["start_uv"]
        state["starts"] += 1
    if 0.0 < j <= 0.01:
        state["idle_uv"] += ELY_V11["idle_uvph"] * dt_h
    state["clock_h"] += dt_h
    state["previous_on"] = on
    return state


def _fc_steadiness_next(steadiness, j, previous_j, dt_h):
    p = FC_V11
    change = abs(float(j) - float(previous_j))
    after_change = float(steadiness) * math.exp(-change / p["change_scale_a_cm2"])
    return 1.0 - (1.0 - after_change) * math.exp(-dt_h / p["steadiness_tau_h"])


def _fc_rates(j, steadiness):
    p = FC_V11
    j = max(float(j), 0.0)
    if j <= 1e-12:
        return 0.0, 0.0, p["recovery_rest_per_h"]
    factor = (j / p["j_ref"]) ** p["current_exponent"]
    s = min(max(float(steadiness), 0.0), 1.0)
    irreversible = (p["irr_dynamic_uvph"] + s * (p["irr_steady_uvph"] - p["irr_dynamic_uvph"])) * factor
    reversible = (p["rev_dynamic_uvph"] + s * (p["rev_steady_uvph"] - p["rev_dynamic_uvph"])) * factor
    return irreversible, reversible, p["recovery_operating_per_h"]


def advance_fc_density(state, j, previous_j, dt_h):
    state = dict(state)
    j, previous_j = max(float(j), 0.0), max(float(previous_j), 0.0)
    steadiness = _fc_steadiness_next(state["steadiness"], j, previous_j, dt_h)
    irr, rev, recovery = _fc_rates(j, steadiness)
    state["irreversible_uv"] += irr * dt_h
    state["reversible_uv"] = _recoverable_step(state["reversible_uv"], rev, recovery, dt_h)
    on = j > 1e-9
    if on and previous_j <= 1e-9:
        state["start_uv"] += FC_V11["start_uv"]
        state["starts"] += 1
    if 0.0 < j <= 0.05:
        state["idle_uv"] += FC_V11["idle_uvph"] * dt_h
    state["steadiness"], state["previous_j"], state["previous_on"] = steadiness, j, on
    return state


def advance_ely_power(state, power_w, previous_power_w, alpha, dt_h):
    return advance_ely_density(state, ely_current_density(abs(float(power_w)), alpha), ely_current_density(abs(float(previous_power_w)), alpha), dt_h)


def advance_fc_power(state, power_w, previous_power_w, alpha, dt_h):
    return advance_fc_density(state, fc_current_density(abs(float(power_w)), alpha), fc_current_density(abs(float(previous_power_w)), alpha), dt_h)


def permanent_uv(state):
    return float(sum(state.get(key, 0.0) for key in (
        "irreversible_uv", "start_uv", "idle_uv"
    )))


def conditioning_uv(state):
    return float(state.get("breakin_uv", 0.0))


def reversible_uv(state):
    return float(state.get("reversible_uv", 0.0))


def total_uv(state):
    return permanent_uv(state) + conditioning_uv(state) + reversible_uv(state)


def voltage_reference(component):
    if component == "fc":
        return FC_VOLTAGE_REFERENCE
    if component == "ely":
        return ELY_VOLTAGE_REFERENCE
    raise ValueError("composant inconnu: %s" % component)


def soh_permanent(component, state):
    return 1.0 - permanent_uv(state) * 1e-6 / voltage_reference(component)


def soh_operando(component, state):
    return 1.0 - total_uv(state) * 1e-6 / voltage_reference(component)


def state_cost_eur(component, state):
    config = I.FC if component == "fc" else I.ELY
    used = (1.0 - soh_permanent(component, state)) / (1.0 - config["SoH_EoL"])
    return float(max(used, 0.0) * config["cost"])


def aging_snapshot(component, state):
    return {
        "permanent_uv": permanent_uv(state),
        "conditioning_uv": conditioning_uv(state),
        "reversible_uv": reversible_uv(state),
        "total_uv": total_uv(state),
        "soh_permanent": soh_permanent(component, state),
        "soh_operando": soh_operando(component, state),
        "cost_eur": state_cost_eur(component, state),
        "starts": int(state.get("starts", 0)),
        "steadiness": float(state.get("steadiness", 1.0)),
    }
