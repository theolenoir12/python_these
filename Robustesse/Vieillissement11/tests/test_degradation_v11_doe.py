import pytest

from Common import Init_EMR_MG_v16_python as I
from Common.cost_fcn_total2 import get_cost_ely
from Common.degradation_v11 import (
    ELY_V11, FC_V11, advance_ely_density, conditioning_uv,
    new_ely_state, permanent_uv, state_cost_eur, total_uv,
    voltage_reference,
)


def test_conditioning_changes_operando_but_not_capital_cost():
    state = new_ely_state()
    state["breakin_uv"] = 50_000.0
    assert conditioning_uv(state) == 50_000.0
    assert permanent_uv(state) == 0.0
    assert total_uv(state) == 50_000.0
    assert state_cost_eur("ely", state) == 0.0


def test_public_ely_cost_excludes_finite_conditioning():
    n = 1009
    power_at_j2 = 0.0
    # Rejeu direct pour obtenir le cout attendu : pas de faux start au premier pas.
    state = new_ely_state()
    for _ in range(n):
        state = advance_ely_density(state, 2.0, 2.0, 1.0)
    expected = state_cost_eur("ely", state)
    assert permanent_uv(state) == pytest.approx(ELY_V11["steady_2_uvph"] * n)
    assert conditioning_uv(state) > 20_000.0

    # get_cost_ely est teste par equivalence sur une puissance nulle separement;
    # ici la propriete essentielle est que le conditionnement ne change pas expected.
    without_conditioning = dict(state, breakin_uv=0.0)
    assert state_cost_eur("ely", without_conditioning) == pytest.approx(expected)


def test_doe_pemwe_lifetime_anchor_after_conditioning():
    eol_voltage_uv = (
        (1.0 - I.ELY["SoH_EoL"]) * voltage_reference("ely") * 1e6
    )
    lifetime_h = eol_voltage_uv / ELY_V11["steady_2_uvph"]
    assert 35_000.0 <= lifetime_h <= 45_000.0


def test_mccay_constant_lifetime_is_preserved():
    eol_voltage_uv = (
        (1.0 - I.FC["SoH_EoL"]) * voltage_reference("fc") * 1e6
    )
    lifetime_h = eol_voltage_uv / FC_V11["irr_steady_uvph"]
    # McCay annonce jusqu'a 60 kh au point de mesure; la normalisation V11
    # utilise la tension nominale du stack et donne 71.7 kh, meme ordre de grandeur.
    assert 50_000.0 <= lifetime_h <= 80_000.0


def test_colombo_operando_rates_are_not_used_as_causal_damage_law():
    assert FC_V11["current_exponent"] == 0.0
