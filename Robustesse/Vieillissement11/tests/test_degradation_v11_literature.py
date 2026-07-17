import math

import pytest

from Common.degradation_v11 import (
    _ely_rates, _fc_rates, advance_ely_density, advance_fc_density,
    new_ely_state, new_fc_state, permanent_uv, reversible_uv, total_uv,
)


def _replay_ely(pattern, dt_h, duration_h=1009.0):
    state = new_ely_state()
    previous = pattern(0.0)
    t = 0.0
    while t < duration_h - 1e-12:
        step = min(dt_h, duration_h - t)
        current = pattern(t)
        state = advance_ely_density(state, current, previous, step)
        previous = current
        t += step
    return total_uv(state) / duration_h


@pytest.mark.parametrize(
    "name,pattern,dt_h,target_uvph,tolerance",
    [
        ("A", lambda t: 1.0, 0.1, 0.0, 0.05),
        ("B", lambda t: 2.0, 0.1, 194.0, 2.0),
        ("C", lambda t: 2.0 if t % 12.0 < 6.0 else 1.0, 0.1, 65.0, 2.0),
        ("D", lambda t: 2.0 if t % 12.0 < 6.0 else 0.0, 0.1, 16.0, 2.0),
        ("E", lambda t: 2.0 if t % (1.0 / 3.0) < 1.0 / 6.0 else 0.0,
         1.0 / 60.0, 50.0, 2.0),
    ],
)
def test_rakousky_table_2_protocols(name, pattern, dt_h, target_uvph, tolerance):
    del name
    assert _replay_ely(pattern, dt_h) == pytest.approx(target_uvph, abs=tolerance)


def test_pemwe_asymptotic_slope_is_doe_anchor_not_short_test_slope():
    irreversible, breakin, _, _ = _ely_rates(2.0, 20_000.0)
    assert irreversible == pytest.approx(4.8)
    assert breakin < 1e-20


def test_mccay_regimes_are_not_mixed():
    irr_static, rev_static, _ = _fc_rates(0.5, 1.0)
    irr_dynamic, rev_dynamic, _ = _fc_rates(0.5, 0.0)
    assert (irr_static, rev_static) == pytest.approx((1.2, 52.0))
    assert (irr_dynamic, rev_dynamic) == pytest.approx((4.8, 22.0))


def test_reversible_loss_recovers_but_permanent_loss_does_not():
    fc = new_fc_state()
    for _ in range(100):
        fc = advance_fc_density(fc, 0.5, 0.5, 1.0)
    permanent_before = permanent_uv(fc)
    reversible_before = reversible_uv(fc)
    for _ in range(10):
        fc = advance_fc_density(fc, 0.0, 0.0, 1.0)
    assert permanent_uv(fc) == pytest.approx(permanent_before)
    assert reversible_uv(fc) < reversible_before * math.exp(-19.0)


def test_dynamic_pemfc_operando_rate_is_in_colombo_order_of_magnitude():
    state = new_fc_state()
    previous = 0.3
    for hour in range(1000):
        current = 0.3 if hour % 2 == 0 else 0.7
        state = advance_fc_density(state, current, previous, 1.0)
        previous = current
    apparent_rate = total_uv(state) / 1000.0
    assert 10.0 <= apparent_rate <= 35.0
