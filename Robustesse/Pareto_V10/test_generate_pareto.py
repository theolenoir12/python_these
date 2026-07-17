import numpy as np

import generate_pareto as gp


def test_v10_sources_and_active_soh_point():
    points = gp.load_base_points()
    front = gp.pareto_front(gp.load_soh_rows())
    selected = gp.select_soh_point(front, points["RB2"]["unified_keur"])

    assert np.isclose(points["RB2"]["lpsp_pct"], 0.7750)
    assert np.isclose(points["RB2"]["degradation_keur"], 63.3746)
    assert selected["unified_keur"] <= points["RB2"]["unified_keur"] + 1e-6
    assert np.isclose(selected["lpsp_pct"], 0.771986)
    assert np.isclose(selected["degradation_keur"], 63.356314)
    assert len(front) >= 10


def test_v10_isocost_slope_uses_total_load_lpsp():
    slope = gp.iso_slope_keur_per_lpsp_point(gp.load_base_points())
    expected = 3.0 * 4058.4885 / 1000.0 / 0.7750
    assert np.isclose(slope, expected)
    assert slope > 15.0
