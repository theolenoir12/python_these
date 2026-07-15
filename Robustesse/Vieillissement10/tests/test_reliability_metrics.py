import os
import sys

import numpy as np
import pytest

V10 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, V10)

from Common import reliability_metrics as R


def test_lpsp_uses_total_load_as_its_only_denominator(monkeypatch):
    monkeypatch.setitem(R.I.LOAD, "Ts", 3600.0)
    data = {
        "P_dc_load": np.array([10_000.0, 20_000.0]),
        "P_dc_pv": np.array([5_000.0, 25_000.0]),
        "lol_tab": np.array([0.5, 1.0]),
    }

    result = R.compute_reliability_metrics(data)

    assert set(result) == {"eens_kwh", "load_energy_kwh", "lpsp_pct"}
    assert result["eens_kwh"] == pytest.approx(2.5)
    assert result["load_energy_kwh"] == pytest.approx(30.0)
    assert result["lpsp_pct"] == pytest.approx(100.0 * 2.5 / 30.0)
