"""Metriques de fiabilite communes aux simulations Vieillissement10."""

import numpy as np

from . import Init_EMR_MG_v16_python as I


def compute_reliability_metrics(data):
    """Calcule l'EENS et la LPSP rapportee a la charge totale.

    Toutes les energies sont evaluees au bus DC et sur le meme pas temporel.
    La convention unique est celle du manuscrit :

        LPSP = energie non servie / energie totale demandee.

    La production PV sert uniquement a determiner la puissance residuelle dont
    le manque eventuel constitue l'EENS ; elle n'apparait pas au denominateur.
    """
    dt_h = I.LOAD["Ts"] / 3600.0
    load_kw = np.clip(np.asarray(data["P_dc_load"], dtype=float) / 1000.0, 0.0, None)
    pv_kw = np.asarray(data["P_dc_pv"], dtype=float) / 1000.0
    residual_kw = np.clip(load_kw - pv_kw, 0.0, None)
    lol = np.clip(np.asarray(data["lol_tab"], dtype=float), 0.0, 1.0)
    unserved_kw = residual_kw * lol

    eens_kwh = float(unserved_kw.sum() * dt_h)
    load_kwh = float(load_kw.sum() * dt_h)
    return {
        "eens_kwh": eens_kwh,
        "load_energy_kwh": load_kwh,
        "lpsp_pct": (
            100.0 * eens_kwh / load_kwh if load_kwh > 0.0 else 0.0
        ),
    }
