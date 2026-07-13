"""Metriques reproductibles de la premiere vie des composants."""

from pathlib import Path
import json
import numpy as np

from .electrochemistry import (
    FC_VOLTAGE_REFERENCE, ELY_VOLTAGE_REFERENCE,
    fc_current_density, ely_current_density, fc_pmax, ely_pmax,
)


def _first_life_end(soh):
    soh = np.asarray(soh, dtype=float)
    replacements = np.flatnonzero((soh[1:] == 1.0) & (soh[:-1] != 1.0))
    if replacements.size:
        return int(replacements[0]) + 1, True
    return len(soh) - 1, False


def _h2_metrics(data, component, ts_h):
    end, eol = _first_life_end(data["SoH_" + component])
    power = np.abs(np.asarray(data["P_" + component][:end], dtype=float))
    alpha = np.asarray(data["alpha_" + component][:end], dtype=float)
    if component == "fc":
        pmax = np.asarray(fc_pmax(alpha), dtype=float)
        density = np.asarray(fc_current_density(power, alpha), dtype=float)
        vref = FC_VOLTAGE_REFERENCE
        idle_key = "idling"
    else:
        pmax = np.asarray(ely_pmax(alpha), dtype=float)
        density = np.asarray(ely_current_density(power, alpha), dtype=float)
        vref = ELY_VOLTAGE_REFERENCE
        idle_key = "maintaining"

    load_fraction = np.divide(power, pmax, out=np.zeros_like(power), where=pmax > 0)
    on = power >= 0.0005 * pmax
    on_h = float(np.sum(on) * ts_h)
    calendar_h = float(end * ts_h)
    efph = float(np.sum(load_fraction) * ts_h)
    starts = int(np.sum((~on[:-1]) & on[1:])) if len(on) > 1 else 0
    degradation = {
        key: float(values[end - 1])
        for key, values in data["deg_" + component].items()
    }
    voltage_uv = {key: value * vref * 1e4 for key, value in degradation.items()}
    permanent_uv = (
        voltage_uv["irreversible"] + voltage_uv["start-stop"] + voltage_uv[idle_key]
    )
    bins = {
        "off": density < 1e-9,
        "0-0.5_A_cm2": (density >= 1e-9) & (density < 0.5),
        "0.5-1_A_cm2": (density >= 0.5) & (density < 1.0),
        "1-2_A_cm2": (density >= 1.0) & (density < 2.0),
        "above_2_A_cm2": density >= 2.0,
    }
    return {
        "eol_reached": eol,
        "calendar_h": calendar_h,
        "calendar_years_8760": calendar_h / 8760.0,
        "on_h": on_h,
        "utilization_on": on_h / calendar_h if calendar_h else None,
        "efph": efph,
        "capacity_factor_equivalent": efph / calendar_h if calendar_h else None,
        "energy_kwh": float(np.sum(power) * ts_h / 1000.0),
        "starts": starts,
        "mean_run_h": on_h / starts if starts else None,
        "mean_load_fraction_on": float(np.mean(load_fraction[on])) if np.any(on) else None,
        "mean_current_density_on_A_cm2": float(np.mean(density[on])) if np.any(on) else None,
        "p95_current_density_on_A_cm2": float(np.percentile(density[on], 95)) if np.any(on) else None,
        "hours_by_current_density": {key: float(np.sum(mask) * ts_h) for key, mask in bins.items()},
        "degradation_pct_end": degradation,
        "voltage_loss_uV_end": voltage_uv,
        "equivalent_rate_total_uV_per_on_h": voltage_uv["total"] / on_h if on_h else None,
        "equivalent_rate_permanent_uV_per_on_h": permanent_uv / on_h if on_h else None,
        "equivalent_rate_irreversible_uV_per_on_h": voltage_uv["irreversible"] / on_h if on_h else None,
    }


def compute_first_life_metrics(data, ts_seconds):
    ts_h = float(ts_seconds) / 3600.0
    bat_end, bat_eol = _first_life_end(data["SoH_bat"])
    return {
        "conventions": {
            "year_h": 8760.0,
            "on_threshold_fraction_pmax": 0.0005,
            "efph_definition": "sum(abs(P)/Pmax(alpha)*dt_h)",
            "life_interval": "from initial unit to first reset; censored if no reset",
        },
        "battery": {
            "eol_reached": bat_eol,
            "calendar_h": float(bat_end * ts_h),
            "calendar_years_8760": float(bat_end * ts_h / 8760.0),
        },
        "fc": _h2_metrics(data, "fc", ts_h),
        "ely": _h2_metrics(data, "ely", ts_h),
    }


def format_first_life_metrics(metrics):
    lines = ["METRIQUES DE PREMIERE VIE", ""]
    for key, label in (("fc", "PEMFC"), ("ely", "PEMWE")):
        item = metrics[key]
        lines.extend([
            label,
            "  EoL atteinte : %s" % item["eol_reached"],
            "  Duree calendaire : %.3f ans (%.0f h)" % (
                item["calendar_years_8760"], item["calendar_h"]),
            "  Temps ON : %.0f h" % item["on_h"],
            "  EFPH : %.1f h" % item["efph"],
            "  Demarrages : %d" % item["starts"],
            "  Energie : %.1f kWh" % item["energy_kwh"],
            "  j moyen ON : %.4f A/cm2" % item["mean_current_density_on_A_cm2"],
            "  Taux total equivalent : %.4f uV/h ON" % (
                item["equivalent_rate_total_uV_per_on_h"]),
            "",
        ])
    return "\n".join(lines)


def export_first_life_metrics(metrics, path):
    path = Path(path)
    path.write_text(
        format_first_life_metrics(metrics)
        + "\n\nDONNEES JSON\n"
        + json.dumps(metrics, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
