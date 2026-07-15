"""Metriques reproductibles de la premiere vie des composants."""

from pathlib import Path
import json
import numpy as np

from . import Init_EMR_MG_v16_python as I
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
    # La loi de cout ne facture que les transitions OFF -> ON observees entre
    # deux pas. Ne pas ajouter artificiellement le premier etat s'il est ON.
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


def _battery_metrics(data, ts_h):
    soh = np.asarray(data["SoH_bat"], dtype=float)
    end, eol = _first_life_end(soh)
    power = np.asarray(data["P_bat"][:end], dtype=float)
    soc = np.asarray(data["SoC"][:end + 1], dtype=float)
    nominal_energy_kwh = (
        I.BAT["Q_bat"] * I.BAT["v_cell_nom"]
        * I.BAT["series_num"] * I.BAT["parallel_num"] / 1000.0
    )
    discharge_kwh = float(np.clip(power, 0.0, None).sum() * ts_h / 1000.0)
    charge_kwh = float(-np.clip(power, None, 0.0).sum() * ts_h / 1000.0)
    throughput_kwh = discharge_kwh + charge_kwh
    current_a = power / (I.BAT["v_cell_nom"] * I.BAT["series_num"])
    c_rate = np.abs(current_a) / (I.BAT["Q_bat"] * I.BAT["parallel_num"])
    active = c_rate > 1e-12
    soh_end = float(soh[max(end - 1, 0)])
    return {
        "eol_reached": eol,
        "calendar_h": float(end * ts_h),
        "calendar_years_8760": float(end * ts_h / 8760.0),
        "nominal_energy_kwh": float(nominal_energy_kwh),
        "discharge_energy_kwh": discharge_kwh,
        "charge_energy_kwh": charge_kwh,
        "throughput_energy_kwh": throughput_kwh,
        "equivalent_full_cycles_discharge": (
            discharge_kwh / nominal_energy_kwh if nominal_energy_kwh else None
        ),
        "equivalent_full_cycles_throughput": (
            throughput_kwh / (2.0 * nominal_energy_kwh)
            if nominal_energy_kwh else None
        ),
        "mean_abs_c_rate": float(np.mean(c_rate)) if len(c_rate) else None,
        "p95_abs_c_rate": float(np.percentile(c_rate, 95)) if len(c_rate) else None,
        "mean_abs_c_rate_active": (
            float(np.mean(c_rate[active])) if np.any(active) else None
        ),
        "p95_abs_c_rate_active": (
            float(np.percentile(c_rate[active], 95)) if np.any(active) else None
        ),
        "soh_end_or_before_reset": soh_end,
        "capacity_loss_pct_end_or_before_reset": 100.0 * (1.0 - soh_end),
        "mean_soc": float(np.mean(soc)) if len(soc) else None,
        "min_soc": float(np.min(soc)) if len(soc) else None,
        "max_soc": float(np.max(soc)) if len(soc) else None,
        "soh_eol_threshold": float(I.BAT["SoH_EoL"]),
    }


def compute_first_life_metrics(data, ts_seconds):
    ts_h = float(ts_seconds) / 3600.0
    return {
        "conventions": {
            "year_h": 8760.0,
            "on_threshold_fraction_pmax": 0.0005,
            "efph_definition": "sum(abs(P)/Pmax(alpha)*dt_h)",
            "battery_efc_discharge_definition": "sum(max(P_bat,0)*dt)/E_nom",
            "battery_efc_throughput_definition": "sum(abs(P_bat)*dt)/(2*E_nom)",
            "life_interval": "from initial unit to first reset; censored if no reset",
        },
        "battery": _battery_metrics(data, ts_h),
        "fc": _h2_metrics(data, "fc", ts_h),
        "ely": _h2_metrics(data, "ely", ts_h),
    }


def format_first_life_metrics(metrics):
    lines = ["METRIQUES DE PREMIERE VIE", ""]
    battery = metrics["battery"]
    lines.extend([
        "BATTERIE",
        "  EoL atteinte : %s" % battery["eol_reached"],
        "  Duree calendaire : %.3f ans (%.0f h)" % (
            battery["calendar_years_8760"], battery["calendar_h"]),
        "  Energie dechargee / chargee : %.1f / %.1f kWh" % (
            battery["discharge_energy_kwh"], battery["charge_energy_kwh"]),
        "  Cycles equivalents (decharge / throughput) : %.1f / %.1f" % (
            battery["equivalent_full_cycles_discharge"],
            battery["equivalent_full_cycles_throughput"]),
        "  C-rate absolu moyen / p95 : %.4f / %.4f C" % (
            battery["mean_abs_c_rate"], battery["p95_abs_c_rate"]),
        "  C-rate actif moyen / p95 : %.4f / %.4f C" % (
            battery["mean_abs_c_rate_active"],
            battery["p95_abs_c_rate_active"]),
        "  SoH final (ou avant reset) : %.4f" % (
            battery["soh_end_or_before_reset"]),
        "  SoC moyen / min / max : %.4f / %.4f / %.4f" % (
            battery["mean_soc"], battery["min_soc"], battery["max_soc"]),
        "",
    ])
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
