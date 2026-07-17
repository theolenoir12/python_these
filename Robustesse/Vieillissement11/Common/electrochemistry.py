"""Point d'entree unique des modeles electrochimiques de Vieillissement10.

Toutes les puissances sont des puissances de stack en W, les courants des
courants de stack en A et les tensions des tensions par cellule en V.
"""

import numpy as np
from scipy.optimize import brentq

from .Init_EMR_MG_v16_python import A, B, ELY, FC, S, j_in

FC_I_MAX_INTERCEPT = 238.8252
FC_I_MAX_SLOPE = 234.8032
ELY_I_MAX_INTERCEPT = 732.6
ELY_I_MAX_SLOPE = 732.6


def fc_i_max(alpha):
    return np.maximum(0.0, FC_I_MAX_INTERCEPT - FC_I_MAX_SLOPE * np.asarray(alpha))


def ely_i_max(alpha):
    return np.maximum(0.0, ELY_I_MAX_INTERCEPT - ELY_I_MAX_SLOPE * np.asarray(alpha))


def fc_voltage_cell(current, alpha=0.0):
    current = np.asarray(current, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    density = current / (S * FC['n_parallel'])
    concentration = np.maximum(1e-12, 1.0 - density / FC['j_L'] / (1.0 - alpha))
    return (FC['E_0'] - FC['R'] * (1.0 + alpha) * current / FC['n_parallel']
            - A * FC['T'] * np.log((density + j_in) / FC['j_0'])
            - B * FC['T'] * np.log(concentration))


def ely_voltage_cell(current, alpha=0.0):
    current = np.asarray(current, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    density = current / (S * ELY['n_parallel'])
    concentration = np.maximum(1e-12, 1.0 - density / ELY['j_L'] / (1.0 - alpha))
    return (ELY['E_0'] + ELY['R'] * (1.0 + alpha) * current / ELY['n_parallel']
            + A * ELY['T'] * np.log((density + j_in) / ELY['j_0'])
            + B * ELY['T'] * np.log(concentration))


def fc_power(current, alpha=0.0):
    return np.asarray(current, dtype=float) * FC['n_parallel'] * FC['n_series'] * fc_voltage_cell(current, alpha)


def ely_power(current, alpha=0.0):
    return np.asarray(current, dtype=float) * ELY['n_parallel'] * ELY['n_series'] * ely_voltage_cell(current, alpha)


def fc_pmax(alpha):
    return fc_power(fc_i_max(alpha), alpha)


def ely_pmax(alpha):
    return ely_power(ely_i_max(alpha), alpha)


def _current_from_power_scalar(power, alpha, power_function, imax_function):
    target = abs(float(power))
    alpha = float(alpha)
    if target <= 0.0:
        return 0.0
    imax = float(imax_function(alpha))
    pmax = float(power_function(imax, alpha))
    if target >= pmax * (1.0 - 1e-12):
        return imax
    return brentq(lambda current: float(power_function(current, alpha)) - target,
                  0.0, imax, xtol=1e-9, rtol=1e-11)


def _array_or_scalar_solver(power, alpha, power_function, imax_function):
    power_array, alpha_array = np.broadcast_arrays(
        np.asarray(power, dtype=float), np.asarray(alpha, dtype=float))
    result = np.empty(power_array.shape, dtype=float)
    for index in np.ndindex(power_array.shape):
        result[index] = _current_from_power_scalar(
            power_array[index], alpha_array[index], power_function, imax_function)
    return float(result) if result.ndim == 0 else result


def fc_current_from_power(power, alpha=0.0):
    return _array_or_scalar_solver(power, alpha, fc_power, fc_i_max)


def ely_current_from_power(power, alpha=0.0):
    return _array_or_scalar_solver(power, alpha, ely_power, ely_i_max)


def fc_current_density(power, alpha=0.0):
    return fc_current_from_power(power, alpha) / (S * FC['n_parallel'])


def ely_current_density(power, alpha=0.0):
    return ely_current_from_power(power, alpha) / (S * ELY['n_parallel'])


FC_I_NOMINAL = 0.75 * FC_I_MAX_INTERCEPT
ELY_I_NOMINAL = 0.75 * ELY_I_MAX_INTERCEPT
FC_VOLTAGE_REFERENCE = float(fc_voltage_cell(FC_I_NOMINAL, 0.0))
ELY_VOLTAGE_REFERENCE = float(ely_voltage_cell(ELY_I_NOMINAL, 0.0))


# Solveurs exacts conserves pour validation ; les simulations utilisent les LUT.
fc_current_from_power_exact = fc_current_from_power
ely_current_from_power_exact = ely_current_from_power

def fc_current_density_exact(power, alpha=0.0):
    return fc_current_from_power_exact(power, alpha) / (S * FC['n_parallel'])

def ely_current_density_exact(power, alpha=0.0):
    return ely_current_from_power_exact(power, alpha) / (S * ELY['n_parallel'])

_ALPHA_GRID = np.linspace(0.0, 0.30, 301)
_LOAD_GRID = np.linspace(0.0, 1.0, 2001)
_DENSITY_LUT = {}


def _build_density_lut(kind):
    if kind in _DENSITY_LUT:
        return _DENSITY_LUT[kind]
    if kind == "fc":
        imax_function, power_function, area, parallel = fc_i_max, fc_power, S, FC['n_parallel']
    else:
        imax_function, power_function, area, parallel = ely_i_max, ely_power, S, ELY['n_parallel']
    table = np.empty((len(_ALPHA_GRID), len(_LOAD_GRID)), dtype=float)
    for row, alpha in enumerate(_ALPHA_GRID):
        imax = float(imax_function(alpha))
        currents = np.linspace(0.0, imax, len(_LOAD_GRID))
        powers = np.asarray(power_function(currents, alpha), dtype=float)
        fractions = powers / powers[-1] if powers[-1] > 0 else np.zeros_like(powers)
        table[row] = np.interp(_LOAD_GRID, fractions, currents / (area * parallel))
    _DENSITY_LUT[kind] = table
    return table


def _density_from_power_lut(power, alpha, kind):
    if np.ndim(power) == 0 and np.ndim(alpha) == 0:
        power_value, alpha_value = abs(float(power)), float(alpha)
        pmax_value = float(fc_pmax(alpha_value) if kind == "fc" else ely_pmax(alpha_value))
        fraction = min(max(power_value / pmax_value if pmax_value > 0 else 0.0, 0.0), 1.0)
        apos = min(max(alpha_value, 0.0), 0.30) / 0.30 * (len(_ALPHA_GRID) - 1)
        fpos = fraction * (len(_LOAD_GRID) - 1)
        ia, jf = int(apos), int(fpos)
        ia1, jf1 = min(ia + 1, len(_ALPHA_GRID) - 1), min(jf + 1, len(_LOAD_GRID) - 1)
        wa, wf = apos - ia, fpos - jf
        table = _build_density_lut(kind)
        d0 = table[ia, jf] * (1.0 - wf) + table[ia, jf1] * wf
        d1 = table[ia1, jf] * (1.0 - wf) + table[ia1, jf1] * wf
        return float(d0 * (1.0 - wa) + d1 * wa)
    power_array, alpha_array = np.broadcast_arrays(
        np.abs(np.asarray(power, dtype=float)), np.asarray(alpha, dtype=float))
    pmax = fc_pmax(alpha_array) if kind == "fc" else ely_pmax(alpha_array)
    fraction = np.divide(power_array, pmax, out=np.zeros_like(power_array), where=pmax > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    apos = np.clip(alpha_array, _ALPHA_GRID[0], _ALPHA_GRID[-1]) / 0.30 * (len(_ALPHA_GRID) - 1)
    fpos = fraction * (len(_LOAD_GRID) - 1)
    ia = np.floor(apos).astype(int)
    jf = np.floor(fpos).astype(int)
    ia1 = np.minimum(ia + 1, len(_ALPHA_GRID) - 1)
    jf1 = np.minimum(jf + 1, len(_LOAD_GRID) - 1)
    wa, wf = apos - ia, fpos - jf
    table = _build_density_lut(kind)
    d0 = table[ia, jf] * (1.0 - wf) + table[ia, jf1] * wf
    d1 = table[ia1, jf] * (1.0 - wf) + table[ia1, jf1] * wf
    result = d0 * (1.0 - wa) + d1 * wa
    return float(result) if result.ndim == 0 else result


def fc_current_density(power, alpha=0.0):
    return _density_from_power_lut(power, alpha, "fc")


def ely_current_density(power, alpha=0.0):
    return _density_from_power_lut(power, alpha, "ely")


def fc_current_from_power(power, alpha=0.0):
    return fc_current_density(power, alpha) * S * FC['n_parallel']


def ely_current_from_power(power, alpha=0.0):
    return ely_current_density(power, alpha) * S * ELY['n_parallel']

