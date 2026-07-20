"""MPC deterministe online pour Vieillissement11, nominal PEMWE ``p=2``.

Le controleur resout a chaque heure un MILP a horizon glissant et n'applique
que la premiere action. Le probleme interne contient explicitement le bilan de
puissance, le delestage, l'ecretage, les dynamiques batterie/H2 et les etats
marche/arret. Le cout PEMWE quadratique V11 est interpole par segments affines
convexes en densite de courant. Tous les resultats finaux restent evalues par
la boucle physique et le ledger V11 exacts.

Deux variantes partagent exactement cette formulation :

``no_soh``
    Les contraintes utilisent l'etat physique courant, mais les poids d'usure
    ne dependent pas de la sante.
``soh``
    Les memes couts internes sont multiplies par la marge de sante normalisee
    ``h**(-beta)``. Avec ``beta_fc=beta_ely=0``, les deux variantes sont
    strictement identiques : c'est le test nul d'attribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from Common import Init_EMR_MG_v16_python as I
from Common.cost_fcn_total2 import deg_cumul1, deg_cumul2
from Common.degradation_v11 import ELY_V11, FC_V11, MODEL_ID, voltage_reference
from Common.electrochemistry import (
    ely_current_density, ely_i_max, ely_pmax, ely_power, fc_pmax, fc_power,
)
from Common.get_lol import get_lol


NOMINAL_ELY_STRESS_EXPONENT = 2.0
if not np.isclose(ELY_V11["stress_exponent"], NOMINAL_ELY_STRESS_EXPONENT):
    raise RuntimeError(
        "MPC V11 attribue au nominal p=2, mais degradation_v11 expose p=%r"
        % ELY_V11["stress_exponent"]
    )


ETA = float(I.CONV["eta"])
BAT_EFF = float(I.BAT["eff"])
DT_H = float(I.LOAD["Ts"] / 3600.0)
BAT_NOMINAL_WH = float(
    I.BAT["series_num"] * I.BAT["parallel_num"]
    * I.BAT["Q_bat"] * I.BAT["v_cell_nom"]
)
BAT_K_DISCHARGE = 1.0 / (ETA * BAT_EFF)
BAT_K_CHARGE = ETA * BAT_EFF
SOC_MIN = 0.20001
SOC_MAX = 0.99499


def _permanent_uv_to_eur(component: str, uv: float | np.ndarray) -> np.ndarray:
    config = I.FC if component == "fc" else I.ELY
    return (
        np.asarray(uv, dtype=float) * 1e-6 / voltage_reference(component)
        / (1.0 - config["SoH_EoL"]) * config["cost"]
    )


def _battery_wear_eur_per_wh() -> float:
    """Pente moyenne du cout batterie V11 sur la fenetre de SoC utile."""
    cumulative = np.interp([SOC_MIN, SOC_MAX], deg_cumul1, deg_cumul2)
    total_eur = (
        abs(float(cumulative[1] - cumulative[0])) / 2.15 * 1e-6
        / (1.0 - I.BAT["SoH_EoL"]) * I.BAT["cost"]
    )
    return total_eur / ((SOC_MAX - SOC_MIN) * BAT_NOMINAL_WH)


BAT_WEAR_EUR_PER_WH = _battery_wear_eur_per_wh()


@dataclass(frozen=True)
class MPCConfig:
    """Configuration immuable d'un controleur MPC V11."""

    horizon_steps: int = 6
    forecast_mode: str = "perfect"  # perfect | persistence | noisy
    forecast_seed: int = 2026
    forecast_sigma_energy_kwh_18h: float = 39.38
    forecast_bias_energy_kwh_18h: float = -2.32
    forecast_error_rho: float = 0.80
    forecast_sigma_scale: float = 1.0
    voll_eur_per_kwh: float = 3.0
    health_mode: str = "no_soh"     # no_soh | soh
    beta_fc: float = 0.0
    beta_ely: float = 0.0
    health_floor: float = 0.05
    terminal_bat_eur_per_kwh: float = 0.60
    terminal_h2_eur_per_kwh: float = 1.00
    battery_wear_scale: float = 1.0
    fc_wear_scale: float = 1.0
    ely_wear_scale: float = 1.0
    fc_dynamic_scale: float = 1.0
    high_soc_hold_eur: float = 0.0
    high_soc_knee: float = 0.60
    mip_rel_gap: float = 1e-5
    time_limit_s: float = 5.0
    fail_hard: bool = True

    def __post_init__(self) -> None:
        if self.horizon_steps < 2:
            raise ValueError("horizon_steps doit etre >= 2")
        if self.forecast_mode not in {"perfect", "persistence", "noisy"}:
            raise ValueError("forecast_mode doit valoir perfect, persistence ou noisy")
        if self.forecast_sigma_energy_kwh_18h < 0.0:
            raise ValueError("forecast_sigma_energy_kwh_18h doit etre non negatif")
        if self.forecast_sigma_scale < 0.0:
            raise ValueError("forecast_sigma_scale doit etre non negatif")
        if not 0.0 <= self.forecast_error_rho < 1.0:
            raise ValueError("forecast_error_rho doit appartenir a [0,1[")
        if self.health_mode not in {"no_soh", "soh"}:
            raise ValueError("health_mode doit valoir no_soh ou soh")
        if min(self.beta_fc, self.beta_ely) < 0.0:
            raise ValueError("les exposants de sante doivent etre positifs")
        if not 0.0 < self.health_floor <= 1.0:
            raise ValueError("health_floor doit appartenir a ]0,1]")
        if self.voll_eur_per_kwh <= 0.0:
            raise ValueError("la VoLL interne doit etre positive")
        if self.time_limit_s <= 0.0:
            raise ValueError("time_limit_s doit etre positif")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class _LinearRows:
    """Petit assembleur sparse de contraintes lower <= A x <= upper."""

    def __init__(self, n_variables: int):
        self.n_variables = int(n_variables)
        self.row: list[int] = []
        self.col: list[int] = []
        self.value: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, terms: list[tuple[np.ndarray | int, np.ndarray | float]],
            lower: float = -np.inf, upper: float = np.inf) -> None:
        r = len(self.lower)
        for indices, values in terms:
            ii = np.atleast_1d(indices).astype(int)
            vv = np.broadcast_to(np.asarray(values, dtype=float), ii.shape)
            self.row.extend([r] * len(ii))
            self.col.extend(ii.tolist())
            self.value.extend(vv.tolist())
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.value, (self.row, self.col)),
            shape=(len(self.lower), self.n_variables),
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


class MPCPolicyV11:
    """Politique callable compatible avec ``Common.main_init_and_loop``."""

    def __init__(self, config: MPCConfig | None = None):
        self.config = config or MPCConfig()
        self.forecast_horizon_steps = self.config.horizon_steps
        self.policy_id = (
            f"mpc_v11_p2_{self.config.health_mode}_h{self.config.horizon_steps}_"
            f"{self.config.forecast_mode}_{self.config.fingerprint}"
        )
        self.reset()

    def reset(self) -> None:
        self.previous_fc_w = 0.0
        self.previous_ely_w = 0.0
        self.previous_fc_on = 0
        self.previous_ely_on = 0
        self.calls = 0
        self.forecast_origins = 0
        self.failures = 0
        self.solve_seconds = 0.0
        self.max_solve_seconds = 0.0
        self.max_constraint_residual = 0.0
        self.max_lol = 0.0
        self.lol_above_one_steps = 0
        self.planned_shed_kwh = 0.0
        self.planned_curtail_kwh = 0.0
        self.last_solution: dict[str, Any] | None = None

    def _forecast_error_w(self, n_future: int) -> np.ndarray:
        """Erreur additive de prevision [W] pour les echeances futures.

        Calibration : biais=-2.32 kWh et sigma=39.38 kWh sur l'energie nette
        cumulee a 18 h (backtest historique du projet). L'erreur croit avec la
        racine de l'echeance et suit un AR(1) le long des leads. Le pas courant
        reste toujours mesure exactement.

        SeedSequence(seed, origine) apparie les realisations entre politiques
        SoH/non-SoH et horizons. Les origines successives sont independantes,
        choix conservatif pour constituer une borne haute d'incertitude.
        """
        n = int(n_future)
        if n <= 0:
            return np.empty(0, dtype=float)
        rho = float(self.config.forecast_error_rho)
        calibration_h = 18
        leads = np.arange(1, n + 1, dtype=float)
        weights = np.sqrt(np.minimum(leads, calibration_h) / calibration_h)
        weights18 = np.sqrt(
            np.arange(1, calibration_h + 1, dtype=float) / calibration_h)
        indices = np.arange(calibration_h)
        correlation = rho ** np.abs(indices[:, None] - indices[None, :])
        energy_norm = DT_H / 1000.0 * float(
            np.sqrt(weights18 @ correlation @ weights18))
        sigma_w = (
            float(self.config.forecast_sigma_scale)
            * float(self.config.forecast_sigma_energy_kwh_18h)
            / energy_norm
        ) if energy_norm > 0.0 else 0.0
        sequence = np.random.SeedSequence([
            int(self.config.forecast_seed) & 0xFFFFFFFF,
            int(self.forecast_origins) & 0xFFFFFFFF,
        ])
        innovations = np.random.default_rng(sequence).standard_normal(n)
        standardized = np.empty(n, dtype=float)
        standardized[0] = innovations[0]
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho * rho))
        for index in range(1, n):
            standardized[index] = (
                rho * standardized[index - 1]
                + innovation_scale * innovations[index]
            )
        bias_w = (
            float(self.config.forecast_bias_energy_kwh_18h)
            / calibration_h / DT_H * 1000.0
        )
        self.forecast_origins += 1
        return bias_w + sigma_w * weights * standardized

    @staticmethod
    def _health_margin(soh: float, eol: float, floor: float) -> float:
        margin = (float(soh) - float(eol)) / (1.0 - float(eol))
        return min(1.0, max(float(floor), margin))

    def _wear_factors(self, soh_fc: float, soh_ely: float) -> tuple[float, float]:
        if self.config.health_mode == "no_soh":
            return 1.0, 1.0
        fc_margin = self._health_margin(
            soh_fc, I.FC["SoH_EoL"], self.config.health_floor)
        ely_margin = self._health_margin(
            soh_ely, I.ELY["SoH_EoL"], self.config.health_floor)
        # Cas nul exact : x**0 vaut exactement 1.0.
        return fc_margin ** (-self.config.beta_fc), ely_margin ** (-self.config.beta_ely)

    @staticmethod
    def _ely_piecewise(alpha_ely: float, ecap: float) -> tuple[np.ndarray, np.ndarray]:
        """Bornes de segments [W DC] et pentes [EUR/(h.W)] du cout V11."""
        jmax = float(ely_i_max(alpha_ely)) / (I.S * I.ELY["n_parallel"])
        densities = np.array([
            0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 1.0,
            1.25, 1.50, 2.0, jmax,
        ])
        densities = np.unique(np.clip(densities, 0.0, jmax))
        powers_all = np.asarray(
            ely_power(densities * I.S * I.ELY["n_parallel"], alpha_ely),
            dtype=float,
        ) / ETA
        keep = powers_all < float(ecap) - 1e-7
        powers, densities = powers_all[keep], densities[keep]
        endpoint_density = float(ely_current_density(float(ecap) * ETA, alpha_ely))
        powers = np.r_[powers, float(ecap)]
        densities = np.r_[densities, endpoint_density]

        uv_per_h = (
            ELY_V11["steady_2_uvph"]
            * np.maximum(densities - 1.0, 0.0) ** NOMINAL_ELY_STRESS_EXPONENT
        )
        eur_per_h = _permanent_uv_to_eur("ely", uv_per_h)
        widths = np.diff(powers)
        slopes = np.divide(
            np.diff(eur_per_h), widths,
            out=np.zeros_like(widths), where=widths > 0.0,
        )
        if np.any(np.diff(slopes) < -1e-11):
            raise RuntimeError("approximation PEMWE non convexe")
        return widths, slopes

    def _build_problem(
        self, forecast_w: np.ndarray, soc: float, h2_kwh: float, h2_capacity_kwh: float,
        soh_bat: float, soh_fc: float, soh_ely: float,
        alpha_fc: float, alpha_ely: float, p_fc_max_w: float, p_ely_max_w: float,
        aging_context: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, Bounds, LinearConstraint, dict[str, Any]]:
        p = np.asarray(forecast_w, dtype=float)
        horizon = len(p)
        if horizon < 2 or not np.isfinite(p).all():
            raise ValueError("fenetre de prevision invalide")

        # Double borne : la boucle fournit normalement les Pmax coherents avec
        # alpha, mais le minimum rend aussi les appels unitaires defensifs.
        fcap = 0.999 * ETA * min(float(p_fc_max_w), float(fc_pmax(alpha_fc)))
        ecap = 0.999 * min(float(p_ely_max_w), float(ely_pmax(alpha_ely))) / ETA
        fc_min = min(
            fcap,
            float(fc_power(0.051 * I.S * I.FC["n_parallel"], alpha_fc)) * ETA,
        )
        ely_min = min(
            ecap,
            float(ely_power(0.011 * I.S * I.ELY["n_parallel"], alpha_ely)) / ETA,
        )
        segment_widths, segment_slopes = self._ely_piecewise(alpha_ely, ecap)
        n_segments = len(segment_widths)

        offset = 0
        blocks: dict[str, np.ndarray] = {}

        def allocate(name: str, size: int) -> np.ndarray:
            nonlocal offset
            idx = np.arange(offset, offset + size, dtype=int)
            blocks[name] = idx
            offset += size
            return idx

        f = allocate("fc_w", horizon)
        e = allocate("ely_w", horizon)
        bd = allocate("bat_discharge_w", horizon)
        bc = allocate("bat_charge_w", horizon)
        shed = allocate("shed_w", horizon)
        curtail = allocate("curtail_w", horizon)
        fc_on = allocate("fc_on", horizon)
        ely_on = allocate("ely_on", horizon)
        fc_start = allocate("fc_start", horizon)
        ely_start = allocate("ely_start", horizon)
        bat_dis = allocate("bat_discharge_mode", horizon)
        df = allocate("fc_delta_w", horizon)
        de = allocate("ely_delta_w", horizon)
        soc_high = allocate("soc_high", horizon)
        ely_segment = allocate("ely_segments", horizon * n_segments).reshape(
            horizon, n_segments)
        n_variables = offset

        objective = np.full(n_variables, 1e-12)
        lower = np.zeros(n_variables)
        upper = np.full(n_variables, np.inf)
        integrality = np.zeros(n_variables, dtype=int)
        for binary in (fc_on, ely_on, fc_start, ely_start, bat_dis):
            upper[binary] = 1.0
            integrality[binary] = 1
        upper[f] = fcap
        upper[e] = ecap
        upper[df] = fcap
        upper[de] = ecap
        upper[soc_high] = 1.0
        for k in range(horizon):
            upper[ely_segment[k]] = segment_widths

        battery_wh = BAT_NOMINAL_WH * float(soh_bat)
        bd_max = (SOC_MAX - SOC_MIN) * battery_wh / (BAT_K_DISCHARGE * DT_H)
        bc_max = (SOC_MAX - SOC_MIN) * battery_wh / (BAT_K_CHARGE * DT_H)
        upper[bd] = bd_max
        upper[bc] = bc_max

        fc_factor, ely_factor = self._wear_factors(soh_fc, soh_ely)
        battery_wear = BAT_WEAR_EUR_PER_WH * self.config.battery_wear_scale
        objective[bd] = (
            battery_wear + self.config.terminal_bat_eur_per_kwh / 1000.0
        ) * BAT_K_DISCHARGE * DT_H
        objective[bc] = (
            battery_wear - self.config.terminal_bat_eur_per_kwh / 1000.0
        ) * BAT_K_CHARGE * DT_H
        objective[shed] = self.config.voll_eur_per_kwh / 1000.0 * DT_H
        objective[curtail] = 1e-9 * DT_H

        g_ely = ETA * float(I.ELY["eff"])
        g_fc = 1.0 / (ETA * float(I.FC["eff"]))
        objective[f] += self.config.terminal_h2_eur_per_kwh * g_fc / 1000.0 * DT_H
        objective[e] -= self.config.terminal_h2_eur_per_kwh * g_ely / 1000.0 * DT_H

        fc_steadiness = 1.0
        if aging_context and isinstance(aging_context.get("fc"), dict):
            fc_steadiness = float(aging_context["fc"].get("steadiness", 1.0))
        fc_steadiness = min(max(fc_steadiness, 0.0), 1.0)
        fc_rate_uvph = (
            FC_V11["irr_dynamic_uvph"]
            + fc_steadiness
            * (FC_V11["irr_steady_uvph"] - FC_V11["irr_dynamic_uvph"])
        )
        fc_rate_eur_h = float(_permanent_uv_to_eur("fc", fc_rate_uvph))
        fc_start_eur = float(_permanent_uv_to_eur("fc", FC_V11["start_uv"]))
        fc_dynamic_eur = float(_permanent_uv_to_eur(
            "fc", FC_V11["irr_dynamic_uvph"] - FC_V11["irr_steady_uvph"]))
        objective[fc_on] += (
            fc_factor * self.config.fc_wear_scale * fc_rate_eur_h * DT_H)
        objective[fc_start] += (
            fc_factor * self.config.fc_wear_scale * fc_start_eur)
        objective[df] += (
            fc_factor * self.config.fc_wear_scale * self.config.fc_dynamic_scale
            * fc_dynamic_eur / max(fcap, 1.0))

        ely_start_eur = float(_permanent_uv_to_eur("ely", ELY_V11["start_uv"]))
        objective[ely_start] += (
            ely_factor * self.config.ely_wear_scale * ely_start_eur)
        for k in range(horizon):
            objective[ely_segment[k]] += (
                ely_factor * self.config.ely_wear_scale * segment_slopes * DT_H)
        objective[soc_high] += self.config.high_soc_hold_eur * DT_H

        rows = _LinearRows(n_variables)
        for k in range(horizon):
            # f - e + bd - bc + shed - curtail = profil net.
            rows.add([
                (f[k], 1.0), (e[k], -1.0), (bd[k], 1.0), (bc[k], -1.0),
                (shed[k], 1.0), (curtail[k], -1.0),
            ], p[k], p[k])

            upto = np.arange(k + 1)
            soc_terms = [
                (bc[upto], np.full(k + 1, BAT_K_CHARGE * DT_H / battery_wh)),
                (bd[upto], np.full(k + 1, -BAT_K_DISCHARGE * DT_H / battery_wh)),
            ]
            rows.add(soc_terms, SOC_MIN - soc, SOC_MAX - soc)

            h2_terms = [
                (e[upto], np.full(k + 1, g_ely * DT_H / 1000.0)),
                (f[upto], np.full(k + 1, -g_fc * DT_H / 1000.0)),
            ]
            rows.add(h2_terms, -h2_kwh, h2_capacity_kwh - h2_kwh)

            rows.add([(f[k], 1.0), (fc_on[k], -fcap)], upper=0.0)
            rows.add([(f[k], -1.0), (fc_on[k], fc_min)], upper=0.0)
            rows.add([(e[k], 1.0), (ely_on[k], -ecap)], upper=0.0)
            rows.add([(e[k], -1.0), (ely_on[k], ely_min)], upper=0.0)
            rows.add([(fc_on[k], 1.0), (ely_on[k], 1.0)], upper=1.0)

            previous_fc_on = self.previous_fc_on if k == 0 else fc_on[k - 1]
            previous_ely_on = self.previous_ely_on if k == 0 else ely_on[k - 1]
            if k == 0:
                rows.add([(fc_on[k], 1.0), (fc_start[k], -1.0)],
                         upper=float(self.previous_fc_on))
                rows.add([(ely_on[k], 1.0), (ely_start[k], -1.0)],
                         upper=float(self.previous_ely_on))
                rows.add([(fc_start[k], 1.0)], upper=1.0 - self.previous_fc_on)
                rows.add([(ely_start[k], 1.0)], upper=1.0 - self.previous_ely_on)
            else:
                rows.add([(fc_on[k], 1.0), (previous_fc_on, -1.0),
                          (fc_start[k], -1.0)], upper=0.0)
                rows.add([(ely_on[k], 1.0), (previous_ely_on, -1.0),
                          (ely_start[k], -1.0)], upper=0.0)
                rows.add([(fc_start[k], 1.0), (previous_fc_on, 1.0)], upper=1.0)
                rows.add([(ely_start[k], 1.0), (previous_ely_on, 1.0)], upper=1.0)
            rows.add([(fc_start[k], 1.0), (fc_on[k], -1.0)], upper=0.0)
            rows.add([(ely_start[k], 1.0), (ely_on[k], -1.0)], upper=0.0)

            previous_fc = self.previous_fc_w if k == 0 else f[k - 1]
            previous_ely = self.previous_ely_w if k == 0 else e[k - 1]
            if k == 0:
                rows.add([(f[k], 1.0), (df[k], -1.0)], upper=self.previous_fc_w)
                rows.add([(f[k], -1.0), (df[k], -1.0)], upper=-self.previous_fc_w)
                rows.add([(e[k], 1.0), (de[k], -1.0)], upper=self.previous_ely_w)
                rows.add([(e[k], -1.0), (de[k], -1.0)], upper=-self.previous_ely_w)
            else:
                rows.add([(f[k], 1.0), (previous_fc, -1.0), (df[k], -1.0)], upper=0.0)
                rows.add([(f[k], -1.0), (previous_fc, 1.0), (df[k], -1.0)], upper=0.0)
                rows.add([(e[k], 1.0), (previous_ely, -1.0), (de[k], -1.0)], upper=0.0)
                rows.add([(e[k], -1.0), (previous_ely, 1.0), (de[k], -1.0)], upper=0.0)

            rows.add([(bd[k], 1.0), (bat_dis[k], -bd_max)], upper=0.0)
            rows.add([(bc[k], 1.0), (bat_dis[k], bc_max)], upper=bc_max)
            rows.add([(e[k], 1.0), (ely_segment[k], -np.ones(n_segments))], 0.0, 0.0)
            rows.add(soc_terms + [(soc_high[k], -1.0)],
                     upper=self.config.high_soc_knee - soc)

        metadata = {
            "blocks": blocks,
            "horizon": horizon,
            "n_segments": n_segments,
            "segment_widths": segment_widths,
            "segment_slopes": segment_slopes,
            "fc_wear_factor": fc_factor,
            "ely_wear_factor": ely_factor,
            "forecast_w": p,
        }
        return objective, integrality, Bounds(lower, upper), rows.constraint(), metadata

    def solve_horizon(
        self, forecast_w: np.ndarray, soc: float, h2_kwh: float,
        h2_capacity_kwh: float, soh_bat: float, soh_fc: float, soh_ely: float,
        alpha_fc: float, alpha_ely: float, p_fc_max_w: float, p_ely_max_w: float,
        aging_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        objective, integrality, bounds, constraint, meta = self._build_problem(
            forecast_w, soc, h2_kwh, h2_capacity_kwh, soh_bat, soh_fc, soh_ely,
            alpha_fc, alpha_ely, p_fc_max_w, p_ely_max_w, aging_context,
        )
        started = time.perf_counter()
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraint,
            options={
                "mip_rel_gap": self.config.mip_rel_gap,
                "time_limit": self.config.time_limit_s,
                "presolve": True,
            },
        )
        elapsed = time.perf_counter() - started
        self.solve_seconds += elapsed
        self.max_solve_seconds = max(self.max_solve_seconds, elapsed)
        if not result.success or result.x is None:
            self.failures += 1
            message = f"echec MILP MPC V11: status={result.status}, {result.message}"
            if self.config.fail_hard:
                raise RuntimeError(message)
            return {"success": False, "message": message, "solve_seconds": elapsed}

        x = np.asarray(result.x, dtype=float)
        matrix = constraint.A
        ax = np.asarray(matrix @ x).ravel()
        lower_violation = np.maximum(np.asarray(constraint.lb) - ax, 0.0)
        upper_violation = np.maximum(ax - np.asarray(constraint.ub), 0.0)
        residual = float(max(lower_violation.max(initial=0.0),
                             upper_violation.max(initial=0.0)))
        self.max_constraint_residual = max(self.max_constraint_residual, residual)
        blocks = meta["blocks"]
        solution = {
            "success": True,
            "objective_eur": float(result.fun),
            "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            "solve_seconds": elapsed,
            "constraint_residual": residual,
            "fc_w": x[blocks["fc_w"]],
            "ely_w": x[blocks["ely_w"]],
            "bat_discharge_w": x[blocks["bat_discharge_w"]],
            "bat_charge_w": x[blocks["bat_charge_w"]],
            "shed_w": x[blocks["shed_w"]],
            "curtail_w": x[blocks["curtail_w"]],
            "fc_on": np.rint(x[blocks["fc_on"]]).astype(int),
            "ely_on": np.rint(x[blocks["ely_on"]]).astype(int),
            "fc_wear_factor": meta["fc_wear_factor"],
            "ely_wear_factor": meta["ely_wear_factor"],
        }
        self.last_solution = solution
        return solution

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "model_id": MODEL_ID,
            "ely_stress_exponent": NOMINAL_ELY_STRESS_EXPONENT,
            "config": asdict(self.config),
            "config_fingerprint": self.config.fingerprint,
            "calls": self.calls,
            "forecast_origins": self.forecast_origins,
            "failures": self.failures,
            "solve_seconds": self.solve_seconds,
            "mean_solve_seconds": self.solve_seconds / self.calls if self.calls else 0.0,
            "max_solve_seconds": self.max_solve_seconds,
            "max_constraint_residual": self.max_constraint_residual,
            "max_lol": self.max_lol,
            "lol_above_one_steps": self.lol_above_one_steps,
            "planned_shed_kwh": self.planned_shed_kwh,
            "planned_curtail_kwh": self.planned_curtail_kwh,
        }

    def __call__(
        self, SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
        RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
        P_tot_ref_future=None, aging_context=None,
    ):
        del lol_tab, RUL_fc_t, RUL_ely_t
        if self.config.forecast_mode == "persistence":
            forecast = np.full(self.config.horizon_steps, float(P_tot_ref_t))
        else:
            if P_tot_ref_future is None or len(P_tot_ref_future) < 2:
                forecast = np.full(self.config.horizon_steps, float(P_tot_ref_t))
            else:
                forecast = np.asarray(P_tot_ref_future, dtype=float)
                forecast = forecast[:self.config.horizon_steps].copy()
            forecast[0] = float(P_tot_ref_t)
            if self.config.forecast_mode == "noisy":
                forecast[1:] += self._forecast_error_w(len(forecast) - 1)
        if len(forecast) < 2:
            forecast = np.r_[forecast, float(P_tot_ref_t)]

        try:
            solution = self.solve_horizon(
                forecast, float(SoC_t), float(E_h2_t), float(E_h2_init),
                float(SoH_bat_t), float(SoH_fc_t), float(SoH_ely_t),
                float(alpha_fc_t), float(alpha_ely_t),
                float(P_fc_max_t), float(P_ely_max_t), aging_context,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}; call={self.calls}; P_ref={float(P_tot_ref_t):.12g}; "
                f"SoC={float(SoC_t):.12g}; E_h2={float(E_h2_t):.12g}; "
                f"SoH=({float(SoH_bat_t):.12g},{float(SoH_fc_t):.12g},"
                f"{float(SoH_ely_t):.12g}); "
                f"alpha=({float(alpha_fc_t):.12g},{float(alpha_ely_t):.12g})"
            ) from exc
        if not solution["success"]:
            raise RuntimeError(solution["message"])

        fc = float(solution["fc_w"][0])
        ely = float(solution["ely_w"][0])
        if "FC" in defaillances:
            fc = 0.0
        if "ELY" in defaillances:
            ely = 0.0
        # La batterie est l'ajustement a l'execution, comme pour RB1/RB2/PD.
        battery = float(P_tot_ref_t) - fc + ely
        action, lol = get_lol(
            SoC_t, (battery, fc, -ely), P_tot_ref_t, defaillances,
            E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t,
        )

        self.calls += 1
        self.previous_fc_w = float(action[1])
        self.previous_ely_w = abs(float(action[2]))
        self.previous_fc_on = int(self.previous_fc_w > 1e-8)
        self.previous_ely_on = int(self.previous_ely_w > 1e-8)
        self.max_lol = max(self.max_lol, float(lol))
        self.lol_above_one_steps += int(float(lol) > 1.0 + 1e-9)
        self.planned_shed_kwh += float(solution["shed_w"][0]) / 1000.0 * DT_H
        self.planned_curtail_kwh += float(solution["curtail_w"][0]) / 1000.0 * DT_H
        return action, float(lol)
