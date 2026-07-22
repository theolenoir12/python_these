"""Petit moteur Mamdani deterministe, auditable et sans dependance externe.

Le moteur est volontairement limite aux besoins du controleur GENIAL :
operateur ET=min, implication=min, aggregation=max et defuzzification par le
centre de gravite sur un univers regulier. Les fonctions d'appartenance et les
regles restent des objets explicites, exportables dans le manuscrit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _return_scalar_if_needed(reference, value):
    if np.asarray(reference).ndim == 0:
        return float(np.asarray(value))
    return value


@dataclass(frozen=True)
class TriangularMF:
    """Fonction d'appartenance triangulaire definie par ``a < b < c``."""

    a: float
    b: float
    c: float

    def __post_init__(self):
        if not float(self.a) < float(self.b) < float(self.c):
            raise ValueError("une MF triangulaire exige a < b < c")

    def __call__(self, x):
        values = np.asarray(x, dtype=float)
        rising = (values - self.a) / (self.b - self.a)
        falling = (self.c - values) / (self.c - self.b)
        membership = np.maximum(np.minimum(rising, falling), 0.0)
        membership = np.minimum(membership, 1.0)
        return _return_scalar_if_needed(x, membership)


@dataclass(frozen=True)
class TrapezoidalMF:
    """Fonction trapezoidale, epaules comprises (``a <= b <= c <= d``)."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        if not float(self.a) <= float(self.b) <= float(self.c) <= float(self.d):
            raise ValueError("une MF trapezoidale exige a <= b <= c <= d")
        if float(self.a) == float(self.d):
            raise ValueError("le support d'une MF ne peut pas etre nul")

    def __call__(self, x):
        values = np.asarray(x, dtype=float)
        membership = np.zeros_like(values, dtype=float)

        plateau = (values >= self.b) & (values <= self.c)
        membership = np.where(plateau, 1.0, membership)

        if self.b > self.a:
            rising = (values > self.a) & (values < self.b)
            membership = np.where(
                rising, (values - self.a) / (self.b - self.a), membership
            )
        else:
            membership = np.where(values == self.a, 1.0, membership)

        if self.d > self.c:
            falling = (values > self.c) & (values < self.d)
            membership = np.where(
                falling, (self.d - values) / (self.d - self.c), membership
            )
        else:
            membership = np.where(values == self.d, 1.0, membership)

        membership = np.where(
            (values < self.a) | (values > self.d), 0.0, membership
        )
        return _return_scalar_if_needed(x, membership)


@dataclass(frozen=True)
class FuzzyVariable:
    name: str
    terms: Mapping[str, object]
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("une variable floue doit avoir un nom")
        if not self.terms:
            raise ValueError("une variable floue doit avoir au moins un terme")
        if not float(self.lower) < float(self.upper):
            raise ValueError("bornes de variable floue invalides")

    def fuzzify(self, value):
        clipped = float(np.clip(float(value), self.lower, self.upper))
        return {name: float(mf(clipped)) for name, mf in self.terms.items()}


@dataclass(frozen=True)
class FuzzyRule:
    antecedents: tuple[str, ...]
    consequent: str
    label: str = ""


class MamdaniSystem:
    """Systeme Mamdani a entrees scalaires et sortie scalaire."""

    def __init__(
        self,
        inputs: Sequence[FuzzyVariable],
        output: FuzzyVariable,
        rules: Sequence[FuzzyRule],
        output_points: int = 401,
        default_output: float = 0.0,
    ):
        self.inputs = tuple(inputs)
        self.output = output
        self.rules = tuple(rules)
        self.default_output = float(default_output)
        if not self.inputs:
            raise ValueError("un systeme Mamdani exige au moins une entree")
        if int(output_points) < 51:
            raise ValueError("output_points doit etre >= 51")
        self.output_universe = np.linspace(
            output.lower, output.upper, int(output_points), dtype=float
        )
        self._output_mfs = {
            name: np.asarray(mf(self.output_universe), dtype=float)
            for name, mf in output.terms.items()
        }
        self._validate_rules()

    def _validate_rules(self):
        for rule in self.rules:
            if len(rule.antecedents) != len(self.inputs):
                raise ValueError("nombre d'antecedents incompatible avec les entrees")
            for variable, term in zip(self.inputs, rule.antecedents):
                if term not in variable.terms:
                    raise ValueError(
                        f"terme inconnu {term!r} pour l'entree {variable.name!r}"
                    )
            if rule.consequent not in self.output.terms:
                raise ValueError(f"consequent inconnu {rule.consequent!r}")

    def infer(self, values: Mapping[str, float], return_trace: bool = False):
        expected = {variable.name for variable in self.inputs}
        missing = expected.difference(values)
        if missing:
            raise KeyError(f"entrees Mamdani manquantes : {sorted(missing)}")

        fuzzified = {
            variable.name: variable.fuzzify(values[variable.name])
            for variable in self.inputs
        }
        activation = {name: 0.0 for name in self.output.terms}
        rule_strengths = []
        for rule in self.rules:
            strength = min(
                fuzzified[variable.name][term]
                for variable, term in zip(self.inputs, rule.antecedents)
            )
            activation[rule.consequent] = max(
                activation[rule.consequent], strength
            )
            if return_trace:
                rule_strengths.append((rule.label, float(strength)))

        aggregated = np.zeros_like(self.output_universe)
        for term, height in activation.items():
            if height > 0.0:
                aggregated = np.maximum(
                    aggregated, np.minimum(height, self._output_mfs[term])
                )
        area = float(np.sum(aggregated))
        if area <= 1e-15:
            crisp = self.default_output
        else:
            crisp = float(np.dot(self.output_universe, aggregated) / area)

        if not return_trace:
            return crisp
        return crisp, {
            "fuzzified": fuzzified,
            "output_activation": activation,
            "rule_strengths": tuple(rule_strengths),
            "aggregated_area_discrete": area,
        }
