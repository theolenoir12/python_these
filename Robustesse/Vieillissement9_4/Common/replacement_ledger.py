"""Comptabilite disjointe des couts de degradation par unite physique.

Une unite remplacee porte les pas de l'intervalle ``[start_step, stop_step)``.
La nouvelle unite commence exactement a ``stop_step`` : aucun pas ne peut donc
etre attribue a la fois a l'ancienne et a la nouvelle unite.
"""

from __future__ import annotations

import math


COMPONENTS = ("bat", "fc", "ely")


class ReplacementLedger:
    """Journal minimal, serialisable, des remplacements et couts par unite."""

    schema_version = 1

    def __init__(self):
        self._retired_eur = {component: 0.0 for component in COMPONENTS}
        self._start_step = {component: 0 for component in COMPONENTS}
        self._events = []

    def retire(self, component, current_cost_eur, stop_step, reason,
               soh_before, soh_after=1.0):
        """Clot l'unite courante juste avant ``stop_step`` et ouvre la suivante."""
        if component not in COMPONENTS:
            raise ValueError("composant inconnu : %s" % component)
        stop_step = int(stop_step)
        start_step = self._start_step[component]
        if stop_step <= start_step:
            raise ValueError(
                "intervalle vide ou inverse pour %s : [%d, %d)"
                % (component, start_step, stop_step)
            )
        cost = float(current_cost_eur)
        soh_before = float(soh_before)
        soh_after = float(soh_after)
        if not all(math.isfinite(value) for value in (cost, soh_before, soh_after)):
            raise ValueError("valeur non finie dans le remplacement de %s" % component)
        if cost < -1e-9:
            raise ValueError("cout negatif pour %s : %g" % (component, cost))
        event = {
            "component": component,
            "reason": str(reason),
            "start_step": start_step,
            "stop_step_exclusive": stop_step,
            "retired_eur": cost,
            "soh_before": soh_before,
            "soh_after": soh_after,
        }
        self._events.append(event)
        self._retired_eur[component] += cost
        self._start_step[component] = stop_step
        return event

    def snapshot(self, current_eur, end_step):
        """Retourne une photographie autonome du ledger a la fin du calcul."""
        current = {component: float(current_eur[component]) for component in COMPONENTS}
        if not all(math.isfinite(value) and value >= -1e-9 for value in current.values()):
            raise ValueError("cout courant negatif ou non fini")
        total = {
            component: self._retired_eur[component] + current[component]
            for component in COMPONENTS
        }
        end_step = int(end_step)
        for component in COMPONENTS:
            if end_step < self._start_step[component]:
                raise ValueError("fin anterieure au dernier remplacement")
        return {
            "schema_version": self.schema_version,
            "accounting": "disjoint_replacement_intervals",
            "retired_eur": dict(self._retired_eur),
            "current_eur": current,
            "total_eur": total,
            "current_start_step": dict(self._start_step),
            "end_step_exclusive": end_step,
            "events": [dict(event) for event in self._events],
        }
