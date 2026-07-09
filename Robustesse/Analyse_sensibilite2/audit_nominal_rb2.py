# -*- coding: utf-8 -*-
"""audit_nominal_rb2.py -- verifie que le CODE ACTUEL reproduit les points
canoniques de generate_pareto.py pour RB2 et RB2(SoH).

But : trancher l'origine de l'ecart constate en repro Windows (2.59 vs 2.454).
Le point NOMINAL est deterministe (aucun Monte-Carlo) : a setpoints + donnees +
code identiques, le LPSP/deg est reproductible au chiffre pres.

Lancer (depuis ce dossier, avec l'interpreteur habituel numpy/scipy/sympy) :
    python audit_nominal_rb2.py

Lecture du resultat :
  - si RB2 ~ (2.4540, 65.42) et RB2(SoH) ~ (2.5475, 59.36)
        -> generate_pareto est CANONIQUE et coherent avec le code actuel.
           L'ecart sur ma machine venait donc des DONNEES (CSV reconstruit).
           => on refait les sensibilites RB2/RB2(SoH) sur cette base.
  - si les valeurs different nettement de generate_pareto (ex. ~2.59 / ~2.91)
        -> generate_pareto est lui-meme OBSOLETE (fige avant le dernier
           deplacement de setpoints) : il faut AUSSI le regenerer, et toutes
           les sensibilites, avec les setpoints actuels.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

from sens_common import load_strategy, init_and_run_loop, metrics  # noqa: E402

# Points canoniques de generate_pareto.py (a comparer)
CANON = {"RB2": (2.4540, 65.4218), "RB2(SoH)": (2.5475, 59.3644)}

print("=== Audit nominal (code actuel) vs generate_pareto ===", flush=True)
for folder in ("RB2", "RB2(SoH)"):
    t0 = time.time()
    strat = load_strategy(folder)
    data = init_and_run_loop(strat)
    lpsp, deg = metrics(data)
    cx, cy = CANON[folder]
    verdict = "OK (~canon)" if (abs(lpsp - cx) < 0.02 and abs(deg - cy) < 0.5) \
        else "ECART -> generate_pareto a revoir"
    print("%-9s  LPSP=%.4f  deg=%.4f   | canon (%.4f, %.4f)  dLPSP=%+.4f  ddeg=%+.4f  [%s]  (%.0fs)"
          % (folder, lpsp, deg, cx, cy, lpsp - cx, deg - cy, verdict, time.time() - t0), flush=True)
