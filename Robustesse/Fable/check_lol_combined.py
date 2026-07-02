# -*- coding: utf-8 -*-
"""
check_lol_combined.py -- le max(lol_pmax, lol_storage, lol_soc) sous-compte-t-il ?
===================================================================================
Question (revue de coherence, point "get_lol") : quand PLUSIEURS contraintes sont
actives au meme pas (batterie au plancher + reservoir H2 vide, typiquement dans
les pires creux), lol = max(...) mesure-t-il bien l'energie non servie ?

Analyse fine : l'ORDRE des corrections dans get_lol propage deja les contraintes
(le bloc SoC corrige P_dc_bat AVANT les blocs pmax/storage, qui recalculent leur
lol avec les valeurs corrigees) -> le max devrait etre tres proche du lol exact
recalcule sur l'action FINALE. Ce script le PROUVE (ou le refute) en comparant
les metriques du RB2 socle avec le flag Common/get_lol.LOL_COMBINED :

    False (historique)  vs  True (lol recalcule sur l'action finale corrigee)

Si l'ecart de LPSP est negligeable (<0.01 pt), la metrique historique est
VALIDEE (phrase utile pour le manuscrit face a un reviewer). Sinon, l'ecart
quantifie le biais et il faudra statuer.

Usage : python check_lol_combined.py [n_years]     (defaut 25 ; test rapide : 2)
"""
import os, sys
import importlib.util
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.abspath(os.path.join(HERE, "..", "Prédictions"))
for _p in (HERE, PRED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_fable import metrics, VOLL
import Common.get_lol as gl
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast


def _socle():
    spec = importlib.util.spec_from_file_location(
        "strat_socle", os.path.join(HERE, "RB2(Prop)", "get_optimal_action_RB.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ENABLE = False          # RB2 socle exact, deterministe
    return m


def main():
    ny = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out = {}
    for combined in (False, True):
        gl.LOL_COMBINED = combined
        s = _socle()
        data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                          n_years=ny)
        lpsp, deg, eens, _ = metrics(data)
        total = deg + VOLL * eens / 1000.0
        out[combined] = (lpsp, deg, eens, total)
        print(f"LOL_COMBINED={combined!s:5s} : LPSP={lpsp:.5f}%  deg={deg:.3f} kEUR"
              f"  EENS={eens:.1f} kWh  total={total:.3f} kEUR")
    gl.LOL_COMBINED = False   # remise au defaut

    d_lpsp = out[True][0] - out[False][0]
    d_tot  = out[True][3] - out[False][3]
    print(f"\necart combine - historique : dLPSP = {d_lpsp:+.5f} pt"
          f"  |  dtotal = {d_tot:+.4f} kEUR")
    if abs(d_lpsp) < 0.01:
        print("[check_lol_combined] OK -- metrique historique VALIDEE (ecart negligeable) :")
        print("  l'ordre des corrections de get_lol propage deja les contraintes simultanees.")
    else:
        print("[check_lol_combined] ATTENTION -- ecart non negligeable : le max(...)")
        print("  biaise le LPSP ; a discuter avant tout changement (touche TOUTES les strategies).")


if __name__ == "__main__":
    main()
