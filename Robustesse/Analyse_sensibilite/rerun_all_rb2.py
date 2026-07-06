# -*- coding: utf-8 -*-
"""rerun_all_rb2.py -- DRIVER SANS ARGUMENT : re-run RB2/RB2(SoH) sur TOUS les axes
de sensibilite impactes par le deplacement des setpoints, avec splice (5 axes
multi-strategies) ou regeneration complete (soh, timestep) dans results_meso/.

POURQUOI CE FICHIER
-------------------
rerun_rb2_splice.py s'utilise en ligne de commande (arguments 'run ...' /
'runfull ...'). Ce driver l'appelle SANS aucun argument -> il se lance :
  - au MESOCENTRE :   sbatch rerun_all_rb2.slurm        (RECOMMANDE, ~1-1.5 h/32 coeurs)
  - dans Spyder (F5): possible mais LONG (~5-6 h/7 coeurs) et le multiprocessing
                      Windows+Spyder est capricieux -> a EVITER, prefere le mesocentre.

Seules RB2 et RB2(SoH) sont re-simulees (setpoints actuels) ; les 8 autres
strategies sont deterministes et inchangees -> conservees telles quelles.
Sauvegardes results_meso/sens_<axe>.txt.bak.<horodatage> avant chaque ecriture.

POUR UN SOUS-ENSEMBLE (ex. depuis Spyder, juste cweights qui est rapide) :
commente les entrees inutiles dans SPLICE_AXES / RUNFULL_AXES ci-dessous.

ETAPE SUIVANTE (leger, dans Spyder, sans multiprocessing -> F5 sans souci) :
    ../Pareto/generate_ellipses.py     (figures a ellipses des 5 axes)
    ../Pareto/generate_pareto.py       (plans de Pareto, RB2/RB2(SoH) deja corriges)
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

from rerun_rb2_splice import run_axis, run_full  # noqa: E402

# --------------------------------------------------------------------------- #
#  CONFIG : axes a relancer. Commenter pour restreindre (ex. tests Spyder).    #
# --------------------------------------------------------------------------- #
SPLICE_AXES = [           # multi-strategies -> re-run RB2/RB2(SoH) + splice
    "eol",
    "hthresholds",
    "sizing",
    "cweights",           # analytique : quasi instantane
    "calendar",
]
RUNFULL_AXES = [          # mono/bi-strategie -> fichier entier regenere
    "soh",
    "timestep",
]


def main():
    t0 = time.time()
    print("##### rerun_all_rb2 : %d axes splice + %d runfull #####"
          % (len(SPLICE_AXES), len(RUNFULL_AXES)), flush=True)
    for axis in SPLICE_AXES:
        run_axis(axis)
    for name in RUNFULL_AXES:
        run_full(name)
    print("\n" + "#" * 78)
    print("# TERMINE en %.0f min." % ((time.time() - t0) / 60))
    print("# Verifie dans results_meso/ que RB2 ~ 2.59 et RB2(SoH) ~ 2.91.")
    print("# Etape suivante (Spyder, leger) :")
    print("#   ../Pareto/generate_ellipses.py   puis   ../Pareto/generate_pareto.py")
    print("#" * 78, flush=True)


if __name__ == "__main__":
    main()
