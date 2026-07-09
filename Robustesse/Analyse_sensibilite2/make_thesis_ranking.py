"""make_thesis_ranking.py -- figure de classement unifie pour le CHAPITRE 2 (these).
============================================================================
Le chapitre 2 du manuscrit traite uniquement des EMS *statiques* : la strategie
adaptative RB2(SoH) y est exclue (reportee au chapitre suivant). Ce script
regenere donc la SEULE figure de classement (voll_ranking.pdf) en retirant
RB2(SoH) de la liste des strategies, avec le VoLL constant defini dans
voll_common (3 EUR/kWh). Il ecrit directement dans le dossier Figures de la
section Sensibilite de la these et NE TOUCHE PAS aux sorties partagees
results_meso/ (donc ni a la figure de l'article).

Aucune simulation : on relit results_meso/*.txt via voll_common.build_cases().
"""
import os
import numpy as np

import voll_common as V
import plot_voll_summary as P

# 1) Perimetre statique : on retire RB2(SoH) de l'ordre des strategies. build_cases
#    et figure_ranking lisent tous deux V.EMS_ORDER -> exclusion propagee partout.
V.EMS_ORDER = [e for e in V.EMS_ORDER if e != "RB2(SoH)"]

# 2) Sortie : dossier Figures de la section Sensibilite de la these.
THESIS_FIG = os.path.normpath(os.path.join(
    V.HERE, "..", "..", "..", "LaTeX", "Manuscrit_post_chap1_v1",
    "Chapitre 2", "Sensibilite", "Figures"))
P.OUT = THESIS_FIG

# 3) Style + donnees + figure.
P.set_pub_style()
cases = V.build_cases()
print("Cas (%d) : %s" % (len(cases), ", ".join(c[0] for c in cases)))
R, mean_rank, labels = P.figure_ranking(cases)

# 4) Recap rang moyen (verification du texte LaTeX).
order = sorted(range(len(V.EMS_ORDER)), key=lambda i: mean_rank[i])
print("\nRang moyen (1 = meilleur), VoLL =", V.VOLL_TIERS)
for i in order:
    print("  %-8s rang_moyen=%.2f  rangs_par_cas=%s"
          % (V.EMS_ORDER[i], mean_rank[i],
             [int(x) if not np.isnan(x) else None for x in R[i]]))
print("\nFigure ecrite -> %s" % os.path.join(P.OUT, "voll_ranking.pdf"))
