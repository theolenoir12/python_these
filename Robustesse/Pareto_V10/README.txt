================================================================================
DOSSIER Pareto_V10/ -- figures LPSP / dégradation du modèle Vieillissement10
================================================================================

PÉRIMÈTRE ACTUEL
----------------
Cette première version contient uniquement :
  - les stratégies de base V10 ;
  - RB2(SoH), défini par le point du front minimisant la dégradation sous la
    contrainte coût unifié <= coût unifié de RB2 ;
  - le front complet des configurations RB2(SoH) normalisées.

RUL, prévisions de profils, combinaison des couches, ellipses de sensibilité et
front de programmation dynamique ne sont volontairement pas repris tant que
leur analyse n'a pas été refaite avec Vieillissement10.

SOURCES
-------
  ../Vieillissement10/rank_base_strategies_25y.txt
  ../Vieillissement10/Optimization_results_psi1/optimization_soh_validated25.csv
  ../Vieillissement10/Optimization_results_psi1/optimization_soh_validated_shapes25.csv

Le script relit ces résultats, reconstruit le front non dominé, sélectionne
RB2(SoH) sous la contrainte d'iso-coût unifié et recalcule la pente des
iso-couts pour VoLL=3 EUR/kWh. Le cas nominal utilise le plancher batterie
psi=1 sous 1C. Le point trace comme compromis est
(strength_fc, strength_ely, shape)=(0.025, 0.025, 1) : la degradation ne baisse
que de 0.029 % sous l'iso-cout. Le meilleur cout unifie observe est le point
(0.025, 0, 4), avec un gain limite a 0.30 %. Une baisse de degradation proche
de 1 % demande environ +4.6 % de cout unifie.

UTILISATION
-----------
  python generate_pareto.py
  python generate_pareto.py base base_soh

SORTIES
-------
Figures directement comparables aux versions V8 :
  figures/base.{pdf,png}
  figures/base_isocost.{pdf,png}
  figures/base_soh.{pdf,png}
  figures/base_soh_isocost.{pdf,png}

Complément spécifique aux nouveaux résultats SoH :
  figures/base_soh_front.{pdf,png}
  figures/base_soh_front_isocost.{pdf,png}

Le fichier points_v10.csv est régénéré à chaque exécution et constitue la table
de contrôle des points effectivement tracés.

Le dossier historique Pareto a été renommé Pareto_V8 et conservé intégralement.
Il ne faut pas mélanger ses données PD, RUL, prévisions ou sensibilités avec les
figures présentes ici.
