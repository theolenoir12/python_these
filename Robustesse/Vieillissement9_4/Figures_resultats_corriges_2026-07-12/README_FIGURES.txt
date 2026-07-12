================================================================================
FIGURES DE LECTURE DES RESULTATS CORRIGES P1 / P3 / P4 -- 2026-07-12
================================================================================

OBJECTIF
--------

Ce dossier met les ameliorations dans les plans d'objectifs, sans relancer de
simulation. Toutes les figures sont regenerees depuis les caches pleine
precision des jobs valides 215089 a 215091.

Il s'agit de plans d'objectifs et non, au sens mathematique strict, de fronts de
Pareto complets : seuls les points des strategies effectivement testees sont
affiches. Une ligne reliant des points montre une famille ou une cascade de
strategies ; elle n'affirme pas que tous les points intermediaires sont
realisables ni non domines.

Convention commune :

  axe horizontal = energie non servie EENS [MWh] ;
  axe vertical   = cout hors defaillance [kEUR] ;
  bas-gauche     = amelioration simultanee des deux objectifs ;
  droite tiretee = meme cout total que la reference a VoLL=3 EUR/kWh.

Dans les plans de differences, la zone verte verifie :

  delta cout = delta cout_hors_defaillance + 3 x delta EENS < 0.


ORDRE DE LECTURE
----------------

1. 00_SYNTHESE_espaces_objectifs

   Vue rapide des trois resultats. Les panneaux ne doivent pas etre fusionnes :
   P1, P3 et P4 ont des protocoles et des incertitudes differents.

2. 01_P1_SoH_espace_objectifs

   - panneau gauche : nuages des 200 mondes et cascade moyenne
     RB2 -> recale -> horloge -> SoH ;
   - panneau droit : differences appariees dans le plan des objectifs.

   Le resultat central est SoH-RB2 = -2,305 kEUR. Ce gain total vient surtout
   de la fiabilite : +0,522 kEUR de degradation mais -0,942 MWh d'EENS.
   La valeur propre face a l'horloge globale est -0,716 kEUR. Le SoH est exact
   et l'horloge n'est pas recalee sur l'age de l'unite apres remplacement.

3. 02_P3_RUL_espace_objectifs

   - panneaux gauche/centre : scenario T=6 mois, C_intervention=1,5 kEUR ;
   - panneau droit : evolution de RUL-correctif avec T=3/6/12 mois.

   Le cout vertical comprend degradation + interventions + gaspillage de vie.
   RUL-correctif vaut -2,625 kEUR a T=6 mois, avec 178/200 gains. La RUL est
   calculee depuis le SoH vrai : il s'agit d'une borne structurelle, pas encore
   d'une performance realiste de pronostic.

4. 03_P4_prevision_espace_objectifs

   - panneau gauche : zone utile N=2 a 6 h ;
   - panneau centre : 32 erreurs de prevision pour N=4 ;
   - panneau droit : gain puis falaise de fiabilite au-dela de N=6.

   N=4 bruite vaut -1,470 kEUR face a la base et -0,387 kEUR face au meilleur
   min-off sans prevision. Cette seconde valeur est la valeur propre de
   l'information previsionnelle dans ce banc.


TRACABILITE
-----------

Le script `generer_figures.py` valide avant trace :

- 200 tirages contigus par strategie P1/P3 ;
- fermeture cout = degradation + VoLL x EENS ;
- fermeture du gaspillage P3 ;
- 32 graines valides pour P4 noisy N=4.

`DONNEES_MOYENNES_FIGURES.tsv` contient les coordonnees exactes des points
moyens affiches. PNG = consultation rapide ; PDF = version vectorielle.

Regeneration locale :

  cd Python/Robustesse/Vieillissement9_4/Figures_resultats_corriges_2026-07-12
  MPLCONFIGDIR=/tmp/mplconfig python generer_figures.py
================================================================================
