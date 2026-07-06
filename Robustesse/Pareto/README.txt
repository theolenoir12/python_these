================================================================================
DOSSIER Pareto/ -- plans de Pareto (LPSP <-> coût de dégradation), regroupés.
Créé le 2026-07-06. Deux générateurs :
  generate_pareto.py    -> figures/           (plans de Pareto par stratégie)
  generate_ellipses.py  -> figures_ellipses/  (plans à ellipses de sensibilité)
================================================================================

BUT
---
Regrouper en un seul endroit tous les plans de Pareto du chapitre robustesse et
permettre de (re)générer directement les figures demandées :
  - une figure avec toutes les stratégies de base                  -> famille "base"
  - une avec ces stratégies + RB2(SoH)                             -> famille "base_soh"
  - l'intégration de la prévision, toutes les variantes            -> famille "pred"
  - chacune AVEC ou SANS le front de Pareto par PD                 -> suffixe _dp
  - chacune AVEC ou SANS les lignes d'iso-coût                     -> suffixe _isocost
Les petits encadrés de texte donnant la VALEUR des iso-coûts ont été RETIRÉS
(demande explicite) : les iso-coûts restent tracés en pointillés, sans étiquette.

UTILISATION (dans l'environnement conda habituel : numpy + matplotlib)
----------------------------------------------------------------------
    python generate_pareto.py            # génère les 12 figures (x pdf+png)
    python generate_pareto.py base pred  # seulement certaines familles
Sorties -> figures/<famille>[_dp][_isocost].{pdf,png}

    python generate_ellipses.py          # plans à ellipses de sensibilité (5 axes)
    python generate_ellipses.py eol      # un seul axe
Sorties -> figures_ellipses/sens_<axe>_pareto.{pdf,png}   (cf. section ELLIPSES)

CATALOGUE DES FIGURES (12 = 3 familles x {rien, _dp} x {rien, _isocost})
------------------------------------------------------------------------
    base                 base_dp                 base_isocost                 base_dp_isocost
    base_soh             base_soh_dp             base_soh_isocost             base_soh_dp_isocost
    pred                 pred_dp                 pred_isocost                 pred_dp_isocost

    base      : 0-100, 25-75, 50-50, 75-25, 100-0, RB2, RB1, SoC1, SoC06, Ideal
    base_soh  : base + RB2(SoH)                 (UNIQUE levier SoH = setpoints H2)
    pred      : base + RB2(SoH) + RB2(Pred) + RB2(RUL) + RB2(SoH+Pred)

DONNÉES
-------
    data/dp_pareto_25y_51x51_v2.npz    copie de
        ../Vieillissement8/DP/results_meso/dp_pareto_25y_51x51_v2.npz
    (front de Pareto de la programmation dynamique ; clés eps/lpsp/deg_keur/
     nondominated). Si absent, les variantes _dp sont simplement ignorées.

ELLIPSES (figures_ellipses/, script generate_ellipses.py)
---------------------------------------------------------
Plans de Pareto à ELLIPSES d'incertitude, RE-TRACÉS sans simulation en lisant les
résultats Monte-Carlo déjà stockés dans ../Analyse_sensibilite/results_meso/
sens_<axe>.txt (une ellipse 1σ/2σ par stratégie, alignée sur les axes : les .txt
ne stockent que moyenne/écart-type marginaux). Modèle : ../Analyse_sensibilite/
plot_eol_pareto_chap2.py, généralisé.
  RÉGÉNÉRÉS (5 axes)   : sens_{eol,hthresholds,sizing,cweights,calendar}_pareto
                         (cweights : LPSP invariante -> ellipse verticale ;
                          calendar : point nominal = calendaire OFF, ellipse = ON)
  STATIQUES (non régén.) dans figures_ellipses/ :
    sens_soh_pareto.pdf          sens_soh.txt est MONO-STRATÉGIE (RB2(SoH) : biais
                                 + bruit) -> hors du plotter multi-stratégies.
    sens_eol_pareto_en.pdf       variante labels anglais (plot_eol_pareto_en.py).
  figures_ellipses/prediction/ : ellipses du BRUIT DE PRÉVISION / pronostic RUL,
    figures STATIQUES (scripts propres dans ../Prédictions/) :
      pareto_ems_rul_ellipses.*  (plot_pareto_rul_ellipses.py ; donnée source
                                  mc_rul_uncertainty_cloud.csv ABSENTE -> non régén.)
      mc_rul_uncertainty.*       (mc_rul_uncertainty.py)
      pred_uncertainty_zoom.*    (plot_pred_uncertainty.py)
Version des données retenue : results_meso/ (2026-06-22, la plus récente ; doublons
plus anciens ignorés : ../Analyse_sensibilite/{results,results_meso_1806,
results_meso_eol}/).

CHIFFRES : CHAPITRE PREDICTIONS (jeu cohérent avec RUL)
------------------------------------------------------
Les points sont ceux du chapitre PREDICTIONS (VoLL=3, horizon 25 ans), identiques
à ../Prédictions/plot_pareto_strategies.py. Les stratégies prévisionnelles sont
prises à leur MOYENNE Monte-Carlo (variante hysteresis, N=200 ;
../Prédictions/sens_pred_noise_N200_meso.txt), pas à la borne omnisciente.
    RB2 = (2.4540, 65.4218) ; RB2(SoH) = (2.5475, 59.3644) ;
    RB2(Pred) = (2.3642, 65.0248) ; RB2(RUL) = (2.5763, 59.9217) ;
    RB2(SoH+Pred) = (2.4796, 59.3898).
    NB : UN SEUL levier SoH (setpoints H2, noté simplement "SoH").

================================================================================
POINTS À TRANCHER / HYPOTHÈSES PRISES EN MODE NON-ASSISTÉ (à valider)
================================================================================
1. DEUX GÉNÉRATIONS DE CHIFFRES COEXISTENT dans Robustesse/. Ce générateur utilise
   la génération "PREDICTIONS" (Gen B) car c'est la SEULE cohérente contenant à la
   fois RB2(RUL), l'unique RB2(SoH), RB2(Pred) et RB2(SoH+Pred) -- conforme à vos
   deux précisions (un seul levier SoH ; inclure le RUL). L'autre génération, dite
   FABLE (Gen A, ../Fable/ + ../Fable_pred/, socle RB2 deg ~59, sans RUL, avec le
   raffinement SoH_bat/SoH_all), reste disponible dans ses dossiers d'origine. Si
   vous préférez les chiffres FABLE pour base/base_soh, dites-le -> j'ajoute une
   variante.

2. RB2(RUL) est désormais INCLUS dans la famille "pred". Les figures à ELLIPSES
   (sensibilité + bruit de prévision/RUL) sont dans figures_ellipses/ (cf. section
   ELLIPSES ci-dessus) ; les 5 axes de sensibilité sont régénérables par
   generate_ellipses.py.

3. UN SEUL levier SoH (confirmé) : "RB2(SoH)" = les setpoints H2. Pas de SoH_bat
   ni SoH_all.

4. TRI ADDITIF (non destructif) : les scripts et figures d'origine (Fable/,
   Fable_pred/, Prédictions/, Vieillissement8/DP/) sont laissés INTACTS car
   référencés par INDEX_MANUSCRIT_FABLE.txt et par des sorties de jobs. Ce dossier
   est le nouvel emplacement canonique. Une fois validé, on peut décider ensemble
   quels originaux retirer (voir "DOUBLONS" ci-dessous).

5. NON EXÉCUTÉ ICI : aucun interpréteur numpy/matplotlib n'était accessible dans
   la session qui a créé ce dossier -> le script n'a pas pu être lancé/vérifié
   visuellement. Il est porté fidèlement de scripts éprouvés ; à confirmer d'un
   `python generate_pareto.py` dans votre environnement.

DOUBLONS (scripts/figures Pareto désormais couverts par generate_pareto.py)
---------------------------------------------------------------------------
    Prédictions/plot_pareto_strategies.py      -> famille pred (Gen B, +_dp) [SOURCE]
    Vieillissement8/DP/plot_pareto_vs_strategies.py -> famille base_dp (Gen B)
    Fable/plot_pareto_fable.py                 -> (Gen A) NON repris (base_soh alt.)
    Fable_pred/plot_pareto_ultime.py           -> (Gen A) NON repris (pred alt.)
    Prédictions/Pareto_2d_25y_{pred,rul,sohpred}.py -> scripts de CALCUL (produisent
                                                       les points ; non ré-exécutés)
    Analyse_sensibilite/plot_eol_pareto_chap2.py -> modèle de generate_ellipses.py
    Analyse_sensibilite/sens_*.py                -> MC + tracé d'origine des ellipses
                                                    (générateurs lourds ; non repris)
================================================================================
