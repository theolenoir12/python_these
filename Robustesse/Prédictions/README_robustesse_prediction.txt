================================================================================
DEMARCHE : QUELS POINTS PREVISIONNELS GARDER DANS LE PLAN DE PARETO ?
Dossier : Robustesse/Predictions/        Maj : 2026-06-30
================================================================================

0. RESUME
--------------------------------------------------------------------------------
La regle methodologique porte sur la VARIANTE des strategies previsionnelles que
l'on a le DROIT de presenter : seule la version ROBUSTE au bruit de prevision.

  GARDE (pour RB2(Pred) et RB2(SoH+Pred)) :
            UNIQUEMENT la version HYSTERESIS (robuste), centree sur la moyenne
            Monte-Carlo, + ses ellipses d'incertitude 1s / 2s.

  RETIRE (jamais tracees) :
            variantes OMNISCIENTES  (prevision parfaite : on "sait exactement le
                                      futur" -> non deployable, non robuste) ;
            variantes BINAIRES      (decision net>0 sans marge : fragiles, le
                                      bruit fait clignoter l'electrolyseur).

  DEUX FIGURES SEPAREES (les ellipses etant minuscules a l'echelle du plan) :
    1) pareto_strategies.{pdf,png}     -- plot_pareto_strategies.py
         plan de Pareto, TOUTES les strategies (0-100, ..., RB1, RB2, RB2(SoH),
         RB2(RUL), SoC..., "Ideal") + FRONT PD NETTOYE, SANS ellipses. Sert a
         situer les positions RELATIVES. RB2(Pred)/RB2(SoH+Pred) y sont a leur
         moyenne Monte-Carlo (hyst).
    2) pred_uncertainty_zoom.{pdf,png} -- plot_pred_uncertainty.py
         ZOOM dedie sur les 2 strategies previsionnelles (hyst) + LEUR baseline
         (RB2, RB2(SoH)), avec les ellipses 1s / 2s. Rien d'autre.


1. POURQUOI RETIRER L'OMNISCIENT
--------------------------------------------------------------------------------
RB2(Pred) / RB2(SoH+Pred) augmentent la regle de base par une PRE-CHARGE batterie
declenchee sur l'energie nette PREVUE a 18 h. Avec une prevision PARFAITE
(omnisciente) le gain est reel mais c'est une BORNE SUPERIEURE non atteignable :
en exploitation on ne connait pas le vrai futur. Montrer ce point laisserait
croire a un gain que l'incertitude de prevision detruit.

Quantification de l'incertitude (backtest LSTM, cf. ../Predictions profils/) :
  energie nette prevue a 18 h = vrai futur + N(biais ~ -2.3 kWh, sigma ~ 39.4 kWh).
Ce sigma est mesure CONTRE LA REALITE : il est IRREDUCTIBLE.


2. POURQUOI LA DECISION BINAIRE EST FRAGILE
--------------------------------------------------------------------------------
La pre-charge d'origine est BINAIRE (pre-charger si net>0). Pres du seuil net~0,
le bruit fait BASCULER la decision d'un pas a l'autre -> l'electrolyseur clignote
(marche/arret) -> degradation start-stop. Resultat : sous bruit reel, le gain
omniscient (-2.06 kEUR) devient une PERTE (+0.33 kEUR, pire que RB2 nu).


3. CE QU'ON GARDE : L'HYSTERESIS (ROBUSTE)
--------------------------------------------------------------------------------
On remplace le seuil binaire par une HYSTERESIS A MARGE calee sur le sigma mesure
  - entrer en pre-charge si  net_pred > +M_SIGMA*sigma   (M_SIGMA = 1.0)
  - sortir              si  net_pred < -M_SIGMA*sigma
  - zone morte entre les deux = rejet du bruit ; + duree de maintien MIN_DWELL=12 h.
Sous bruit reel : +1.44 kEUR vs RB2 nu (GAIN), ~71 % du gain omniscient recupere,
clignotement de l'electrolyseur ramene au niveau omniscient.

Parametres de PRODUCTION (dans RB2(Pred)/ et RB2(SoH)/get_optimal_action_RB.py) :
  NOISE_ENABLE=True, HYST_ENABLE=True, M_SIGMA=1.0, MIN_DWELL=12.

Caveat (a garder dans la redaction) : l'hysteresis est un DISPOSITIF DE ROBUSTESSE
AU BRUIT, pas un meilleur EMS dans l'absolu (sans bruit, la bande +-sigma est trop
large et le plateau retombe sur RB2 nu). On la presente comme la facon de
PRESERVER la valeur de la prevision face a son incertitude reelle.


4. CHIFFRES DE LA FIGURE  (Monte-Carlo N=200, mesocentre, 25 ans)
--------------------------------------------------------------------------------
Source : sens_pred_noise_N200_meso.txt
sigma_inject ~ U([0.5,1.5]*39.38) kWh (test de misestimation), bande hyst figee.

  point (variante hyst)   LPSP %  +/- sLPSP   deg kEUR +/- sdeg
  RB2(Pred)               2.3642   0.0430      65.025    0.161
  RB2(SoH+Pred)           2.4796   0.0515      59.390    0.035
  RB2          (ref.)     2.4509   --          65.415    --
  RB2(SoH)     (ref.)     2.5474   --          59.359    --

Lecture :
  - les ellipses sont MINUSCULES (sdeg ~ 0.04-0.16 kEUR) : l'agregat 25 ans
    auto-moyenne le bruit horaire (~219 000 pas) -> dispersion quasi nulle ;
  - RB2(Pred) hyst DOMINE RB2 nu (meilleur LPSP ET meilleure degradation) ;
  - RB2(SoH+Pred) hyst gagne sur le LPSP a degradation ~egale vs RB2(SoH) ;
  - le front PD (borne optimale, foresight parfait sur 25 ans) reste nettement
    en-dessous/a gauche : une regle reste sous-optimale, c'est attendu.


5. FICHIERS
--------------------------------------------------------------------------------
Figures (a relancer en local) :
  plot_pareto_strategies.py     -> pareto_strategies.{pdf,png}
      plan de Pareto toutes strategies + front PD NETTOYE (non domine), sans ellipses.
      lit sens_pred_noise_N200_meso.txt + le front PD.
  plot_pred_uncertainty.py      -> pred_uncertainty_zoom.{pdf,png}
      zoom dedie : 2 strategies pred (hyst) + leur baseline, avec ellipses 1s/2s.
      lit sens_pred_noise_N200_meso.txt.
  (les deux lisent le front PD via ../Vieillissement8/DP/.../dp_pareto_25y_51x51.npz)

Front PD : coordonnees + dominance (../Vieillissement8/DP/) :
  export_pareto_points.py       -> results_*/dp_pareto_points_25y.txt
                                   + dp_pareto_25y_51x51_clean.npz (8 pts non domines)
      Le front stocke a 15 points dont 7 GLOBALEMENT DOMINES (a gauche/bas LPSP) ;
      les figures tracent le front NETTOYE.

Donnees / calcul (mesocentre) :
  sens_pred_noise_N200_meso.txt   means/stds Monte-Carlo N=200 (source de la figure)
  sens_pred_noise.py              harnais de calcul (compute + stats) + slurm
  mc_noise_pred.py                constat de fragilite (decision binaire)
  mc_noise_hyst.py                balayage M_SIGMA x MIN_DWELL (reglage hyst)
  mc_soh_pred.py                  transfert du levier a RB2(SoH)
  run_meso_pred.slurm             job mesocentre

Analyse detaillee (le "pourquoi" complet) :
  robustesse_bruit_prevision.txt

SUPPRIME le 2026-06-30 (recuperable via git) -- figures/scripts montrant
l'omniscient ou les strategies non robustes, devenus source de confusion :
  sens_pred_noise.{pdf,png,txt}            (doublons de _N200_meso, + omniscient)
  sens_pred_noise_N200_meso.{pdf,png}      (figure bin/omni/hyst, remplacee)
  pareto_strategies_vs_DP.{pdf,png}        (nuage "toutes strategies", non robuste)
  plot_pareto_vs_strategies.py             (script de la figure ci-dessus)
================================================================================
