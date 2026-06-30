================================================================================
RB2(SoH)(Pred) : AUGMENTATION DE RB2(SoH) PAR LA PREVISION
================================================================================
Date    : 2026-06-30
Methode : EXACTEMENT celle de RB2(Pred) (cf. ../robustesse_bruit_prevision.txt),
          appliquee a la baseline RB2(SoH) au lieu de RB2 nu.

--------------------------------------------------------------------------------
0. BASELINE RB2(SoH)
--------------------------------------------------------------------------------
RB2 dont les setpoints sont MODULES PAR L'ETAT DE SANTE :
    P_fc_set  = 0.440 * FC_max  * SoH_fc^0      (FC : pas de modulation)
    P_ely_set = 0.320 * ELY_max * SoH_ely^0.5   (ELY : setpoint baisse en vieillissant)

--------------------------------------------------------------------------------
1. AUGMENTATION (get_optimal_action_RB.py)
--------------------------------------------------------------------------------
UN SEUL levier 100% previsionnel, IDENTIQUE a RB2(Pred) : la PRE-CHARGE BATTERIE.
Si un DEFICIT NET est prevu sur H_PRE=18h et que le SoC a de la marge -> on COUPE
l'ELY (P_ely_set=0) pour rediriger le surplus PV courant vers la batterie (~95%
aller-retour) au lieu de la chaine H2 (lossy) -> moins de LPSP sur le creux a venir.

  - Bruit de prevision : net_pred = net_vrai + N(biais, sigma), sigma=39.38 kWh @18h
    (backtest LSTM, MEME valeur que RB2(Pred)). NOISE_ENABLE flag.
  - Anti-clignotement : hysteresis +-M_SIGMA*sigma + gel MIN_DWELL. Optimum HERITE
    de RB2(Pred) : M_SIGMA=1.0 / MIN_DWELL=12. HYST_ENABLE flag.
  - Test nul : ENABLE=False -> RB2(SoH) pur a l'identique. Tout gain est
    ATTRIBUABLE a la prevision, EN PLUS de la modulation SoH de la baseline.

--------------------------------------------------------------------------------
2. RESULTATS (harnais mc_soh_pred.py, N=8 graines, gain = baseline - variante)
--------------------------------------------------------------------------------
  horizon | RB2(SoH) | +Pred OMNI | +Pred BRUITE bin | +Pred BRUITE+HYST(1.0/12)
  --------+----------+------------+------------------+--------------------------
   2 ans  |  23.329  |   +1.912   |     +1.285       |     +1.010
   5 ans  |  31.454  |   +2.296   |     +1.025       |     +1.223
  10 ans  |  43.183  |   +1.802   |     +0.230       |     +0.920
  25 ans  |  80.258  |   +1.428   |     -0.368       |     +0.695   (binaire en PERTE)

  -> A 25 ans le scenario d'horizon est COMPLET : le bruite binaire passe en PERTE
     (-0.368, comme RB2(Pred) -0.33) et l'hysteresis le recupere en GAIN (+0.695).
  Points 25 ans (LPSP %, deg kEUR) pour le FRONT DE PARETO :
     RB2(SoH)                 (2.5475, 59.3644)   [validation : == point connu]
     +Pred omniscient         (2.3449, 59.5982)   borne sup.
     +Pred reel (bruite+hyst) (2.4580, 59.4033)   <- trace sur le front
  (RB2(Pred) reel = (2.3297, 65.0030) ; cf. plot_pareto_vs_strategies.py.)

ENSEIGNEMENTS :
(1) LE LEVIER PREVISIONNEL TRANSFERE. La pre-charge apporte +1.8 a +2.3 kEUR
    omniscient sur RB2(SoH), comparable au +2.06 de RB2(Pred) sur RB2 nu. La
    modulation SoH et la prevision sont COMPLEMENTAIRES (la 2e s'ajoute a la 1ere).
(2) LE BINAIRE S'EFFONDRE AVEC L'HORIZON (+1.285 -> +1.025 -> +0.230). Le
    clignotement ELY (start-stop) coute d'autant plus que la DEGRADATION s'accumule
    (deg 4.6 -> 11.1 -> 22.7). Extrapole a 25 ans, il passerait en PERTE comme
    RB2(Pred) (ou le binaire 25 ans donnait -0.33).
(3) L'HYSTERESIS CROISE LE BINAIRE VERS ~5 ANS. A 2 ans elle est en-dessous (son
    cout LPSP > l'economie de degradation encore minuscule) ; a 5 ans elle passe
    devant (+1.223 vs +1.025) ; a 10 ans elle domine (+0.920 vs +0.230).

--------------------------------------------------------------------------------
3. SIMULATION COURTE : CE QU'ELLE PERMET / CE QU'ELLE BIAISE
--------------------------------------------------------------------------------
- VALABLE en simu courte (2-5 ans, ~35s a 2min) : mesurer le LEVIER previsionnel
  (gain omniscient) et iterer sur la strategie.
- BIAISE en simu courte : le verdict ROBUSTESSE (hysteresis vs binaire) car il est
  pilote par la DEGRADATION, qui a besoin d'ANNEES pour s'accumuler. A 2 ans
  l'hysteresis semble meme contre-productive ; il faut >= 5-10 ans pour la voir
  gagner. Le chiffre de tete reste donc a etablir sur l'horizon long (25 ans).

--------------------------------------------------------------------------------
4. FICHIERS
--------------------------------------------------------------------------------
  RB2(SoH)/get_optimal_action_RB.py   strategie RB2(SoH)+pre-charge+bruit+hyst.
  mc_soh_pred.py                      harnais : baseline / omni / bruite / hyst,
                                      n_years PARAMETRABLE (simu courte).
  Lancer (env simu_env) :
    cd "Robustesse/Predictions"
    /home/theo/miniconda3/envs/simu_env/bin/python mc_soh_pred.py 8 5   # N=8, 5 ans

  NB : l'ancien readme (stub "identique a RB5 le 22/09 ... copies de la trame")
       est conserve dans l'historique git.
================================================================================
