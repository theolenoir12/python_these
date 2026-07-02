================================================================================
RB2(RUL) : MODULATION DU SETPOINT ELY PAR LA RUL EXTRAPOLEE
================================================================================
(remplace le stub "identique a RB5 le 22/09..." ; note du 2026-07-02)

Strategie : P_ely_set = 0.320 * ELY_max * (min(RUL_ely/RUL_REF, 1))^p, analogue
de RB2(SoH) ou le signal est la RUL (extrapolation lineaire du SoH jusqu'a
l'EoL, calculee dans la boucle Commune) au lieu du SoH instantane.
Parametres retenus par sweep 25 ans : RUL_ELY_REF = 1000 j, EXP_ELY = 0.1.

--------------------------------------------------------------------------------
RESULTAT HONNETE (baselines cost-min, cf ../sweep_rul_attribution.txt)
--------------------------------------------------------------------------------
A baseline egale (best-vs-best, socle 0.440/0.310 = 80.108 kEUR), le levier RUL
apporte ~0 (-0.001 kEUR). Le "-5.3 % vs RB2 nu (85.55)" cite dans l'en-tete de
get_optimal_action_RB.py compare au RB2 NOMINAL (0.450/0.330) : ce gain vient
en realite de la re-optimisation des constantes, pas de la RUL. Dans le
manuscrit, n'utiliser QUE la comparaison a baseline cost-min.

Argument complementaire (../mc_rul_uncertainty.py + Pronostic SoH/) : la RUL
est une grandeur EXTRAPOLEE (sigma/RUL ~5-12 %, courbe en U) la ou le SoH
instantane est ~2-3 ordres de grandeur plus precis -> RB2(SoH) domine RB2(RUL)
comme signal de derating temps reel. Repositionnement possible de la RUL : la
PLANIFICATION des remplacements (aligner l'EoL sur une fenetre de maintenance),
usage a horizon long coherent avec la courbe sigma_RUL(horizon).

--------------------------------------------------------------------------------
A REFAIRE (fix RUL du 2026-07-02, cf ../../Fable/README_fable.txt sect. 4)
--------------------------------------------------------------------------------
L'estimateur de RUL en ligne etait ancre sur SoH[j_new] (valeur EoL ~0.9 de
l'ancienne unite) -> RUL figee a sa valeur par defaut (3000 j) apres le 1er
remplacement ELY : le levier etait DESACTIVE en silence sur la fin de l'horizon
et les ellipses de mc_rul_uncertainty partiellement sous-estimees. Le fix
(ancres j_rul_* = j+1) est dans les boucles Communes ; relancer :
    sweep_rul.py, sweep_rul_attribution.py, mc_rul_uncertainty.py
(verdict attendu inchange sur le fond, chiffres a rafraichir ; test de
non-regression : ../../Fable/check_rul_fix.py).
================================================================================
