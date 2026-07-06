================================================================================
MPC(LP) -- COMMANDE PREDICTIVE A HORIZON GLISSANT vs STRATEGIES A REGLES
Dossier Robustesse/MPC/ ; redige le 2026-07-05.
================================================================================

1. OBJET
--------
Ajouter au comparatif du manuscrit une famille de commande fondamentalement
differente des regles (RB1, RB2 et derivees SoH/Pred) : un MPC qui, a chaque
pas horaire, resout un programme LINEAIRE sur les MPC_H prochaines heures a
partir de la prevision du profil net, applique la premiere action, et
recommence (horizon glissant). Il se place entre :
    - les RB (en ligne, information locale + fenetre de prevision),
    - la DP (offline, omnisciente sur 25 ans = borne d'optimalite).
Le MPC utilise la MEME information que RB2(SoH_all+Pred) (fenetre 48 h,
meme bruit backtest) : l'ecart mesure est donc attribuable a la DECISION,
pas a l'information.

2. COMPARAISON A ARMES EGALES (garanties d'honnetete)
-----------------------------------------------------
  - meme boucle : Common/main_init_and_loop_forecast (Predictions), H_forecast=48 ;
  - meme surface d'action que les RB : la strategie choisit (P_fc, P_ely), la
    batterie prend TOUT le residu ; l'action retournee est toujours equilibree ;
  - ecretage et comptage du lol par Common/get_lol, comme pour les RB : le LP
    contient un slack de delestage pour rester faisable, mais ce slack n'est
    JAMAIS applique a l'execution (pas de sous-service "silencieux") ;
  - memes metriques (bench_fable) : LPSP, deg kEUR (modeles REELS de
    cost_fcn_total2 sur la trajectoire), EENS, cout unifie VoLL=3 ;
  - meme protocole CRN : graines communes MC_SEED+i, ancrages RB2 socle /
    RB2(SoH_all) / RB2(SoH_all+Pred) re-executes dans le meme job.

3. FORMULATION DU LP (par pas, variables cote DC en W)
------------------------------------------------------
  Variables (blocs de H) : f=P_fc, e=|P_ely|, bd/bc=decharge/charge batterie,
  s=delestage, c=ecretage PV, zf/ze=depassements de seuil, df/de=|Delta P|.
  Contraintes :
    - equilibre : bd-bc+f-e+s-c = P_net_prevu(k) ;
    - SoC via cumsum (rendements CONV 0.9 et BAT 0.95 asymetriques exacts),
      bornes [SOC_MIN, soc_max_vieilli] lues dans Common.get_lol (marge 0.005) ;
    - H2 via cumsum (rendements constants FC['eff']/ELY['eff'], comme les
      plafonds RB2 ; la centrale applique les vraies LUT), bornes [0.5, 199.5] ;
    - f <= 0.999*eta*P_fc_max(t), e <= 0.999*P_ely_max(t)/eta (Pmax VIEILLIS).
  Objectif (EUR, calibre A L'IMPORT sur les modeles reels) :
    - delestage : MPC_VOLL (3 EUR/kWh par defaut ; c'est le bouton du front) ;
    - batterie : ~85 EUR/MWh cycle (pente moyenne de la table cumulative) ;
    - ELY : 0 sous le genou 30 % Pmax, charniere calee sur (a2+b2) a 60 %
      (~12.1 EUR/h) -- le LP retrouve de lui-meme le "genou" qui retro-explique
      le setpoint 0.310 de RB2 ; demarrages via cout sur |Delta e| ;
    - FC : charniere > 80 % Pmax (~0.17 EUR/h), transitoires + on/off via
      |Delta f| ; l'idling est supprime par bande morte a l'execution ;
    - valeurs terminales : credit MPC_V_BAT=0.60 et MPC_V_H2=1.00 EUR/kWh sur
      l'energie restante en fin d'horizon (sinon vidange myope). Garde
      anti-arbitrage verifiee a l'import (cycler batterie->ELY pour le credit
      ne doit pas etre rentable) ; MPC_V_H2 < VoLL*eta_chaine ~ 1.35 (pas de
      thesaurisation H2 pendant un delestage).
  Solveur : scipy.optimize.linprog(method='highs') -- dans l'anaconda du meso,
  AUCUNE dependance nouvelle. Fallback regles RB2 socle si echec (compteur
  LP_FAILURES, reste a 0 sur tous les tests).

4. PREVISION BRUITEE
--------------------
  Fenetre omnisciente de la boucle + bruit AR(1) PAR PAS (rho=0.8), re-tire a
  chaque appel, k=0 exact. Calibration : l'erreur AGREGEE sur 18 h retrouve
  le backtest de RB2(Pred) : sigma=39.38 kWh, biais=-2.32 kWh (memes
  constantes de design que Fable_pred). MPC_NOISE_ENABLE=False = omniscient.
  --> Discussion de fond (omniscience, echelle d'information, limite du bruit
      a lever de facon unifiee avec RB2(Pred), role des ancrages RB2) :
      voir METHODO_mpc.txt.

5. LIMITES ASSUMEES DU MODELE INTERNE (payees dans les metriques reelles)
-------------------------------------------------------------------------
  - pas de binaires on/off (LP pur, pas MILP) : approxime par les couts
    |Delta P| + bandes mortes + gel optionnel MPC_ELY_MIN_DWELL (0 par defaut,
    12 = convention production : sur les tests locaux, -40 % de demarrages ELY
    ET meilleur cout unifie sous bruit -> variante mesuree dans le bench) ;
  - charniere ELY lineaire au-dela de 60 % (le vrai taux sature) : conservatif ;
  - pas de recuperation reversible ELY dans le LP ;
  - rendements H2 constants (vs LUT) ; densite de dommage batterie moyennee
    sur [0.2, 0.995] (vs ~5x plus chere au-dessus de SoC 0.5) ;
  - horizon 24-48 h : pas de vision saisonniere du stock H2 (la DP l'a).

6. FICHIERS
-----------
  MPC(LP)/get_optimal_action_RB.py   strategie (interface identique aux RB,
                                     set_noise_seed/reset, reglages par setattr)
  bench_mpc.py                       banc CRN (formats bench_ultime) :
                                     RB2 socle, RB2(SoH_all), RB2(SoH_all+Pred),
                                     MPC H=24, H=24+gel12h, H=48 (+ --omni)
                                     sweeps : h / pareto (VoLL interne) / vh2
  run_meso_mpc.slurm                 job Helios (smp, 32 coeurs, 24 h)
  plot_pareto_mpc.py                 figure LPSP vs deg (+ front DP v2 en fond)
  README_mpc.txt                     cette note

7. LANCEMENT
------------
  Local (fumee, ~2 min)  : python bench_mpc.py --quick
  Meso (nominal)         : sbatch run_meso_mpc.slurm            # N=100, 25 ans
  Meso (+ omni)          : sbatch run_meso_mpc.slurm 100 25 --omni
  Meso (front MPC)       : sbatch run_meso_mpc.slurm 32 25 --sweep pareto
  Couts indicatifs : run 25 ans H=24 ~ 35-45 min (LP ~5.7 ms/pas),
  H=48 ~ 80-95 min ; bench nominal N=100 ~ 12-16 h sur 32 coeurs.
  Prerequis meso inchanges : GENIAL_DATA_DIR=$WORK/genial_data + dos2unix.

8. VALIDATION LOCALE (donnees SYNTHETIQUES -- chiffres NON representatifs)
--------------------------------------------------------------------------
  Le CSV sidelec et les LUT de rendement ne sont pas dans le git (*.csv
  ignores) : les tests locaux ont tourne sur un profil synthetique au meme
  format. Verifie : 0 echec LP sur tous les runs ; hierarchie attendue
  (MPC bruite >= ultime RB ; H=48 > H=24 ; omni >> bruite) ; interactions
  vieillissement OK sur 3 ans ; ELY starts eleves sans gel -> variante gel 12h.
  LES CHIFFRES A CITER SORTIRONT DU RUN MESOCENTRE SUR DONNEES REELLES.

9. PERSPECTIVES
---------------
  - MILP (binaires on/off exacts) sur les cas ou le LP chatouille ses bandes
    mortes (scipy.optimize.milp, deja utilise par milp_weekly.py) ;
  - valeurs terminales apprises de la DP (fonction de valeur saisonniere) ;
  - MPC_V_H2 saisonnier (ete vs hiver) pour donner au MPC la vision que son
    horizon n'a pas ; balayage --sweep vh2 comme premiere exploration.
================================================================================
