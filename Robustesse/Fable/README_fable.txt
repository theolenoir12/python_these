================================================================================
FABLE : PROTOTYPES D'AMELIORATION DES STRATEGIES PREVISIONNELLES + NOTES
Dossier : Robustesse/Fable/          Cree : 2026-07-01 (revue de coherence)
================================================================================

0. RESUME
--------------------------------------------------------------------------------
Ce dossier contient les prototypes issus de la revue de coherence :

  RB2(Proba)/   levier previsionnel declenche par une hysteresis sur la
                PROBABILITE de deficit  P(deficit) = Phi(net_pred/sigma)
                (generalise l'hysteresis en energie ; seuils reglables et
                asymetriques ; sans gel par defaut).

  RB2(Prop)/    levier previsionnel PROPORTIONNEL : le setpoint ELY est module
                CONTINUMENT par w = Phi(net_pred/(TAU*sigma)) au lieu d'etre
                coupe en tout-ou-rien. Anti-clignotement PAR CONSTRUCTION,
                utilise l'AMPLITUDE prevue et pas seulement le signe.

  bench_fable.py        banc d'essai Monte-Carlo (common random numbers),
                        memes metriques que sens_pred_noise / reopt_pred.
  run_meso_fable.slurm  job mesocentre.

Les deux strategies partagent le socle cost-min RB2 (0.440/0.310), les
conventions de bruit de RB2(Pred) (biais -2.32 / sigma 39.38 kWh @18h) et la
propriete de TEST NUL : ENABLE=False ou prevision neutre -> RB2 socle EXACT.

Reperes chiffres (reopt_pred.txt, 25 ans, VoLL=3) a battre :
    RB2 socle ................. 80.108 kEUR
    RB2(Pred) hyst (prod) ..... 79.717 kEUR   <- levier actuel
    RB2(Pred) omniscient ...... 79.027 kEUR   <- borne superieure du levier
CRITERE DE SUCCES : total < 79.717 ; ideal : tendre vers 79.03 sans
augmenter les demarrages ELY (colonne ELY_starts du bench).

LANCER (depuis Fable/, env simu_env, GENIAL_DATA_DIR ou layout Data historique) :
    python bench_fable.py --quick          # fumee (1 an, N=2) : verifie que ca tourne
    python bench_fable.py 16 25 --omni     # bench local complet
    sbatch run_meso_fable.slurm            # N=200 au mesocentre
    sbatch run_meso_fable.slurm 64 25 --sweep prop    # balayage TAU
    sbatch run_meso_fable.slurm 64 25 --sweep proba   # balayage (P_HI,P_LO,gel)


1. RB2(Proba) : POURQUOI PASSER EN PROBABILITE
--------------------------------------------------------------------------------
L'hysteresis actuelle filtre le bruit par une bande en ENERGIE +-M_SIGMA*sigma.
Comme sigma est CONNU (backtest), la meme information s'exprime en probabilite :
    P(deficit) = Phi(net_pred / sigma)
    ENTRER si P > P_HI ; SORTIR si P < P_LO ; sinon garder l'etat.

Correspondance : (P_HI, P_LO) = (p, 1-p)  <=>  bande +-Phi^-1(p)*sigma.
    P_HI=0.84/P_LO=0.16 == M_SIGMA=1.0 (controle de non-regression dans le sweep).

Ce que ca apporte :
  a) REGLAGE PORTABLE : le seuil est en unites d'erreur de prevision. Si sigma
     change (autre modele, autre horizon, sigma dependant du regime meteo, ou
     sigma fourni pas a pas par le predicteur), le reglage P_HI/P_LO reste
     valable sans re-sweep. C'est aussi le chainon naturel vers un sigma_t
     VARIABLE (MC-dropout / quantiles par horizon).
  b) BANDE PLUS ETROITE TESTABLE : hypothese = le caveat "plateau omniscient"
     (bande +-1sigma trop large qui bloque les vrais declenchements) se traite
     en reduisant la bande (P_HI=0.60-0.70) SANS gel, plutot qu'en comptant sur
     le dithering du bruit. Le sweep tranchera.
  c) SEUILS ASYMETRIQUES possibles (confiance pour agir != pour arreter).
  d) sigma -> 0  =>  Phi -> echelon  =>  retombe sur l'omniscient binaire :
     le dispositif de robustesse DISPARAIT de lui-meme quand la prevision
     devient parfaite (le caveat disparait par construction).

Parametres : P_HI=0.70, P_LO=0.30, MIN_DWELL=0 (defauts a confirmer par sweep).


2. RB2(Prop) : MODULATION CONTINUE (LE VRAI CHANGEMENT DE MECANISME)
--------------------------------------------------------------------------------
Constat sur la chaine actuelle : la pre-charge est BINAIRE (ELY coupe ou pas).
C'est la bascule qui cree le clignotement sous bruit (start-stop ELY), qui a
impose hysteresis+gel, dont la bande large bloque a son tour les declenchements
"doux". RB2(Prop) supprime la bascule :

    w = Phi( net_pred / (TAU*sigma) )        dans (0,1)
    P_ely_set = C_ELY * P_ely_max * (1-w)

  a) ANTI-CLIGNOTEMENT PAR CONSTRUCTION : le bruit fait varier w de facon
     lisse ; l'ELY ne passe par OFF que si le deficit est quasi certain
     (w~1). Ni zone morte ni gel necessaires. Verif : colonne ELY_starts.
  b) AMPLITUDE, PAS SEULEMENT SIGNE : un deficit prevu PETIT (< sigma) reduit
     l'ELY partiellement -> pre-charge partielle proportionnee au risque.
     Ce sont exactement les cas que la bande +-1sigma jetait.
  c) CONVERGENCE OMNISCIENTE : sigma -> 0 => w -> echelon => binaire omniscient.
  d) TAU = temperature (defaut 1.0) : petit = reactif/quasi-binaire, grand =
     doux/robuste. Balayage --sweep prop (0.25 a 2.0).

Limite attendue (honnetete) : sous bruit fort, w moyen ~0.5 en zone ambigue ->
l'ELY tourne a mi-regime dans des heures de surplus reel -> un peu moins d'H2
stocke que l'hysteresis qui, elle, tranche. Le bench dira si le gain LPSP des
declenchements doux paie plus que ce manque a stocker. Si RB2(Prop) et
RB2(Proba) gagnent tous deux, tester ensuite le COMBO (modulation continue +
petite zone morte).


3. INTEGRATION DE SoH_bat DANS RB2 : DECOMPOSITION DU COUT ET PROPOSITIONS
--------------------------------------------------------------------------------
Decomposition du cout de degradation 25 ans (couts unitaires du fichier d'init,
durees de vie observees ~5 ans BAT / ~12 ans FC / ~13 ans ELY, cf all_aging_2) :

    composant   cout/remplacement   vies/25 ans   cout 25 ans   part
    BAT         7.00 kEUR           ~5.0          ~35.0 kEUR    ~63 %
    ELY         9.34 kEUR           ~1.9          ~18.0 kEUR    ~32 %
    FC          1.18 kEUR           ~2.1          ~ 2.5 kEUR    ~ 4 %
    (total ~55 kEUR : coherent avec les deg ~55-65 kEUR des tableaux)

=> LA BATTERIE PORTE ~2/3 DU COUT DE DEGRADATION et aucune strategie ne
   l'exploite. C'est mecaniquement le plus gros gisement restant. A l'inverse,
   moduler la FC ne peut presque rien rapporter (4 %) : coherent avec
   l'EXP_FC=0 trouve par les sweeps.

POURQUOI C'EST STRUCTURELLEMENT DUR : dans RB2 la batterie est la variable
d'AJUSTEMENT (elle absorbe tout ce que la chaine H2 ne prend pas). Elle n'a pas
de setpoint propre -> la modulation "P_set * SoH^gamma" de RB2(SoH) ne se
transpose pas. Les leviers doivent donc DEPLACER DU FLUX hors de la batterie,
pas la "derater". Quatre propositions, de la plus prometteuse a la plus
speculative (toutes gardent le test nul : SoH_bat=1 -> RB2 exact) :

  (P1) CROSS-MODULATION (implementee en hook dans RB2(Proba)/RB2(Prop),
       parametres BETA_FC_BAT / BETA_ELY_BAT, defaut 0=OFF) :
           P_fc_set  = c_fc  * Pmax * SoH_fc^g  * SoH_bat^(-BETA_FC_BAT)
           P_ely_set = c_ely * Pmax * SoH_ely^g * SoH_bat^(-BETA_ELY_BAT)
       Quand la batterie vieillit, on REMONTE les setpoints H2 : la chaine H2
       prend une part croissante du throughput, la batterie cycle moins (moins
       de deg_SoC cumulee), au prix d'un peu plus d'usure ELY/FC et de pertes.
       L'arbitrage est favorable sur le papier (1 kWh cycle par la batterie
       coute ~7 kEUR/51.84 kWh/0.3 de fenetre utile >> le cout marginal ELY),
       mais c'est le sweep qui tranchera : balayer BETA_ELY_BAT en premier
       (l'ELY est le vrai concurrent de la batterie sur les surplus).
       NB : c'est le symetrique exact de RB2(SoH) -> se raconte tres bien dans
       le manuscrit ("modulation croisee des setpoints par l'etat de sante").

  (P2) FENETRE SoC DEPENDANTE DU SoH_bat. Le modele de deg batterie est une
       densite de dommage par niveau de SoC (Cumulative_degradation_bat) :
       deplacer/retrecir la fenetre [0.2, 0.995] vers la zone la moins
       dommageable quand SoH_bat baisse reduit le dommage par kWh cycle.
       ATTENTION : bornes codees en dur dans get_lol.py (0.2/0.995) ->
       demande de les promouvoir en parametres module (petit refactor).
       Cout : perte de capacite utile -> LPSP ; a ne tester qu'apres (P1).

  (P3) PRE-CHARGE CONSCIENTE DU SoH_bat (synergie avec la prevision) : la
       marge d'energie utile de la batterie fond comme Q*SoH_bat ; a SoH=0.7
       il faut pre-charger PLUS TOT pour stocker la meme energie avant un
       creux. Concretement : SOC_TARGET ou le seuil P_HI (ou TAU) fonction de
       SoH_bat (ex. P_HI = 0.70*SoH_bat^k). Peu de code, s'ajoute a (P1).

  (P4) SI le vieillissement CALENDAIRE est active (BAT_CAL_TCAL_Y, analyse
       R3) : baisser le SoC MOYEN de sejour quand la batterie est jeune
       (surplus routes vers H2 plus tot). Sans calendaire ce levier est nul
       par construction -> seulement pour la variante calendaire du chapitre.

  Verif rapide avant sweep (P1) : la part deg_bat de RB2 vs 75-25 dans les
  pie-charts existants (degradation_pie_charts_final) pour confirmer le ~63 %.

  NB : un plafond de C-rate SoH-dependant a ete envisage puis ecarte : a
  l'echelle du systeme (pack 51.84 kWh, flux ~qq kW), C_rate ~0.1-0.4 << 1,
  la penalite super-lineaire C>1 du modele n'est presque jamais active.


4. BUGS CORRIGES A COTE (fichiers Communs, hors Fable/)
--------------------------------------------------------------------------------
RUL EN LIGNE (Vieillissement8/Common/main_init_and_loop.py,
              Predictions/Common/main_init_and_loop.py et _forecast.py) :
    l'ancre de l'extrapolation lineaire etait j_new_* (pas du remplacement),
    or SoH[j_new] est la valeur EoL (~0.9) de l'ANCIENNE unite ; delta_soh
    restait negatif pendant toute la vie des unites suivantes -> RUL figee a
    sa valeur par defaut (8000/3000 j) des le 1er remplacement -> levier
    RB2(RUL) silencieusement desactive ensuite (et ellipses mc_rul_uncertainty
    partiellement sous-estimees, le bruit x(1+eps) sur RUL=3000 ne franchissant
    jamais le seuil de 1000 j).
    FIX : nouvelles ancres j_rul_* = j+1 au remplacement (SoH=1 par
    construction). Le telescopage des couts (j_new_*) est INCHANGE ; aucune
    strategie autre que RB2(RUL) ne lit la RUL -> aucun autre resultat modifie.
    A REFAIRE apres fix : sweep_rul.py, sweep_rul_attribution.py,
    mc_rul_uncertainty.py (les conclusions peuvent bouger un peu ; l'argument
    sigma_RUL(horizon) en U reste valable).

5. POINTS NOTES MAIS PAS TOUCHES (choix assume : ne pas invalider l'existant)
--------------------------------------------------------------------------------
  a) get_lol : lol = max(lol_pmax, lol_storage, lol_soc), chaque terme calcule
     avec les AUTRES ressources non corrigees -> sous-compte l'energie non
     servie quand DEUX contraintes sont actives au meme pas (batterie au
     plancher + reservoir vide, typiquement dans les pires creux). Patch
     suggere (a chiffrer avant adoption, change le LPSP de TOUTES les
     strategies) : recalculer lol FINAL = 1-(P_bat_corr+P_h2_corr)/P_tot_ref
     apres toutes les corrections, plutot que le max des trois.
  b) Nommage Predictions/ : RB2(SoH)/ y contient en realite RB2(SoH)+Pred
     (bruit+hyst) tandis que RB2(SoH+Pred)/ est la variante binaire sans
     bruit. A renommer un jour de calme.
  c) Baselines : ne garder dans le manuscrit que le socle cost-min
     (80.108) ; le readme de RB2(RUL) cite encore -5.3 % vs le nominal 85.55
     alors que l'attribution honnete (sweep_rul_attribution) donne ~0.
  d) sigma=39.38 kWh mesure sur sidelec_csv2 (2 ans, conso bruitee) alors que
     la simulation tourne sur l'ancien CSV (1 an, tuile x51) : approximation a
     assumer dans la redaction (majorant plutot realiste).

6. FICHIERS / SORTIES
--------------------------------------------------------------------------------
  bench_fable.py             banc d'essai (voir usage en tete de fichier)
  check_rul_fix.py           test de non-regression du fix RUL (sect. 4) :
                             usure FC acceleree x150, verifie que la RUL est
                             re-estimee apres le 2e remplacement (valide OK
                             avec le fix / ECHEC reproduit sur l'ancien code)
  bench_fable.txt            tableau stats (apres run nominal)
  bench_fable_cloud.csv      nuage brut (label;seed;lpsp;deg;eens;total;starts)
  sweep_fable_prop.txt       balayage TAU        (apres --sweep prop)
  sweep_fable_proba.txt      balayage seuils/gel (apres --sweep proba)
  RB2(Proba)/, RB2(Prop)/    strategies (main.py = run unitaire 25 ans + plots)

  Lecture du bench : comparer 'total' (deg + VoLL*EENS) a la ref RB2(Pred)
  hyst ; surveiller ELY_starts (clignotement) et sLPSP/sdeg (dispersion MC).
  La config Proba 0.84/0.16 d12 du sweep == prod actuelle (non-regression).
================================================================================
