# MPC online V11-p=2

## Statut

Le noyau déterministe est corrigé et le screening annuel doit être rejoué. Il
s'agit d'un MPC mixte linéaire résolu à chaque heure par HiGHS via
`scipy.optimize.milp`. Seule la première action est appliquée par la boucle V11.
La formulation courante est identifiée par
`mpc-v11-p2-milp-v2-delta-capacity-fade-2026-07-20`. Le diagnostic qui invalide
les caches MPC v1 est `analysis/DIAGNOSTIC_MPC_DELTA_BOUND_2026-07-20.md`.

Les anciens résultats de `Python/Robustesse/MPC3/` ne sont pas réutilisés comme
résultats : ils reposent sur les anciennes fonctions de coût. Leur architecture
à horizon glissant a seulement servi de point de départ.

## Formulation

Le problème interne contient explicitement :

- bilan de puissance avec délestage et écrêtage ;
- dynamiques linéarisées de la batterie et du réservoir H2 ;
- modes exclusifs charge/décharge batterie et FC/ELY ;
- variables binaires de marche et de démarrage FC/ELY ;
- dommage permanent FC V11 et coût de démarrage ;
- approximation affine convexe du dommage PEMWE quadratique V11, avec les
  ancres obligatoires `j=1` et `j=2` ;
- coûts terminaux communs pour la batterie et H2.

La borne de variation au premier pas couvre la puissance réellement exécutée
à l'heure précédente, même si la puissance maximale vient de diminuer avec le
vieillissement. Sans cette précaution, l'arrêt complet d'un convertisseur
précédemment à sa puissance maximale pouvait devenir artificiellement
infaisable.

Le surrogate sert uniquement à choisir l'action. Le coût rapporté reste celui
du ledger V11 exact après exécution dans `Common/main_init_and_loop.py`.

## Attribution du SoH

Deux modes partagent toutes les contraintes, prévisions et paramètres :

- `no_soh` utilise le SoH uniquement pour les contraintes physiques inévitables
  — capacité et puissances disponibles — mais pas pour pondérer l'objectif ;
- `soh` multiplie les coûts internes FC et ELY par
  `h^(-beta)`, où `h` est la marge de santé normalisée entre l'état neuf et
  l'EoL.

Le test nul `health_mode=soh, beta_fc=beta_ely=0` reproduit bit-à-bit le mode
`no_soh`, y compris la trajectoire complète et le ledger.

## Validations locales

Commande unitaire :

```bash
python -m unittest -v test_mpc_v11.py
```

Les dix tests vérifient : `p=2`, convexité du surrogate PEMWE, présence des
ancres `j=1/j=2`, bilan et modes du MILP, action exécutée équilibrée, pondération
SoH, test nul exact, prévisions bruitées appariées, `lol=0` en surplus et arrêt
possible après une décroissance de capacité.

Le smoke v1 historique est :

`runs/smoke_7d_f209c53dbce2/`

Il contient 167 décisions horaires par stratégie, sans échec solveur, sans
`lol>1` et avec bilan de déficit fermé. Les résultats économiques de cette
semaine ne sont pas scientifiques. Les temps de décision observés localement
sont :

- H=6 : 42,9 ms en moyenne, 68,0 ms au maximum ;
- H=24 : 546 ms en moyenne, 2,15 s au maximum.

Ces temps v1 indiquent que les horizons sont compatibles avec le pas horaire.
Le smoke v2 `runs/smoke_1d_9a11b7e02867/` passe ensuite sur 23 décisions pour
H6 et H24, sans échec, `lol>1` ni résidu de bilan ; son test nul est exact. Un
mini-screening séparé confirme que l'identifiant v2 est enregistré dans le
protocole et la trajectoire.

## Résultats annuels v1, conservés uniquement comme diagnostic

Le screening `runs/screen_1y_d840744e29c7/` compare :

- RB1 `(0.20,0.40)` et RB2 `(0.574,0.465)` ;
- MPC sans SoH, H=6 et H=24 ;
- à H=6, `beta_fc=1`, `beta_ely=1` et leur combinaison ;
- à H=24, la combinaison `beta_fc=beta_ely=1`.

Les 8/8 points utilisent le même profil annuel, le ledger V11-p=2 et une
prévision parfaite. Leurs métriques sont reproductibles, mais ils ont été
produits avec l'ancienne borne de variation. Ils ne constituent donc plus des
résultats acquis. À titre de trace, le MPC H24 sans SoH atteignait
0,262979 % de LPSP, 2,375768 kEUR de dégradation et 2,541018 kEUR pour J3. Il
réduit J3 de 5,231 % face à H6 et de 12,034 % face à RB1, qu'il domine avec RB2
sur les deux axes. Son temps de résolution vaut 201,6 ms en moyenne et 3,74 s
au maximum. H24 reste l'hypothèse de travail à retester, pas encore une base
définitivement retenue.

La référence DP annuelle exacte contient 19 points. Tous les points MPC/RB du
screening sont dominés par au moins un point DP. À la LPSP du MPC H24, son
surcoût de dégradation interpolé au front vaut 0,632001 kEUR (+36,24 %) ; son
J3 est 37,71 % au-dessus du meilleur J3 DP échantillonné. Le DP connaît toute
l'année, tandis que le MPC ne connaît que sa fenêtre de 24 h.

Dans les caches v1, à H24, la pondération SoH `beta_fc=beta_ely=1` change J3 de +0,062 % sous
prévision parfaite. Dans le banc d'incertitude apparié, les variations moyennes
sont +0,074 % (bruit x0,5), -0,200 % (x1), -0,311 % (x1,5) et -1,585 % sous
persistance. Aucun cas n'atteint le seuil pratique de quelques pourcents : cette
injection simple du SoH semblait non matérielle. Cette conclusion doit être
confirmée sur la formulation v2.

Le banc d'incertitude v1 `runs/forecast_uncertainty_1y_d0a7f75d0466/` reste
un diagnostic legacy : 33/34 points sont terminés et
`mpc_no_soh_h24_noisy_s1p0_r202604` a échoué. Les comparaisons à x1 utilisent
donc seulement quatre graines communes. Les trajectoires terminées ne laissent
aucun déficit après LOL ; elles présentent 4,353 à 15,177 kWh/an d'écrêtage
implicite, désormais mesuré séparément. Trois anciens `lol>1` surviennent
uniquement en surplus et n'affectent pas EENS/LPSP. `Common/get_lol.py` borne
maintenant la LOL à zéro en surplus, sans modifier les états physiques.
Face à la prévision parfaite sans SoH, J3 augmente en moyenne de 4,84 % au
bruit x0,5, 10,36 % à x1 (quatre graines seulement), 14,32 % à x1,5 et 50,03 %
sous persistance. Ces écarts suggèrent que la qualité de prévision est un levier
matériel, mais ils doivent être recalculés avec la formulation v2.

## Prochaine expérience

Réimporter les contenus des dossiers canoniques `Common/` et `MPC/`, puis
relancer successivement `run_screen_mpc_v11.slurm` et
`run_forecast_uncertainty_mpc_v11.slurm`. L'identifiant de formulation entre
dans les empreintes : de nouveaux dossiers sont créés et aucun cache MPC v1
n'est réutilisé. Le tuning ne commencera qu'après audit complet de ces deux
runs. La variante SoH ne sera conservée que si elle franchit le seuil de
quelques pourcents.


## Information future, reference DP et incertitude

Les runs H=6/H=24 sont omniscients sur leur horizon fini : a chaque heure, le
controleur recoit les valeurs realisees du profil net des 6 ou 24 prochains pas.
Le pas courant est une mesure exacte. Le DP V11 est lui aussi clairvoyant, mais
sur une fenetre annuelle reelle reconstruite chaque annee. Il reste discretise
(51x51 et grille de commandes) : c'est la reference de performance du projet,
sans etre une preuve mathematique d'optimalite globale dans l'espace continu du
MPC.

Une comparaison numerique n'est valide que sur le meme horizon d'evaluation.
Le front DP comparable au screening se lance par :

~~~bash
sbatch run_dp_reference_for_mpc.slurm
~~~

Puis la comparaison, qui refuse automatiquement tout melange 1 an/25 ans, est :

~~~bash
python compare_mpc_dp_v11.py \\
  --mpc-run runs/screen_1y_<empreinte> \\
  --dp ../DP/results/mpc_reference_1y_<empreinte>/dp_reference_1y_51x51_v2.npz
~~~

Le mode `forecast_mode=noisy` remplace l'omniscience par un profil horaire
bruite. Sa calibration reprend le backtest du projet a 18 h : biais cumule
-2.32 kWh et ecart-type 39.38 kWh. L'erreur augmente comme la racine de
l'echeance, est correlee AR(1) avec rho=0.8, et le pas courant reste exact. Les
origines de prevision successives sont independantes, choix volontairement
conservatif qui sert de borne haute. Les facteurs 0.5, 1 et 1.5 et cinq graines
communes aux variantes SoH/non-SoH se lancent avec :

~~~bash
sbatch run_forecast_uncertainty_mpc_v11.slurm
~~~
