# MPC online V11-p=2

## Statut

Le premier noyau déterministe est implémenté et validé sur un smoke de sept
jours. Il s'agit d'un MPC mixte linéaire résolu à chaque heure par HiGHS via
`scipy.optimize.milp`. Seule la première action est appliquée par la boucle V11.

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

Les six tests vérifient : `p=2`, convexité du surrogate PEMWE, présence des
ancres `j=1/j=2`, bilan et modes du MILP, action exécutée équilibrée, pondération
SoH et test nul exact.

Smoke canonique :

`runs/smoke_7d_f209c53dbce2/`

Il contient 167 décisions horaires par stratégie, sans échec solveur, sans
`lol>1` et avec bilan de déficit fermé. Les résultats économiques de cette
semaine ne sont pas scientifiques. Les temps de décision observés localement
sont :

- H=6 : 42,9 ms en moyenne, 68,0 ms au maximum ;
- H=24 : 546 ms en moyenne, 2,15 s au maximum.

Les deux formulations sont donc online au pas horaire. H=24 est environ treize
fois plus coûteux et doit encore justifier ce surcoût sur un profil annuel.

Le moteur commun n'a pas été modifié : le front PD reste attribuable à la même
physique. Le défaut historique de `lol_tab` est borné explicitement dans les
nouveaux bancs par le résidu de bilan, le nombre de `lol>1` et l'énergie au-delà
du clipping. Toute valeur non négligeable invalidera le point concerné.

## Prochaine expérience

Le screening d'un an compare en parallèle :

- RB1 `(0.20,0.40)` et RB2 `(0.574,0.465)` ;
- MPC sans SoH, H=6 et H=24 ;
- à H=6, `beta_fc=1`, `beta_ely=1` et leur combinaison ;
- à H=24, la combinaison `beta_fc=beta_ely=1`.

Ce plan factoriel sert à sélectionner horizon et canal de santé. Il ne constitue
pas encore un tuning final et ne permettra pas seul de conclure à la valeur du
SoH.

Lancement mésocentre :

```bash
sbatch run_screen_mpc_v11.slurm
```

Le banc écrit chaque point immédiatement dans
`runs/screen_1y_<empreinte>/`. Une relance identique réutilise les points déjà
terminés. Après ce screening seulement, le budget de tuning sera figé et
appliqué symétriquement aux finalistes avec/sans SoH.
