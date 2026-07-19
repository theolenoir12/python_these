# Audit des résultats PD V11-p=2 rapatriés du mésocentre

Date : 18 juillet 2026.

Modèle : `v11-doe-rakousky-mccay-colombo-2026-07-16`. Sources : jobs Slurm `216232` et `216233`.
Le préflight V11-p=2 a réussi dans les deux logs.

## Qualification

Les caches 25 ans sont complets, finis et cohérents avec les ledgers.
Le rollout à epsilon=3 valide le port V11 de la méthode V2 qui a produit
le front historique Pareto_V8. Le balayage multi-epsilon peut être lancé
avec cette variante unique ; les variantes BoL et lookup restent des
diagnostics hors front.

Le balayage Pareto 25 ans n'est pas présent. Les seuls caches
`dp_pareto` disponibles sont les smokes `1y_7x7`; ils ne doivent pas
être tracés comme résultat. La figure produite ici est donc un plan des
objectifs à `epsilon=3`, pas encore le front multi-epsilon.

## Métriques indépendamment recalculées

| Stratégie | Dégradation (kEUR) | EENS (kWh) | LPSP (%) | J@VoLL3 (kEUR) | Rempl. B/FC/ELY |
|---|---:|---:|---:|---:|---:|
| RB1 optimisee | 60.833 | 4242.6 | 0.8102 | 73.561 | 6/4/0 |
| RB2 optimisee | 63.091 | 4210.0 | 0.8040 | 75.721 | 6/5/1 |
| Ablation PD, modele BoL | 42.810 | 9825.1 | 1.8762 | 72.285 | 3/5/1 |
| Controleur PD annuel, lookup | 44.470 | 2635.9 | 0.5034 | 52.377 | 3/5/1 |
| Controleur PD annuel, rollout | 45.111 | 1232.7 | 0.2354 | 48.809 | 4/5/1 |

Les identités `total = retired + current` sont exactes à la précision
machine pour chaque composant et chaque stratégie.

## Lecture scientifique

- Les ablations BoL et séquentielle connaissent exactement la même fenêtre
  future de puissance de 8760 h. Elles sont identiques pendant la première
  année et leur première différence d'action apparaît au pas 8768.
- L'ablation BoL reconstruit chaque année avec SoH=1, alpha=0 et Pmax
  nominaux. Le contrôleur séquentiel reconstruit avec l'état courant, chaque
  année et après remplacement. Il y a donc 25 contre 34 reconstructions :
  l'écart descriptif de 27.54 % ne peut
  pas être attribué au seul SoH et ne compare pas deux solutions PD optimales.
- Le rollout passe ensuite de 52.377 à 48.809 kEUR à VoLL=3, soit 6.81 %. Ce gain change l'algorithme dès le pas 19 et n'est pas un effet du SoH.
- Le rollout V2 domine RB1 et RB2 sur les deux axes au point central ;
  c'est la variante unique retenue pour le balayage du front.

## Contrôle de grille

| Grille | Nu | LPSP (%) | J (kEUR) | Temps (s) |
|---:|---:|---:|---:|---:|
| 21x21 | 28 | 0.1238 | 1.851 | 24 |
| 31x31 | 40 | 0.0926 | 1.815 | 57 |
| 41x41 | 51 | 0.1033 | 1.803 | 121 |
| 51x51 | 63 | 0.1282 | 1.812 | 217 |
| 71x71 | 75 | 0.0889 | 1.782 | 465 |

Entre les grilles 31x31 et 71x71, J reste dans [1.782; 1.815] kEUR, soit une étendue de 1.83 % autour de la moyenne. La LPSP reste dans [0.0889; 0.1282] %, sans convergence monotone. La grille 51x51 suffit pour distinguer un
écart entre ces contrôleurs et RB1/RB2, mais ce contrôle numérique ne
qualifie pas leur statut d'optimum. Tout futur point devra aussi être rejoué
avec une grille plus fine ou une sensibilité séparant état et contrôle.

## Réserves

- Le port conserve volontairement le backward annuel et le rollout V2 de
  Pareto_V8 ; `PROVENANCE_PARETO_V8.md` en donne les empreintes sources.
- `lol_tab` dépasse parfois 1 avant clipping, surtout pendant les surplus.
  En déficit, cela concerne 19 h pour le contrôleur annuel lookup et 3 h pour le rollout. Le déséquilibre auxiliaire cumulé correspondant vaut respectivement 1.461 et 0.309 kWh sur 25 ans : il est négligeable face aux écarts centraux mais doit être corrigé avant les
nouveaux EMS.
- `PD_BoL` reste une ablation de diagnostic, pas un second résultat PD central.

## Décision

Lancer `run_dp_pareto.slurm` avec la variante canonique V2
(`recompute='yearly'`, projection, rollout). Après rapatriement, vérifier
les ledgers, le masque non dominé, le point epsilon=3 et le coude avant de
tracer le front. Le MPC vient après cette étape.
