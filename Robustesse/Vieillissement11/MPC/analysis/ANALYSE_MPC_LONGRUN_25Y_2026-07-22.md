# Analyse provisoire du longrun MPC 25 ans

Date : 22 juillet 2026.

## Provenance et complétude

Le dossier analysé est `runs/tune_longrun_25y_eab8dde5d5d0/`. Le protocole
attend quatre trajectoires appariées : baseline et `combo_top3`, chacune avec
les graines de prévision 202604 et 202605, au bruit nominal x1.

Le job Slurm 219129 a atteint sa limite de 24 h après trois trajectoires :

| Cas | Graine | LPSP (%) | Dégradation (kEUR) | EENS (kWh) | J3 (kEUR) |
|---|---:|---:|---:|---:|---:|
| baseline | 202604 | 0,466239 | 61,400610 | 2 441,517 | 68,725161 |
| combo_top3 | 202604 | 0,443415 | 60,238982 | 2 321,997 | 67,204973 |
| combo_top3 | 202605 | 0,447373 | 60,198177 | 2 342,720 | 67,226335 |

La trajectoire baseline 202605 manque. Les fichiers `comparison.tsv`,
`comparison.json`, `invalid.json` et `failures.json` ne pouvaient donc pas être
produits. La comparaison appariée à deux graines et la décision finale restent
ouvertes.

## Contrôle physique bloquant

La baseline 202604 ferme le bilan à la précision numérique
(`5,7e-11 W`). En revanche, les deux trajectoires `combo_top3` présentent une
électrolyse résiduelle pendant un délestage total :

| Graine | Défaut maximal après LOL (W) | Nombre d'occurrences observé | Énergie non fermée (kWh) |
|---|---:|---:|---:|
| 202604 | 35,884 | 1 | 0,035884 |
| 202605 | 35,887 | 10 | 0,358846 |

Le seuil de validation annoncé est `1e-4 W`. Ces deux trajectoires sont donc
invalides en l'état. Le job ayant été tué avant le post-traitement final, leur
présence dans `points.tsv` ne constitue pas une validation.

Le mécanisme est le suivant : lorsque la demande est positive, que la batterie
est à sa borne basse et que la pile est éteinte, une consigne minimale
d'électrolyse peut encore être exécutée. La LOL est bornée à 1 et ne peut alors
plus compenser cette consommation. Il faut corriger cette garde d'exécution et
invalider les deux caches `combo_top3` avant une relance scientifique.

## Signal de performance, sous réserve de correction

Sur la seule paire disponible (graine 202604), `combo_top3` améliore toutes les
métriques par rapport à la baseline :

| Métrique | Écart absolu | Écart relatif |
|---|---:|---:|
| LPSP | -0,022824 point | -4,895 % |
| Dégradation | -1,161628 kEUR | -1,892 % |
| EENS | -119,520 kWh | -4,895 % |
| J3 | -1,520188 kEUR | -2,212 % |

Le signal dépasse le seuil de matérialité de 1 %, mais il ne peut pas encore
être déclaré confirmé : il manque la baseline 202605 et les deux trajectoires
tunées doivent être recalculées après correction.

## Comparaison au front DP 25 ans

Le front DP disponible est extrait de `DP/dp_pareto.216257.out`. Ses valeurs
sont arrondies dans le log, donc les écarts suivants sont indicatifs à quelques
unités sur le dernier chiffre.

À la LPSP de `combo_top3` 202604 (0,443415 %), l'interpolation du segment DP
entre epsilon 0,75 et 0,5 donne environ 44,601 kEUR de dégradation et
51,567 kEUR de J3 avec l'EENS du MPC. Le MPC reste donc au-dessus du front de :

- +15,638 kEUR de dégradation, soit +35,1 % ;
- +15,638 kEUR de J3, soit +30,3 %.

Le meilleur J3 échantillonné sur le front DP vaut 48,085 kEUR (epsilon 20),
contre 67,205 kEUR pour ce point MPC provisoire, soit +39,8 %. Le MPC ne domine
donc pas globalement le front DP, conformément à l'attendu. En revanche, même
provisoire, il domine les points RB1 et RB2 du même calcul 25 ans en LPSP et en
dégradation.

## Relance minimale recommandée

Ne pas relancer immédiatement le même job : le cache reprendrait les deux
trajectoires `combo_top3` invalides et ne calculerait que la baseline 202605,
puis le contrôle final rejetterait le banc.

La suite minimale est :

1. ajouter une garde d'exécution MPC qui coupe l'électrolyse résiduelle lorsque
   la LOL totale ne suffit plus à fermer un déficit ;
2. ajouter un test de régression et faire rejeter les caches dont
   `max_deficit_shortage_after_lol_w > 1e-4` ;
3. recalculer les deux `combo_top3` et la baseline 202605 ;
4. récupérer les trois trajectoires corrigées et les fichiers agrégés de
   comparaison/validation.

Le coût énergétique du défaut observé est très faible et ne devrait pas changer
qualitativement la position du MPC par rapport au DP. La correction reste
néanmoins indispensable pour conserver un front physiquement admissible et une
provenance reproductible.
