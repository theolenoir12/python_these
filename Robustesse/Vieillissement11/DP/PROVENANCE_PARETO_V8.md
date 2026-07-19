# Provenance de la chaîne PD reprise pour V11

Date de vérification : 18 juillet 2026.

## Source historique retenue

Les figures archivées dans `Python/Robustesse/Pareto_V8/` lisent le cache :

`Pareto_V8/data/dp_pareto_25y_51x51_v2.npz`

Le `README.txt` de cette archive l'identifie comme une copie de :

`Vieillissement8/DP/results_meso/dp_pareto_25y_51x51_v2.npz`

La comparaison binaire confirme cette provenance. Le cache présent dans
`Vieillissement8/DP2/DP/results/` est aussi identique, car `DP2` était un
doublon de travail de la même chaîne.

SHA-256 commun aux trois caches :

`7447b233f23159eb8944cea6050a06bad2b64853ca1b2a739159226963dae0b7`

Le calcul final est le job mésocentre `213210`, lancé avec
`DP_PARETO_V2=1`, 19 valeurs de `epsilon`, une grille `51x51`, 10 contrôles FC,
50 contrôles ELY et un rejeu de 25 ans. Son log historique est
`Vieillissement8/DP2/DP/dp_pareto.213210.out`. Les sources définitives ont été
consolidées dans `Vieillissement8/DP/` par le nettoyage Git `7104674` du
5 juillet 2026.

Empreintes des quatre fichiers historiques centraux :

| Fichier | SHA-256 |
|---|---|
| `dp_core.py` | `5a96b17d953598e5ef0dd8295373a93d75229118ed4f3c01fede75761426d0cd` |
| `dp_aging.py` | `b1e4d0c2726ab8df9af7ac91b1d5cbfdeab4dd74fb15df203b10fc66c4189a21` |
| `dp_pareto.py` | `c644d4e2c3bf33dc818f210b6268bdf1a6f8c3e3d53649607806daf39fc60cfc` |
| `run_dp_pareto.slurm` | `7986dd7c70f674bcdff61dba739fb0433670213844da788fd0daf09b7bdf817f` |

## Port V11 canonique

La seule branche active est désormais `Vieillissement11/DP/`. Elle conserve
la structure de calcul qui a produit le front V8 :

- backward annuel cyclique et politique séquentielle ;
- recalcul au vieillissement courant ;
- variante V2 avec projection et rollout horaire ;
- balayage parallèle des mêmes 19 valeurs de `epsilon` ;
- masque des points non dominés et sauvegarde incrémentale.

Les changements de modèle sont volontairement limités au port V11 :

- coût permanent PEMFC/PEMWE de `Common/degradation_v11.py` ;
- noyau PEMWE nominal fixé à `p=2` et ancres de contrôle `j=1`, `j=2` ;
- coût batterie canonique V11 ;
- boucle physique V11 et métriques issues du ledger corrigé ;
- références RB1 `(0.20, 0.40)` et RB2 `(0.574, 0.465)` réoptimisées sous V11 ;
- métadonnées du modèle et ledgers sauvegardés avec chaque cache.

`check_dp_v11.py` vérifie avant calcul que les coûts élémentaires du backward
coïncident avec les transitions V11, que `p=2` est actif et qu'aucune constante
de coût legacy n'est encore importée par les modules exécutables.

## Organisation retenue

`Vieillissement11/DP2/` était une copie historique sans résultat unique. Il est
retiré de la branche V11 afin qu'un seul dossier soit importé, exécuté et cité.
Les arbres anciens de Vieillissement8 restent des archives de provenance et ne
doivent plus être copiés dans une nouvelle version.
