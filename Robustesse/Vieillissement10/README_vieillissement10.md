# Vieillissement10

Version expérimentale créée le 13 juillet 2026 pour tester des modèles de vieillissement PEMFC/PEMWE pilotés par la densité de courant et pour produire des métriques reproductibles de première vie.

## Origine et préservation de Vieillissement9_4

Le dossier est une copie du dernier état de travail de `Vieillissement9_4`, incluant ses évolutions récentes sur l'intégration de connaissances dans les règles EMS (SoH, prévision et expérimentations associées). Il ne s'agit pas d'une copie du seul commit `c8ad6185ea0c407b338b1404a3230fe0b52e86f1` : les modifications locales présentes le 13 juillet 2026 ont été conservées.

Les deux fichiers alors en conflit Git dans `Vieillissement9_4` n'ont volontairement pas été copiés :

- `bench_dwell_ely.py`
- `run_meso_dwell.slurm`

Aucun fichier de `Vieillissement9_4` n'a été modifié ou résolu pendant la création de cette version.

## Changements scientifiques

### Référentiel électrochimique commun

`Common/electrochemistry.py` centralise désormais :

- les courbes de polarisation PEMFC et PEMWE ;
- les courants maximaux et les puissances maximales en fonction de `alpha` ;
- l'inversion puissance vers courant ;
- le calcul de la densité de courant `j`.

Les boucles rule-based n'utilisent plus `P/Pmax` comme substitut de `j` pour calculer les vitesses de vieillissement.

Une table d'interpolation accélère l'inversion. La comparaison à la résolution exacte donne une erreur maximale de l'ordre de `5.2e-6 A/cm²` pour la PEMFC et `6.0e-5 A/cm²` pour le PEMWE.

### PEMFC

Le modèle réversible/irréversible est exprimé en fonction de `j`. La référence actuelle à `0.5 A/cm²` est :

- irréversible : `1.2 µV/h` ;
- réversible : `22 µV/h`.

Les démarrages, l'idling et la récupération restent traités séparément.

### PEMWE

Le modèle de récupération inspiré de Rakousky est exprimé directement en fonction de `j` :

- pas de génération `a/b` sous `1 A/cm²` ;
- rampe entre `1` et `2 A/cm²` ;
- calibration complète de Rakousky à `2 A/cm²` ;
- saturation au-dessus de `2 A/cm²`.

Le paramètre à tuner se trouve dans `Common/cost_fcn_total2.py` :

```python
ELY_REC["scale"] = 1.0
```

`1.0` conserve la calibration Rakousky. Une première estimation purement linéaire pour rapprocher la première vie RB2 de 40 000 h ON serait `0.89`, mais cette valeur n'est pas fixée : elle doit être validée par balayage car récupération, démarrages et point de fonctionnement rendent la relation non linéaire.

Le passage µV vers pourcentage utilise la tension nominale du modèle électrochimique, environ `2.0509 V`, au lieu de la constante historique `1.5 V`.

## Métriques de première vie

`Common/lifetime_metrics.py` calcule pour la PEMFC et le PEMWE :

- temps calendaire et années à 8 760 h ;
- temps ON ;
- EFPH ;
- énergie ;
- nombre de démarrages et durée moyenne d'une séquence ON ;
- fraction de charge moyenne ;
- densité de courant moyenne et percentile 95 ;
- heures par plage de densité de courant ;
- contributions à la dégradation en fin de première vie ;
- taux équivalents en µV/h ON.

Conventions :

```text
ON   : |P| >= 0.0005 * Pmax(alpha)
EFPH : somme(|P| / Pmax(alpha) * dt_h)
vie  : de l'unité initiale au premier reset du SoH
```

Les données sont accessibles dans :

```python
data["first_life_metrics"]
```

`Common/main_plot.py` exporte également `first_life_metrics.txt` dans le dossier de figures de la stratégie exécutée.

## Premier test RB2, horizon 25 ans

Avec `ELY_REC["scale"] = 1.0` :

| Composant | Calendrier | Temps ON | EFPH | Démarrages | j moyen ON |
|---|---:|---:|---:|---:|---:|
| PEMFC | 8.095 ans | 38 916 h | 19 641 h | 3 133 | 0.352 A/cm² |
| PEMWE | 14.143 ans | 35 518 h | 9 309 h | 5 236 | 1.089 A/cm² |

Le détail scientifique et les sources se trouvent dans `RESUME_PREMIERE_APPROCHE_VIEILLISSEMENT10.txt`.

## Exécution

Depuis le dossier `Robustesse/Vieillissement10` :

```powershell
C:\Users\tlenoi01\AppData\Local\anaconda3\python.exe .\RB2\main.py
```

Test court sans génération de figures :

```powershell
C:\Users\tlenoi01\AppData\Local\anaconda3\python.exe -c "import sys; sys.path.insert(0,'.'); from Common.main_init_and_loop import init_and_run_loop; from RB2.get_optimal_action_RB import get_optimal_action_RB; d=init_and_run_loop(get_optimal_action_RB,n_years=1); print(d['first_life_metrics'])"
```

## Périmètre validé et limites

Validé pour cette première étape :

- analyse syntaxique des neuf fichiers modifiés ;
- imports du cœur électrochimique, des coûts, des métriques, des deux boucles rule-based, de la physique et des graphiques ;
- cohérence puissance vers courant vers puissance ;
- comparaison interpolation/résolution exacte ;
- simulation RB2 sur un an ;
- simulation RB2 sur 25 ans et détection des premières vies ;
- cohérence du ledger de remplacements.

Non certifié dans Vieillissement10 :

- `DP/` et `DP2/` réimplémentent encore le vieillissement historique en fraction de puissance ;
- le calcul de `Pmax` du chemin MILP est unifié, mais sa dégradation n'est pas entièrement migrée vers `j` ;
- `Common/milp_weekly.py` contient un import historique de `j_0` absent au niveau module de `Init_EMR_MG_v16_python.py`, défaut également présent dans la base V9_4.

Jusqu'à migration, les comparaisons scientifiques V10 doivent donc utiliser les stratégies rule-based.

## Fichiers principaux ajoutés ou modifiés

Ajoutés :

- `Common/electrochemistry.py`
- `Common/lifetime_metrics.py`
- `RESUME_PREMIERE_APPROCHE_VIEILLISSEMENT10.txt`
- `README_vieillissement10.md`

Adaptés :

- `Common/cost_fcn_total2.py`
- `Common/main_init_and_loop.py`
- `Common/main_init_and_loop_maintenance.py`
- `Common/physics.py`
- `Common/main_plot.py`
- `Common/milp_weekly.py`
- `Common/milp_weekly (1).py`



## Mise a jour RB2 et fond PEMWE long terme (13 juillet 2026)

Le modele PEMWE utilise desormais un fond irreversible
`ELY_REC["a_background"] = 3.5` microV/h ON sous 1 A/cm2. Le point de
Rakousky a 2 A/cm2 reste inchange et la rampe 1--2 A/cm2 part de ce fond.
Cette hypothese evite d'assimiler une variation nette nulle sur 1 009 h a une
absence de vieillissement sur plusieurs dizaines de milliers d'heures.

La RB2 active a ete optimisee uniquement sur le cout unifie, sans contrainte
LPSP et sans information SoH :

- consignes normales FC/ELY : 0.31/0.22 Pmax ;
- plafonds de secours FC/ELY : 0.90/0.225 Pmax ;
- LPSP : 1.7978 % ;
- degradation : 67.284 kEUR ;
- cout unifie de balayage : 82.029 kEUR.

Elle est non dominee face a RB1-costopt-V8 (1.7204 %, 68.621 kEUR).
Les details et les grilles sont dans `RB2/README_RB2_V10.md`.

Metriques de premiere vie finales :

| Composant | Calendrier | Temps ON | EFPH | Energie | Demarrages | j moyen ON |
|---|---:|---:|---:|---:|---:|---:|
| PEMFC | 8.136 ans | 40 066 h | 17 671 h | 27 250 kWh | 3 119 | 0.314 A/cm2 |
| PEMWE | 11.389 ans | 30 003 h | 6 649 h | 105 007 kWh | 4 229 | 0.944 A/cm2 |

`all_aging_2.pdf` et `all_aging_2.png` presentent maintenant les metriques
a cote de chaque diagramme circulaire.
