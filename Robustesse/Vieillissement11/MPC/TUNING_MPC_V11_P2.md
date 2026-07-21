# Tuning du MPC H24 V11-p=2

Date : 21 juillet 2026.

## Objet

Le tuning porte uniquement sur le MPC H24 sans pondération SoH, retenu après le
screening v2. Le coût publié reste
`J3 = coût de dégradation exact du ledger + 3 EUR/kWh × EENS`. Les paramètres
ci-dessous modifient seulement le surrogate utilisé pour choisir les actions.

Le protocole est identifié par
`mpc-v11-p2-h24-tuning-v1-2026-07-21`. Il sépare strictement les graines de
sélection et de validation.

## Étage 1 — sélection

La baseline est `(Vbat, VH2)=(0,60 ; 1,00) EUR/kWh`, avec tous les poids d'usure
à 1. Douze variantes modifient chacune un seul levier :

- valeur terminale batterie : 0,30 ou 1,20 EUR/kWh ;
- valeur terminale H2 : 0,50 ou 1,25 EUR/kWh ;
- poids d'usure batterie, FC et ELY : 0,5 ou 2,0 ;
- poids de dynamique FC : 0 ou 3.

Les 13 cas sont évalués avec le bruit nominal x1 sur les graines 202601,
202602 et 202603, soit 39 trajectoires. La métrique de classement est le J3
moyen. Les trois meilleurs réglages portant sur trois leviers distincts sont
retenus afin de ne pas sélectionner simultanément les bornes haute et basse
d'un même bouton.

Un cas est exclu dans son ensemble si au moins une de ses trois trajectoires
échoue à un garde-fou physique (solveur, déficit non fermé après LOL ou LOL
supérieure à 1). Une trajectoire rejetée est consignée dans `invalid.json` et
le cas correspondant dans `excluded_cases.json`; une baseline invalide reste
une erreur bloquante.

## Étage 2 — validation aveugle

La baseline, les trois réglages simples, leur combinaison top-2 et leur
combinaison top-3 sont évalués sur les graines réservées 202604 et 202605 aux
bruits x0,5, x1 et x1,5, ainsi qu'en prévision parfaite et sous persistance.
Cela représente 6 cas × 8 scénarios = 48 trajectoires.

Le critère primaire est le J3 moyen au bruit x1 sur les deux graines réservées.
Un réglage n'est admissible que si sa pénalité face à la baseline reste :

- inférieure ou égale à 1 % sous prévision parfaite ;
- inférieure ou égale à 2 % dans chacun des diagnostics x0,5, x1,5 et
  persistance.

Même s'il est premier, un réglage n'est conservé que s'il améliore la baseline
d'au moins 1 % sur le critère primaire. Sinon, la baseline est gardée. Cette
règle évite de promouvoir un optimum numérique sub-pourcent.

## Budget et caches

Le protocole évalue 87 trajectoires, mais réutilise 11 trajectoires baseline
déjà validées dans `runs/forecast_uncertainty_1y_1acc8ef7e9d2/`. Le calcul
nouveau porte donc sur 76 trajectoires : 36 en sélection et 40 en validation.
Les sorties sont empreintées et chaque trajectoire candidate possède son NPZ,
son ledger et son résumé. Les références externes sont consignées dans
`external_cache_manifest.json`.

Le screening complet doit être créé dans
`runs/tune_screen_1y_97e636e32db7/`. Le nom du dossier de validation dépend des
trois cas sélectionnés et n'est donc connu qu'après l'étage 1.

## Résultat de l'étage 1

Le job 218548 a produit les 39/39 trajectoires, sans échec solveur. L'audit
indépendant retrouve un seul profil pour les 39 points et recalcule exactement
les métriques et les coûts depuis les NPZ et les ledgers.

`terminal_h2_1p25` est exclu : sur la graine 202603, 35,873 W d'électrolyse
subsistent pendant deux pas avec `lol=1`, laissant autant de déficit non fermé.
Les trois leviers sélectionnés sont `battery_wear_0p5`,
`terminal_bat_1p2` et `fc_wear_2`, avec respectivement -1,132 %, -0,686 % et
-0,547 % de J3 moyen face à la baseline sur les graines d'apprentissage. Les
trois gagnent sur 3/3 graines. Le dossier de validation attendu est désormais
`runs/tune_validation_1y_9c728d3d847a/`.

## Exécution mésocentre

Le dossier `Common/` n'a pas changé pour cette étape. Dans le dossier distant
`Vieillissement11/MPC/` existant, mettre à jour uniquement :

- `mpc_v11.py` ;
- `benchmark_mpc_v11.py` ;
- `benchmark_tuning_mpc_v11.py` ;
- `test_mpc_v11.py` ;
- `test_tuning_mpc_v11.py` ;
- `run_tuning_mpc_v11.slurm`.

Ne pas déposer un nouveau dossier `MPC` dans le dossier `MPC` ouvert. Conserver
sur place `runs/forecast_uncertainty_1y_1acc8ef7e9d2/`, nécessaire à la reprise
des onze références baseline.

Depuis le dossier canonique `Vieillissement11/MPC` :

```bash
sbatch run_tuning_mpc_v11.slurm
```

Le job exécute les seize tests, les deux étages et la décision automatique.
Une interruption est reprise depuis les caches complets. Le résultat à examiner
en premier est `decision.json` dans le dossier `tune_validation_1y_*`, puis
`ranking.tsv`, `validation_stats.tsv`, les trajectoires et leurs ledgers.

Pour reprendre après le job 218548, mettre à jour seulement
`benchmark_tuning_mpc_v11.py` et `test_tuning_mpc_v11.py`, puis resoumettre le
même lanceur. Les 39 trajectoires de sélection sont relues depuis les caches ;
seules les 40 trajectoires non-baseline de validation sont nouvelles.

Le rejeu 25 ans n'est pas inclus : il ne sera lancé qu'après audit de la
validation annuelle et seulement pour la baseline et le réglage effectivement
retenu.
