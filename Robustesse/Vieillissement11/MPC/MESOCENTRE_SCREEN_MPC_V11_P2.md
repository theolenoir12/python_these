# Calculs MPC V11-p=2 sur le mésocentre

## Statut

Les jobs v2 sont terminés : 218546 pour le screening (8/8) et 218547 pour le
banc d'incertitude (34/34). Les caches canoniques sont respectivement
`runs/screen_1y_718d8fe28384/` et
`runs/forecast_uncertainty_1y_1acc8ef7e9d2/`. Ne pas les resoumettre à protocole
identique. Les commandes ci-dessous sont conservées pour la reproductibilité.

Les jobs 218548 et 218935 ont terminé la sélection et la validation annuelle.
`combo_top3` est retenu. La dernière étape MPC est le rejeu 25 ans apparié
contre la baseline. Conserver sur le mésocentre le dossier
`runs/tune_validation_1y_9c728d3d847a/` et importer :

- `benchmark_tuning_mpc_v11.py` ;
- `test_tuning_mpc_v11.py` ;
- `benchmark_longrun_tuning_mpc_v11.py` ;
- `test_longrun_tuning_mpc_v11.py` ;
- `run_longrun_tuning_mpc_v11.slurm`.

Le job 218935 s'étant arrêté avant le post-traitement, copier aussi dans
`runs/tune_validation_1y_9c728d3d847a/` les trois petits fichiers produits
localement sans nouveau calcul :

- `decision.json` ;
- `validation_stats.tsv` ;
- `excluded_validation_cases.json`.

Puis, depuis le dossier `MPC/` existant, lancer :

```bash
sbatch run_longrun_tuning_mpc_v11.slurm
```

Le job calcule quatre trajectoires de 25 ans en parallèle et doit créer
`runs/tune_longrun_25y_eab8dde5d5d0/`. Le temps mural estimé est de 12 à 16 h ;
la limite est 24 h et une relance reprend les trajectoires déjà terminées.

## Dossiers à importer

Remplacer sur le mésocentre :

- `Python/Robustesse/Vieillissement11/Common/`
- `Python/Robustesse/Vieillissement11/MPC/`

Les données restent dans `$WORK/genial_data`.

Attention : transférer le **contenu** du dossier local `MPC/` vers le dossier
remote `MPC/`, ou supprimer d'abord l'ancienne copie remote. Ne pas déposer le
dossier `MPC` dans un dossier `MPC` déjà ouvert, sinon on recrée le doublon
`MPC/MPC` observé au rapatriement du 20 juillet.

## Relance v2 obligatoire

L'ancienne borne de variation pouvait interdire artificiellement l'arrêt de la
FC ou de l'électrolyseur après une légère décroissance de leur puissance
maximale. Les caches `screen_1y_d840744e29c7` et
`forecast_uncertainty_1y_d0a7f75d0466` sont v1 et ne doivent pas être repris.
L'identifiant v2 est inclus dans les empreintes, donc les scripts créent de
nouveaux dossiers. Depuis `Vieillissement11/MPC`, lancer d'abord :

```bash
dos2unix run_screen_mpc_v11.slurm  # seulement si nécessaire
sbatch run_screen_mpc_v11.slurm
```

Après succès des 8/8 points, lancer :

```bash
dos2unix run_forecast_uncertainty_mpc_v11.slurm  # seulement si nécessaire
sbatch run_forecast_uncertainty_mpc_v11.slurm
```

Le screening vérifie `scipy.optimize.milp`, exécute les dix tests, puis lance
huit trajectoires d'un an. Le banc d'incertitude lance 34 trajectoires avec les
graines communes. Si un job est interrompu, seuls les caches v2 complets du
nouveau dossier empreinté sont repris.

## Sorties à rapatrier

Rapatrier :

- `mpc_v11_screen.<JOBID>.out` ;
- `mpc_v11_pred.<JOBID>.out` ;
- le dossier complet `runs/screen_1y_<empreinte>/`.
- le dossier complet `runs/forecast_uncertainty_1y_<empreinte>/`.

Chaque point possède un cache `.npz`, un ledger JSON et un résumé JSON. Le
fichier `summary.txt` est réécrit après chaque point terminé. Si le job est
interrompu, resoumettre le même script : les caches complets sont détectés et
ne sont pas recalculés.
