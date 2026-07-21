# Calculs MPC V11-p=2 sur le mésocentre

## Statut

Les jobs v2 sont terminés : 218546 pour le screening (8/8) et 218547 pour le
banc d'incertitude (34/34). Les caches canoniques sont respectivement
`runs/screen_1y_718d8fe28384/` et
`runs/forecast_uncertainty_1y_1acc8ef7e9d2/`. Ne pas les resoumettre à protocole
identique. Les commandes ci-dessous sont conservées pour la reproductibilité.

Le prochain job est le tuning décrit dans `TUNING_MPC_V11_P2.md`. Il réutilise
11 trajectoires du banc d'incertitude ; conserver impérativement le dossier
`runs/forecast_uncertainty_1y_1acc8ef7e9d2/` sur le mésocentre, importer les
nouveaux scripts dans le dossier `MPC/` existant, puis lancer :

```bash
sbatch run_tuning_mpc_v11.slurm
```

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
