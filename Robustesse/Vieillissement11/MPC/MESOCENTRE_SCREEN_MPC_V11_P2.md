# Calculs MPC V11-p=2 sur le mésocentre

## Dossiers à importer

Remplacer sur le mésocentre :

- `Python/Robustesse/Vieillissement11/Common/`
- `Python/Robustesse/Vieillissement11/MPC/`

Les données restent dans `$WORK/genial_data`.

Attention : transférer le **contenu** du dossier local `MPC/` vers le dossier
remote `MPC/`, ou supprimer d'abord l'ancienne copie remote. Ne pas déposer le
dossier `MPC` dans un dossier `MPC` déjà ouvert, sinon on recrée le doublon
`MPC/MPC` observé au rapatriement du 20 juillet.

## Reprise prioritaire du banc d'incertitude

Le cache canonique local
`runs/forecast_uncertainty_1y_d0a7f75d0466/` contient 33/34 trajectoires. Il
doit être présent au même chemin sur le mésocentre. Depuis `Vieillissement11/MPC` :

```bash
dos2unix run_forecast_uncertainty_mpc_v11.slurm  # seulement si nécessaire
sbatch run_forecast_uncertainty_mpc_v11.slurm
```

Le script relit les 33 caches et ne recalcule que
`mpc_no_soh_h24_noisy_s1p0_r202604`. Si ce MILP échoue encore, le nouveau log
contiendra le pas, le SoC, le stock H2, les SoH et les paramètres de polarisation
à l'origine de l'infaisabilité.

## Relance éventuelle du screening parfait

Depuis `Vieillissement11/MPC` :

```bash
dos2unix run_screen_mpc_v11.slurm  # seulement si nécessaire
sbatch run_screen_mpc_v11.slurm
```

Le script vérifie la présence de `scipy.optimize.milp`, exécute les tests
unitaires, puis lance huit trajectoires d'un an en parallèle. Le budget demandé
est de 8 cœurs, 16 Go et 3 heures.

## Sorties à rapatrier

Rapatrier :

- `mpc_v11_screen.<JOBID>.out` ;
- le dossier complet `runs/screen_1y_<empreinte>/`.

Chaque point possède un cache `.npz`, un ledger JSON et un résumé JSON. Le
fichier `summary.txt` est réécrit après chaque point terminé. Si le job est
interrompu, resoumettre le même script : les caches complets sont détectés et
ne sont pas recalculés.
