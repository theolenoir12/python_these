# Screening MPC V11-p=2 sur le mésocentre

## Dossiers à importer

Remplacer sur le mésocentre :

- `Python/Robustesse/Vieillissement11/Common/`
- `Python/Robustesse/Vieillissement11/MPC/`

Les données restent dans `$WORK/genial_data`.

## Soumission

Depuis `Vieillissement11/MPC` :

```bash
dos2unix run_screen_mpc_v11.slurm  # seulement si nécessaire
sbatch run_screen_mpc_v11.slurm
```

Le script vérifie la présence de `scipy.optimize.milp`, exécute les six tests
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
