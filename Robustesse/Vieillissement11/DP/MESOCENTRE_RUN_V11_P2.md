# Exécution du front de Pareto PD V11-p=2 sur le mésocentre

## Dossiers à réimporter

Remplacer sur le mésocentre les deux dossiers suivants par leurs versions
locales :

- `Python/Robustesse/Vieillissement11/Common/`
- `Python/Robustesse/Vieillissement11/DP/`

`Vieillissement11/DP/` est désormais l'unique dossier PD actif. Ne pas recopier
`DP2` : ce doublon historique a été supprimé.

Les profils d'entrée restent dans `$WORK/genial_data`, référencé par
`GENIAL_DATA_DIR` dans le script Slurm.

## Vérification et soumission

Depuis le dossier `Vieillissement11/DP` du mésocentre :

```bash
python check_dp_v11.py
sbatch run_dp_pareto.slurm
```

Si le transfert a converti les fins de ligne au format Windows, exécuter d'abord :

```bash
dos2unix run_dp_pareto.slurm
```

Le script lance la méthode V2 ayant produit `Pareto_V8`, portée sur les coûts
V11-p=2 : 19 valeurs d'epsilon, grille 51 x 51, horizon 25 ans, projection du
vieillissement et rollout V11. Le réglage `DP_PARETO_V2=1` est déjà fixé dans le
script.

## Sorties à rapatrier

Rapatrier le journal `dp_pareto.<JOBID>.out` ainsi que :

- `runs/dp_pareto_v11_p2_25y_51x51_rollout.txt`
- `runs/dp_pareto_v11_p2_25y_51x51_rollout.npz`
- `runs/dp_pareto_v11_p2_25y_51x51_rollout_ledgers.json`
- `runs/dp_pareto_traj_v11_p2_25y_51x51_rollout.npz`

Le tableau `.txt` est mis à jour après chaque epsilon terminé. Il permet donc de
conserver les résultats partiels si le job atteint la limite de temps.

Après rapatriement, la figure se génère localement avec :

```bash
python plot_pareto.py
```

La provenance exacte des anciens résultats et des sources reprises est consignée
dans `PROVENANCE_PARETO_V8.md` et vérifiable avec
`verify_pareto_v8_provenance.py`.
