# Relance longrun MPC V3

Date : 22 juillet 2026.

## Objet du correctif

La formulation V3 ajoute une garde d'exécution qui coupe l'électrolyse lorsque
la LOL totale laisserait encore un déficit de puissance. Le seuil de fermeture
reste `1e-4 W`. Le nombre d'activations est enregistré dans le diagnostic
`execution_balance_guard_steps`.

Les résumés physiquement invalides ne sont plus repris comme caches. La
formulation porte désormais l'identifiant :

`mpc-v11-p2-milp-v3-execution-balance-guard-2026-07-22`

Ce nouvel identifiant force un dossier longrun indépendant et le recalcul des
quatre trajectoires. La durée Slurm est portée de 24 h à 48 h.

## Fichiers à envoyer au mésocentre

Depuis `Robustesse/Vieillissement11/MPC/`, remplacer exactement :

1. `mpc_v11.py`
2. `benchmark_tuning_mpc_v11.py`
3. `test_mpc_v11.py`
4. `test_tuning_mpc_v11.py`
5. `run_longrun_tuning_mpc_v11.slurm`

Les autres scripts et les anciens résultats n'ont pas besoin d'être renvoyés.
En particulier, il ne faut ni supprimer ni déplacer
`runs/tune_longrun_25y_eab8dde5d5d0/` : le nouvel identifiant de formulation
empêche sa réutilisation.

## Commande de relance

Depuis le dossier `Robustesse/Vieillissement11/MPC/` :

```bash
sbatch run_longrun_tuning_mpc_v11.slurm
```

Le job commence par exécuter :

```bash
python -m unittest -v \
  test_mpc_v11.py test_tuning_mpc_v11.py test_longrun_tuning_mpc_v11.py
```

Le run attendu est :

`runs/tune_longrun_25y_7f4a35bbc101/`

## Fichiers à récupérer après le job

Récupérer :

1. le log `mpc_v11_long.<JOBID>.out` ;
2. le dossier complet `runs/tune_longrun_25y_7f4a35bbc101/`.

Le dossier complet doit notamment contenir :

- `protocol.json`, `points.tsv`, `summary.json` ;
- `failures.json`, `invalid.json` ;
- `comparison.json`, `comparison.tsv` ;
- `external_cache_manifest.json` ;
- pour chacune des quatre trajectoires, le triplet `.npz`, `_ledger.json` et
  `_summary.json`.

Les quatre labels attendus sont :

- `long_baseline_s1p0_r202604` ;
- `long_baseline_s1p0_r202605` ;
- `long_combo_top3_s1p0_r202604` ;
- `long_combo_top3_s1p0_r202605`.

Le résultat n'est complet que si `failures.json` et `invalid.json` sont vides et
si `comparison.json` annonce `n_paired_seeds = 2`.
