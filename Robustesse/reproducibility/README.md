# Reproductibilité des expériences de robustesse

Le fichier `../EXPERIMENTS_MANIFEST.json` distingue les stratégies, les
protocoles et les artefacts actuellement cités. Les sorties historiques sont
gelées par leur SHA-256 mais restent marquées `legacy_unfingerprinted` lorsque
le commit et les sources effectivement chargées au moment du calcul ne sont
pas prouvables.

Règles appliquées à partir de ce chantier :

1. une stratégie publiée ne change plus de paramètres ; un nouveau réglage
   reçoit un nouvel identifiant ;
2. une sortie ou un cache nouveau porte une empreinte du contenu de toutes les
   sources, tables, données et paramètres qui l'ont produit ;
3. une ancienne sortie sans empreinte ne peut être reprise automatiquement
   comme cache ;
4. les nouvelles exécutions vont dans un dossier `runs/<id>_<empreinte>/` et
   ne remplacent une sortie canonique qu'après relecture et promotion explicite ;
5. chaque comparaison scientifique conserve ses tests nuls et ses CRN.

Validation légère, sans dépendance externe :

```bash
python Robustesse/reproducibility/validate_manifest.py
```

La provenance de calcul est implémentée dans `provenance.py`; les résumés de
différences appariées dans `paired_stats.py`.

## Ordre des calculs corrigés V9_4

Précondition Helios : le profil Sidelec doit être lisible avant soumission.
Le lanceur utilise par défaut `$WORK/genial_data` ; pour un autre emplacement :

```bash
mkdir -p "$WORK/genial_data"
# Copier une seule fois sidelec_roche_plate_csv.csv dans ce dossier,
# ou exporter GENIAL_DATA_DIR vers un dossier qui le contient.
test -r "$WORK/genial_data/sidelec_roche_plate_csv.csv"
export GENIAL_DATA_DIR="$WORK/genial_data"
```

```bash
cd Robustesse/Vieillissement9_4

# 1. Validation physique et comptable, 1 coeur.
sbatch run_meso_invariance.slurm

# 2. Après succès du job précédent (remplacer JOB_INV).
sbatch --dependency=afterok:JOB_INV run_meso_valeur_info.slurm
sbatch --dependency=afterok:JOB_INV run_meso_maintenance_matrix.slurm
```

La matrice P3 soumet cinq tâches : T=3/6/12 mois à marge 1, puis T=6 mois à
marges 1,5 et 2. Les trois caches T=3/6/12 à marge 1 alimentent ensuite :

```bash
python ../reproducibility/postprocess_corrected_p1_p3.py \
  --p1 /chemin/p1/results_raw.tsv \
  --p3-t3 /chemin/p3_T3/results_raw.tsv \
  --p3-t6 /chemin/p3_T6/results_raw.tsv \
  --p3-t12 /chemin/p3_T12/results_raw.tsv \
  --p3-t6-m15 /chemin/p3_T6_m15/results_raw.tsv \
  --p3-t6-m2 /chemin/p3_T6_m2/results_raw.tsv
```

Chaque job affiche son dossier empreint. Pour les retrouver ensuite sans
deviner l'empreinte :

```bash
find runs -maxdepth 2 -name provenance.json -print
```

Le post-traitement lit ces fiches et refuse un fichier mal étiqueté (mauvaise
période, marge, horizon, portée préventive ou mode de comptabilité). Les deux
jobs de marge sont des sensibilités auxiliaires mais figurent dans le rapport
apparié, avec tests nuls sur les trois politiques non-RUL.

Les tests nuls P1 et P3 sont exécutés avant les Monte-Carlo et arrêtent le job
en cas d'écart. `results_raw.tsv` est le seul cache numérique ; `results.txt`
est un rapport lisible, jamais une entrée automatique.
