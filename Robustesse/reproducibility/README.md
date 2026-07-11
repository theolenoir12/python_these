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
