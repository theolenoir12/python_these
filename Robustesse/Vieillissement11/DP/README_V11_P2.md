# Programmation dynamique V11 — nominal p=2

## Statut

La méthode canonique est la chaîne V2 qui a produit les valeurs réellement
utilisées dans les figures `Pareto_V8`, portée sur V11-p=2. Sa provenance
binaire et ses empreintes sont consignées dans `PROVENANCE_PARETO_V8.md`.
Le portage technique, le point central à `epsilon=3` et les métriques réalisées
du job 216233 sont validés. Le balayage multi-epsilon du job 216257 est terminé
et audité dans `AUDIT_PARETO_V11_P2_2026-07-19.md`. Les fichiers
`note_dp_front_v2.txt` et `DEMARCHE_dp_front_v2.txt`
décrivent l'ancien modèle de coût et restent utiles seulement pour l'historique
de la méthode V2.

Vérification locale de la provenance :

```bash
python verify_pareto_v8_provenance.py
```

Le socle exécutable canonique est :

- `dp_core.py` : backward annuel au coût permanent V11 ;
- `dp_aging.py` : politique online dans la boucle physique V11 et reporting par
  ledger corrigé ;
- `dp_gridcheck.py` : convergence de grille à l'état neuf ;
- `dp_pareto.py` : sensibilité au poids fiabilité `epsilon` ;
- `check_dp_v11.py` : préflight obligatoire et court.

Toutes les sorties nouvelles portent `v11_p2` et vont dans `DP/runs/`. Les
trajectoires pleine précision sont accompagnées d'un fichier
`*_ledgers.json`, afin que le total reste vérifiable par unité physique.

## Invariants d'attribution

- Exposant PEMWE fixé à `p=2`; le module refuse de démarrer si le défaut V11 a
  été modifié silencieusement.
- Références obligatoires : RB1 `(SoC_low, SoC_high)=(0.20, 0.40)` et RB2
  `(FC, ELY)=(0.574, 0.465)`, réoptimisées séparément sous V11-p=2.
- Coût réalisé : somme `degradation_ledger.total_eur`.
- Fiabilité : `Common.reliability_metrics.compute_reliability_metrics`.
- VoLL de reporting : 3 EUR/kWh. Dans `dp_pareto.py`, `epsilon` ne change que le
  poids de fiabilité pendant la résolution.
- Le front retient une seule méthode : `recompute='yearly'`,
  `aging_proj=True`, `rollout=True`, comme le front final de Pareto_V8.
  `recompute='never'` et le lookup restent des diagnostics et ne sont pas lancés
  dans le multi-epsilon canonique.

## Approximation interne

La grille d'état ne contient pas la stabilité PEMFC. Le backward suppose une
PEMFC stabilisée quand elle était déjà allumée et applique la transition V11
exacte lors d'un démarrage. Le rollout reçoit au contraire la stabilité
courante par `aging_context` et utilise la densité de courant précédente
réalisée. Le coût final n'est jamais celui du backward approché : il est mesuré
par la boucle et le ledger V11.

Les ancres PEMWE `j=1` et `j=2` sont ajoutées explicitement à chaque grille de
contrôle et recalculées avec `alpha_ely`.

## Vérifications locales du 2026-07-17

Commande :

```bash
conda run -n simu_env python check_dp_v11.py
```

Elle valide à la précision numérique : `p=2`, les coûts élémentaires PEMFC et
PEMWE contre `advance_*_density`, les ancres `j=1`/`j=2`, les paramètres RB1 et
RB2 et l'absence des anciennes constantes de coût dans les modules exécutables.

Deux smokes d'un an, grille `7x7`, trois niveaux FC, six niveaux ELY et une seule
itération de valeur ont également passé :

- chaîne comparative : `runs/dp_aging_v11_p2_1y_7x7.*` ;
- Pareto parallèle : `runs/dp_pareto_v11_p2_1y_7x7_rollout.*`.

Ces sorties démontrent seulement l'exécutabilité et la cohérence du ledger. Les
gains qu'elles affichent ne sont pas des résultats scientifiques : la grille,
l'horizon et la convergence de valeur sont volontairement insuffisants.

## Calculs mésocentre

Les étapes suivantes sont terminées :

- contrôle de grille : job 216232 ;
- comparaison 25 ans `51x51` : job 216233 ;
- front V2 sur 19 valeurs d'epsilon : job 216257.

Le coût unifié varie de moins de 2 % entre les grilles `31x31` et `71x71`, mais
la LPSP ne converge pas monotonement. Le point central V11 reproduit le chemin
V2 attendu et les grands écarts sont résolus ; les points du front proches du
coude seront relus avec cette sensibilité en tête.

Le front complet est dans :

- `runs/dp_pareto_v11_p2_25y_51x51_rollout.{txt,npz}` ;
- `runs/dp_pareto_v11_p2_25y_51x51_rollout_ledgers.json` ;
- `runs/dp_pareto_traj_v11_p2_25y_51x51_rollout.npz`.

Les 19 points sont non dominés. À VoLL=3, le minimum réalisé est observé pour
`epsilon=20` à 48,085 kEUR, mais toute la bande `epsilon=10--50` reste à moins
de 0,211 % de ce minimum. Cette bande est la conclusion défendable ; la
sensibilité de grille ne permet pas d'attribuer un optimum précis à 20.

Le `lol_tab` brut dépasse 1 de façon significative sur l'extrémité très peu
fiable (`epsilon <= 0,15`). Cette réserve porte sur l'interprétation fine de la
queue gauche. Dans la zone de décision `epsilon >= 1,5`, l'excès cumulé au-delà
du clipping reste inférieur à 0,310 kWh et 0,027 % de l'EENS.

Les figures canoniques sont générées par `plot_pareto.py`, notamment
`runs/pareto_lpsp_deg_v11_p2.{pdf,png}` et son zoom. La PD est maintenant close
comme référence offline ; la prochaine étape est le MPC online V11-p=2.
