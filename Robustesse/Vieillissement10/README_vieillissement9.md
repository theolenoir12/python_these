# Vieillissement9 — nouvelle fonction de coût PEMFC (réversible / irréversible)

Copie de `Vieillissement8` où **seule la fonction de coût PEMFC** a été réécrite.
Le reste (batterie, PEMWE, EMS rule-based, DP, plotting) est identique, à
l'exception des adaptations nécessaires à la nouvelle décomposition FC.

## Ce qui change par rapport à Pei et al. (2008)

L'ancien `get_cost_fc` classait chaque pas dans 4 régimes discrets (start-stop,
haute puissance > 80 % Pmax, idling < 1 % Pmax, transitoire) avec des
coefficients fixes en % de tension, **toute la dégradation étant permanente**.

Le nouveau modèle est **stateful** et structurellement **symétrique de celui de
l'électrolyseur** (`_ely_advance`). Par cellule, avec `f = |P_fc| / P_fc_max` :

```
dV_irr/dt = a(f)                 # permanent  : Ostwald / perte d'ECSA
dV_rev/dt = b(f) - k(f)·V_rev    # se construit en charge, RÉCUPÈRE au repos
V_ss   += s      à chaque démarrage OFF→ON            # permanent, sévère
V_idle += idle·dt  à très basse puissance (haut potentiel, dissolution Pt)
```

Conséquence clé : `SoH_fc` **peut remonter** quand la FC est à l'arrêt (la part
réversible se récupère), exactement comme le PEMWE. C'est le levier naturel d'un
micro-réseau (les périodes OFF de la FC sont abondantes).

## Calibration (sources)

- **McCay et al., J. Power Sources 665 (2026) 239011** — short-stack 10 cellules,
  profils maritimes (charge lente = cas le plus proche d'un micro-réseau
  stationnaire). Sépare réversible/irréversible à j = 0,5 A/cm² :
  réversible **22–52 µV/h** (dominant, récupérable), irréversible **1,2–4,8 µV/h**
  (permanent, ~20 % de perte d'ECSA cathode / 1500 h).
- **Colombo et al., J. Power Sources 553 (2023) 232246** — cellule segmentée,
  1000 h de cycle automobile réaliste. Table 4 : taux de perte **croissant avec
  le courant** (CCM B : 2,4 → 31,7 µV/h de j = 0,095 à 1,748 A/cm²). Donne la
  **forme** de la dépendance au courant (remplace les seuils 1 %/80 %).

Correspondance modèle ↔ mesures (par cellule) :

| f    | a_irr (µV/h) | b_rev (µV/h) | référence                         |
|------|-------------:|-------------:|-----------------------------------|
| 0,46 |         1,27 |         20,7 | McCay j≈0,5 : irr 1,2 / rev 22    |
| 1,0  |         6,0  |         45,0 | McCay dyn irr 4,8 ; rev max ~52   |

## Coefficients ajustables (`cost_fcn_total2.py`, dict `FC_REC`)

| clé      | défaut | signification                                             |
|----------|-------:|-----------------------------------------------------------|
| `a_irr`  |   6,0  | irréversible à f=1 ; loi `a(f)=a_irr·f²` (convexe)        |
| `b_rev`  |  45,0  | génération réversible à f=1 ; loi `b(f)=b_rev·f`          |
| `k_rest` |   2,0  | récupération à l'arrêt (τ ≈ 0,5 h)                        |
| `k_op`   |  0,002 | récupération en fonctionnement (→ plateau réversible ~1,2 %) |
| `s`      |  20,0  | µV/cycle de démarrage (start-stop, ~Fletcher 24 µV)      |
| `idle`   |   3,0  | µV/h de maintien à très basse puissance (haut potentiel) |

Seuils : `FC_F_OFF = 0,01` (en dessous → récupération), `FC_F_IDLE = 0,05`
(en dessous → pénalité idle). `UV_TO_PCT_FC` est calculé sur la tension de
cellule BoL au courant nominal (~0,86 V), robuste aux paramètres.

## Hypothèses de modélisation à connaître (pour la soutenance)

1. **Récupération activée à l'arrêt (f≈0)**. Physiquement, McCay/Colombo
   récupèrent le réversible par excursion à bas potentiel (CV, court-circuit
   résistif). On assimile ici l'arrêt FC à une séquence d'arrêt avec purge /
   étape réductrice (standard dans un système bien géré). La récupération par
   excursion à **fort courant** (mécanisme McCay « constant > dynamique ») n'est
   **pas** modélisée — activable en rendant `k(f)` croissant à haut f.
2. **Plateau réversible** ~1,2 % de tension (= `b/k_op`) en fonctionnement
   continu prolongé, cohérent avec Colombo (« few to 20 mV »). Mettre `k_op=0`
   pour une croissance linéaire non bornée.
3. **Rodage (break-in)** : le pic d'irréversible des ~250 premières heures
   (Ostwald) n'est pas modélisé ; on utilise les taux en régime établi.
4. Écart d'échelle McCay vs Colombo (McCay ~2× Colombo CCM B) : magnitude calée
   sur **McCay** (cas d'usage le plus proche), forme sur **Colombo**.

## Portée / limites

- Vaut pour les stratégies **rule-based** (via `get_cost_total` et
  `init_and_run_loop`) et le **MILP** (`milp_weekly.py`, clés mises à jour).
- Le **DP** (`DP/`, `DP2/`) réimplémente encore le modèle de Pei *inline*
  (vectorisé) et importe les constantes legacy `FC_FHIGH/FC_FLOW/FC_ALPHA_*`
  (conservées). Migrer le DP vers le nouveau modèle nécessiterait de réécrire
  `dp_core.py` / `dp_aging.py` — **hors périmètre de cette version**.

## Décomposition `deg_fc`

Anciennes clés `['start-stop','idling','transient','high']` →
nouvelles `['start-stop','idling','reversible','irreversible']` (+ `total`).
`main_plot.py` et `milp_weekly.py` ont été adaptés en conséquence.

## Statut de la comptabilité des remplacements

Depuis l'audit du 2026-07-11, la boucle rule-based corrigée ne rejoue plus sur
l'unité neuve le pas qui vient de déclencher son remplacement. Elle expose un
ledger par composant : coût des unités retirées + coût de l'unité courante,
avec intervalles demi-ouverts disjoints. Les métriques rule-based et les sweeps
V9_4 lisent ce ledger ; le recalcul sur une trace complète n'est plus un oracle
valide après des resets.

Le calcul de LOL a également été corrigé : après une panne FC/ELY, la puissance
du composant hors service est retirée du bilan avant l'écrêtage batterie. Aux
bornes du réservoir H2, l'inversion rendement–puissance est maintenant résolue
avec la même expression que la transition directe ; l'ancienne interpolation
entre nœuds pouvait consommer quelques Wh de trop et rendre l'état suivant
infaisable vers 12,6 ans.

Les sorties P1/P3/P4 de juillet antérieures aux jobs 215088-215091 sont des
archives legacy. Le rerun corrigé a maintenant passé le protocole d'acceptation
et est audité dans `AUDIT_RERUN_CORRIGE_2026-07-11.txt`. P4 corrige en plus son
bruit de prévision : une graine définit désormais une trajectoire AR(1) horaire
persistante, commune à tous les horizons N. Le protocole d'acceptation était :

1. `run_meso_invariance.slurm` sur 25 ans : ledger = somme des segments,
   absence de rejeu, gels correctifs et boucle instantanée = boucle de base ;
2. P1/P3/P4 dans `runs/<id>_<empreinte>/`, avec cache brut pleine précision ;
3. statistiques appariées VoLL=1/3/10, puis promotion explicite.

Ces trois étapes passent. Le CSV Sidelec identifié par la provenance est
exactement reproductible depuis la copie locale datée en supprimant la colonne
date ; aucune récupération du mésocentre n'est nécessaire.

Le mode `replacement_accounting="legacy_overlap"` est conservé uniquement pour
diagnostiquer les anciennes sorties ; il ne doit pas produire un nouveau
résultat scientifique.
