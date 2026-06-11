# Analyse de sensibilité des EMS

Dossier dédié aux **analyses de sensibilité** demandées par les reviewers (APEN),
en réponse aux remarques sur le caractère déterministe des résultats et les
hypothèses des modèles de dégradation.

> ⚠️ Ces scripts **n'écrivent jamais** dans `Vieillissement8`. Ils importent le
> code de base (boucle de simulation, modèles de coût, stratégie RB2(SoH)) **en
> lecture seule** via un chemin absolu (voir `sens_common.py`). Les paramètres de
> base ne sont pas modifiés.

## Organisation

| Fichier | Rôle |
|---|---|
| `sens_common.py` | Helpers partagés : import (portable Win/Linux) du code de base, **chargeur dynamique de stratégie** (`load_strategy`), métriques (`metrics`, `metrics_components`), durées de vie, exécution parallèle, ellipses de confiance. |
| `sens_soh_estimation.py` | **Étape 1** — robustesse de RB2(SoH) à l'erreur d'estimation du SoH (R2-6, R3-min2). |
| `sens_eol.py` | **Étape 2** — sensibilité aux seuils de fin de vie (EoL), **toutes stratégies** (R3-major3-iii, R4-9/R4-10, R1-6). |
| `sens_cweights.py` | **Étape 3** — sensibilité aux poids de coût (C-weights), **toutes stratégies** (R3-major3, R1-6). |
| `sens_hthresholds.py` | **Étape 4** — sensibilité aux seuils de dégradation H2 (PEMFC/PEMWE), **toutes stratégies** (R3-major3-i). |
| `results/` | Sorties (figures PDF + résumés `.txt`). |

## Étape 1 — Erreur d'estimation du SoH

L'EMS calcule ses setpoints à partir du SoH **estimé**. On remplace le SoH exact
par `SoH_est = clip(SoH_vrai · (1 + e))`, rafraîchi chaque semaine (cohérent avec
l'hypothèse « SoH estimé au moins 1×/semaine » de l'article). Le **vrai** SoH se
dégrade normalement ; LPSP et coût sont mesurés sur les vraies trajectoires.

- **Régime 1 — biais systématique** : `e` constant de −10 % à +10 %.
- **Régime 2 — bruit gaussien** : `e ~ N(0, σ)`, σ ∈ {2, 5, 10 %}, Monte Carlo.

Sorties :
- `results/sens_soh_bias.pdf` — LPSP & coût en fonction du biais (double axe).
- `results/sens_soh_pareto.pdf` — plan de Pareto : locus du biais + **intervalles
  de confiance** (ellipses 1σ/2σ) des nuages Monte Carlo autour du baseline.
- `results/sens_soh.txt` — chiffres (moyennes, écarts-types, min/max).

## Étape 2 — Seuils de fin de vie (EoL)

Contrairement à l'estimation du SoH (propre à RB2(SoH)), le **seuil EoL est un
paramètre global** : il agit sur **toutes** les stratégies via (i) le
déclenchement des remplacements dans la boucle, (ii) la normalisation du coût
(`coût = indicateur / (1−SoH_EoL) · coût_remplacement`) et (iii) la conversion
indicateur→SoH + les bornes `alpha`.

On reproduit donc le **front de Pareto complet** (les 10 EMS de `batch_pareto`)
mais où **chaque point porte sa propre ellipse de confiance**, obtenue en
échantillonnant conjointement les seuils EoL des 3 composants (Monte-Carlo,
mêmes triplets pour toutes les stratégies → comparaison non polluée par le bruit
MC). C'est la « bande d'incertitude par stratégie » demandée par R3.

- **Plages MC** : batterie `[0.60, 0.80]` (0.60 = 40 % de perte = valeur article ;
  0.80 = 20 %, borne haute suggérée par R3) ; FC/ELY `[0.90, 0.96]`.
- **Limite FC/ELY** : les bornes `alpha` (brentq dans
  `Vieillissement8/Common/main_init_and_loop.py`, l. ~78-84) sont calées juste
  sous une singularité du modèle de tension et calibrées jusqu'à ~10-20 % de
  perte. **On ne descend pas FC/ELY sous 0.90** sans recalibrer (sinon
  extrapolation). Pour élargir : modifier les brackets brentq **puis**
  `MC_RANGES` dans `sens_eol.py`. Tout échantillon infaisable est abandonné
  proprement (`try/except`).

Sorties :
- `results/sens_eol_pareto.pdf` — front des 10 EMS, point = seuils nominaux,
  ellipses 1σ/2σ = incertitude EoL autour de chaque point.
- `results/sens_eol_oat.pdf` — figure d'appui : effet OAT d'un seuil à la fois
  (coût & durée de vie) sur la stratégie de référence (RB2(SoH)).
- `results/sens_eol.txt` — chiffres (nominal + moyennes/écarts-types MC, OAT).

## Étape 3 — Poids de coût (C-weights)

L'axe Y (coût de dégradation) pondère les trois dégradations par les **coûts de
remplacement** des composants (dérivés des CAPEX : BAT 150 €/kWh, FC stack
750 €/kW, ELY stack 563 €/kW). On fait varier ces poids (±30 % par défaut,
centré) pour mesurer la robustesse du front.

**Propriété clé** : la trajectoire SoH, les remplacements et la **LPSP sont
invariants** aux poids (le facteur `['cost']` se simplifie dans le calcul du
SoH), et le coût total est **linéaire** en chaque poids. Donc :
- **1 seule simulation par stratégie** (10 simus) ; le Monte-Carlo est du
  **post-traitement analytique** (`N_MC=5000`, instantané) ;
- la bande par point est **purement verticale** (barres d'erreur, pas
  d'ellipses) — *résultat en soi* : le classement LPSP est insensible aux
  hypothèses de coût, seule l'amplitude de l'axe dégradation se dilate.

Sorties :
- `results/sens_cweights_pareto.pdf` — front des 10 EMS, barres verticales
  (IC 95 %) = incertitude due aux poids de coût.
- `results/sens_cweights_breakdown.pdf` — composition du coût (bat/FC/ELY) par
  stratégie : montre quel composant domine la dégradation de chaque EMS.
- `results/sens_cweights.txt` — chiffres (nominal, composantes, IC).

## Étape 4 — Seuils de dégradation H2 (PEMFC / PEMWE)

Réponse à R3-major3-i (« linear superposition of four mechanisms with seemingly
arbitrary thresholds (1 %, 80 %, 60 %) »). On balaie les **4 seuils de régime** :

| Seuil | Nominal | Rôle |
|---|---|---|
| `FC_FHIGH` | 0.80 | PEMFC, seuil haute puissance (« 80 % ») |
| `FC_FLOW`  | 0.01 | PEMFC, seuil idling (« 1 % ») |
| `ELY_F30`  | 0.30 | PEMWE, début de dégradation (30 % Pmax) |
| `ELY_F60`  | 0.60 | PEMWE, saturation au *rated* (« 60 % ») |

**Refactor** : ces 4 seuils sont désormais des **constantes de module** de
`cost_fcn_total2.py` (les FC ont été promus depuis des littéraux ; valeurs par
défaut **inchangées**, comportement nominal identique — vérifié). La boucle ET le
calcul de coût appellent les mêmes fonctions qui relisent ces constantes ; on les
**mute donc dans chaque worker** (override cohérent, sans dupliquer le modèle).

Contrairement aux C-weights, ces seuils pilotent dégradation → SoH →
**remplacements → LPSP** : il faut **re-simuler**, et la bande par point est **2D
(ellipses)** comme l'EoL.

Sorties :
- `results/sens_hthresholds_pareto.pdf` — front des 10 EMS + ellipses 1σ/2σ.
- `results/sens_hthresholds_oat.pdf` — OAT 1×4 (coût & durée de vie vs chaque
  seuil) sur la stratégie de référence.
- `results/sens_hthresholds.txt` — chiffres (nominal + MC + OAT).

## Lancer

```bash
cd Robustesse/Analyse_sensibilite
~/miniconda3/envs/simu_env/bin/python sens_soh_estimation.py   # étape 1
~/miniconda3/envs/simu_env/bin/python sens_eol.py              # étape 2
~/miniconda3/envs/simu_env/bin/python sens_cweights.py         # étape 3
~/miniconda3/envs/simu_env/bin/python sens_hthresholds.py      # étape 4
```

> Sous Windows, remplacer par le Python anaconda local
> (`C:\Users\tlenoi01\AppData\Local\anaconda3\python.exe`).

`N_WORKERS` est **auto-détecté** (`os.cpu_count()−1`) : le code s'adapte seul à
une machine plus grosse.

- **Étape 1** : baseline (1) + biais (9) + Monte Carlo σ∈{2,10 %}×N=20 (40) =
  **50 runs** (~15-20 min sur 7 cœurs).
- **Étape 2** : 10 nominaux + 10×`N_MC` + OAT. Avec `N_MC=15` → **~170 runs**
  (~50-70 min sur 7 cœurs). Monter `N_MC` pour lisser les ellipses (sur grosse
  machine, rien d'autre à changer).
- **Étape 3** : **10 runs seulement** (~3-5 min) — le MC sur les poids est
  analytique. Ajuster `DELTA` (par composant) pour la largeur des bandes.
- **Étape 4** : 10 nominaux + 10×`N_MC` + OAT (re-simulation requise). Avec
  `N_MC=15` → **~176 runs** (~50-70 min sur 7 cœurs). Ajuster `MC_RANGES` /
  `N_MC`. Sécurité : tout tirage gardant `f30 < f60` (toujours vrai ici).
