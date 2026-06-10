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
| `sens_common.py` | Helpers partagés : import du code de base, calcul des métriques (LPSP, coût de dégradation — identiques à `batch_pareto`), durées de vie, exécution parallèle, ellipses de confiance. |
| `sens_soh_estimation.py` | **Étape 1** — robustesse de RB2(SoH) à l'erreur d'estimation du SoH (R2-6, R3-min2). |
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

## Lancer

```bash
cd Robustesse/Analyse_sensibilite
~/miniconda3/envs/simu_env/bin/python sens_soh_estimation.py
```

Budget calibré machine : ~50 simulations/batch. L'étape 1 = baseline (1) +
biais (9) + Monte Carlo σ∈{2,10 %}×N=20 (40) = **50 runs** (~15-20 min sur 7
cœurs). Augmenter `N_MC` pour lisser les nuages de la figure finale.
