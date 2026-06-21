# Étude de robustesse des EMS sous défaillance (composants H₂)

Remplace les anciens scripts `Robustesse/Défaillances.py` et
`Robustesse/Défaillances_50%.py` (qui n'étaient que des nuages de points avec
des LPSP codées en dur). Ce dossier régénère ces résultats proprement, de bout
en bout, par Monte-Carlo.

## Idée

On considère qu'une **défaillance d'un composant hydrogène** (PEMFC ou PEMWE,
beaucoup plus probable que la batterie vu leurs MTBF plus faibles) peut survenir
en cours d'exploitation. On veut savoir **comment chaque stratégie de gestion
d'énergie (EMS) encaisse la panne**, et quelle stratégie est la plus robuste.

## Méthodologie

1. **Régime permanent.** On simule **une seule fois** la stratégie de référence
   **RB2** sur 2 ans (`YEARS_BASELINE`). On en retient, à chaque heure, l'état
   complet : `SoC, E_h2, SoH_bat/fc/ely, alpha_fc/ely`. Le **1ᵉʳ mois**
   (`SETTLE_HOURS ≈ 730 h`) sert à établir le régime permanent et est exclu des
   instants de panne.

2. **Défaillance.** Une panne survient à un instant `t0` tiré aléatoirement dans
   les **mois 2 → 24** et dure **1 semaine** (`WEEK_HOURS = 168 h`) avant
   réparation. Deux sévérités : `total` (0 %) ou `50` (moitié de puissance).
   Quatre scénarios : **PEMFC totale, PEMFC 50 %, PEMWE totale, PEMWE 50 %**.

3. **Branche + dispatcher conscient de la panne.** À `t0` on **repart du snapshot
   RB2** et on simule la fenêtre de panne, SoH/alpha gelés. La **panne est
   connue de l'EMS** (détectée) : à chaque pas, la stratégie candidate calcule
   son intention normale, puis on **plafonne le composant défaillant à sa
   capacité disponible (0 % ou 50 %) et on reroute le manque vers la batterie** ;
   le référé `Common.get_lol` (code de base, réutilisé tel quel) tranche
   (écrêtage SoC + réservoir H₂). Par construction, **plus de capacité ⇒ moins de
   stress batterie**, donc **LPSP(50 %) ≤ LPSP(total)** : la monotonie physique
   est garantie. Les stratégies candidates sont exactement celles du nuage de
   Pareto (`Pareto_2d_25y.py`) : `0-100, 25-75, 50-50, 75-25, 100-0, RB2, RB1,
   SoC1, SoC06`.

4. **Métrique.** **LPSP** sur la fenêtre d'évaluation `EVAL_HOURS` (défaut = la
   semaine de panne), calculée comme `sens_common.metrics`. On calcule aussi,
   **pour chaque tirage, la LPSP de la même fenêtre SANS panne** (contrefactuel,
   même stratégie) → le **surcoût de robustesse** = LPSP(panne) − LPSP(normale)
   isole l'effet propre de la panne et répond à « comment une stratégie pourrait
   être meilleure que la marche normale » (impossible à fenêtre/méthode égales).

5. **Monte-Carlo & statistiques.** `N_DRAWS = 200` instants tirés uniformément,
   **identiques pour toutes les stratégies/scénarios** (comparaison *appariée*).
   Distribution de la LPSP, meilleure stratégie par scénario (LPSP moyenne
   minimale), et fréquence « parmi les meilleures » par tirage (ex æquo partagés).

> On **ne** considère **pas** RB2(vieillissement) — RB2(SoH)/RB2(RUL) — pour le
> moment, ni `LP_filter` (stateful). C'est volontaire et facile à rajouter.

## Fichiers

| Fichier | Rôle |
|---|---|
| `robustesse_common.py` | Harness : import **lecture seule** de `Vieillissement8`, baseline RB2, simulateur de semaine de panne, Monte-Carlo. |
| `run_robustesse.py`    | Pilote : baseline → tirages → simulations → `results/robustesse_results.npz` + résumé texte. |
| `plot_robustesse.py`   | Figures : boxplots, heatmap LPSP moyenne, CDF. |
| `results/`             | Sorties (cache baseline `.npz`, résultats, figures PDF). |

## Lancer

```bash
# environnement conda : simu_env
python run_robustesse.py    # ~quelques minutes (parallélisé)
python plot_robustesse.py
```

Paramètres en tête de `run_robustesse.py` : `N_DRAWS`, `SEED`, `STRATEGIES`,
`SCENARIOS`. Le cache baseline (`results/baseline_rb2_2y.npz`) n'est recalculé
que s'il est absent.

## Choix de modélisation à connaître

- **Panne détectée (total ET 50 %).** L'EMS connaît la panne et reroute vers la
  batterie via le dispatcher conscient (cf. point 3). Total et 50 % passent par
  le **même** mécanisme (dérating + reroutage), ce qui garantit
  `LPSP(50 %) ≤ LPSP(total)` et un surcoût de robustesse `≥ 0` (à fenêtre/méthode
  égales).

- **⚠️ La métrique 1-semaine sous-estime les pannes ELY.** Quand l'électrolyseur
  tombe, le surplus PV non électrolysé **recharge la batterie**, ce qui *réduit*
  la LPSP des déficits courts de la semaine ; le **vrai coût** (réservoir H₂ vidé
  → famine de la FC) tombe **après** la semaine de panne. Sur 1 semaine, le
  surcoût ELY peut donc être quasi nul voire légèrement négatif. Allonger
  `EVAL_HOURS` à ~3–4 semaines (la panne reste 1 semaine, on mesure la **reprise**
  post-réparation) rend tous les surcoûts positifs et physiques. Réglable en tête
  de `robustesse_common.py` (`EVAL_HOURS = k * WEEK_HOURS`).
