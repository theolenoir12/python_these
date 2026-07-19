# Vieillissement10

Version expérimentale créée le 13 juillet 2026 pour tester des modèles de vieillissement PEMFC/PEMWE pilotés par la densité de courant et pour produire des métriques reproductibles de première vie.

## Origine et préservation de Vieillissement9_4

Le dossier est une copie du dernier état de travail de `Vieillissement9_4`, incluant ses évolutions récentes sur l'intégration de connaissances dans les règles EMS (SoH, prévision et expérimentations associées). Il ne s'agit pas d'une copie du seul commit `c8ad6185ea0c407b338b1404a3230fe0b52e86f1` : les modifications locales présentes le 13 juillet 2026 ont été conservées.

Les deux fichiers alors en conflit Git dans `Vieillissement9_4` n'ont volontairement pas été copiés :

- `bench_dwell_ely.py`
- `run_meso_dwell.slurm`

Aucun fichier de `Vieillissement9_4` n'a été modifié ou résolu pendant la création de cette version.

## Changements scientifiques

### Référentiel électrochimique commun

`Common/electrochemistry.py` centralise désormais :

- les courbes de polarisation PEMFC et PEMWE ;
- les courants maximaux et les puissances maximales en fonction de `alpha` ;
- l'inversion puissance vers courant ;
- le calcul de la densité de courant `j`.

Les boucles rule-based n'utilisent plus `P/Pmax` comme substitut de `j` pour calculer les vitesses de vieillissement.

Une table d'interpolation accélère l'inversion. La comparaison à la résolution exacte donne une erreur maximale de l'ordre de `5.2e-6 A/cm²` pour la PEMFC et `6.0e-5 A/cm²` pour le PEMWE.

### PEMFC

Le modèle réversible/irréversible est exprimé en fonction de `j`. La référence actuelle à `0.5 A/cm²` est :

- irréversible : `1.2 µV/h` ;
- réversible : `22 µV/h`.

Les démarrages, l'idling et la récupération restent traités séparément.

### PEMWE

Le modèle de récupération inspiré de Rakousky est exprimé directement en fonction de `j` :

- pas de génération `a/b` sous `1 A/cm²` ;
- rampe entre `1` et `2 A/cm²` ;
- calibration complète de Rakousky à `2 A/cm²` ;
- saturation au-dessus de `2 A/cm²`.

Le paramètre à tuner se trouve dans `Common/cost_fcn_total2.py` :

```python
ELY_REC["scale"] = 1.0
```

`1.0` conserve la calibration Rakousky. Une première estimation purement linéaire pour rapprocher la première vie RB2 de 40 000 h ON serait `0.89`, mais cette valeur n'est pas fixée : elle doit être validée par balayage car récupération, démarrages et point de fonctionnement rendent la relation non linéaire.

Le passage µV vers pourcentage utilise la tension nominale du modèle électrochimique, environ `2.0509 V`, au lieu de la constante historique `1.5 V`.

## Métriques de première vie

`Common/lifetime_metrics.py` calcule pour la PEMFC et le PEMWE :

- temps calendaire et années à 8 760 h ;
- temps ON ;
- EFPH ;
- énergie ;
- nombre de démarrages et durée moyenne d'une séquence ON ;
- fraction de charge moyenne ;
- densité de courant moyenne et percentile 95 ;
- heures par plage de densité de courant ;
- contributions à la dégradation en fin de première vie ;
- taux équivalents en µV/h ON.

Conventions :

```text
ON   : |P| >= 0.0005 * Pmax(alpha)
EFPH : somme(|P| / Pmax(alpha) * dt_h)
vie  : de l'unité initiale au premier reset du SoH
```

Les données sont accessibles dans :

```python
data["first_life_metrics"]
```

`Common/main_plot.py` exporte également `first_life_metrics.txt` dans le dossier de figures de la stratégie exécutée.

## Premier test RB2, horizon 25 ans

Avec `ELY_REC["scale"] = 1.0` :

| Composant | Calendrier | Temps ON | EFPH | Démarrages | j moyen ON |
|---|---:|---:|---:|---:|---:|
| PEMFC | 8.095 ans | 38 916 h | 19 641 h | 3 133 | 0.352 A/cm² |
| PEMWE | 14.143 ans | 35 518 h | 9 309 h | 5 236 | 1.089 A/cm² |

Le détail scientifique et les sources se trouvent dans `RESUME_PREMIERE_APPROCHE_VIEILLISSEMENT10.txt`.

## Exécution

Depuis le dossier `Robustesse/Vieillissement10` :

```powershell
C:\Users\tlenoi01\AppData\Local\anaconda3\python.exe .\RB2\main.py
```

Test court sans génération de figures :

```powershell
C:\Users\tlenoi01\AppData\Local\anaconda3\python.exe -c "import sys; sys.path.insert(0,'.'); from Common.main_init_and_loop import init_and_run_loop; from RB2.get_optimal_action_RB import get_optimal_action_RB; d=init_and_run_loop(get_optimal_action_RB,n_years=1); print(d['first_life_metrics'])"
```

## Périmètre validé et limites

Validé pour cette première étape :

- analyse syntaxique des neuf fichiers modifiés ;
- imports du cœur électrochimique, des coûts, des métriques, des deux boucles rule-based, de la physique et des graphiques ;
- cohérence puissance vers courant vers puissance ;
- comparaison interpolation/résolution exacte ;
- simulation RB2 sur un an ;
- simulation RB2 sur 25 ans et détection des premières vies ;
- cohérence du ledger de remplacements.

Non certifié dans Vieillissement10 au moment de cette étape historique :

- les anciens `DP/` et `DP2/` réimplémentaient encore le vieillissement historique en fraction de puissance ;
- le calcul de `Pmax` du chemin MILP est unifié, mais sa dégradation n'est pas entièrement migrée vers `j` ;
- `Common/milp_weekly.py` contient un import historique de `j_0` absent au niveau module de `Init_EMR_MG_v16_python.py`, défaut également présent dans la base V9_4.

Cette limite ne décrit plus la PD active de Vieillissement11 : la méthode V2 de
`Pareto_V8` a été portée dans l'unique dossier `Vieillissement11/DP/`, avec les
coûts et le ledger V11. Le doublon `Vieillissement11/DP2/` a été supprimé.

## Fichiers principaux ajoutés ou modifiés

Ajoutés :

- `Common/electrochemistry.py`
- `Common/lifetime_metrics.py`
- `RESUME_PREMIERE_APPROCHE_VIEILLISSEMENT10.txt`
- `README_vieillissement10.md`

Adaptés :

- `Common/cost_fcn_total2.py`
- `Common/main_init_and_loop.py`
- `Common/main_init_and_loop_maintenance.py`
- `Common/physics.py`
- `Common/main_plot.py`
- `Common/milp_weekly.py`
- `Common/milp_weekly (1).py`



## Recalage RB2 et vieillissement PEMWE (13 juillet 2026)

Cette mise a jour remplace l'ancienne regularisation par fond constant ainsi
que la RB2 avec plafonds de secours.

La RB2 active est strictement une regle a deux consignes fixes :
0.59 Pmax pour la PEMFC et 0.49 Pmax pour le PEMWE. Elle n'utilise ni SoH,
ni reserve SoC, ni plafond conditionnel.

Le vieillissement PEMWE separe une loi irreversible longue duree, ancree sur
la cible DOE de 4.8 microV/h a 2 A/cm2, de la contribution reversible issue
des essais courts de Rakousky. Au-dessus de 2 A/cm2, un terme de stress
quadratique explicite (high_current_accel = 100) evite de plafonner
artificiellement le vieillissement des strategies a fort courant. Ce
coefficient definit un scenario de sensibilite et ne doit pas etre presente
comme une constante universelle.

Resultat RB2 25 ans :

- EENS : 4 058.49 kWh ;
- LPSP sur charge totale : 0.7750 % ;
- LPSP historique sur charge residuelle : 1.4846 % ;
- degradation : 63.375 kEUR ;
- cout unifie avec VoLL = 3 EUR/kWh : 75.550 kEUR.

RB1 atteint 0.898 % de LPSP charge, 68.340 kEUR de degradation et
82.449 kEUR de cout unifie. RB2 redevient donc la meilleure strategie de base,
avec a la fois moins d'energie non fournie et moins de cout unifie que RB1.

Les definitions, limites, sources et commandes reproductibles sont detaillees
dans RB2/README_RB2_V10.md. Le classement complet est produit par
rank_base_strategies.py.

## Augmentations RB2 (15 juillet 2026)

Les quatre variantes `RB2(SoH)`, `RB2(RUL)`, `RB2(Pred)` et
`RB2(SoH+RUL+Pred)` utilisent maintenant le dispatch commun
`Common/rb2_policy.py`. Elles restent strictement pilotees par deux setpoints H2
dynamiques, sans plafond de strategie.

Le balayage 25 ans (`optimize_rb2_augmentations.py`) conclut a :

- SoH : la loi `SoH^gamma` a ete remplacee pour les nouveaux calculs par une
  loi fondee sur l'usure normalisee entre SoH=1 et SoH_EoL. Le cout unifie
  minimal reste RB2, mais un front LPSP/degradation apparait. Sous
  `LPSP <= 1.10 %`, le point retenu (`strength_fc=strength_ely=0.25`, formes
  lineaires) donne `LPSP=1.09760 %`, `degradation=62.20838 kEUR` et
  `cout unifie=79.45155 kEUR`, contre respectivement `0.77502 %`,
  `63.37461 kEUR` et `75.55008 kEUR` pour RB2 ;
- RUL : cas nul optimal, donc strategie identique a RB2 (`75.5501 kEUR`) ;
- Pred : `H=24 h`, cible SoC `0.99`, bande `1.5 sigma`, aucun maintien,
  moyenne `75.3940 kEUR` sur trois graines ;
- cumul : SoH + Pred ci-dessus et RUL nul, moyenne `75.2836 kEUR`, LPSP
  charge totale `0.7560 %`.

Les definitions, commandes et resultats complets sont dans
`AUGMENTATION_RB2_V10.txt` et `Optimization_results/`.

Le diagnostic detaille et le front sont produits par
`analyze_rb2soh_tradeoff.py`. La baisse des setpoints reduit bien la degradation
H2 (jusqu'a -7.07 % sur le cout total de degradation dans la grille), mais elle
augmente aussi les heures de fonctionnement H2, laisse presque intact le cout
des demarrages-arrets et reporte une partie de l'effort sur la batterie. Avec
VoLL=3 EUR/kWh, l'EENS supplementaire rend ainsi les points SoH plus couteux au
sens du critere unifie.

Le bruit utilise encore les statistiques du backtest historique a 18 h alors
que l'horizon optimal est 24 h. Un backtest 24 h et un Monte-Carlo plus large
sont donc requis avant validation scientifique definitive de Pred et du cumul.
