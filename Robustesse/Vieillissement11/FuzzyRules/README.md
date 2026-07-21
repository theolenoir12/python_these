# FuzzyRules V11-p=2

Ce dossier développe les EMS floues et les règles apprises sans modifier les
dossiers PD ou MPC. Le protocole scientifique est décrit dans
`PLAN_FUZZY_RULE_LEARNING_V11_P2.md`.

## Baseline disponible

`flc-mamdani-expert-v11-p2-i0-v1-2026-07-21` est une FLC experte sans SoH et
sans prévision. Elle comprend deux systèmes Mamdani de 27 règles :

- branche déficit : commande PEMFC ;
- branche surplus : commande PEMWE ;
- batterie : fermeture du bilan de puissance.

Les trois entrées normalisées sont la puissance nette courante, le SoC et le
remplissage H2. Le moteur utilise `min` pour ET et l'implication, `max` pour
l'agrégation et un centre de gravité discret pour la défuzzification. Les
tables de règles et les appartenances sont figées dans `flc_policy_v11.py` ;
les plafonds, échelles de puissance et la zone morte ont ensuite été réglés.

Les SoH et puissances maximales vieillies ne sont pas des entrées de décision
de cette variante I0. Ils restent appliqués par `Common.get_lol` comme bornes
physiques communes à toutes les stratégies.

## Vérification locale

Depuis `Vieillissement11/`, dans l'environnement scientifique contenant NumPy,
SciPy et SymPy :

```bash
python -m unittest discover -s FuzzyRules/tests -v
python -m FuzzyRules.analyze_flc_surfaces
python -m FuzzyRules.run_smoke_flc_v11 --days 7
python -m FuzzyRules.audit_smoke_flc_v11 FuzzyRules/runs/<run_id>
```

Le smoke compare la FLC aux références attribuables RB1 `(0,20 ; 0,40)` et
RB2 `(0,574 ; 0,465)`. Ses sorties pleine précision sont empreintées dans
`FuzzyRules/runs/`. Un smoke court vérifie l'intégration ; il ne permet aucune
conclusion de performance ni de Pareto.

Le screening central d'un an de la v1 est consigné dans
`BASELINE_FLC_V1_AUDIT_2026-07-21.md`. Il passe les invariants mécaniques mais
est dominé par RB1 et RB2 ; il sert donc de diagnostic avant un réglage à
budget et partitions temporelles préannoncés, et non de conclusion sur la
famille FLC.

## FLC I0 réglée

Le protocole et les résultats du réglage sont respectivement :

- `TUNING_PROTOCOL_FLC_I0_V11_P2_2026-07-21.md` ;
- `TUNING_FLC_I0_RESULTS_2026-07-21.md`.

Le parent promu `flc_8126e6f729c6` est construit par
`make_tuned_expert_flc_policy_v11()`. Sur le profil canonique 25 ans, il domine
RB2 sur LPSP et dégradation et forme un compromis non dominé avec RB1. Son
léger avantage de J3 face à RB1 reste inférieur au seuil matériel de 1 %.

## Extension SoH attribuable

La couche hiérarchique FLC-IS est implémentée dans `flc_soh_policy_v11.py`.
Son protocole et ses résultats sont :

- `TUNING_PROTOCOL_FLC_IS_SOH_V11_P2_2026-07-21.md` ;
- `TUNING_FLC_IS_SOH_RESULTS_2026-07-21.md`.

Le test nul reproduit le parent bit-à-bit sur 25 ans. Les trois variantes SoH
actives promues depuis le screening cinq ans sont toutes dominées par I0 à
25 ans ; aucune n'est retenue. Le cache canonique est
`runs/promoted_flc_is_soh_25y_34bdb5fbe2af/`, avec audit `PASS`.

La prochaine couche IF utilisera une énergie nette prévue agrégée à H18, puis
la combinaison ISF sera jugée par ablation par rapport à IF.
