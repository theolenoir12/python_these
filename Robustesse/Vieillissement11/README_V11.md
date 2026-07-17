# Vieillissement11 — modèle de coût auditable et RB2 informée par le SoH

## Statut

V11 est le dossier expérimental construit à partir de V10. Le manuscrit n'a pas
été consulté ni modifié. La synthèse faisant foi est
`RESULTATS_CENTRAUX_V11.md`.

Correction importante : l'ancienne `RB2(Aging)` utilisait comme parent la RB2
historique `(0,59 ; 0,49)`, et non la RB2 dite « retunée » `(0,57 ; 0,45)`.
Les nouveaux essais `RB2(SoH)` utilisent toujours comme parent la meilleure
RB2 statique du même modèle.

## Résultat central à 25 ans

Le critère est `J = coût de dégradation + 3 EUR/kWh * EENS`.

| Modèle PEMWE | Meilleure RB1 | Meilleure RB2 | Classement |
|---|---:|---:|---|
| Quadratique par défaut, `p=2` | 73 560,88 € | 75 721,37 € | RB1 gagne de 2,85 % |
| Overload cubique, `p=3` | 76 197,57 € | 74 351,70 € | RB2 gagne de 2,42 % |

Dans le scénario cubique, la meilleure `RB2(SoH)` vaut 74 247,11 €, soit un
gain de 104,59 € ou 0,1407 % sur son parent. Ce gain reste autour de 0,14 % à
20 et 30 ans et autour de 0,12 % lorsque les coûts non identifiés de démarrage
et de quasi-idle sont annulés. Le SoH n'apporte donc pas les quelques pourcents
recherchés.

## Interprétation scientifique

Le modèle distingue :

1. le dommage permanent, seul monétisé et utilisé pour le remplacement ;
2. le conditionnement fini ;
3. la perte réversible, utilisée pour la performance operando mais non
   capitalisée.

Le noyau cubique conserve exactement les points 0, 1 et 2 A/cm² de Rakousky et
l'ancre DOE de 4,8 mV/kh à 2 A/cm². Il extrapole toutefois plus fortement le
dommage au-delà de 2 A/cm², zone non mesurée dans l'article. Il reste donc une
sensibilité explicite et n'est pas devenu la valeur par défaut.

Les détails bibliographiques et les limites d'identification sont dans
`LITERATURE_COST_MODEL_V11.md`.

## Attribution du gain SoH

`Common/rb2_soh_policy_v11.py` appelle exactement le même
`dispatch_rb2_setpoints` que RB2. Le SoH ne modifie que les deux consignes.
Avec les six coefficients SoH nuls, les actions et les métriques reproduisent
exactement le parent.

La meilleure loi trouvée dans le scénario cubique conserve la consigne ELY à
0,465 et fait passer progressivement la consigne FC de 0,574 à 0,5615 près de
l'EOL, avec une forme quartique. Elle réduit surtout l'EENS et ne change pas le
nombre de remplacements.

## Points d'entrée canoniques

- `Common/degradation_v11.py` : paramètres et noyaux littérature/DOE ;
- `Common/main_init_and_loop.py` : accumulation physique et économique ;
- `Common/cost_fcn_total2.py` : replay public avec le même noyau ;
- `Common/rb2_policy.py` : dispatch RB2 commun ;
- `Common/rb2_soh_policy_v11.py` : consignes dépendantes du SoH ;
- `Common/rb1_policy_v11.py` : RB1 locale, sans import de V8 ;
- `run_v11_candidates.py` : simulations JSON autonomes ;
- `summarize_v11_results.py` : déduplication, classement et crédit terminal ;
- `study/FINAL_RESULTS_V11.csv` : principaux résultats numériques.

Les scripts historiques copiés avec V10 restent présents pour traçabilité. Les
nouveaux résultats centraux n'utilisent ni `Analyse_sensibilite` ni V8.

## Validation

Depuis `Robustesse/Vieillissement11` :

```text
python -m pytest tests -q -p no:cacheprovider
```

Les scénarios cubiques portent explicitement dans leurs fichiers candidats :

```json
{"model": {"ely": {"stress_exponent": 3.0}}}
```

La prochaine étape rationnelle est d'ajouter des prévisions de puissance à la
même structure RB2 et de mesurer leur valeur incrémentale par rapport au parent
statique et à ce faible gain SoH.

