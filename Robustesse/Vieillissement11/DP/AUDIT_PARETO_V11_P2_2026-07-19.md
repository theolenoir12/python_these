# Audit du front de Pareto PD V11-p=2

Date : 19 juillet 2026. Job mésocentre : `216257`.

## Verdict

Le calcul est complet et exploitable : 19/19 valeurs d'epsilon, modèle V11-p=2, variante V2 avec projection et rollout, caches finis, métriques reproductibles et ledgers corrigés exacts. Les 19 points sont non dominés.

Le point epsilon=3 reproduit bit-à-bit après conversion float32 les trajectoires du job 216233 et son ledger est exactement identique.

## Résultats centraux

- Minimum réalisé à VoLL=3 : epsilon=20, J=48.085 kEUR, dégradation=45.402 kEUR, EENS=894.6 kWh et LPSP=0.1708 %.
- Bande epsilon=10--50 : écart maximal au minimum 0.211 %.
- Le point résolu avec epsilon=3 vaut 48.809 kEUR, soit 1.504 % au-dessus du minimum réalisé.
- Gain du meilleur point PD sur RB1 à VoLL=3 : 25.475 kEUR (34.63 %).
- Gain sur RB2 : 27.636 kEUR (36.50 %).

`epsilon` est ici le poids de fiabilité du backward discrétisé, et non l'exposant de vieillissement `p`, qui reste fixé à 2. Le coût final est recalculé par le rollout physique et le ledger : la correspondance entre epsilon interne et VoLL de reporting n'est donc pas une identité. La sélection défendable est la bande epsilon=10--50, pas un optimum précis à epsilon=20, car son plateau est plus étroit que la sensibilité de grille déjà observée.

## Contrôles

- Erreurs bloquantes : 0.
- Coût de dégradation strictement croissant avec epsilon : True.
- EENS strictement décroissante avec epsilon : True.
- Points non dominés : 19/19.
- Identités vérifiées : ledger = coût sauvegardé ; J@3 = dégradation + 3 EENS/1000 ; LPSP = EENS/demande.
- Intervalles de remplacement disjoints et nombres de remplacements cohérents.
- Toutes les trajectoires SoH et H2 sont finies et dans leurs bornes.

## Réserve sur l'extrémité très peu fiable

La métrique canonique borne `lol_tab` entre 0 et 1 avant de calculer l'EENS. Le `lol_tab` brut dépasse 1 lorsque les contraintes réduisent davantage la puissance que la charge résiduelle. Ce déséquilibre au-delà du clipping devient significatif pour epsilon <= 0,15 : jusqu'à 10334.7 kWh, soit 16.6 % de l'EENS. Ces points décrivent bien la convention du simulateur historique, mais ne doivent pas être interprétés finement sans correction du rebouclage de puissance.

Cette réserve ne touche pas la zone de décision : pour epsilon >= 1,5, l'excès maximal est 0.310 kWh, soit 0.027 % de l'EENS.

## Conclusion scientifique

Le front fournit une référence offline omnisciente unique ; il n'existe pas ici de variante avec/sans SoH à comparer. La région utile du front est suffisamment propre pour servir de plafond de performance aux EMS online. RB1 et RB2 restent des références online : la PD ne doit pas être présentée comme une comparaison à information égale.

Données détaillées : `runs/pareto_audit_v11_p2.csv`.
