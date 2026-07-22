# Protocole de réglage FLC-IS — couche SoH

Date de préannonce : 21 juillet 2026, avant le lancement du balayage.

## Question attribuable

À logique I0, paramètres I0 et profil identiques, l'ajout des SoH courants
permet-il de déplacer favorablement le point réalisé dans le plan
`(LPSP, coût de dégradation)` ?

Le parent est strictement `flc_8126e6f729c6`, d'empreinte
`71c0531744f2ecf0b6cde6ee97a7ed0ba0d3d2468cebca06caa75643c2bd162d`.
Ses 54 règles, fonctions d'appartenance, plafonds, échelles de sévérité et zone
morte restent figés.

## Couche d'information IS

L'usure de chaque composant est normalisée sur son intervalle SoH utile :

`wear_i = clip((1 - SoH_i)/(1 - SoH_EoL_i), 0, 1)`.

Une couche Mamdani de neuf règles compare l'usure du composant H2 actif à celle
de la batterie :

- composant H2 plus usé : réduction de sa fraction de commande ;
- usures comparables : correction neutre ;
- batterie plus usée : augmentation de la fraction H2.

La branche déficit utilise uniquement SoH batterie + PEMFC ; la branche surplus
uniquement SoH batterie + PEMWE. La correction multiplie la commande I0 après
la zone morte et reste bornée. Elle ne peut pas activer une branche arrêtée par
le parent.

Deux intensités indépendantes sont réglées : `strength_FC` et `strength_ELY`.
La valeur zéro désactive exactement la branche correspondante.

## Tests nuls

Les contrôles obligatoires sont :

1. `strength_FC=strength_ELY=0` appelle directement le parent I0 ;
2. la trajectoire nulle de cinq ans, son ledger et toutes ses métriques doivent
   être bit-à-bit identiques à un rejeu du parent ;
3. avec tous les SoH constants à 1, une couche non nulle est exactement neutre ;
4. le SoH ELY n'influence pas la branche déficit et le SoH FC n'influence pas
   la branche surplus.

## Budget de screening

Le screening emploie le profil canonique injecté depuis
`DP/runs/dp_aging_v11_p2_25y_51x51.npz`, V11-p=2 et le ledger corrigé.

- horizon : cinq ans, soit 43 799 pas ;
- grille déterministe : produit cartésien
  `{0 ; 0,025 ; 0,05 ; 0,10 ; 0,20 ; 0,40}²` ;
- 36 variantes IS, cas nul inclus, plus un rejeu I0 dédié au test nul ;
- aucun raffinement ajouté après lecture des résultats ;
- caches par paramètres et empreinte de protocole.

Le front non dominé cinq ans est construit sur LPSP et dégradation. `J3` reste
une scalarisation auxiliaire à `VoLL=3 EUR/kWh`.

## Promotion 25 ans

Après audit, au plus quatre rôles, dédupliqués, pourront être promus :

1. minimum de J3, cas nul compris ;
2. meilleur J3 strictement non nul ;
3. meilleure LPSP avec dégradation au plus 1 % au-dessus du parent ;
4. plus faible dégradation avec LPSP au plus 0,05 point au-dessus du parent.

Le parent I0 25 ans est réutilisé depuis le cache audité
`promoted_flc_25y_5d6c177f02a7`. Le rejeu 25 ans des variantes IS utilisera
bit-à-bit le même profil. Les conclusions porteront séparément sur les deux
axes ; un faible gain de J3 ne suffira pas à conclure à une valeur matérielle
du SoH.

L'incertitude de mesure SoH n'est pas incluse dans cette première expérience :
le SoH parfait constitue la borne haute informationnelle de la couche. Une
variante non nulle devra d'abord montrer un effet nominal avant tout banc de
biais ou de bruit.

## Résultat du protocole

Le protocole a été exécuté sans raffinement post-hoc. Les résultats et la
décision sont consignés dans `TUNING_FLC_IS_SOH_RESULTS_2026-07-21.md` : le
test nul 25 ans est exact, les trois variantes actives sont dominées par I0 et
aucune n'est retenue.
