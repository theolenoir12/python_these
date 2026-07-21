# Protocole de réglage de la FLC experte I0

Date de préannonce : 21 juillet 2026, avant le lancement du balayage.

## Objet

L'objet scientifique final est le point réalisé sur le plan de Pareto

\[
\bigl(LPSP_{25\,ans},\;C_{deg,25\,ans}\bigr).
\]

L'EENS est conservée en kWh et la LPSP est calculée par
`100 EENS / énergie totale de charge`. Le coût unifié

\[
J_3=C_{deg}+3\,EENS
\]

est une scalarisation auxiliaire à `VoLL=3 EUR/kWh`. Le suffixe 3 désigne la
VoLL ; il ne s'agit ni d'un troisième objectif ni d'une puissance cubique. Le
minimum de `J3` sert à repérer un compromis, mais le front non dominé en
LPSP--dégradation reste le résultat principal.

## Périmètre de ce premier réglage

La base de 54 règles et les fonctions d'appartenance de la v1 restent fixes.
Seuls cinq paramètres interprétables sont réglés :

| Paramètre | Borne basse | Borne haute | Rôle |
|---|---:|---:|---|
| plafond PEMFC | 0,35 | 0,90 | fraction de la puissance nominale DC |
| plafond PEMWE | 0,15 | 0,70 | fraction de la puissance nominale DC |
| échelle déficit | 0,60 | 1,60 | multiplicateur de l'échelle nominale de sévérité |
| échelle surplus | 0,60 | 1,60 | multiplicateur de l'échelle nominale de sévérité |
| zone morte | 0,02 | 0,30 | fraction de commande annulée |

La FLC v1 non réglée et trois ancres explicites sont ajoutées au plan, même si
la v1 se trouve hors des bornes de plafonds. Aucun SoH, aucune prévision et
aucune information calendaire ne sont ajoutés.

## Budget et échantillonnage

Le screening utilise exactement le profil central d'un an déjà diagnostiqué,
le modèle V11-p=2, le ledger corrigé et les références RB1 `(0,20 ; 0,40)` et
RB2 `(0,574 ; 0,465)`.

- étage grossier : 48 points Latin hypercube déterministes, graine 20260721,
  plus quatre ancres ;
- étage local : au plus 30 voisins, soit deux perturbations coordonnées par
  paramètre autour d'au plus trois parents couvrant fiabilité, durabilité et
  minimum de J3 ;
- budget total maximal : 82 évaluations FLC d'un an ;
- les calculs sont mis en cache par paramètres et empreinte de protocole.

L'année a déjà servi au diagnostic de la v1 : elle est donc explicitement une
donnée de calibration, pas une validation aveugle. L'étage local ne reçoit
aucun budget supplémentaire après lecture des résultats.

## Promotion

Le screening produit le front non dominé d'un an et reporte séparément LPSP,
EENS, coût total et composantes de dégradation, démarrages et stock H2.
Il ne lance aucun calcul 25 ans automatiquement.

Après audit mécanique, au plus trois représentants distincts pourront être
promus :

1. le minimum de `J3` sur le front ;
2. un point orienté fiabilité ;
3. un point orienté durabilité.

Les représentants ne seront pas dupliqués si le front est trop court. Le
calcul 25 ans devra rejouer ces candidats et RB1/RB2 sur le même socle, puis
construire le front réalisé `(LPSP, C_deg)`. Une amélioration sur `J3` seule ne
suffira pas à déclarer une domination de Pareto.

Ce premier réglage ne démontre aucune généralisation à un autre profil
météorologique. Une modification ultérieure des règles ou des appartenances
constituera une nouvelle version et un nouveau budget, pas une extension
silencieuse de ce balayage.
