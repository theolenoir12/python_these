# Baseline FLC experte I0 v1 — implémentation et screening

Date : 21 juillet 2026.

Statut : baseline technique auditée, **non optimisée et non promue**. Les
résultats ci-dessous décrivent uniquement la spécification v1 ; ils ne
constituent ni une conclusion sur la famille FLC ni un résultat de manuscrit.

## Spécification figée

- stratégie : `flc-mamdani-expert-v11-p2-i0-v1-2026-07-21` ;
- empreinte :
  `5f9ebd0d8f1e0eea9c0fafbb19bf6386c80081b60b4db5a59928a015eeeb9c13` ;
- deux branches Mamdani de 27 règles, respectivement PEMFC et PEMWE ;
- entrées I0 : puissance nette courante, SoC et remplissage H2 ;
- aucune entrée calendrier, prévision ou SoH dans la décision ;
- batterie résiduelle et faisabilité imposée par le `get_lol` commun ;
- AND/implication `min`, agrégation `max`, centre de gravité discret sur
  401 points et zone morte de commande égale à 0,10.

Les 16 tests unitaires passent. Le smoke de 7 jours et le screening d'un an
passent les contrôles de finitude, bornes SoC/H2, exclusivité PEMFC/PEMWE,
fermeture du déficit et identités du ledger corrigé.

## Surfaces de décision

Les surfaces sont sauvegardées dans
`figures/flc_surfaces_i0_5f9ebd0d8f1e.png`, avec leur diagnostic pleine
précision dans
`figures/flc_surfaces_i0_5f9ebd0d8f1e_diagnostics.json`. Leur structure
globale suit les tables linguistiques. Sur une grille 31 x 31 x 31, la
défuzzification Mamdani et la zone morte introduisent cependant de petites
inversions locales : la plus grande vaut 0,0151 en fraction de commande. Cette
propriété est conservée dans l'audit de la v1 et devra être traitée
explicitement si la monotonie stricte devient une contrainte de conception.

## Screening central d'un an

Cache canonique de ce screening :
`runs/smoke_flc_i0_b7bda4ee3399/`. Les trois stratégies partagent les mêmes
8 759 pas du profil empreinté, le modèle V11-p=2, le ledger corrigé et
`VoLL=3 EUR/kWh`.

| Stratégie | Dégradation (EUR) | EENS (kWh) | LPSP (%) | J3 (EUR) | Starts FC/ELY |
|---|---:|---:|---:|---:|---:|
| FLC experte I0 v1 | 3 189,91 | 223,88 | 1,0689 | 3 861,56 | 389 / 424 |
| RB1 (0,20 ; 0,40) | 2 511,15 | 125,83 | 0,6007 | 2 888,64 | 329 / 319 |
| RB2 (0,574 ; 0,465) | 2 676,64 | 129,19 | 0,6168 | 3 064,20 | 386 / 367 |

Pour cette v1, `J3` est supérieur de 33,68 % à RB1 et de 26,02 % à RB2.
L'EENS est également supérieure de 77,92 % et 73,30 %. L'écart de dégradation
est principalement associé au PEMWE : 1 159,43 EUR, soit 3,97 fois RB1 et
2,14 fois RB2. Le stock H2 atteint sa borne basse, contre 20,86 kWh au minimum
pour RB2.

Ces observations diagnostiquent une sollicitation trop agressive de la chaîne
H2 et une mauvaise gestion de la réserve saisonnière dans la spécification v1.
Elles ne permettent pas d'attribuer cet échec à la logique floue en général.

## Décision et prochaine expérience

La v1 reste une baseline reproductible mais ne passe pas le seuil de promotion
vers 25 ans. Avant tout réglage, il faut figer :

1. des blocs temporels distincts pour réglage et validation ;
2. un budget d'évaluations fermées identique aux variantes I0/IS ;
3. les paramètres réglables, limités d'abord aux plafonds FC/ELY, à la zone
   morte et aux partitions d'appartenance ;
4. une pénalisation ou une contrainte sur la sollicitation PEMWE, le stock H2
   terminal et, si retenue, la monotonie des surfaces.

L'année ci-dessus devient une donnée de diagnostic : une optimisation qui
l'utilise ne pourra plus la présenter ensuite comme validation aveugle.

Mise à jour : ce réglage a ensuite été préannoncé et exécuté. Son résultat est
consigné dans `TUNING_FLC_I0_RESULTS_2026-07-21.md`. La présente note reste
l'audit historique de la v1 non optimisée.
