# Réglage multiobjectif de la FLC experte I0

Date : 21 juillet 2026.

Statut : screening d'un an complet et deux points préannoncés rejoués sur
25 ans. Le candidat `flc_8126e6f729c6` est promu comme parent FLC I0 ; le point
de durabilité extrême reste un diagnostic.

## Rappel des objectifs

Le résultat principal est le couple
`(LPSP [%], coût total de dégradation [EUR])`. L'EENS est conservée en kWh et

`J3 = C_deg + 3 EUR/kWh x EENS`

sert uniquement à repérer un compromis à la VoLL de référence. Le suffixe 3
désigne la VoLL et non un troisième objectif.

Le protocole préannoncé est
`TUNING_PROTOCOL_FLC_I0_V11_P2_2026-07-21.md`. Les 54 règles et les fonctions
d'appartenance de la v1 sont restées fixes. Le balayage porte sur les plafonds
FC/ELY, les échelles de sévérité déficit/surplus et la zone morte.

## Screening d'un an

Le cache `runs/tune_flc_i0_1y_c21a6da6c16c/` contient 82/82 évaluations
valides : 48 points Latin hypercube, quatre ancres et 30 voisins locaux. Le
front FLC interne compte 18 points.

Le compromis retenu est `flc_8126e6f729c6` :

| Paramètre | Valeur |
|---|---:|
| plafond FC | 0,78375339 |
| plafond ELY | 0,52282887 |
| multiplicateur échelle déficit | 1,28769941 |
| multiplicateur échelle surplus | 0,62373324 |
| zone morte | 0,26735578 |

À un an, il donne LPSP=0,476655 %, dégradation=2 639,41 EUR,
EENS=99,84 kWh et J3=2 938,93 EUR. Par rapport à la v1 non réglée, J3 diminue
de 23,89 %, la LPSP de 55,41 % et la dégradation de 17,26 %.

À cet horizon, il domine RB2 sur les deux axes. Face à RB1, il améliore la
LPSP de 20,66 % mais augmente la dégradation de 5,11 % ; les deux points sont
donc non dominés l'un par l'autre.

## Rejeu canonique sur 25 ans

Le cache accepté est `runs/promoted_flc_25y_5d6c177f02a7/` et son audit est
`PASS`. Les 218 999 couples charge/PV sont bit-à-bit ceux du cache canonique
RB1/RB2/PD, le moteur est V11-p=2 et le ledger est corrigé.

| Stratégie | LPSP (%) | Dégradation (kEUR) | EENS (kWh) | J3 (kEUR) |
|---|---:|---:|---:|---:|
| FLC I0 réglée `8126e6f729c6` | 0,721251 | 62,139 | 3 776,92 | 73,470 |
| RB1 `(0,20 ; 0,40)` | 0,810172 | 60,833 | 4 242,56 | 73,561 |
| RB2 `(0,574 ; 0,465)` | 0,803955 | 63,091 | 4 210,01 | 75,721 |

La FLC réglée domine RB2 : LPSP -10,29 %, dégradation -1,51 % et J3 -2,97 %.
Face à RB1, elle constitue un nouveau compromis : LPSP -10,98 % pour un coût
de dégradation +2,15 %. Son J3 est inférieur de seulement 90,73 EUR, soit
0,123 %. Cet écart est sous le seuil de criblage de 1 % et ne doit pas être
présenté comme un gain matériel face à RB1.

La dégradation de la FLC se répartit en 43,272 kEUR batterie, 8,320 kEUR FC et
10,548 kEUR ELY. Le stock H2 atteint ponctuellement sa borne basse mais finit à
189,63 kWh ; les bornes, le bilan de puissance, l'exclusivité FC/ELY, les
métriques et les ledgers sont tous validés.

Le point de durabilité extrême `flc_d0b94c87f073` atteint 56,936 kEUR de
dégradation, mais sa LPSP vaut 9,0209 % et son J3 198,654 kEUR. Il documente la
queue du front FLC ; il n'est pas un réglage opérationnel recommandé.

## Écart de profil détecté par l'audit

Un premier rejeu, conservé dans `runs/promoted_flc_25y_fa10e2ba91da/`, a été
invalidé avant interprétation. Le profil local courant est périodique sur
8 760 heures, alors que le cache canonique RB1/RB2/PD est périodique sur
8 761 points et duplique le dernier point à chaque jonction annuelle. Les
profils sont identiques pendant la première année puis se décalent.

Le runner 25 ans injecte désormais dans chaque worker les tableaux charge/PV
du cache canonique. Toute comparaison future au front PD doit utiliser cette
voie ou reconstruire toutes les références sur un nouveau profil commun.

## Décision scientifique

`flc_8126e6f729c6`, d'empreinte de spécification
`71c0531744f2ecf0b6cde6ee97a7ed0ba0d3d2468cebca06caa75643c2bd162d`, devient
le parent FLC I0 reproductible. Il est exposé par
`make_tuned_expert_flc_policy_v11()`.

Cette première optimisation rend la baseline compétitive et crée un point
Pareto distinct de RB1. Une optimisation des appartenances ou des tables de
règles constituerait une v2 avec un nouveau budget. Pour tester la valeur du
SoH, l'extension IS devra partir exactement de ce parent, conserver un test
nul coefficient zéro et recevoir un budget de réglage annoncé séparément.
