# Couche SoH de la FLC experte — résultats nominaux

Date : 21 juillet 2026.

Statut : screening cinq ans et promotion 25 ans terminés. Aucun candidat SoH
actif n'est retenu ; le parent FLC I0 `flc_8126e6f729c6` reste la stratégie de
référence.

## Question et périmètre

L'expérience mesure uniquement la valeur nominale de l'information SoH pour
une correction hiérarchique donnée. Les 54 règles, les appartenances et les
paramètres du parent I0 restent figés. Une seconde couche Mamdani de neuf règles
compare l'usure normalisée du composant H2 actif à celle de la batterie et
multiplie la consigne FC ou ELY. Deux intensités indépendantes règlent les
branches déficit et surplus.

Le résultat principal est le couple `(LPSP, coût total de dégradation)` sur
25 ans. L'indicateur auxiliaire est

`J3 = coût de dégradation + 3 EUR/kWh x EENS`.

Le protocole préannoncé est
`TUNING_PROTOCOL_FLC_IS_SOH_V11_P2_2026-07-21.md`. Le moteur est V11-p=2, le
ledger est corrigé et charge/PV sont injectés depuis le cache DP canonique.

## Tests nuls

Le cas `strength_FC=strength_ELY=0` appelle directement le parent I0.

- sur cinq ans, l'empreinte complète de trajectoire et du ledger est identique
  au rejeu parent ; toutes les métriques hors durée d'exécution sont égales ;
- sur 25 ans, les quinze tableaux sauvegardés sont égaux bit-à-bit au cache
  parent, ainsi que le ledger et les métriques hors durée d'exécution ;
- le test fonctionnel au BoL et l'isolation des SoH FC/ELY entre branches
  passent également.

Le contrôle 25 ans est `flc_is_be3fc49db506`. Son audit indépendant est
`PASS` dans `runs/promoted_flc_is_soh_25y_34bdb5fbe2af/`.

## Screening préannoncé sur cinq ans

Les 36 combinaisons de
`strength_FC,strength_ELY in {0 ; 0,025 ; 0,05 ; 0,10 ; 0,20 ; 0,40}²` ont été
évaluées, plus un rejeu parent dédié. Les trois candidats actifs dédupliqués par
les rôles de promotion sont :

| Candidat | `(strength_FC ; strength_ELY)` | LPSP (%) | Dégradation (kEUR) | EENS (kWh) | J3 (kEUR) |
|---|---:|---:|---:|---:|---:|
| parent I0 | `(0 ; 0)` | 0,688755 | 12,604879 | 721,318 | 14,768831 |
| `8f0cea73d3df` | `(0 ; 0,025)` | 0,680580 | 12,625127 | 712,757 | 14,763397 |
| `f7d06e4ee38b` | `(0 ; 0,10)` | 0,665479 | 12,693907 | 696,941 | 14,784730 |
| `b7ad87cca2c0` | `(0,20 ; 0)` | 0,711564 | 12,601085 | 745,205 | 14,836700 |

Le meilleur J3 actif, `8f0cea73d3df`, améliore la LPSP de 0,008175 point
mais augmente la dégradation de 20,25 EUR. Son J3 ne baisse que de 5,43 EUR,
soit 0,0368 %, très sous le seuil de 1 %. Le screening ne montrait donc déjà
qu'un effet faible et dépendant du compromis choisi.

## Rejeu canonique sur 25 ans

Le cache accepté est `runs/promoted_flc_is_soh_25y_34bdb5fbe2af/` et son audit
est `PASS`.

| Candidat | `(strength_FC ; strength_ELY)` | LPSP (%) | Dégradation (kEUR) | EENS (kWh) | J3 (kEUR) |
|---|---:|---:|---:|---:|---:|
| parent I0 | `(0 ; 0)` | 0,721251 | 62,139391 | 3 776,918 | 73,470146 |
| `8f0cea73d3df` | `(0 ; 0,025)` | 0,721348 | 62,161543 | 3 777,423 | 73,493812 |
| `f7d06e4ee38b` | `(0 ; 0,10)` | 0,723032 | 62,213895 | 3 786,242 | 73,572620 |
| `b7ad87cca2c0` | `(0,20 ; 0)` | 0,807406 | 62,239071 | 4 228,080 | 74,923310 |

Écarts au parent :

| Candidat | Delta LPSP (point) | Delta dégradation (EUR) | Delta J3 (EUR) | Delta J3 (%) |
|---|---:|---:|---:|---:|
| `8f0cea73d3df` | +0,000096 | +22,15 | +23,67 | +0,0322 |
| `f7d06e4ee38b` | +0,001780 | +74,50 | +102,47 | +0,1395 |
| `b7ad87cca2c0` | +0,086155 | +99,68 | +1 453,16 | +1,9779 |

Les trois variantes actives sont donc strictement dominées par le parent sur
les deux axes principaux. Le petit avantage observé à cinq ans s'inverse après
les remplacements successifs.

La variante FC `b7ad87cca2c0` illustre le transfert d'usure : elle réduit le
coût FC de 40,90 EUR et compte six remplacements FC au lieu de sept, mais
augmente les coûts batterie et ELY de respectivement 89,97 et 50,61 EUR. Elle
augmente en outre l'EENS de 451,16 kWh. Les corrections ELY augmentent quant à
elles le coût ELY de 20,40 EUR pour `8f0cea73d3df` et de 67,73 EUR pour
`f7d06e4ee38b`, sans amélioration de fiabilité à 25 ans.

## Décision scientifique

Aucun candidat SoH actif n'est promu. Le parent I0 reste le socle de la future
couche de prévision. Un banc bruit/biais SoH n'est pas lancé : le SoH parfait,
qui constitue ici le cas informationnel le plus favorable, ne produit déjà pas
de déplacement favorable pour cette architecture.

Cette conclusion est spécifique à la correction d'usure relative testée. Elle
ne démontre pas que le SoH est sans valeur pour toute FLC, une règle apprise ou
un ANFIS. Elle montre que la modulation multiplicative locale choisie ne crée
pas de gain durable et que l'horizon de cinq ans est insuffisant pour la
sélection finale.

## Suite : couche de prévision IF puis combinaison ISF

La prochaine expérience partira à nouveau du parent I0. L'entrée candidate est
l'énergie nette prévue, agrégée plutôt que la fenêtre brute :

`E_net,H(t) = sum_k [P_load_hat(t+k) - P_PV_hat(t+k)] Delta t`.

L'horizon principal sera H18, cohérent avec le levier de pré-charge déjà étudié
dans le manuscrit. Le backtest LSTM existant sur 216 origines donne à H18 un
biais de -2,32 kWh et un écart-type de 39,38 kWh ; H48, avec un écart-type de
80,03 kWh, restera une sensibilité. Comme ces modèles n'utilisent pas de
prévisions météorologiques exogènes, leur erreur peut être traitée comme un
scénario conservateur à confronter ensuite à une prévision opérationnelle plus
riche, et non comme une précision garantie.

La couche IF devra satisfaire `forecast_strength=0 = I0` bit-à-bit et être
comparée sous futur parfait, erreur empirique LSTM et persistance. La couche
ISF ne sera construite qu'ensuite. Comme IS seul n'est pas retenu, sa valeur
devra être mesurée par l'ablation `ISF-IF`; une éventuelle interaction ne devra
pas être présentée comme un effet propre du SoH.
