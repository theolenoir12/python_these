# Résultats centraux V11 — coûts littérature, RB2 et apport du SoH

Date des calculs : 16 juillet 2026.

## Verdict

1. L'ancienne variante `RB2(Aging)` n'était pas construite sur la RB2
   « retunée » : son parent était la RB2 historique `(0,59 ; 0,49)`. Cette
   comparaison ne permettait donc pas d'attribuer proprement le gain au
   vieillissement.
2. Dans les nouveaux calculs, chaque `RB2(SoH)` est construite sur la meilleure
   RB2 statique du même modèle. Avec tous les coefficients SoH nuls, elle
   reproduit exactement le parent, et elle appelle le même dispatch RB2.
3. Avec le noyau PEMWE quadratique par défaut, la RB1 optimisée reste meilleure
   que RB2 de 2,85 %. Le cahier des charges « RB2 meilleure stratégie de base »
   n'est donc pas vérifié.
4. Une variante explicite d'overload PEMWE cubique, qui conserve exactement les
   points expérimentaux 0, 1 et 2 A/cm² et l'ancre DOE à 2 A/cm², donne RB2
   meilleure que toutes les stratégies de base : 2,42 % devant RB1 optimisée et
   6,91 % devant la meilleure stratégie à partage fixe.
5. Sur ce scénario favorable, le SoH améliore RB2 de 0,141 % seulement. Le gain
   est stable à 20, 25 et 30 ans et survit au crédit des stocks terminaux, mais
   il est très inférieur aux quelques pourcents recherchés.

La conclusion défendable est donc conditionnelle : une pénalisation forte des
surcharges PEMWE suffit à rendre RB2 meilleure parmi les règles de base, mais le
SoH seul n'apporte pas d'information actionnable importante à la structure RB2.

## Fonction de coût retenue

Le coût unifié vaut

\[
J=C_{\mathrm{deg}}+3\,EENS,
\]

avec `C_deg` en euros, `EENS` en kWh et un VoLL de 3 €/kWh. Le ledger
attribue chaque heure de vieillissement à une seule unité physique.

### PEMWE

Le dommage permanent de longue durée est

\[
\dot D_{ELY}=4{,}8\max(j-1,0)^p\quad[\mu V/h],
\]

où `j` est en A/cm². Le modèle par défaut conserve `p=2`. Les pertes très
élevées observées par Rakousky sur 1009 h sont représentées séparément comme
conditionnement fini et perte réversible ; elles ne sont pas capitalisées comme
une pente permanente sur 25 ans.

Le scénario central favorable à RB2 utilise `p=3`. Il est exactement identique
à `p=2` aux points `j=0`, `j=1` et `j=2` :

- zéro sous le seuil de 1 A/cm² ;
- 4,8 µV/h à 2 A/cm², soit l'ancre DOE de 4,8 mV/kh ;
- mêmes protocoles Rakousky aux niveaux de courant testés.

Il change l'interpolation et surtout l'extrapolation au-delà de 2 A/cm². Cette
zone n'est pas identifiée par Rakousky. Dans les simulations, elle pénalise les
pointes PEMWE de RB1, qui peuvent atteindre environ 3,33 A/cm², alors que RB2
borne la puissance par sa consigne. Le scénario `p=3` doit donc être annoncé
comme hypothèse hybride d'overload, pas comme coefficient mesuré dans l'article.

### PEMFC

Le dommage permanent interpole les deux régimes identifiés par McCay :

\[
\dot D_{FC}=4{,}8+s(1{,}2-4{,}8)\quad[\mu V/h],\qquad 0\leq s\leq1.
\]

La perte réversible interpole séparément 22 et 52 µV/h et n'entre pas dans le
coût de remplacement. Les pentes de Colombo mesurées à plusieurs courants sur
le même composant vieilli ne sont pas utilisées comme loi causale instantanée ;
le noyau nominal fixe donc l'exposant de courant PEMFC à zéro.

Les termes hybrides nominaux `start_uv` et `idle_uvph` ne sont identifiés par
aucun des trois articles. Ils sont conservés pour reproduire les ordres de
grandeur des premières vies, puis testés à zéro.

## Classement best-vs-best sur 25 ans

### Noyau quadratique par défaut

| Politique optimisée | Paramètres | Dégradation (€) | EENS (kWh) | J (€) |
|---|---:|---:|---:|---:|
| RB1 | `SoC_low=0,20; SoC_high=0,40` | 60 833,19 | 4 242,56 | **73 560,88** |
| RB2 | `FC=0,574; ELY=0,465` | 63 091,35 | 4 210,01 | 75 721,37 |

RB1 devance RB2 de 2 160,49 €, soit 2,85 % du coût RB2. Ce classement reste
inchangé sur 12 combinaisons de constantes de stabilité McCay. Lorsque les
coûts de démarrage et de quasi-idle sont annulés, RB1 reste meilleure de 2,11 %.

### Scénario d'overload cubique

| Politique optimisée | Paramètres | Dégradation (€) | EENS (kWh) | J (€) |
|---|---:|---:|---:|---:|
| RB2 | `FC=0,574; ELY=0,465` | 61 717,34 | 4 211,45 | **74 351,70** |
| RB1 | `SoC_low=0,20; SoC_high=0,37` | 62 661,68 | 4 511,96 | 76 197,57 |
| 100-0 | — | — | — | 79 868,71 |
| 75-25 | — | — | — | 91 562,42 |
| SoC1 | — | — | — | 104 688,32 |
| 50-50 | — | — | — | 132 379,83 |
| 0-100 | — | — | — | 150 738,39 |
| 25-75 | — | — | — | 231 925,38 |
| SoC06 | — | — | — | 306 828,26 |

RB2 devance RB1 optimisée de 1 845,87 €, soit 2,42 % du coût de RB1. Elle
réduit simultanément la dégradation de 944,33 € et l'EENS de 300,51 kWh. Elle
devance la meilleure règle à partage fixe de 5 517,01 €, soit 6,91 %.

Le changement de classement se produit entre `p=2` et `p=3`. Il est presque
entièrement porté par le coût PEMWE et par le franchissement discret d'un
remplacement. Le classement n'est donc pas robuste à l'extrapolation de la loi
de dommage au-delà du domaine expérimental.

## RB2(SoH) attribuable au SoH

Le parent est exactement la RB2 cubique optimisée `(0,574 ; 0,465)`. La
meilleure variante trouvée conserve la même structure de règle et le même
dispatch. Seule la consigne PEMFC varie :

\[
c_{FC}(t)=0{,}574-0{,}0125
\left(\frac{1-SoH_{FC}(t)}{1-SoH_{FC,EOL}}\right)^4.
\]

La consigne ELY reste à 0,465. La baisse maximale de consigne FC n'est donc que
de 0,0125 Pmax, concentrée près de la fin de vie.

| Politique, 25 ans | Dégradation (€) | EENS (kWh) | J (€) |
|---|---:|---:|---:|
| RB2 parent | 61 717,34 | 4 211,45 | 74 351,70 |
| RB2(SoH) | 61 754,90 | 4 164,07 | **74 247,11** |
| Écart SoH − parent | +37,56 | −47,38 | **−104,59** |

Le gain vaut 0,1407 %. Il vient d'une baisse d'EENS de 142,15 € partiellement
annulée par 37,56 € de dégradation supplémentaire. Le nombre de remplacements
reste identique : 6 batteries, 5 PEMFC et 1 PEMWE. La règle compense donc
l'approche de l'EOL du PEMFC plutôt qu'elle ne prolonge matériellement sa vie.

Après crédit de la batterie et du stock H₂ terminaux, les coûts deviennent
73 993,36 € pour le parent et 73 888,10 € pour RB2(SoH), soit un gain de
105,25 € (0,1423 %). Les stocks finaux n'expliquent pas le résultat.

### Transfert temporel des réglages 25 ans

| Horizon | RB2 (€) | RB2(SoH) (€) | Gain SoH | RB1 (€) | Avance RB2 / RB1 |
|---|---:|---:|---:|---:|---:|
| 20 ans | 59 391,00 | 59 310,06 | 0,1363 % | 60 782,87 | 2,2899 % |
| 25 ans | 74 351,70 | 74 247,11 | 0,1407 % | 76 197,57 | 2,4225 % |
| 30 ans | 89 240,36 | 89 112,34 | 0,1435 % | 91 803,84 | 2,7923 % |

Ce tableau transfère les réglages optimisés à 25 ans ; il ne prétend pas
réoptimiser chaque politique à chaque horizon. L'ordre est néanmoins stable.

## Sensibilité aux termes non identifiés de démarrage et de quasi-idle

Dans le scénario cubique avec ces quatre termes fixés à zéro :

- RB2 optimisée `(0,574 ; 0,465)` : 66 502,86 € ;
- RB1 optimisée `(0,19 ; 0,43)` : 68 926,13 € ;
- RB2 devance RB1 de 2 423,27 €, soit 3,52 % ;
- RB2(SoH) gagne 76,36 € sur son parent, soit 0,1148 %.

Le classement favorable à RB2 ne dépend donc pas des pénalités de démarrage.
En revanche, l'absence d'effet SoH de quelques pourcents est encore renforcée.

## Cohérence des premières vies

Pour la RB2 cubique optimisée :

| Composant | Vie calendaire | Temps ON | EFPH | Énergie |
|---|---:|---:|---:|---:|
| PEMFC | 37 854 h | 20 564 h | 14 906 h | 22 990,9 kWh |
| PEMWE | 178 974 h | 43 821 h | 21 030 h | 331 351 kWh |

La première vie PEMWE de 43,8 kh ON est du même ordre que le statut DOE de
40 kh à +10 % de tension. La première vie PEMFC dépasse la cible DOE backup de
10 kh. Ces comparaisons valident un ordre de grandeur, pas l'identité des duty
cycles ni des technologies.

## Conclusion de travail

Le résultat à retenir n'est pas « les articles démontrent que RB2 est toujours
meilleure ». Les données permettent deux lectures :

- lecture conservatrice `p=2` : RB1 est meilleure et le cahier des charges de
  classement échoue ;
- lecture hybride avec dommage cubique en overload : RB2 est la meilleure
  stratégie de base de quelques pourcents, avec des premières vies cohérentes
  avec les repères DOE.

Dans les deux lectures, l'information SoH seule améliore RB2 de moins de 0,2 %.
Il n'est donc pas justifié d'annoncer quelques pourcents pour RB2(SoH). La suite
logique est de conserver le SoH comme variable disponible, puis de mesurer la
valeur ajoutée de prévisions de puissance dans une RB2 dont la structure de
dispatch reste inchangée.

## Reproductibilité

- Couche de coûts : `Common/degradation_v11.py`.
- RB2 statique : `Common/rb2_policy.py`.
- RB2(SoH) attribuable : `Common/rb2_soh_policy_v11.py`.
- Lanceur autonome V11 : `run_v11_candidates.py`.
- Résultats bruts : `study/*.jsonl`.
- Tests : `python -m pytest tests -q -p no:cacheprovider`.

Le modèle par défaut reste volontairement quadratique. Chaque calcul cubique
porte explicitement `model.ely.stress_exponent=3.0` afin d'éviter de transformer
une analyse de sensibilité favorable en vérité bibliographique implicite.

