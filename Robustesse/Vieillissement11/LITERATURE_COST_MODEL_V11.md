# Interprétation bibliographique et modèle de coût V11

## Périmètre

Cette note documente le code de `Vieillissement11`. Elle ne modifie pas le
manuscrit. L'objectif est de séparer trois objets qui étaient auparavant
confondus :

1. le dommage permanent qui consomme le capital et déclenche le remplacement ;
2. le conditionnement fini après mise en service ;
3. la perte réversible qui dégrade temporairement la performance.

Le coût unifié utilisé pour les optimisations est

\[
J=C_{\mathrm{deg,ledger}}+3\,EENS,
\]

avec les coûts et l'EENS en euros et kWh. Le ledger corrigé attribue chaque pas
de temps à une seule unité physique.

## Ce que permettent réellement les trois articles

### PEMWE — Rakousky

Rakousky compare pendant 1009 h cinq profils : 1 A/cm² constant, 2 A/cm²
constant, 2↔1 A/cm² par blocs de 6 h, 2↔0 A/cm² par blocs de 6 h, et 2↔0
A/cm² par blocs de 10 min. Les pertes nettes du tableau 2 sont respectivement
0, 194, 65, 16 et 50 µV/h (`rakousky.pdf`, PDF p. 2 et p. 4).

Ces valeurs ne sont pas cinq pentes permanentes :

- l'interruption EIS vers 500 h récupère 54 mV pour la cellule B et 15 mV pour
  C ;
- à 1 A/cm², une amélioration de contact masque une dégradation catalytique ;
- la différence entre les pauses de 6 h et de 10 min confond récupération et
  fréquence des commutations.

Le papier autorise donc une mémoire récupérable et un conditionnement borné. Il
n'identifie ni une pente permanente de 194 µV/h, ni un coût universel par
démarrage. Le coefficient de 11,7 µV/démarrage qui reproduit le protocole E est
un résidu de calibration hybride, pas une mesure extraite de l'article.

### PEMFC — McCay

McCay alterne des blocs de 125 h à courant constant et selon un profil maritime
dynamique de 360 s, avec le même courant moyen de 0,5 A/cm² et la même charge
totale (`mccay.pdf`, PDF p. 2). Après les 250 premières heures :

- irréversible : 1,2 µV/h constant et 4,8 µV/h dynamique ;
- réversible : 52 µV/h constant et 22 µV/h dynamique.

Le rapport quatre entre régimes stable et dynamique est causalement
exploitable. Les pertes réversibles ne doivent pas être monétisées. Les quelque
7 mV irréversibles des 250 premières heures constituent un rodage fini ; ils ne
doivent pas être répétés tous les 250 h.

McCay évite volontairement les arrêts/redémarrages. L'article ne fournit donc
aucun coefficient de dommage par démarrage PEMFC.

### PEMFC — Colombo

Colombo applique pendant 1000 h opératoires un cycle automobile couplant
courant, humidité, pression, température et procédures d'arrêt. Pour le CCM B,
le tableau 4 donne 2,4, 13,5, 21,9 et 31,7 µV/h aux différents points de mesure
(`colombo.pdf`, PDF p. 6).

Ces quatre pentes sont observées aux différents courants pendant le même cycle
vieillissant. Elles décrivent la sensibilité de la tension d'une cellule
vieillie ; elles ne sont pas quatre lois instantanées de génération du dommage.
V11 fixe donc `current_exponent=0` dans le coût PEMFC. La dépendance au courant
reste portée par la courbe de polarisation et la puissance disponible.

## Formulation implémentée

### PEMWE

Avec la densité de courant `j` en A/cm², le noyau permanent est

\[
\dot D_{ELY}=4{,}8\max(j-1,0)^p\quad[\mu V/h].
\]

La valeur par défaut `p=2` vérifie l'ancre DOE de 4,8 mV/kh à 2 A/cm² et ne
prolonge pas la pente courte de 194 µV/h. Le reste de Rakousky est représenté
par :

- un conditionnement exponentiel fini, exclu du coût capital ;
- un état réversible avec récupération plus rapide à faible courant et à
  l'arrêt ;
- dans la variante hybride, 11,7 µV par démarrage et 1,5 µV/h au quasi-idle.

Les deux derniers termes permanents ne sont pas identifiés par Rakousky et sont
obligatoirement annulés ou variés dans l'analyse de sensibilité.

La sensibilité `p=3` conserve exactement le noyau `p=2` aux niveaux
expérimentaux 0, 1 et 2 A/cm² et à l'ancre DOE. Elle change l'interpolation
entre 1 et 2 A/cm² et l'extrapolation au-delà de 2 A/cm². Rakousky ne permet
pas d'identifier cet exposant d'overload. Il est donc toujours enregistré comme
override explicite et non comme valeur nominale.

### PEMFC

Un état de stabilité `s` interpole entre les deux régimes McCay :

\[
\dot D_{FC}=4{,}8+s(1{,}2-4{,}8)\quad[\mu V/h],\qquad 0\leq s\leq1.
\]

La perte réversible interpole de la même façon entre 22 et 52 µV/h, mais elle
n'entre ni dans le coût de remplacement ni dans le SoH permanent. La variante
hybride conserve 20 µV par démarrage et 3 µV/h au quasi-idle ; ces deux valeurs
ne viennent pas de McCay ou Colombo et sont aussi testées à zéro.

## Deux SoH distincts

\[
SoH_{\mathrm{perm}}=1-\frac{D_{\mathrm{perm}}}{V_{\mathrm{ref}}},
\qquad
SoH_{\mathrm{op}}=1-
\frac{D_{\mathrm{perm}}+D_{\mathrm{cond}}+R_{\mathrm{rev}}}
{V_{\mathrm{ref}}}.
\]

- `SoH_perm` détermine le coût et le remplacement ;
- `SoH_op` décrit la performance visible et peut informer une variante EMS
  explicitement marquée `operando`.

Le conditionnement et le réversible peuvent modifier la puissance disponible,
mais ne sont jamais transformés directement en euros de remplacement.

## Cohérence des premières vies

Pour la RB2 optimale avec le noyau quadratique, les premières vies sont
d'environ :

- PEMFC : 20 562 h ON et 37 852 h calendaires ;
- PEMWE : 39 060 h ON et 159 584 h calendaires.

Pour la sensibilité cubique favorable à RB2 :

- PEMFC : 20 564 h ON et 37 854 h calendaires ;
- PEMWE : 43 821 h ON et 178 974 h calendaires.

Le PEMWE reste voisin du statut DOE de 40 kh à +10 % de tension. La PEMFC est
au-dessus de la cible DOE backup de 10 kh. Ce sont des validations d'ordre de
grandeur ; les duty cycles et critères EOL ne sont pas identiques.

## Conséquence sur le classement EMS

Le noyau quadratique donne RB1 meilleure que RB2 de 2,85 %. La sensibilité
cubique donne RB2 meilleure que RB1 de 2,42 %, et de 3,52 % lorsque les termes
de démarrage et de quasi-idle non identifiés sont annulés. Le classement dépend
donc de l'extrapolation PEMWE hors domaine expérimental.

Dans la sensibilité cubique, RB2(SoH) n'améliore son parent que de 0,141 % sur
25 ans. Le gain reste compris entre 0,136 % et 0,143 % entre 20 et 30 ans. Le
SoH seul ne produit donc pas le gain de quelques pourcents recherché.

## Règle d'interprétation des résultats EMS

Une amélioration RB2(SoH) n'est retenue que si :

- son parent est la meilleure RB2 statique trouvée directement sur 25 ans ;
- tous les coefficients SoH nuls reproduisent exactement le parent ;
- seules les deux consignes de RB2 varient, le dispatch restant identique ;
- le gain survit au crédit des stocks SoC/H₂ finaux ;
- il ne disparaît pas lorsque les pénalités de démarrage non identifiables sont
  annulées ;
- il dépasse le bruit pratique associé au plateau de réglage statique.

Le gain observé satisfait les quatre premiers contrôles et survit à l'annulation
des coûts de démarrage, mais son amplitude reste non matérielle. La conclusion
correcte est que le SoH n'apporte pas d'information actionnable importante à
cette structure de règles et à ces profils.

## Sources

- Rakousky et al., `rakousky.pdf`.
- McCay et al., `mccay.pdf`.
- Colombo et al., `colombo.pdf`.
- U.S. DOE, Technical Targets for Proton Exchange Membrane Electrolysis:
  https://www.energy.gov/cmei/fuels/technical-targets-proton-exchange-membrane-electrolysis
- U.S. DOE, Technical Targets for Fuel Cell Backup Power Systems:
  https://www.energy.gov/cmei/fuels/doe-technical-targets-fuel-cell-backup-power-systems

