# RB2 - Vieillissement10

## Regle active

RB2 est revenue a sa definition historique : deux consignes fixes de puissance
H2 et aucun plafond conditionnel, aucune reserve SoC et aucune information SoH
dans la decision.

- consigne PEMFC : 0.59 Pmax ;
- consigne PEMWE : 0.49 Pmax.

Les seules limitations restantes sont physiques et communes aux strategies :
energie disponible dans le reservoir H2, capacite libre du reservoir,
defaillances, puissances disponibles des composants et referee get_lol.

Le couple 0.59/0.49 est le meilleur couple teste lors des balayages successifs.
Il ne doit pas etre presente comme un optimum mathematique global continu.

## Definition du vieillissement PEMWE

Le modele distingue maintenant trois objets qui etaient auparavant confondus :

1. la degradation irreversible longue duree, quadratique en densite de courant
   et ancree a 4.8 microV/h a 2 A/cm2 (cible DOE PEM 2022) ;
2. la contribution reversible observee sur les essais courts de Rakousky et
   recuperee pendant les arrets ;
3. une acceleration irreversible explicite au-dessus de 2 A/cm2 :
   100 * (j - 2)^2 microV/h.

Le coefficient d'acceleration est un parametre de scenario, pas une constante
universelle identifiee. Il represente le changement de regime documente a fort
courant. Il donne 104.8 microV/h irreversible a 3 A/cm2. Il faut le conserver
dans les analyses de sensibilite.

Sources principales :

- DOE, Technical Targets for Proton Exchange Membrane Electrolysis:
  https://www.energy.gov/cmei/fuels/technical-targets-proton-exchange-membrane-electrolysis
- Rakousky et al., J. Power Sources 342 (2017), essai de 1009 h:
  https://juser.fz-juelich.de/record/810841/
- etude de stabilite PEMWE a forte densite de courant:
  https://www.sciencedirect.com/science/article/abs/pii/S0013468618309150

## LPSP et cout unifie

Une seule normalisation est utilisee :

- LPSP = EENS / energie totale demandee au bus DC.

Cette definition est celle du manuscrit. L'ancienne normalisation par la
charge residuelle positive charge-PV a ete retiree du code actif ; les anciens
fichiers qui l'utilisent doivent etre recalcules et non convertis.

Le classement economique utilise directement
degradation + VoLL * EENS, avec VoLL = 3 EUR/kWh. Il est donc invariant au
choix du denominateur de LPSP.

## Resultat 25 ans

Pour RB2 0.59/0.49 :

- EENS : 4 058.49 kWh ;
- LPSP : 0.7750 % ;
- degradation batterie / PEMFC / PEMWE :
  42.919 / 5.161 / 15.294 kEUR ;
- degradation totale : 63.375 kEUR ;
- cout unifie : 75.550 kEUR.

RB1 donne 0.898 % de LPSP charge, 68.340 kEUR de degradation et
82.449 kEUR de cout unifie. RB2 est donc la meilleure strategie de base dans
ce scenario et possede egalement une EENS inferieure a RB1.

Premieres vies RB2 :

| Composant | Calendrier | Temps ON | EFPH | j moyen ON | j p95 |
|---|---:|---:|---:|---:|---:|
| PEMFC | 5.722 ans | 27 263 h | 20 295 h | 0.565 A/cm2 | 0.581 A/cm2 |
| PEMWE | 15.299 ans | 32 235 h | 16 220 h | 1.855 A/cm2 | 1.939 A/cm2 |

RB2 ne depasse jamais 2 A/cm2 sur la premiere vie du PEMWE. L'acceleration
fort courant penalise donc les strategies qui sollicitent reellement cette
zone, notamment RB1, et non RB2 par construction.

## Reproductibilite

- rb2_policy.py : unique definition de la regle a deux consignes ;
- sweep_setpoints_rb2.py : balayage reproductible des deux consignes ;
- diagnose_fixed_setpoints.py : EENS, LPSP, remplacements et premieres vies ;
- ../rank_base_strategies.py : classement des neuf strategies de base ;
- ../rank_base_strategies_25y.txt : resultat trie ;
- sweep_setpoints_rb2.txt, .pdf, .png : dernier raffinement local.

Les anciens scripts de plafonds, reserves SoC et variantes riches ont ete
supprimes pour eviter toute reintroduction accidentelle de cette approche.
