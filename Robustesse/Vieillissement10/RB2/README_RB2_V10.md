# RB2 - Vieillissement10

## Regle active

RB2 conserve des consignes fixes en regime normal et ajoute un secours de
faisabilite lorsque la batterie atteindrait une borne physique de SoC.

Parametres retenus sur 25 ans :

- PEMFC normale : 0.31 Pmax ;
- PEMWE normal : 0.22 Pmax ;
- plafond PEMFC de secours : 0.90 Pmax ;
- plafond PEMWE de secours : 0.225 Pmax ;
- bornes de reserve : SoC 0.20 et 0.995.

Le secours ne remplace que le reliquat que la batterie ne peut fournir ou
absorber. Les disponibilites de puissance, le niveau du reservoir H2 et les
defaillances restent imposes par le referee physique commun. Aucun SoH de la
PEMFC ou du PEMWE n'entre dans la decision.

## Selection par cout unifie

Les parametres ont ete choisis avec le meme cout unifie que les balayages EMS :
cout de degradation plus valorisation de la LPSP. Aucune contrainte LPSP n'a
ete ajoutee.

Point final RB2 sur 25 ans :

- LPSP : 1.7978 % ;
- degradation : 67.284 kEUR ;
- cout unifie de balayage : 82.029 kEUR.

Reference du batch avec le meme modele :

- RB1-costopt-V8 : LPSP 1.7204 %, degradation 68.621 kEUR.

RB2 est donc non dominee : elle accepte 0.0774 point de LPSP supplementaire
pour reduire la degradation de 1.337 kEUR. Son cout unifie est plus faible
dans la ponderation retenue.

## Reproductibilite

- sweep_reserve_rb2.py : raffinement des plafonds de secours ;
- sweep_base_rb2.py : raffinement des consignes normales ;
- sweep_soc_reserve_rb2.py : test negatif d'une anticipation de la reserve
  haute ; le declenchement a 0.995 reste preferable ;
- sweep_*.txt : resultats complets tries par cout unifie.

RB2(SoH) n'est pas modifiee dans cette etape.
