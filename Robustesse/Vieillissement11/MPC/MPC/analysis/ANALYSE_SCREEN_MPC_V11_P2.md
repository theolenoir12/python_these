# Analyse du screening MPC V11-p=2

Source : /Work/Users/tlenoir/genial/Robustesse/Vieillissement11/MPC/runs/screen_1y_d840744e29c7.

## Validite

Les 8 points utilisent le meme profil sur 1 an(s),
le modele v11-doe-rakousky-mccay-colombo-2026-07-16 et une prevision parfaite. Cette analyse ne
doit pas etre superposee a un front DP de 25 ans.

## Resultats

- Passage H6 vers H24 sans SoH : variation de J3 = -5.231 %.
- Effet SoH FC+ELY a H24 : variation de J3 = +0.062 %.

## Front non domine du screening

- MPC H24 sans SoH
- MPC H6 SoH-ELY
- MPC H6 SoH-FC+ELY
- MPC H24 SoH-FC+ELY

La comparaison scientifique avec le DP doit etre effectuee avec
`compare_mpc_dp_v11.py` sur un horizon strictement identique.
