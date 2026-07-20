# Comparaison DP–MPC V11-p=2 sur un an

La source canonique détaillée est `AUDIT_MPC_V11_P2_2026-07-20.md`. La figure
`dp_mpc_pareto_1y.{png,pdf}` est calculée à partir des valeurs pleine précision
du front DP `dp_reference_1y_51x51_v2.npz` et du screening MPC
`screen_1y_d840744e29c7`.

Les deux ensembles utilisent le même profil annuel de 20 945,908 kWh, le même
modèle `v11-doe-rakousky-mccay-colombo-2026-07-16` et le ledger V11-p=2. Le DP
est toutefois clairvoyant sur l'année et discrétisé, tandis que le MPC ne voit
que sa fenêtre glissante de 6 h ou 24 h.

Tous les points MPC et rule-based du screening sont dominés par au moins un
point DP. Pour le MPC H24 sans SoH :

- LPSP = 0,262979 %, dégradation = 2,375768 kEUR, J3 = 2,541018 kEUR ;
- dégradation du front DP interpolée à cette LPSP = 1,743767 kEUR ;
- écart au front = +0,632001 kEUR, soit +36,24 % ;
- meilleur J3 DP échantillonné à epsilon=3 : 1,845232 kEUR ;
- écart de J3 du MPC = +37,71 %.

Le MPC H24 avec SoH donne pratiquement le même point et ne ferme pas l'écart.
La comparaison établit une marge de progression pour le MPC, pas une
incohérence : l'horizon d'information du DP est beaucoup plus long.

Le banc de prévision bruitée est analysé séparément et reste à 33/34 points ;
il ne doit pas être mélangé au front parfait dans une conclusion définitive.
