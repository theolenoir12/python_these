# Comparaison DP-MPC V11-p=2 sur un an

## Sources et validite

Le job DP 217276 est termine avec 19 valeurs de epsilon, une grille 51x51,
10 commandes FC, 50 commandes ELY et `n_iter=3`. Le screening MPC annuel
utilise le meme profil de 20 945,908 kWh et le meme ledger V11-p=2.

Les valeurs DP ci-dessous sont extraites du log, donc arrondies a quatre
decimales pour la LPSP et trois decimales pour la degradation. Le fichier NPZ
exact du job doit etre copie pour la figure scientifique finale.

## Resultat de dominance

Tous les points MPC et rule-based sont domines par au moins un point du front
DP. Aucun point MPC ne domine globalement le front DP ; le resultat est donc
coherent avec la reference attendue.

Pour le MPC H24 sans SoH :

- MPC : LPSP 0,262979 %, degradation 2,375768 kEUR, J3 2,541018 kEUR ;
- front DP interpole a la meme LPSP : degradation 1,743785 kEUR ;
- ecart au front : +0,631983 kEUR, soit +36,24 % relativement au DP ;
- meilleur J3 DP echantillonne : epsilon=3, LPSP 0,1262 %, degradation
  1,766 kEUR et J3 environ 1,8453 kEUR ;
- le J3 du MPC H24 est 37,70 % plus eleve que cette reference DP.

Le MPC H24 avec SoH donne pratiquement le meme resultat : son ecart de
degradation au front interpole vaut +0,633963 kEUR (+36,35 %). La couche SoH
testee ne ferme donc pas l'ecart au DP.

## Interpretation

Le gain de H24 par rapport a H6 est reel, mais l'horizon annuel du DP apporte
une valeur d'information et une anticipation terminale bien superieures. Le
MPC reste aussi pilote par un surrogate local et des couts terminaux fixes.
L'ecart observe fournit une marge de progression substantielle pour des
horizons emboites, une valeur terminale apprise du DP ou des couches de
prevision/sante mieux structurees.

## Previsions incertaines

Le job 216874 n'a produit aucun point annuel complet. Un MILP H24 a atteint la
limite de cinq secondes sous 20 processus concurrents, puis le processus parent
a interrompu le banc. Il ne faut donc tirer aucune conclusion statistique de ce
job.

La relance corrigee utilise dix workers, une limite solveur de 30 secondes, un
gap relatif de 1e-4 et conserve tous les points termines meme si une autre
trajectoire echoue. Les erreurs restent calibrees a 0,5/1/1,5 fois 39,38 kWh
sur 18 h, avec cinq graines appariees.
