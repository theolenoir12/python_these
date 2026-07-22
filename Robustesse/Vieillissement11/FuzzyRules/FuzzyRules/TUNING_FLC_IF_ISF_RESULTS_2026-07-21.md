# Clôture de la FLC experte — couches prévision IF et SoH+prévision ISF

Date : 21 juillet 2026.

## Décision

La couche de prévision `IF` est retenue avec `forecast_strength=1`, un horizon
H18 et l'hystérésis de largeur `±39,3768 kWh` sous erreur. Sur huit graines
iid représentant l'erreur du backtest LSTM, elle réduit en moyenne la LPSP de
0,126251 point et augmente le coût de dégradation de 0,776 kEUR par rapport à
la FLC I0. Le coût unifié `J3 = Cdeg + 3*EENS` baisse de 1,208 kEUR, soit
1,644 %, au-dessus du seuil matériel préannoncé de 1 %.

La combinaison `ISF` n'est pas promue. Face à IF et à bruit strictement
apparié, elle gagne 0,000732 point de LPSP mais ajoute 22,79 EUR de dégradation
et 11,29 EUR de J3 en moyenne. Elle prolonge donc marginalement le compromis
du plan à deux objectifs, sans dominer IF et sans justifier le remplacement de
la variante plus simple.

Le constructeur canonique de la stratégie retenue est
`make_selected_if_policy_v11()` dans `flc_forecast_policy_v11.py`. La branche
FLC experte est close à ce stade ; rule-learning et ANFIS restent des familles
séparées et aucun résultat ne leur est encore attribué.

## Protocole et attribution

Le protocole a été figé avant les calculs dans
`TUNING_PROTOCOL_FLC_IF_ISF_V11_P2_2026-07-21.md`. Les simulations utilisent
V11-p=2, le ledger corrigé et le profil charge/PV injecté bit-à-bit depuis
`DP/runs/dp_aging_v11_p2_25y_51x51.npz`.

La couche IF ne modifie que la commande ELY déjà demandée par le parent I0
`flc_8126e6f729c6` : si l'énergie nette prévue à H18 annonce un déficit
suffisamment confiant et que le SoC est inférieur à 0,99, la production H2
courante est réduite afin de précharger la batterie. Les règles, appartenances
et paramètres de la FLC I0 restent figés.

Le test nul `forecast_strength=0` reproduit exactement les quinze tableaux du
parent, ses métriques hors temps de calcul et son ledger. Les huit comparaisons
ISF-IF emploient les mêmes graines et les mêmes suites de bruit. L'audit final
est `PASS`.

## Screening cinq ans

Les 22 trajectoires prévues ont été calculées dans
`runs/tune_flc_if_5y_7863060e5115/`. Le parent vaut J3=14,769 kEUR. Sous erreur
iid, la réponse est monotone sur la grille préannoncée :

| Force IF | LPSP moyenne (%) | Dégradation moyenne (kEUR) | J3 moyen (kEUR) |
|---:|---:|---:|---:|
| 0,00 | 0,688755 | 12,605 | 14,769 |
| 0,25 | 0,653207 | 12,635 | 14,687 |
| 0,50 | 0,617097 | 12,670 | 14,609 |
| 0,75 | 0,584479 | 12,712 | 14,549 |
| 1,00 | 0,553090 | 12,766 | 14,503 |

La force 1 est donc sélectionnée sans raffinement post-hoc. Sous futur parfait,
elle atteint J3=14,485 kEUR, soit -1,922 % face au parent cinq ans.

## Résultats finaux sur 25 ans

Le cache canonique est
`runs/final_flc_if_isf_25y_87baec18c287/` (24/24 trajectoires). Les intervalles
entre crochets sont les IC t à 95 % de la moyenne sur les graines simulées.

| Stratégie/scénario | n | LPSP (%) | Dégradation (kEUR) | EENS (kWh) | J3 (kEUR) |
|---|---:|---:|---:|---:|---:|
| FLC I0 | 1 | 0,721251 | 62,139 | 3 776,92 | 73,470 |
| IF futur parfait | 1 | 0,530232 | 63,433 | 2 776,62 | 71,762 |
| IF erreur LSTM iid | 8 | 0,595000 [0,588116 ; 0,601884] | 62,915 [62,887 ; 62,943] | 3 115,79 [3 079,74 ; 3 151,84] | 72,262 [72,163 ; 72,362] |
| IF erreur LSTM AR(1), rho=0,8 | 4 | 0,618626 [0,612352 ; 0,624901] | 62,694 [62,648 ; 62,740] | 3 239,51 [3 206,66 ; 3 272,37] | 72,412 [72,311 ; 72,514] |
| IF persistance | 1 | 0,721676 | 62,129 | 3 779,14 | 73,466 |
| ISF erreur LSTM iid | 8 | 0,594268 [0,587495 ; 0,601041] | 62,938 [62,912 ; 62,964] | 3 111,95 [3 076,49 ; 3 147,42] | 72,274 [72,175 ; 72,373] |

Les huit réalisations iid d'IF ont un J3 inférieur au parent. La corrélation
temporelle `rho=0,8` érode le gain mais laisse encore -1,440 % de J3 face à I0.
La persistance n'apporte rien de matériel (-0,005 % de J3), ce qui confirme
que le résultat provient bien d'une information future utile et non de la seule
présence de la couche.

## Lecture dans le plan de Pareto

Le point IF-LSTM moyen ne domine pas I0 : il réduit la LPSP de 17,50 % mais
augmente la dégradation de 1,25 %. Il constitue un nouveau compromis interne à
la famille FLC.

Face aux règles de référence sur le même profil :

- IF-LSTM domine RB2, avec -25,99 % de LPSP et -0,28 % de dégradation ;
- face à RB1, il réduit la LPSP de 26,56 % mais augmente la dégradation de
  3,42 % ; son J3 est inférieur de 1,77 %.

La FLC avec prévision reste très éloignée du front PD offline. Ce front sert de
borne d'optimalité avec information complète et non de concurrent online à
budget informationnel équivalent.

## Ablation ISF-IF

Sur les huit graines appariées, `ISF-IF` vaut :

| Écart ISF-IF | Moyenne | IC95 | Nombre de gains |
|---|---:|---:|---:|
| LPSP (point) | -0,000732 | [-0,001336 ; -0,000128] | 7/8 |
| Dégradation (EUR) | +22,79 | [+20,62 ; +24,96] | 0/8 |
| EENS (kWh) | -3,83 | [-7,00 ; -0,67] | 7/8 |
| J3 (EUR) | +11,29 | [+1,85 ; +20,73] | 1/8 |

Aucune graine n'améliore simultanément LPSP et dégradation. Le petit gain de
fiabilité est donc payé par une hausse systématique du coût de vieillissement ;
à VoLL=3, le bilan scalaire est défavorable. Cela confirme, pour cette FLC et
ce mécanisme SoH précis, le rejet obtenu sur IS seule. Ce n'est pas une preuve
que le SoH est inutile pour toute règle apprise ou toute architecture ANFIS.

## Portée de la prévision

Le futur parfait donne la borne haute de valeur de l'anticipation dans ce
protocole. L'erreur gaussienne iid reprend seulement le biais et l'écart-type
du backtest LSTM H18 sur 216 origines ; l'IC quantifie la variabilité des
tirages conditionnellement à ce modèle d'erreur, pas l'incertitude sur le
modèle de prévision lui-même. La sensibilité AR(1) traite partiellement les
erreurs corrélées des fenêtres chevauchantes.

Le LSTM historique ne reçoit aucune météo exogène. Son erreur peut donc être
interprétée comme un scénario conservateur plausible si une exploitation
réelle dispose de meilleures prévisions météo, mais cette supériorité future
n'est pas démontrée ici. Une validation opérationnelle exigerait des prévisions
hors échantillon alignées sur le même profil et le même horizon.

## Artefacts canoniques

- protocole : `TUNING_PROTOCOL_FLC_IF_ISF_V11_P2_2026-07-21.md` ;
- screening : `runs/tune_flc_if_5y_7863060e5115/` ;
- résultats 25 ans : `runs/final_flc_if_isf_25y_87baec18c287/summary.json` ;
- statistiques appariées : `runs/final_flc_if_isf_25y_87baec18c287/paired_isf_minus_if.json` ;
- décision : `runs/final_flc_if_isf_25y_87baec18c287/decision.json` ;
- audit : `runs/final_flc_if_isf_25y_87baec18c287/AUDIT.md` ;
- figure : `runs/final_flc_if_isf_25y_87baec18c287/pareto_25y.png`.
