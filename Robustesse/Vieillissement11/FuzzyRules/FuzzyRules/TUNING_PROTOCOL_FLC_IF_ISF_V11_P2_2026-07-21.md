# Protocole FLC-IF puis FLC-ISF — prévision de puissance

Date de préannonce : 21 juillet 2026, avant toute simulation IF/ISF.

## Questions attribuables

Deux écarts sont étudiés dans cet ordre :

1. `IF-I0` : la prévision du bilan de puissance déplace-t-elle favorablement
   le point FLC dans le plan `(LPSP, coût de dégradation)` ?
2. `ISF-IF` : la couche SoH rejetée isolément crée-t-elle néanmoins une
   interaction utile avec la prévision ?

Le parent est strictement la FLC I0 `flc_8126e6f729c6`, d'empreinte
`71c0531744f2ecf0b6cde6ee97a7ed0ba0d3d2468cebca06caa75643c2bd162d`.
Ses règles, appartenances et cinq paramètres réglés restent figés.

## Couche IF

La seule information ajoutée est l'énergie nette prévue sur 18 pas horaires :

`E_net,18(t) = sum(P_load_hat - P_PV_hat) * Delta t`.

La convention du simulateur est la fenêtre de 18 valeurs
`P_net[t:t+18]`. Une valeur positive annonce un déficit énergétique. Quand ce
déficit est suffisamment confiant et que `SoC < 0,99`, la couche réduit la
commande ELY du parent d'une fraction `forecast_strength`; la batterie absorbe
le surplus courant libéré. La branche FC n'est pas modifiée.

Le déclencheur robuste reprend le mécanisme déjà documenté pour RB2(Pred) :

- entrée en pré-charge si `E_net,18 > +sigma_design` ;
- sortie si `E_net,18 < -sigma_design` ;
- maintien de l'état entre les deux seuils ;
- aucun gel temporel supplémentaire.

Le seuil vaut zéro avec futur parfait. Avec erreur LSTM,
`sigma_design=39,3768 kWh`. La force est le seul paramètre FLC-IF réglé :
`{0 ; 0,25 ; 0,50 ; 0,75 ; 1,00}`.

## Scénarios de prévision

Les scénarios restent explicitement distincts :

- `oracle` : vrai futur H18, borne théorique de valeur de l'anticipation ;
- `gaussian_iid` : énergie oracle augmentée de
  `N(-2,3172 ; 39,3768²) kWh`, paramètres du backtest LSTM sur 216 origines ;
- `gaussian_ar1_rho0p8` : même loi marginale et bruit AR(1) de corrélation 0,8,
  sensibilité aux fenêtres chevauchantes ;
- `persistence` : la puissance nette courante est répétée sur H18.

Le contrôleur ne lit que l'énergie cumulée : perturber cette grandeur est donc
suffisant et évite d'inventer des profils horaires non disponibles. Le modèle
LSTM n'utilise pas de météo exogène. Son erreur est traitée comme un scénario
conservateur plausible, à confronter plus tard à une prévision météorologique
opérationnelle ; ce statut est une hypothèse, pas une garantie.

## Tests nuls et CRN

Les contrôles obligatoires sont :

1. `forecast_strength=0` appelle directement le parent choisi, sans lire la
   prévision ni avancer le générateur aléatoire ;
2. `forecast_strength=0` et SoH nul reproduisent I0 bit-à-bit ;
3. `forecast_strength=0` avec SoH actif reproduit IS bit-à-bit ;
4. une fenêtre absente laisse le parent inchangé ;
5. `reset()` restaure l'hystérésis et le générateur à la graine initiale ;
6. IF et ISF utilisent les mêmes graines et la même suite de bruit pour les
   écarts appariés.

## Screening cinq ans

Le profil est injecté depuis le cache DP canonique, V11-p=2 et ledger corrigé.
Le budget est figé à :

- un rejeu parent dédié ;
- cinq forces sous prévision oracle, cas nul inclus ;
- quatre forces actives sous bruit iid et quatre graines communes
  `20260701--20260704` ;
- soit 22 trajectoires de cinq ans, sans raffinement post-hoc.

Le meilleur candidat actif sous bruit est le minimum de J3 moyen. Les rôles
complémentaires sont la meilleure LPSP moyenne sous +1 % de dégradation et la
plus faible dégradation sous +0,05 point de LPSP. Les points sont dédupliqués.

## Évaluation finale 25 ans

La force active sélectionnée est figée avant le rejeu long. Le budget maximal
est :

- un contrôle nul 25 ans ;
- IF oracle, IF persistance et ISF oracle ;
- huit graines iid `20260701--20260708` pour IF et ISF, appariées ;
- quatre graines `20260701--20260704` pour IF avec `rho=0,8` ;
- soit 24 trajectoires de 25 ans au maximum.

Le scénario ISF fixe `strength_FC=0` et `strength_ELY=0,025`. C'est la couche
SoH active la moins défavorable à 25 ans et la seule agissant sur la même
branche surplus que la pré-charge. Elle n'est pas reréglée après observation
de IF. L'ablation `strength_ELY=0` est exactement IF.

Le reporting donne séparément LPSP, EENS, coût de dégradation total et par
composant, remplacements, démarrages et
`J3 = coût de dégradation + 3 EUR/kWh * EENS`. Pour les scénarios aléatoires,
les écarts IF-I0 et ISF-IF sont rapportés par graine, avec moyenne, écart-type,
intervalle t à 95 % et nombre de gains. Un gain J3 de 1 % reste le seuil de
criblage ; la conclusion principale demeure le plan à deux objectifs.
