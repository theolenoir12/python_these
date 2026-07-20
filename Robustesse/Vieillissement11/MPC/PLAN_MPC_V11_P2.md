# Plan canonique du MPC online V11-p=2

Date : 20 juillet 2026.

Statut : noyau, smoke et screening annuel validés. H24 sans SoH est retenu ; la
pondération SoH à beta=1 reste sous le seuil de matérialité. Le banc de
prévision est à 33/34 et doit être clos avant le tuning. Voir
`analysis/AUDIT_MPC_V11_P2_2026-07-20.md`.

## Statut et objectif

Cette branche doit construire un EMS MPC exécutable en ligne sur le moteur
V11, avec `p=2` figé. Les résultats de `Python/Robustesse/MPC3/` sont legacy :
ils utilisent les anciens coûts de dégradation et l'ancienne boucle forecast.
Seule leur architecture générale peut être réemployée.

L'objectif scientifique principal est une comparaison attribuable entre :

- `mpc_v11_p2_no_soh`, qui reçoit le même état physique réalisable mais ne
  module pas son objectif avec la santé ;
- `mpc_v11_p2_soh`, qui exploite le SoH et le contexte de vieillissement dans
  son objectif interne.

Les deux variantes doivent partager horizon, prévision, solveur, budget de
réglage, graines et métriques. Le test nul est `beta_fc=beta_ely=0`, qui doit
reproduire exactement `mpc_v11_p2_no_soh`.

## Références et métriques communes

- moteur : `Vieillissement11/Common/main_init_and_loop.py` ;
- modèle : `v11-doe-rakousky-mccay-colombo-2026-07-16`, avec assertion
  `PEMWE_OVERLOAD_P=2` ;
- coût rapporté : ledger V11 corrigé, plus EENS/LPSP de
  `Common/reliability_metrics.py` ;
- objectif économique central : `J = C_deg + 3 EUR/kWh * EENS` ;
- références exécutables : RB1 `(0.20, 0.40)` et RB2 `(0.574, 0.465)` ;
- borne clairvoyante : front PD V11-p=2 du job 216257, identifié comme tel et
  non comme une politique online. La bande de référence à VoLL=3 est
  `epsilon=10--50` ; elle reste à moins de 0,211 % du minimum réalisé.

Le moteur commun conserve la même physique que le front PD. Le 20 juillet, une
garde de reporting a été ajoutée à `get_lol.py` : `lol=0` lorsque le profil net
est en surplus et clipping dans `[0,1]` en déficit. La LOL n'entre pas dans la
transition d'état et la métrique de fiabilité la clippait déjà ; cette correction
ne modifie donc ni les actions, ni les états, ni EENS/LPSP des caches existants.
Le MPC contient un délestage explicite et chaque banc contrôle séparément le
déficit après LOL et l'écrêtage implicite.

## Formulation initiale proposée

Le premier noyau est un MPC déterministe mixte linéaire à horizon glissant : le
problème est résolu à chaque heure et seule la première action est appliquée.
La boucle V11 transmet `P_tot_ref_future` et `aging_context` à la politique.

Le problème interne doit rester suffisamment rapide pour être online :

- dynamique linéaire de batterie et réservoir H2, avec bornes physiques
  calculées à partir de l'état réel pour les deux variantes ;
- délestage explicite, payé par une VoLL interne réglable ;
- dommage batterie représenté par la pente moyenne V11 en throughput, avec
  pénalité convexe optionnelle de résidence à haut SoC ;
- surcharge PEMWE quadratique V11 représentée par des segments affines convexes
  en densité de courant, avec nœuds imposés à `j=1` et `j=2` ;
- états binaires exclusifs FC/ELY et charge/décharge, démarrages explicites ;
- dommage PEMFC et coûts de transition construits à partir du contexte V11,
  sans reprendre les anciennes charnières Pei de `MPC3` ;
- valeurs terminales batterie/H2 communes et test anti-arbitrage.

La fonction objectif interne est un surrogate de décision. Tous les résultats
sont évalués après exécution par le ledger V11 exact ; l'écart surrogate--coût
exact doit être rapporté.

## Échelle d'information

Développer dans cet ordre, sans cumuler les nouveautés dans le premier test :

1. prévision exacte sur horizon court, afin de valider formulation et bilan ;
2. persistance, comme null d'information prévisionnelle online ;
3. prévisions bruitées avec graines communes et le même modèle d'erreur pour
   les versions avec/sans SoH ;
4. éventuellement MPC multi-scénarios/CVaR si la version déterministe est
   trop fragile au bruit.

La prévision exacte à horizon fini reste une expérience d'information, pas la
borne PD : elle ne connaît que la fenêtre disponible à l'heure courante.

## Protocole de calcul progressif

1. tests unitaires du bilan, des bornes et du test nul sur quelques pas — terminé ;
2. smoke d'une semaine, horizons 6 h et 24 h — terminé, cache
   `runs/smoke_7d_f209c53dbce2/` ;
3. banc d'un an pour éliminer les formulations dominées et vérifier le temps
   de calcul — terminé, H24 sans SoH retenu ;
4. clôture du banc d'incertitude — 33/34, un point à reprendre ;
5. réglage apparié à budget identique des deux variantes ;
6. rejeu 25 ans uniquement pour les finalistes ;
7. robustesse au bruit/biais du SoH, aux erreurs de prévision et aux profils
   hors réglage.

Les nouveaux caches doivent être pleine précision et empreintés dans
`runs/<id>_<empreinte>/`. Aucun calcul long ne doit être lancé avant validation
des smokes et réutilisation des caches existants.

## Critère de passage à la méthode suivante

Le SoH est matériel pour le MPC si la variante santé apporte au moins quelques
pourcents sur J, avec un écart supérieur à la sensibilité numérique et sans
dégradation rédhibitoire de LPSP/EENS. Si le gain reste voisin de celui de
RB2(SoH), conserver seulement le MPC sans SoH et passer à la distillation de
règles depuis la PD, puis à la logique floue et éventuellement ANFIS.
