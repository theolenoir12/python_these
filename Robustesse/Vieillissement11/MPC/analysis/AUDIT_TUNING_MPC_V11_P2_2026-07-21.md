# Audit du tuning MPC H24 V11-p=2

Date : 21 juillet 2026.

## Périmètre et provenance

Le tuning concerne le MPC H24 sans pondération SoH. Le surrogate de décision
est réglé, mais tous les coûts publiés sont recalculés depuis le ledger V11
exact avec `p=2` et `VoLL=3 EUR/kWh`.

- sélection : `runs/tune_screen_1y_97e636e32db7/` ;
- validation aveugle : `runs/tune_validation_1y_9c728d3d847a/` ;
- modèle : `v11-doe-rakousky-mccay-colombo-2026-07-16` ;
- formulation : `mpc-v11-p2-milp-v2-delta-capacity-fade-2026-07-20` ;
- protocole : `mpc-v11-p2-h24-tuning-v1-2026-07-21`.

Les 48 points de validation sont complets : 40 trajectoires nouvelles et 8
baselines réutilisées. Aucun solveur n'échoue. Les 48 profils sont bit-à-bit
identiques et les métriques, coûts et J3 sont retrouvés depuis les NPZ et les
ledgers sans écart.

## Rejets physiques

Deux cas sont inadmissibles car au moins une trajectoire ne ferme pas le bilan
après une LOL bornée à 1 :

- `terminal_bat_1p2`, graine 202605 au bruit x1 : 35,873 W d'électrolyse
  pendant un délestage total ;
- `combo_top2`, graine 202605 au bruit x1 : 35,873 W d'électrolyse pendant un
  délestage total. La graine 202604 à x0,5 présente aussi 0,023 W de charge
  batterie résiduelle, numériquement négligeable mais déjà couverte par le
  rejet précédent.

La baseline et les cas `battery_wear_0p5`, `fc_wear_2` et `combo_top3` ferment
tous leurs déficits. Le post-traitement exclut désormais un cas invalide de la
décision tout en conservant une baseline invalide comme erreur bloquante.

## Validation aveugle

Le critère primaire est le J3 moyen au bruit x1 sur les graines réservées
202604 et 202605. Les variations ci-dessous sont relatives à la baseline ; une
valeur négative est un gain.

| Cas | J3 x1 moyen (kEUR) | Gain x1 | Parfait | Persistance | x0,5 | x1,5 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2,809834 | 0,000 % | 0,000 % | 0,000 % | 0,000 % | 0,000 % |
| battery_wear_0p5 | 2,783250 | 0,946 % | -2,141 % | -1,115 % | -0,467 % | -1,445 % |
| fc_wear_2 | 2,799886 | 0,354 % | +0,442 % | -2,455 % | -0,232 % | -0,826 % |
| combo_top3 | 2,769282 | **1,443 %** | -5,232 % | -1,365 % | -1,409 % | -1,658 % |

`combo_top3` est le seul cas physiquement valide qui dépasse le seuil de
promotion de 1 %. Il respecte aussi les gardes de robustesse et gagne dans les
cinq régimes de prévision. Il est donc retenu par la règle préannoncée.

Sa configuration diffère de la baseline par trois paramètres :

- `battery_wear_scale = 0,5` au lieu de 1 ;
- `terminal_bat_eur_per_kwh = 1,2` au lieu de 0,6 ;
- `fc_wear_scale = 2` au lieu de 1.

Tous les autres paramètres, informations online, prévisions et graines restent
communs. Le gain annuel est attribuable mais modeste et estimé sur seulement
deux graines réservées.

## Décision et dernière étape MPC

Le réglage `combo_top3` est promu pour la confirmation longue. Le protocole
final compare uniquement la baseline et ce réglage sur 25 ans, au bruit nominal
x1 et avec les graines réservées 202604/202605 communes, soit quatre
trajectoires. Le lanceur est `run_longrun_tuning_mpc_v11.slurm` et la sortie
attendue `runs/tune_longrun_25y_eab8dde5d5d0/`.

Le seuil de matérialité de 1 % est hérité pour lire ce dernier contrôle, mais la
conclusion 25 ans reste ouverte tant que les quatre trajectoires ne sont pas
auditées. Aucun autre tuning MPC n'est prévu avant ce résultat.
