# Diagnostic de la borne de variation MPC — 20 juillet 2026

## Verdict

Le job mésocentre 218545 n'a pas complété le banc d'incertitude. Il a reproduit
l'échec du point `mpc_no_soh_h24_noisy_s1p0_r202604` après 7 680 décisions :

- profil net : 1 494,444 W ;
- SoC : 0,20001 ;
- stock H2 : 0,178820 kWh ;
- SoH batterie/FC/ELY : 0,931046 / 0,977214 / 0,995734 ;
- alpha FC/ELY : 0,065103 / 0,012489.

À cet état, la puissance minimale FC pendant une heure exige environ
0,254753 kWh de H2, au-delà du stock disponible. La FC doit donc pouvoir
s'arrêter et le délestage explicite garantit par ailleurs une solution physique.
L'infaisabilité venait de la borne de la variation au premier pas : elle était
limitée à la capacité courante, légèrement inférieure à la puissance exécutée à
l'heure précédente du fait du vieillissement. L'arrêt complet exigeait donc une
variation qui dépassait cette borne de quelques dixièmes de watt.

## Portée sur les anciens caches

Le script `check_mpc_delta_bound.py` recalcule, sans simulation, les contacts
avec cette ancienne borne. Les résultats complets sont dans
`delta_bound_screen_cache_audit.tsv` et
`delta_bound_forecast_cache_audit.tsv`.

- screening v1 : 6 trajectoires MPC, 2 575 contacts FC et 305 contacts ELY ;
- prévision v1 : 33 trajectoires MPC, 62 564 contacts FC et 944 contacts ELY.

Un contact ne prouve pas que l'optimum corrigé changera à ce pas, mais il prouve
que l'ensemble admissible v1 interdisait l'arrêt complet. Les résultats MPC v1
ne peuvent donc pas être combinés avec le point corrigé ni servir de résultats
finaux. RB1, RB2 et la PD n'utilisent pas cette formulation et restent valides.

## Correction et validation

La formulation `mpc-v11-p2-milp-v2-delta-capacity-fade-2026-07-20` borne la
variation du premier pas par le maximum entre la capacité courante et la
puissance réellement exécutée au pas précédent. Les neuf tests antérieurs et un
nouveau test de régression à l'état exact de l'échec passent ; ce dernier
vérifie que la FC s'arrête et que le délestage maintient le MILP faisable.

L'identifiant de formulation est enregistré dans les protocoles, les
trajectoires et les empreintes. La prochaine étape est un rejeu complet v2 du
screening (8 points), puis du banc de prévision (34 points). Aucun tuning ne
doit précéder leur audit.
