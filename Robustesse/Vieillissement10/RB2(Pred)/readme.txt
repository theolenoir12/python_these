RB2(Pred)
=========

La prevision est la puissance nette future charge-PV. Si son energie cumulee
annonce un deficit et que le SoC est sous la cible, la consigne ELY devient
nulle : le surplus present est conserve dans la batterie. La prevision ne cree
aucun plafond de puissance. Sans prevision, la strategie retombe sur RB2.

Le bruit agrege a 18 h reprend le backtest historique (biais -2.32 kWh,
sigma 39.38 kWh), avec hysteresis et temps de maintien contre le clignotement.

Optimisation : python ../optimize_rb2_augmentations.py --layer pred

Optimum V10 moyen sur trois graines : H=24 h, SoC_cible=0.99, bande=1.5 sigma,
maintien minimal=0 h. Cout unifie moyen 75.3940 kEUR (sigma 0.0614), contre
75.5501 kEUR pour RB2.

Limite : le biais et le sigma injectes proviennent du backtest historique a
18 h. Comme l'optimum de commande est a 24 h, ces deux statistiques devront
etre recalculees a 24 h avant de qualifier definitivement le resultat Pred.
