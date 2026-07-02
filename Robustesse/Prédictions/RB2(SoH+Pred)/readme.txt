================================================================================
ATTENTION NOMMAGE (note du 2026-07-02, revue de coherence)
================================================================================
Ce dossier RB2(SoH+Pred)/ contient la variante HISTORIQUE de la pre-charge sur
base RB2(SoH) : decision BINAIRE (net>0), SANS bruit de prevision et SANS
hysteresis. Elle a servi aux premiers sweeps (H_PRE, SOC_TARGET) mais n'est PAS
la version presentee (le binaire s'effondre sous bruit reel, cf.
../robustesse_bruit_prevision.txt).

La version de PRODUCTION (bruit backtest + hysteresis M_SIGMA=1.0/MIN_DWELL=12),
celle qui alimente les figures et les points de Pareto "RB2(SoH+Pred)", vit dans
le dossier ../RB2(SoH)/ (cf. son readme.txt). Oui, c'est l'inverse de ce que les
noms suggerent -- a renommer un jour de calme, en mettant a jour les chemins de
sens_pred_noise.py, mc_soh_pred.py et des scripts de figures qui pointent vers
"RB2(SoH)".
================================================================================
