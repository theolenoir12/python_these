# -*- coding: utf-8 -*-
"""
check_rul_fix.py -- test de NON-REGRESSION du fix d'ancre RUL (cf README sect. 4).
===================================================================================
Bug corrige : l'extrapolation lineaire de la RUL dans les boucles Communes etait
ancree sur SoH[j_new] (valeur EoL ~0.9 de l'ANCIENNE unite) au lieu de SoH=1 de
l'unite neuve -> RUL figee a sa valeur par defaut (8000/3000 j) pour toutes les
unites APRES le premier remplacement.

Principe du test (rapide, ~1 min, n'importe quel profil de donnees) :
  - on ACCELERE artificiellement l'usure FC (FC['cost'] /= ACCEL) pour provoquer
    plusieurs remplacements FC en quelques mois simules ;
  - une strategie "espionne" enregistre la RUL_fc vue a chaque pas puis delegue
    a RB2 socle ;
  - VERDICT : il doit exister des valeurs de RUL NON-DEFAUT (< 8000 j) APRES le
    2e remplacement. Avant le fix : impossible (tout reste a 8000).

Usage :  python check_rul_fix.py       (depuis Fable/, env avec donnees dispo)
"""
import os, sys
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.abspath(os.path.join(HERE, "..", "Prédictions"))
for _p in (HERE, PRED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Common.Init_EMR_MG_v16_python import FC
from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
import importlib.util

ACCEL   = 150.0   # acceleration des TAUX de degradation FC (assez pour >=3
                  # remplacements meme sur profil doux ; vie >> 20 h = min RUL)
N_YEARS = 0.5     # ~4380 h simulees -> plusieurs remplacements FC

# Usure FC acceleree : multiplier les TAUX (les coefficients alpha_*). NB : ne
# PAS passer par FC['cost'] -- il apparait au numerateur (via get_cost_fc) ET
# au denominateur du SoH, donc se simplifie et n'accelere rien.
import Common.cost_fcn_total2 as _cf
_cf.FC_ALPHA_ON_OFF *= ACCEL
_cf.FC_ALPHA_HIGH   *= ACCEL
_cf.FC_ALPHA_LOW    *= ACCEL
_cf.FC_ALPHA_SHIFT  *= ACCEL

# Strategie de base = RB2(Prop) avec levier OFF (== RB2 socle exact)
spec = importlib.util.spec_from_file_location(
    "strat_base", os.path.join(HERE, "RB2(Prop)", "get_optimal_action_RB.py"))
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)
_base.ENABLE = False

rul_log = []   # RUL_fc vue par la strategie a chaque pas


def spy(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
        RUL_ely_t, SoH_fc_t, SoH_ely_t, P_tot_ref_future=None):
    rul_log.append(RUL_fc_t)
    return _base.get_optimal_action_RB(
        SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
        SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t, RUL_fc_t,
        RUL_ely_t, SoH_fc_t, SoH_ely_t, P_tot_ref_future)


def main():
    print(f"[check_rul_fix] simulation {N_YEARS} an(s), usure FC x{ACCEL:.0f} ...")
    data = init_and_run_loop_forecast(spy, H_forecast=48, n_years=N_YEARS)

    soh = data["SoH_fc"]
    # Indices de remplacement = SAUT du SoH vers le haut (retour a 1 depuis ~EoL).
    # (Ne pas tester soh==1 : le plateau initial a SoH=1 serait compte a tort.)
    resets = np.flatnonzero(np.diff(soh) > 0.05)  # index j du pas de remplacement
    rul = np.array(rul_log)
    print(f"  remplacements FC detectes : {len(resets)} (pas {list(resets[:6])} ...)")
    if len(resets) < 2:
        print("  [INCONCLUSIF] moins de 2 remplacements : augmenter ACCEL ou N_YEARS.")
        sys.exit(2)

    j2 = int(resets[1])  # apres le 2e remplacement, le bug d'origine garantissait RUL=defaut
    seg = rul[j2 + 1:]
    n_nondefault = int(np.sum(seg < 7999.0))
    print(f"  pas apres le 2e remplacement : {len(seg)} ; RUL non-defaut : {n_nondefault}")
    if n_nondefault > 0:
        print(f"  exemples RUL estimees [j] : {np.round(seg[seg < 7999.0][:5], 1)}")
        print("[check_rul_fix] OK -- la RUL est bien re-estimee apres remplacement (fix actif).")
    else:
        print("[check_rul_fix] ECHEC -- RUL figee au defaut apres remplacement (bug present).")
        sys.exit(1)


if __name__ == "__main__":
    main()
