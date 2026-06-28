import numpy as np
import sympy
from timeit import default_timer as timer
from Common.Init_EMR_MG_v16_python import *
from Common.get_lol import *
from scipy.interpolate import interp1d

# ==========================================================================
#  RB2(Pred) : RB2 (NU, FIXEE) augmentee par la PREVISION OMNISCIENTE 48h.
#
#  UN SEUL LEVIER, 100% PREVISIONNEL : la PRE-CHARGE BATTERIE.
#    Si la prevision annonce un DEFICIT NET sur l'horizon proche H_PRE et que
#    le SoC est sous une cible -> on COUPE l'electrolyseur (P_ely_set=0) pour
#    que le surplus PV courant charge la BATTERIE (rendement aller-retour ~95%)
#    au lieu de partir dans la chaine H2 (ELY 0.7 x FC 0.5, tres lossy). On
#    aborde ainsi le creux a venir avec une marge de SoC -> moins de passages
#    au plancher -> moins de LPSP, sans surcout de degradation.
#
#  PROPRIETE METHODO (essentielle) : RB2(Pred) ne differe de RB2 que par une
#  FONCTION DE P_tot_ref_future. Si la prevision est NEUTRE / non informative
#  (pas de deficit prevu, ou ENABLE/USE_FORECAST = False), la strategie retombe
#  EXACTEMENT sur RB2 nu -> tout gain est ATTRIBUABLE a la prevision (test nul).
#
#  Choix valides par sweep 25 ans (VoLL=3) : declenchement LARGE (des qu'un
#  deficit net est prevu) > declenchement conditionnel a l'etat batterie ;
#  optimum H_PRE=18 h (echelle DIURNE jour/nuit), SOC_TARGET peu sensible.
#  Resultat : 85.55 -> 83.49 kEUR (-2.06, -2.4 %), gain purement LPSP
#  (2.454 -> 2.228 %), degradation inchangee.
#
#  NB : le levier "hysteresis sur le niveau H2" (zone zero-irreversible ELY)
#  explore precedemment N'EST PAS previsionnel (il depend de l'etat E_h2, pas
#  du forecast) -> il ne respecte pas la regle "retombe sur RB2 si prevision
#  neutre" et a donc ete retire de RB2(Pred). Il reste un resultat d'EMS a part
#  entiere (point de fonctionnement conscient de la degradation), a valoriser
#  separement, pas comme valeur de la prediction.
# ==========================================================================

# --- Reglages ---
ENABLE       = True     # False -> RB2 nu a l'identique (test nul)
USE_FORECAST = True     # False -> idem (pas de levier sans prevision)
H_PRE        = 18       # horizon de pre-charge [pas = h], optimum diurne
SOC_TARGET   = 0.99     # on ne pre-charge que si SoC < cette cible


def reset():
    """Conserve pour compat avec la boucle/les harnais (pas d'etat interne)."""
    pass


def _precharge(P_tot_ref_future, SoC_t):
    """True s'il faut pre-charger la batterie : un deficit net est prevu sur
    l'horizon H_PRE et le SoC a de la marge. 100% fonction de la prevision."""
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= SOC_TARGET:
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h
    return net > 0.0  # P_tot_ref>0 = deficit (charge nette > production)


def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_tot_ref_future=None) :

    ######################### RULES ##########################
    # Setpoints RB2 nu (FIXES, ancrage)
    P_fc_set  = 0.450 * FC['P_fc_max']
    P_ely_set = 0.330 * ELY['P_ely_max']

    # --- AUGMENTATION PREVISION : pre-charge batterie (seule modif vs RB2) ---
    # On coupe l'ELY pour rediriger le surplus courant vers la batterie.
    if _precharge(P_tot_ref_future, SoC_t):
        P_ely_set = 0.0
    # ------------------------------------------------------------------------

    dt_h         = LOAD['Ts'] / 3600.0
    P_fc_h2_max  = max(E_h2_t, 0.0)               / dt_h * FC['eff']  * CONV['eta'] * 1000   # [W]
    P_ely_h2_max = max(E_h2_init - E_h2_t, 0.0)   / dt_h / (ELY['eff'] * CONV['eta']) * 1000 # [W]

    if P_tot_ref_t > 0 :
        P_fc_avail = min(P_fc_set, P_fc_h2_max)
        if P_tot_ref_t > P_fc_avail :
            P_dc_fc_t  = P_fc_avail
            P_dc_bat_t = P_tot_ref_t - P_fc_avail
        else :
            P_dc_fc_t  = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_ely_t = 0
    if P_tot_ref_t < 0 :
        P_ely_avail = min(P_ely_set, P_ely_h2_max)
        if P_tot_ref_t < - P_ely_avail :
            P_dc_ely_t = - P_ely_avail
            P_dc_bat_t = P_tot_ref_t + P_ely_avail
        else :
            P_dc_ely_t = 0
            P_dc_bat_t = P_tot_ref_t
        P_dc_fc_t  = 0
    ##################################

    if 'FC' in defaillances :
        if P_tot_ref_t > 0 :
            P_dc_bat_t = P_tot_ref_t
    if 'ELY' in defaillances :
        if P_tot_ref_t < 0 :
            P_dc_bat_t = P_tot_ref_t

    action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t

    action, lol = get_lol(SoC_t,action,P_tot_ref_t,defaillances,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,SoH_bat_t)

    return action, lol
