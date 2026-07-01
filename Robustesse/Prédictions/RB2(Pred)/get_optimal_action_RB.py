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

# --- Bruit de prevision (incertitude REALISTE sur l'energie nette prevue) -----
# Jusqu'ici la prevision etait OMNISCIENTE (P_tot_ref_future = vrai futur). On
# degrade la grandeur reellement utilisee pour decider -- l'ENERGIE NETTE cumulee
# sur l'horizon de decision H_PRE -- par un bruit gaussien dont les parametres
# (biais, sigma) viennent du backtest empirique des previsions LSTM
# (Predictions profils/pv_profils_backtest_h18.py), MESURES A L'HORIZON H_PRE=18h :
#     biais ~ -2.3 kWh (negligeable) | sigma ~ 39.4 kWh.
# L'ecart caracterise est "energie predite - energie reelle" -> la prevision
# bruitee est : net_pred = net_vrai + N(biais, sigma).
# La pre-charge ne lisant que le SIGNE de net, le bruit ne change la decision
# qu'au voisinage du seuil (net ~ 0), ce qui est le comportement physique attendu.
#
# NOISE_ENABLE = False -> on retombe EXACTEMENT sur l'omniscient (test nul preserve).
# Evaluation Monte-Carlo : tirages INDEPENDANTS a chaque pas ; set_noise_seed(s)
# reseede le generateur avant chaque run pour des realisations reproductibles.
NOISE_ENABLE = True
BIAS_E_KWH   = -2.32    # biais du backtest a 18h [kWh]
SIGMA_E_KWH  = 39.38    # ecart-type du backtest a 18h [kWh] (valeur de DESIGN)
# Sigma du bruit REELLEMENT INJECTE. None -> = SIGMA_E_KWH (cas nominal). Le
# DECOUPLER de SIGMA_E_KWH permet de tester la robustesse a une MISESTIMATION de
# sigma : la bande d'hysteresis reste calee sur SIGMA_E_KWH (design fige) tandis
# que le vrai bruit varie (cf. sens_pred_noise.py, ellipses de sensibilite).
SIGMA_INJECT_KWH = None

# --- Anti-clignotement (robustesse au bruit) ---------------------------------
# Le bruit fait basculer la decision binaire net>0 d'un pas a l'autre pres du
# seuil -> l'ELY clignote (marche/arret) -> degradation start-stop. On stabilise
# la decision par une HYSTERESIS a deux seuils sur l'energie nette prevue :
#   - on ENTRE en pre-charge (ELY coupe) si net_pred > +M_SIGMA*sigma  (deficit confiant)
#   - on SORT (ELY rallume)            si net_pred < -M_SIGMA*sigma  (surplus confiant)
#   - entre les deux : on GARDE l'etat courant (zone morte = rejet du bruit).
# Une duree minimale de maintien MIN_DWELL [pas/h] gele l'etat apres chaque
# bascule (borne la frequence de clignotement). La bande +-M_SIGMA*sigma utilise
# directement le sigma mesure : large bande = plus robuste mais moins reactif.
#
# HYST_ENABLE=False -> decision binaire net>0 d'origine (omniscient 83.49 /
# bruite 85.88 preserves). M_SIGMA=0 et MIN_DWELL=0 redonnent aussi le binaire.
# Defaut PRODUCTION (robuste au bruit) : optimum du sweep Monte-Carlo 25 ans,
# M_SIGMA=1.0 / MIN_DWELL=12 -> 84.11 kEUR sous bruit realiste (+1.44 vs RB2 nu),
# contre 85.88 pour la decision binaire fragile. C'est la BANDE DE MARGE (+-sigma)
# qui apporte la robustesse ; le maintien 12h (~demi-journee diurne) n'aide qu'a
# la marge. HYST_ENABLE=False -> binaire net>0 (omniscient 83.49 / bruite 85.88).
HYST_ENABLE = True
# --- Socle RB2 : RE-OPTIMISE sur le cost-min (0.440/0.310) au lieu du nominal
#     0.450/0.330 (comparaison honnete best-vs-best). Parametrable pour le sweep.
C_FC_BASE   = 0.440     # etait 0.450 (nominal) -> 0.440 (cost-min RB2)
C_ELY_BASE  = 0.310     # etait 0.330 (nominal) -> 0.310 (cost-min RB2)
GAMMA_FC    = 0.0       # exposant SoH_fc (0 = RB2(Pred) ; 1 = RB2(SoH+Pred))
GAMMA_ELY   = 0.0       # exposant SoH_ely (0 = RB2(Pred) ; 2 = RB2(SoH+Pred))
M_SIGMA     = 1.0       # demi-largeur de bande = M_SIGMA * sigma [-]
MIN_DWELL   = 12        # duree minimale de maintien d'un etat [pas/h]

_rng      = np.random.default_rng(0)
_state_on = False       # etat courant de la pre-charge (ELY coupe ?)
_dwell    = 0           # compteur de maintien restant [pas]


def set_noise_seed(seed):
    """(Re)seede le generateur du bruit de prevision. A appeler avant chaque run
    Monte-Carlo pour une realisation independante et reproductible."""
    global _rng
    _rng = np.random.default_rng(seed)


def reset():
    """Reinitialise l'etat de l'hysteresis. A APPELER avant chaque run (les
    workers d'un pool sont reutilises -> sinon l'etat fuit d'un run a l'autre)."""
    global _state_on, _dwell
    _state_on = False
    _dwell    = 0


def _precharge(P_tot_ref_future, SoC_t):
    """True s'il faut pre-charger la batterie : un deficit net est prevu sur
    l'horizon H_PRE et le SoC a de la marge. 100% fonction de la prevision."""
    global _state_on, _dwell
    if not (ENABLE and USE_FORECAST):
        return False
    if P_tot_ref_future is None or len(P_tot_ref_future) == 0:
        return False
    if SoC_t >= SOC_TARGET:
        return False
    dt_h = LOAD['Ts'] / 3600.0
    net = float(np.sum(np.asarray(P_tot_ref_future[:H_PRE], dtype=float))) * dt_h  # [Wh]
    # Bruit de prevision : net_pred = net_vrai + N(biais, sigma_inject) (kWh -> Wh).
    if NOISE_ENABLE:
        sig_inj = SIGMA_E_KWH if SIGMA_INJECT_KWH is None else SIGMA_INJECT_KWH
        net += (BIAS_E_KWH + sig_inj * _rng.standard_normal()) * 1000.0

    if not HYST_ENABLE:
        return net > 0.0  # decision binaire d'origine (P_tot_ref>0 = deficit)

    # --- Hysteresis a deux seuils + maintien minimal -------------------------
    th = M_SIGMA * SIGMA_E_KWH * 1000.0  # demi-largeur de bande [Wh]
    if _dwell > 0:
        _dwell -= 1                       # etat gele
    elif (not _state_on) and net > th:
        _state_on = True;  _dwell = MIN_DWELL
    elif _state_on and net < -th:
        _state_on = False; _dwell = MIN_DWELL
    return _state_on


def get_optimal_action_RB(SoC_t,P_tot_ref_t,defaillances,lol_tab,alpha_fc_t,alpha_ely_t,SoH_bat_t,E_h2_t,E_h2_init,P_fc_max_t,P_ely_max_t,RUL_fc_t,RUL_ely_t,SoH_fc_t,SoH_ely_t,P_tot_ref_future=None) :

    ######################### RULES ##########################
    # Setpoints RB2 nu (FIXES, ancrage)
    P_fc_set  = C_FC_BASE * FC['P_fc_max'] * SoH_fc_t ** GAMMA_FC
    P_ely_set = C_ELY_BASE * ELY['P_ely_max'] * SoH_ely_t ** GAMMA_ELY

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
