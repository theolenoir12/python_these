"""
robustesse_common.py -- harness pour l'etude de robustesse (comportement sous
defaillance) des EMS du micro-reseau H2.
=======================================================================
SOURCE 100% ASCII (volontaire, cf. sens_common.py) : evite tout mojibake si
l'interpreteur decode le .py en latin-1.

Ce module N'ECRIT JAMAIS dans Vieillissement8 : il importe le code de base EN
LECTURE SEULE (chemin ABSOLU/relatif), exactement comme Analyse_sensibilite.

------------------------------------------------------------------------
METHODOLOGIE (validee avec l'auteur, juin 2026)
------------------------------------------------------------------------
1. REGIME PERMANENT. On simule UNE seule fois la strategie de reference RB2 sur
   `YEARS_BASELINE` ans (defaut 2). Cette trajectoire fournit, a chaque heure j,
   l'etat complet du systeme : SoC, E_h2, SoH_bat/fc/ely, alpha_fc/ely. Le
   premier mois (`SETTLE_HOURS`) sert a etablir le regime permanent et n'est PAS
   utilise comme instant de panne.

2. DEFAILLANCE. On considere les composants hydrogene (PEMFC / PEMWE), bien plus
   sujets aux pannes que les batteries (MTBF plus faibles). Une panne survient a
   un instant t0 tire aleatoirement dans les mois 2->24 et dure 1 semaine
   (`WEEK_HOURS`) avant reparation. Deux severites :
     - 'total' : composant a 0 % (ajout de 'FC'/'ELY' a `defaillances`).
     - '50'    : composant a 50 % de puissance (P_max derate x0.5).

3. BRANCHE. A t0 on REPART du snapshot RB2 (SoC, E_h2, SoH...) et on simule la
   SEULE semaine de panne, en gelant les SoH/alpha (degradation negligeable sur
   1 semaine). Pendant cette semaine on applique une STRATEGIE DE REACTION
   candidate (RB2, RB1, 100-0, 50-50, ...) : on mesure ainsi comment chaque EMS
   s'adapte a la panne, a regime permanent identique.

4. METRIQUE. LPSP (Loss of Power Supply Probability) sur la semaine de panne,
   calculee EXACTEMENT comme sens_common.metrics / batch_pareto (memes clips).
   On renvoie aussi l'energie non servie [kWh].

5. MONTE-CARLO. On tire `N_DRAWS` instants t0 (memes tirages pour TOUTES les
   strategies/scenarios -> comparaison APPARIEE) puis etude statistique
   (distribution de LPSP, meilleure strategie par scenario).

NB physique : `get_lol` (le "referee") plafonne les puissances FC/ELY a leur
P_max effectif mais ne re-route PAS automatiquement vers la batterie l'exces
commande au-dela du plafond -> ce surplus devient de l'energie non servie. La
robustesse mesuree reflete donc la LOGIQUE PROPRE de chaque strategie (une
strategie qui sur-sollicite le composant en panne est penalisee), ce qui est
precisement l'objet de la comparaison.
"""
import os
import sys
import time
import importlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- Chemin vers le code de base (LECTURE SEULE), PORTABLE Win/Linux ----------
HERE = os.path.dirname(os.path.abspath(__file__))
_VIEIL8_REL   = os.path.normpath(os.path.join(HERE, os.pardir, "Vieillissement8"))
_VIEIL8_LINUX = "/home/theo/Documents/Doctorat/GENIAL/Python/Robustesse/Vieillissement8"
VIEIL8 = _VIEIL8_REL if os.path.isdir(_VIEIL8_REL) else _VIEIL8_LINUX
if VIEIL8 not in sys.path:
    sys.path.insert(0, VIEIL8)

from Common import Init_EMR_MG_v16_python as I            # noqa: E402
from Common.simulate_transition import simulate_transition  # noqa: E402
from Common.get_lol import get_lol                          # noqa: E402

eta = I.CONV["eta"]

# --- Dossier de sortie --------------------------------------------------------
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# PARAMETRES DE L'ETUDE (modifiables ; run_robustesse.py peut les surcharger)
# =============================================================================
YEARS_BASELINE = 2.0                       # horizon de la trajectoire RB2
SETTLE_HOURS   = 730                        # ~1 mois : pas d'instant de panne avant
WEEK_HOURS     = 7 * 24                     # duree d'une panne (1 semaine)
# Fenetre sur laquelle on mesure la LPSP. Par defaut = la semaine de panne
# (specification initiale). La mettre a un multiple de WEEK_HOURS permet de
# capter la REPRISE post-reparation (utile pour les pannes ELY, dont le cout
# reel -- reservoir H2 vide -> famine FC -- tombe APRES la semaine de panne).
# Le composant est en panne pendant WEEK_HOURS puis repare ; la mesure continue.
EVAL_HOURS     = 4 * WEEK_HOURS            # panne 1 sem + 3 sem de reprise
E_H2_INIT      = 200.0                      # [kWh] reservoir initial (== main loop)
PLOT_FLAG      = 1                          # arg `plot` de simulate_transition (inerte)

# Constantes RUL passees aux strategies (gelees ; ignorees par les strategies
# rule-based candidates -- seules RB2(RUL)/(SoH), exclues, les utiliseraient).
RUL_FC_DEFAULT  = 8000.0
RUL_ELY_DEFAULT = 3000.0

# Severite -> facteur de derating de la puissance max du composant en panne.
SEVERITY_DERATE = {"total": 0.0, "50": 0.5}

# Scenarios de defaillance : cle -> (composant, severite)
SCENARIOS = {
    "FC_total":  ("FC",  "total"),
    "FC_50":     ("FC",  "50"),
    "ELY_total": ("ELY", "total"),
    "ELY_50":    ("ELY", "50"),
}

# Strategies candidates : EXACTEMENT celles du nuage de Pareto
# (Vieillissement8/Pareto_2d_25y.py), a l'exception de RB2(vieillissement) =
# RB2(SoH)/RB2(RUL) (hors scope). SoC09 n'en fait pas partie -> exclue.
DEFAULT_STRATEGIES = [
    "0-100",
    "25-75",
    "50-50",
    "75-25",
    "100-0",
    "RB2",
    "RB1",
    "SoC1",
    "SoC06",
]

# Etiquettes des figures = noms reels des strategies (Full-H2/Full-bat obsoletes).
STRATEGY_LABELS = {s: s for s in DEFAULT_STRATEGIES}
SCENARIO_LABELS = {
    "FC_total": "PEMFC totale", "FC_50": "PEMFC 50%",
    "ELY_total": "PEMWE totale", "ELY_50": "PEMWE 50%",
}

# Fichiers de cache (partages entre process workers)
BASELINE_CACHE = os.path.join(RESULTS_DIR, "baseline_rb2_%gy.npz" % YEARS_BASELINE)
MC_SETUP_CACHE = os.path.join(RESULTS_DIR, "mc_setup.npz")

Ts_h = I.LOAD["Ts"] / 3600.0


# =============================================================================
# CHARGEMENT DYNAMIQUE D'UNE STRATEGIE (cf. sens_common.load_strategy)
# =============================================================================
STRATEGY_FILENAME  = "get_optimal_action_RB"
STRATEGY_FUNC_NAME = "get_optimal_action_RB"


def load_strategy(folder_name):
    """Importe get_optimal_action_RB depuis VIEIL8/<folder_name>. La fonction
    renvoyee conserve ses propres globals (Common.get_lol, FC, ...) ; on peut
    donc charger plusieurs strategies homonymes dans le meme process et garder
    chaque fonction independamment."""
    folder_path = os.path.join(VIEIL8, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError("Strategie introuvable : %s" % folder_path)
    if folder_path in sys.path:
        sys.path.remove(folder_path)
    sys.path.insert(0, folder_path)
    sys.modules.pop(STRATEGY_FILENAME, None)
    module = importlib.import_module(STRATEGY_FILENAME)
    return getattr(module, STRATEGY_FUNC_NAME)


# =============================================================================
# PUISSANCES MAX EN FONCTION DE alpha (== formules de main_init_and_loop)
# =============================================================================
def p_fc_max_of_alpha(a):
    i = -234.8032 * a + 238.8252
    return (i * I.FC["n_parallel"] * I.FC["n_series"]
            * (I.FC["E_0"] - I.FC["R"] * (1 + a) * i / I.FC["n_parallel"]
               - I.A * I.FC["T"] * np.log((i / I.S / I.FC["n_parallel"] + I.j_in) / I.FC["j_0"])
               - I.B * I.FC["T"] * np.log(1 - i / I.S / I.FC["n_parallel"] / I.FC["j_L"] / (1 - a))))


def p_ely_max_of_alpha(a):
    i = -732.6 * a + 732.6
    return (i * I.ELY["n_parallel"] * I.ELY["n_series"]
            * (I.ELY["E_0"] + I.ELY["R"] * (1 + a) * i / I.ELY["n_parallel"]
               + I.A * I.ELY["T"] * np.log((i / I.S / I.ELY["n_parallel"] + I.j_in) / I.ELY["j_0"])
               + I.B * I.ELY["T"] * np.log(1 - i / I.S / I.ELY["n_parallel"] / I.ELY["j_L"] / (1 - a))))


# =============================================================================
# BASELINE RB2 (regime permanent) -- une seule fois, mis en cache
# =============================================================================
def run_baseline_rb2(years=YEARS_BASELINE, cache=True, verbose=True):
    """Trajectoire RB2 nominale sur `years` ans. Renvoie un dict d'arrays
    (SoC, E_h2, SoH_*, alpha_*) indexes par l'heure j. Reutilise la physique
    EXACTE de Common.main_init_and_loop via un patch temporaire de l'horizon
    (T = SIM['Tend'] * 25 dans le code de base)."""
    path = os.path.join(RESULTS_DIR, "baseline_rb2_%gy.npz" % years)
    if cache and os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}

    import Common.main_init_and_loop as M
    rb2 = load_strategy("RB2")
    one_year = 3600 * 24 * 365
    saved_sim = M.SIM
    if verbose:
        print("[baseline] simulation RB2 sur %g ans ..." % years, flush=True)
    t0 = time.time()
    try:
        M.SIM = dict(I.SIM)
        M.SIM["Tend"] = (years / 25.0) * one_year   # T = Tend * 25
        data = M.init_and_run_loop(rb2)
    finally:
        M.SIM = saved_sim
    keep = ("temps", "SoC", "E_h2", "alpha_fc", "alpha_ely",
            "SoH_bat", "SoH_fc", "SoH_ely")
    out = {k: np.asarray(data[k], dtype=float) for k in keep}
    if verbose:
        print("[baseline] %d pas en %.1fs (SoH_bat fin = %.3f)"
              % (len(out["temps"]), time.time() - t0, out["SoH_bat"][-2]), flush=True)
    if cache:
        np.savez_compressed(path, **out)
    return out


# =============================================================================
# TIRAGE DES INSTANTS DE PANNE (memes tirages pour toutes les strategies)
# =============================================================================
def sample_failure_times(n_baseline_steps, n_draws, seed=0):
    """Instants de panne (en heures) tires UNIFORMEMENT dans la fenetre
    [SETTLE_HOURS, n_baseline_steps - WEEK_HOURS) : apres le regime permanent et
    en laissant 1 semaine complete avant la fin de trajectoire."""
    lo = SETTLE_HOURS
    hi = n_baseline_steps - EVAL_HOURS
    if hi <= lo:
        raise ValueError("Trajectoire trop courte pour la fenetre de panne.")
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n_draws, dtype=np.int64)


# =============================================================================
# SIMULATION D'UNE SEMAINE (branche depuis le snapshot RB2)
# =============================================================================
#
# DISPATCHER CONSCIENT DE LA PANNE (panne DETECTEE par l'EMS, cf. discussion)
# --------------------------------------------------------------------------
# A chaque pas : la strategie calcule son INTENTION normale (appelee sans
# drapeau de panne et avec les P_max sains) ; on PLAFONNE ensuite le composant
# defaillant a sa capacite disponible (derate x P_max) et on REROUTE le manque
# vers la batterie ; enfin le refere `Common.get_lol` (code de base, REUTILISE
# tel quel) tranche : ecretage SoC + gestion du reservoir H2 -> lol.
#
# Cette construction garantit la MONOTONIE PHYSIQUE : plus la capacite restante
# est grande, moins la batterie est sollicitee, donc
#     LPSP(50 %)  <=  LPSP(totale)
# et la marche normale (derate = 1) est le meilleur cas. On passe les MEMES
# P_max sains a get_lol (le plafonnement panne est deja applique en amont), si
# bien que get_lol ne re-ecrete pas le composant : il ne fait que l'ecretage
# SoC/H2, identique a la marche de base.
def _run_week(strat_func, baseline, t0, fc_derate, ely_derate):
    """Simule la fenetre [t0, t0+EVAL_HOURS) et renvoie (LPSP %, ENS kWh) sur
    cette fenetre. Le composant est DERATE (fc_derate/ely_derate, 1.0 = sain)
    pendant les WEEK_HOURS premieres heures (la panne) puis REPARE (derate=1)
    jusqu'a EVAL_HOURS -> capte la reprise. fc_derate=ely_derate=1.0 sur toute
    la fenetre = contrefactuel "memes heures SANS panne"."""
    SoC_t   = float(baseline["SoC"][t0])
    E_h2_t  = float(baseline["E_h2"][t0])
    a_fc    = float(baseline["alpha_fc"][t0])
    a_ely   = float(baseline["alpha_ely"][t0])
    SoH_bat = float(baseline["SoH_bat"][t0])
    SoH_fc  = float(baseline["SoH_fc"][t0])
    SoH_ely = float(baseline["SoH_ely"][t0])

    P_fc_max0  = p_fc_max_of_alpha(a_fc)     # P_max sains (vieillissement gele)
    P_ely_max0 = p_ely_max_of_alpha(a_ely)

    P_ref = I.LOAD["P_ref"]
    P_pv  = I.PV["P"]
    lol_dummy = np.zeros(1)                    # lol_tab : jamais indexe par les strategies

    planned = 0.0
    unserved = 0.0
    for k in range(EVAL_HOURS):
        j = t0 + k
        # Panne active seulement pendant la semaine, puis reparation
        fc_d  = fc_derate  if k < WEEK_HOURS else 1.0
        ely_d = ely_derate if k < WEEK_HOURS else 1.0
        fc_cap  = fc_d  * P_fc_max0
        ely_cap = ely_d * P_ely_max0

        P_dc_load_t = P_ref[j] / eta
        P_dc_pv_t   = P_pv[j]
        P_tot_ref_t = P_dc_load_t - P_dc_pv_t

        # 1. Intention NORMALE de la strategie (sans panne, P_max sains)
        action_nom, _ = strat_func(
            SoC_t, P_tot_ref_t, [], lol_dummy, a_fc, a_ely, SoH_bat,
            E_h2_t, E_H2_INIT, P_fc_max0, P_ely_max0,
            RUL_FC_DEFAULT, RUL_ELY_DEFAULT, SoH_fc, SoH_ely)

        # 2. Plafonnement panne (DETECTEE) + reroutage du manque vers la batterie.
        #    FC (deficit) : ne fournit que fc_cap ; ELY (surplus) : n'absorbe que ely_cap.
        P_dc_fc_t  = min(action_nom[1], fc_cap  * eta * 0.999) if action_nom[1] > 0 else 0.0
        P_dc_ely_t = max(action_nom[2], -ely_cap / eta * 0.999) if action_nom[2] < 0 else 0.0
        P_dc_bat_t = P_tot_ref_t - P_dc_fc_t - P_dc_ely_t        # batterie comble (bilan)

        # 3. Refere physique du code de base (ecretage SoC + reservoir H2).
        #    P_max sains passes ici : le composant n'est PAS re-ecrete par get_lol.
        action, lol = get_lol(
            SoC_t, (P_dc_bat_t, P_dc_fc_t, P_dc_ely_t), P_tot_ref_t, [],
            E_h2_t, E_H2_INIT, P_fc_max0, P_ely_max0, SoH_bat)

        # 4. Propagation de l'etat (SoC, reservoir)
        SoC_tp1, simOut = simulate_transition(
            SoC_t, action, P_tot_ref_t, PLOT_FLAG, lol, a_fc, a_ely, SoH_bat,
            E_h2_t, E_H2_INIT, P_fc_max0, P_ely_max0)
        if simOut:
            SoC_t = SoC_tp1
            E_h2_t = simOut["E_h2_tp1"]
        else:                                   # garde-fou (ne devrait pas arriver)
            SoC_t = min(max(SoC_t, 0.2), 0.995)

        net = P_dc_load_t - P_dc_pv_t           # charge nette [W]
        p = max(net, 0.0) / 1000.0              # kW planifie (deficit seul)
        r = max(net * (1.0 - lol), 0.0) / 1000.0
        planned  += p
        unserved += max(p - r, 0.0)

    lpsp = (unserved / planned * 100.0) if planned > 0 else 0.0
    ens  = unserved * Ts_h                       # kWh non servis sur la semaine
    return float(lpsp), float(ens)


def derates_of(component, severity):
    """(fc_derate, ely_derate) pour un scenario donne (1.0 = sain)."""
    d = SEVERITY_DERATE[severity]
    if component == "FC":
        return d, 1.0
    if component == "ELY":
        return 1.0, d
    raise ValueError("composant inconnu : %r" % component)


def simulate_failure_week(strat_func, baseline, t0, component, severity):
    """LPSP/ENS de la semaine [t0, t0+WEEK_HOURS) AVEC la panne."""
    fc_d, ely_d = derates_of(component, severity)
    return _run_week(strat_func, baseline, t0, fc_d, ely_d)


def simulate_nominal_week(strat_func, baseline, t0):
    """Contrefactuel : meme semaine, MEME strategie, SANS panne (derate = 1)."""
    return _run_week(strat_func, baseline, t0, 1.0, 1.0)


# =============================================================================
# WORKER MULTIPROCESS : une tache = (scenario, strategie) -> tous les tirages
# =============================================================================
_BL = None
_TIMES = None


def _ensure_loaded():
    """Charge baseline + instants de panne UNE fois par process worker."""
    global _BL, _TIMES
    if _BL is None:
        d = np.load(BASELINE_CACHE)
        _BL = {k: d[k] for k in d.files}
    if _TIMES is None:
        _TIMES = np.load(MC_SETUP_CACHE)["t"]


def evaluate(task):
    """task = (scenario_key, component, severity, strategy_folder).
    Renvoie (scenario_key, strategy_folder, lpsp_array, ens_array) AVEC panne."""
    scenario_key, component, severity, strat_folder = task
    _ensure_loaded()
    strat = load_strategy(strat_folder)
    n = len(_TIMES)
    lpsp = np.empty(n)
    ens  = np.empty(n)
    for i, t0 in enumerate(_TIMES):
        lpsp[i], ens[i] = simulate_failure_week(strat, _BL, int(t0), component, severity)
    return scenario_key, strat_folder, lpsp, ens


def evaluate_nominal(strat_folder):
    """Contrefactuel SANS panne (meme semaines). Independant du scenario.
    Renvoie (strategy_folder, lpsp_array, ens_array)."""
    _ensure_loaded()
    strat = load_strategy(strat_folder)
    n = len(_TIMES)
    lpsp = np.empty(n)
    ens  = np.empty(n)
    for i, t0 in enumerate(_TIMES):
        lpsp[i], ens[i] = simulate_nominal_week(strat, _BL, int(t0))
    return strat_folder, lpsp, ens


def _detect_workers():
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm:
        return max(1, int(n_slurm))
    return max(1, (os.cpu_count() or 2) - 1)


N_WORKERS = _detect_workers()


def run_pool(fn, items, title, line_fmt=None, workers=None):
    """Execute `fn` (module-level, picklable) sur chaque element de `items` en
    parallele. `line_fmt(result)` -> str pour l'affichage de progression."""
    workers = workers or N_WORKERS
    print("--- %s : %d taches (%d workers) ---" % (title, len(items), workers), flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, items), 1):
            res.append(r)
            if line_fmt:
                print("  [%2d/%d] %s" % (i, len(items), line_fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)
    return res
