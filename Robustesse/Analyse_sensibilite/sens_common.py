"""
sens_common.py -- helpers partages pour l'analyse de sensibilite des EMS.
=======================================================================
SOURCE 100% ASCII (volontaire) : certains interpreteurs decodent le .py en
latin-1 et mojibakent les caracteres speciaux (accents, euro, sigma). En ASCII
pur, la sortie est identique quel que soit l'encodage de l'interpreteur.

Ce module N'ECRIT JAMAIS dans Vieillissement8 : il importe le code de base EN
LECTURE SEULE via un chemin ABSOLU. Tous les scripts de sensibilite de ce
dossier s'appuient dessus.

Fournit :
  - I            : le module Init_EMR_MG_v16_python (parametres BAT/FC/ELY/...)
  - init_and_run_loop, get_cost_total, BASE_STRAT (RB2(SoH))
  - metrics(data)    -> (LPSP %, cout de degradation kEUR)   [== batch_pareto]
  - lifetimes(data)  -> (vie_bat, vie_fc, vie_ely) en annees
  - run_pool(...)    -> execution parallele d'une fonction evaluate
  - confidence_ellipse(...) -> ellipse de covariance pour les figures
  - RESULTS_DIR      -> dossier de sortie (cree au besoin)
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- Chemin vers le code de base (LECTURE SEULE), PORTABLE Win/Linux ---
# Vieillissement8 est le dossier FRERE de Analyse_sensibilite : on le resout en
# relatif (marche sur n'importe quelle machine ou le depot est clone). On garde
# l'ancien chemin Linux en repli pour ne rien casser sur l'ancienne machine.
HERE = os.path.dirname(os.path.abspath(__file__))
_VIEIL8_REL   = os.path.normpath(os.path.join(HERE, os.pardir, "Vieillissement8"))
_VIEIL8_LINUX = "/home/theo/Documents/Doctorat/GENIAL/Python/Robustesse/Vieillissement8"
VIEIL8 = _VIEIL8_REL if os.path.isdir(_VIEIL8_REL) else _VIEIL8_LINUX
if VIEIL8 not in sys.path:
    sys.path.insert(0, VIEIL8)
_RB2SOH = os.path.join(VIEIL8, "RB2(SoH)")
if _RB2SOH not in sys.path:
    sys.path.insert(0, _RB2SOH)

from Common import Init_EMR_MG_v16_python as I            # noqa: E402
from Common.main_init_and_loop import init_and_run_loop   # noqa: E402
from Common.cost_fcn_total2 import get_cost_total          # noqa: E402
from get_optimal_action_RB import get_optimal_action_RB as BASE_STRAT  # noqa: E402

# --- Dossier de sortie (a cote de ce fichier) ---
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# --- Chargement DYNAMIQUE d'une strategie par dossier (cf. batch_pareto.run_one) ---
STRATEGY_FILENAME  = "get_optimal_action_RB"   # nom du fichier .py SANS extension
STRATEGY_FUNC_NAME = "get_optimal_action_RB"   # nom de la fonction dedans


def load_strategy(folder_name):
    """Importe get_optimal_action_RB depuis VIEIL8/<folder_name> et renvoie la
    fonction. A APPELER DANS LE WORKER : on purge un eventuel module homonyme
    (process reutilise pour une autre strategie) puis on met le bon dossier en
    tete de sys.path -> on importe toujours la bonne strategie. Common reste
    accessible via VIEIL8 (deja dans sys.path)."""
    import importlib
    folder_path = os.path.join(VIEIL8, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError("Strategie introuvable : %s" % folder_path)
    if folder_path in sys.path:
        sys.path.remove(folder_path)
    sys.path.insert(0, folder_path)
    sys.modules.pop(STRATEGY_FILENAME, None)
    module = importlib.import_module(STRATEGY_FILENAME)
    return getattr(module, STRATEGY_FUNC_NAME)


def metrics(data):
    """LPSP [%] et cout de degradation [kEUR], EXACTEMENT comme
    batch_pareto._compute_metrics de Vieillissement8 (interpolation SoH_bat
    aux remplacements pour ne pas compter la degradation a travers un saut)."""
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc = data["alpha_fc"][:-1]; alpha_ely = data["alpha_ely"][:-1]
    SoH_bat = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)), SoH_bat[~np.isnan(SoH_bat)])
    P_planned = (P_dc_load - P_dc_pv) / 1000.0
    P_real    = (P_dc_load - P_dc_pv) * (1 - lol) / 1000.0
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    cost = get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
                          I.LOAD, I.BAT, I.FC, I.ELY, SoH_bat) / 1000.0
    return float(lpsp), float(cost)


def lifetimes(data):
    """Premier remplacement de chaque composant (annees) ; None si aucun."""
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        out.append(float(rep[0] * yr) if len(rep) > 0 else None)
    return out  # [bat, fc, ely]


def run_pool(evaluate, param_list, title, line_fmt=None, workers=None):
    """Execute `evaluate` sur chaque element de param_list en parallele.
    `evaluate` doit etre une fonction module-level (picklable)."""
    workers = workers or N_WORKERS
    print("\n--- %s : %d runs (%d workers) ---" % (title, len(param_list), workers), flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(evaluate, param_list), 1):
            res.append(r)
            if line_fmt:
                print("  [%3d/%d] %s" % (i, len(param_list), line_fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)
    return res


def confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    """Trace l'ellipse de covariance (n_std ecarts-types) du nuage (x, y).
    Renvoie le patch. Standard : valeurs/vecteurs propres de la covariance 2x2."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0.0
    rx = np.sqrt(1 + pearson); ry = np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std; sy = np.sqrt(cov[1, 1]) * n_std
    tr = (transforms.Affine2D().rotate_deg(45).scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    return ax.add_patch(ell)
