"""
bench_valeur_info.py -- VALEUR INFORMATIONNELLE DU SoH SOUS INCERTITUDE DE
                        VIEILLISSEMENT : boucle fermee vs boucle ouverte.
=============================================================================
SOURCE 100% ASCII (convention mesocentre, cf. Analyse_sensibilite/sens_common).

MOTIVATION (note Robustesse/ANALYSE_CRITIQUE_integration_vieillissement.txt,
proposition P1 + diagnostic 1.c)
---------------------------------
Sur le scenario de vieillissement NOMINAL et deterministe, la trajectoire
SoH(t) est previsible : une loi de setpoints PROGRAMMEE dans le temps peut
mimer la modulation SoH^gamma sans aucun capteur. La valeur INFORMATIONNELLE
du SoH (ce que le jumeau numerique de la tache 3.1 apporte reellement a la
tache 3.2) n'est donc pas identifiable en nominal. Ce banc la mesure la ou
elle doit apparaitre : quand le VRAI modele de vieillissement s'ecarte du
modele de conception.

PROTOCOLE (v2 : 4 competiteurs -> decomposition en cascade)
-----------------------------------------------------------
Quatre competiteurs, memes tirages (common random numbers) :
  RB2          statique, constantes d'origine (dossier RB2 : 0.400/0.260) ;
  RB2(Recale)  statique, constantes RE-CALEES = celles de RB2(SoH) a gamma=0
               (wrapper de RB2(SoH) avec SoH_fc=SoH_ely=1 force) : isole la
               valeur du simple RE-CALAGE des constantes, a information nulle ;
  RB2(Sched)   boucle ouverte PROGRAMMEE : clone exact de RB2(SoH) dont les
               signaux SoH_fc/SoH_ely sont REMPLACES par la trajectoire SoH
               enregistree lors d'un run NOMINAL de RB2(SoH) (design-time).
               Ajoute a Recale la NON-STATIONNARITE (l'horloge), sans capteur ;
  RB2(SoH)     boucle FERMEE : la strategie voit le SoH VRAI du monde perturbe
               (estimation supposee parfaite ; le bruit d'estimation est
               traite a part par sens_soh_estimation.py).
Les deux wrappers heritent de la vraie fonction RB2(SoH) : zero duplication,
toute retouche des setpoints est automatiquement propagee.
Le programme Sched est indexe par le TEMPS GLOBAL nominal ; il ne se recale
pas sur l'age de l'unite apres un remplacement decale. Cette convention mesure
la valeur d'une horloge de conception et reste une limite explicite de P1.

DECOMPOSITION EN CASCADE du gain total RB2(SoH) - RB2 :
  re-calage = RB2(Recale) - RB2         (retuning statique, aucun capteur)
  horloge   = RB2(Sched)  - RB2(Recale) (non-stationnarite programmee)
  capteur   = RB2(SoH)    - RB2(Sched)  (valeur informationnelle propre)

Monde perturbe = multiplicateurs log-uniformes U_log[MC_LO, MC_HI] tires
CONJOINTEMENT sur les taux des modeles de degradation (le "vrai" monde) :
  FC  (cost_fcn_total2.FC_REC)  : a_irr, b_rev, s ;
  ELY (cost_fcn_total2.ELY_REC) : a2,    b2,    s ;
  BAT                           : echelle globale de la table de dommage
                                  cumulatif (deg_cumul2).
Plage par defaut x[0.5, 2] : ordre de grandeur des ecarts entre etudes
(McCay vs Colombo ~x2 pour la FC ; modes Rakousky pour l'ELY).
La physique de la boucle (Pmax vieilli, remplacements, LPSP) ET les metriques
finales utilisent le MEME monde perturbe : la strategie ne connait que ses
signaux, le monde est juge sur sa realite.

SEVERITE : sev(tirage) = moyenne geometrique des 7 multiplicateurs (>1 = le
monde vieillit plus vite que le modele de conception). Le banc trace et
tabule la valeur du capteur en fonction de sev (terciles + correlation).

TEST NUL INTEGRE : au tirage nominal (tous multiplicateurs = 1), RB2(Sched)
rejoue EXACTEMENT le run qui a produit son programme -> RB2(Sched) et RB2(SoH)
doivent etre IDENTIQUES (verifie et imprime). Tout ecart = bug.

REUTILISATION : les sorties historiques a la racine ne sont jamais des caches.
Chaque protocole est identifie par le contenu des sources, donnees et
parametres, ecrit dans runs/p1_value_info_<empreinte>/ et reprend uniquement
son results_raw.tsv pleine precision si l'empreinte est identique. Le programme
Sched est dans .science_cache/ avec fiche de provenance et SHA verifies.

LECTURE ATTENDUE
----------------
  - si capteur ~ 0 meme sous perturbation -> la valeur du SoH est
    essentiellement du re-calage + une horloge : resultat negatif honnete ;
  - si capteur < 0 sur les tirages perturbes (P95, CVaR, regret) -> la boucle
    fermee S'AUTO-CORRIGE quand le modele de conception est faux : c'est la
    valeur de ROBUSTESSE du jumeau numerique attendue par l'AAPG 3.2. La
    croissance du gain capteur avec la severite renforce l'argument.

SORTIES (runs/p1_value_info_<empreinte>/)
-----------------------------------------
  results_raw.tsv               cache pleine precision, empreinte
  results.txt                   table + cascade + severite + resume
  figure.pdf/.png               (1) nuages LPSP/deg + ellipses 1s/2s
                                (2) histogrammes des increments horloge/capteur
                                (3) gain capteur vs severite du monde
  provenance.json               sources, donnees, parametres et SHA artefacts

LANCER
------
  local (fumee)          : python bench_valeur_info.py --quick
  mesocentre (nominal)   : sbatch run_meso_valeur_info.slurm
                           (avec le txt du run precedent present : seuls les
                            runs manquants sont lances, ex. ~200 pour ajouter
                            RB2(Recale) ; sans txt : ~800 runs, ~8-10 h / 32 c.)
Options : --nmc N | --years N | --seed S | --lo x --hi x | --workers N | --fresh
"""
import os
import re
import sys
import time
import argparse
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROBUSTESSE_DIR = str(Path(HERE).parent)
if ROBUSTESSE_DIR not in sys.path:
    sys.path.insert(0, ROBUSTESSE_DIR)

from Common import Init_EMR_MG_v16_python as I            # noqa: E402
from Common.main_init_and_loop import init_and_run_loop   # noqa: E402
from Common import cost_fcn_total2 as C                   # noqa: E402
from reproducibility.provenance import (                  # noqa: E402
    acquire_run_lock,
    build_provenance,
    fingerprinted_run_dir,
    provenance_header_lines,
    validate_append_only_artifact,
    validate_cache,
    validate_sidecar_artifact,
    write_provenance_sidecar,
)
from reproducibility.paired_stats import cvar_high        # noqa: E402

# ============================ CONFIGURATION ============================
N_MC    = 200          # tirages Monte-Carlo (mesocentre ; --quick le reduit)
MC_SEED = 2026         # graine des multiplicateurs (CRN entre strategies)
MC_LO   = 0.5          # borne basse des multiplicateurs (log-uniforme)
MC_HI   = 2.0          # borne haute
N_YEARS = 25           # horizon
VOLL    = 3.0          # EUR/kWh -- cout unifie = deg + VOLL * EENS (comme Fable)
REPLACEMENT_ACCOUNTING = "corrected"

# Ordre = cascade : chaque strategie ajoute UN ingredient a la precedente.
STRATS  = ["RB2", "RB2(Recale)", "RB2(Sched)", "RB2(SoH)"]

# Multiplicateurs du "vrai" monde (cles = noms de colonnes du .txt)
FACTOR_KEYS = ["m_fc_a", "m_fc_b", "m_fc_s", "m_ely_a", "m_ely_b", "m_ely_s", "m_bat"]

STRATEGY_FILENAME  = "get_optimal_action_RB"
STRATEGY_FUNC_NAME = "get_optimal_action_RB"
# ======================================================================

# --- copies PRISTINES des parametres nominaux (prises a l'import) ---
FC_REC_BASE     = dict(C.FC_REC)
ELY_REC_BASE    = dict(C.ELY_REC)
DEG_CUMUL2_BASE = np.array(C.deg_cumul2, dtype=float).copy()

NOMINAL_WORLD = {k: 1.0 for k in FACTOR_KEYS}
TRACE_KEYS = (
    "SoC", "E_h2", "P_bat", "P_fc", "P_ely", "P_dc_load", "P_dc_pv",
    "P_dc_bat", "P_dc_fc", "P_dc_ely", "lol_tab", "alpha_fc",
    "alpha_ely", "SoH_bat", "SoH_fc", "SoH_ely",
)


def provenance_files(extra=()):
    """Sources et donnees dont le contenu peut changer les trajectoires."""
    here = Path(HERE)
    common = here / "Common"
    reproducibility = here.parent / "reproducibility"
    files = [
        here / "bench_valeur_info.py",
        common / "Init_EMR_MG_v16_python.py",
        common / "main_init_and_loop.py",
        common / "simulate_transition.py",
        common / "cost_fcn_total2.py",
        common / "get_soh.py",
        common / "get_lol.py",
        common / "replacement_ledger.py",
        here / "RB2" / "get_optimal_action_RB.py",
        here / "RB2(SoH)" / "get_optimal_action_RB.py",
        common / "Cumulative_degradation_bat.txt",
        common / "FC_efficiency_LU_table_power.csv",
        common / "ELY_efficiency_LU_table_power.csv",
        reproducibility / "provenance.py",
        reproducibility / "paired_stats.py",
        ("data/sidelec_roche_plate_csv.csv", I.csv_path),
    ]
    return files + list(extra)


def trajectory_digest(data):
    """Empreinte bit-a-bit des trajectoires utiles aux tests nuls."""
    digest = hashlib.sha256()
    for key in TRACE_KEYS:
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def apply_world(w):
    """Installe le monde 'w' (multiplicateurs) dans cost_fcn_total2. Valeurs
    ABSOLUES depuis les copies pristines -> idempotent, pas de derive quand un
    worker enchaine plusieurs taches. Les dicts FC_REC/ELY_REC sont mutes EN
    PLACE (partages par reference avec la boucle) ; deg_cumul2 est rebinde au
    niveau du module (get_cost_bat le resout dans C a l'appel)."""
    C.FC_REC['a_irr'] = FC_REC_BASE['a_irr'] * w['m_fc_a']
    C.FC_REC['b_rev'] = FC_REC_BASE['b_rev'] * w['m_fc_b']
    C.FC_REC['s']     = FC_REC_BASE['s']     * w['m_fc_s']
    C.ELY_REC['a2']   = ELY_REC_BASE['a2']   * w['m_ely_a']
    C.ELY_REC['b2']   = ELY_REC_BASE['b2']   * w['m_ely_b']
    C.ELY_REC['s']    = ELY_REC_BASE['s']    * w['m_ely_s']
    C.deg_cumul2      = DEG_CUMUL2_BASE * w['m_bat']


def load_strategy(folder_name):
    """Importe get_optimal_action_RB depuis HERE/<folder_name> (meme logique
    que Analyse_sensibilite/sens_common.load_strategy : purge du module
    homonyme + dossier en tete de sys.path, worker reutilisable)."""
    import importlib
    folder_path = os.path.join(HERE, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError("Strategie introuvable : %s" % folder_path)
    if folder_path in sys.path:
        sys.path.remove(folder_path)
    sys.path.insert(0, folder_path)
    sys.modules.pop(STRATEGY_FILENAME, None)
    module = importlib.import_module(STRATEGY_FILENAME)
    return getattr(module, STRATEGY_FUNC_NAME)


# --------------- RB2(Recale) : re-calage statique, zero information ---------------
# Wrapper de RB2(SoH) avec SoH_fc = SoH_ely = 1 force : les setpoints valent
# les constantes re-calees de RB2(SoH) (c_fc, c_ely) SANS modulation -> isole
# la valeur du re-calage pur. Les grandeurs physiques restent celles du monde.
_RECALE = {"base": None}


def recale_reset():
    if _RECALE["base"] is None:
        _RECALE["base"] = load_strategy("RB2(SoH)")


def recale_strategy(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                    alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                    P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
    return _RECALE["base"](SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                           alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                           P_ely_max_t, RUL_fc_t, RUL_ely_t, 1.0, 1.0)


# --------------------- RB2(Sched) : clone open-loop de RB2(SoH) ---------------------
# Wrapper : a chaque pas, on substitue aux signaux SoH_fc_t / SoH_ely_t la
# valeur du PROGRAMME nominal (enregistre design-time), puis on appelle la
# VRAIE fonction RB2(SoH). Les grandeurs PHYSIQUES (SoH_bat pour la capacite,
# P*_max vieillis, alphas, plafonds H2, get_lol) restent celles du monde reel :
# seule l'INFORMATION de decision est mise en boucle ouverte.
# Compteur de pas module-level (la boucle appelle la strategie exactement une
# fois par pas) ; sched_reset() OBLIGATOIRE avant chaque run (workers reutilises).
_SCHED = {"fc": None, "ely": None, "j": 0, "base": None, "path": None}


def sched_load(npz_path):
    _SCHED["path"] = npz_path


def sched_reset():
    if _SCHED["fc"] is None:
        if _SCHED["path"] is None or not os.path.isfile(_SCHED["path"]):
            raise FileNotFoundError("Programme SoH nominal absent : %s" % _SCHED["path"])
        z = np.load(_SCHED["path"])
        _SCHED["fc"]  = z["SoH_fc"]
        _SCHED["ely"] = z["SoH_ely"]
    if _SCHED["base"] is None:
        _SCHED["base"] = load_strategy("RB2(SoH)")
    _SCHED["j"] = 0


def sched_strategy(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                   alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                   P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
    j = _SCHED["j"]
    _SCHED["j"] = j + 1
    return _SCHED["base"](SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                          alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                          P_ely_max_t, RUL_fc_t, RUL_ely_t,
                          float(_SCHED["fc"][j]), float(_SCHED["ely"][j]))


# ------------------------------- metriques -------------------------------
def metrics(data):
    """(LPSP %, deg kEUR, EENS kWh, unifie kEUR).

    Le cout canonique vient du ledger suivi par la boucle, qui connait les
    remplacements et les gels. Le recalcul sur la trace complete est conserve
    uniquement comme repli pour une ancienne boucle sans ledger.
    """
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc = data["alpha_fc"][:-1]; alpha_ely = data["alpha_ely"][:-1]
    Ts_h = I.LOAD['Ts'] / 3600.0
    P_planned = (P_dc_load - P_dc_pv) / 1000.0
    P_real    = (P_dc_load - P_dc_pv) * (1 - lol) / 1000.0
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    eens_kwh = float(np.clip(p - r, 0, None).sum() * Ts_h)
    ledger = data.get("degradation_ledger")
    if ledger is not None:
        deg = sum(ledger["total_eur"].values()) / 1000.0
    else:
        if data.get("replacement_accounting") == "corrected":
            raise RuntimeError("boucle corrigee sans degradation_ledger")
        # Repli strictement historique pour diagnostiquer une sortie legacy.
        soh_bat = data["SoH_bat"][:-1].copy()
        for k in range(1, len(soh_bat)):
            if soh_bat[k] == 1:
                soh_bat[k - 1] = np.nan
        if np.isnan(soh_bat).any():
            soh_bat[np.isnan(soh_bat)] = np.interp(
                np.flatnonzero(np.isnan(soh_bat)),
                np.flatnonzero(~np.isnan(soh_bat)), soh_bat[~np.isnan(soh_bat)]
            )
        deg = C.get_cost_total(
            alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
            I.LOAD, I.BAT, I.FC, I.ELY, soh_bat
        ) / 1000.0
    unified = float(deg) + VOLL * eens_kwh / 1000.0
    return float(lpsp), float(deg), eens_kwh, unified


def lifetimes(data):
    """Premier remplacement de chaque composant (annees) ; None si aucun."""
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        # rep contient l'index du pas precedant l'etat neuf ; le remplacement
        # physique est donc a l'etat rep+1.
        out.append(float((rep[0] + 1) * yr) if len(rep) > 0 else None)
    return out  # [bat, fc, ely]


# ------------------------------- worker -------------------------------
def evaluate(task):
    """task = dict(strat, world, draw, years, sched_npz). Installe le monde,
    charge la strategie, simule, mesure SOUS CE MEME monde."""
    try:
        apply_world(task['world'])
        if task['strat'] == "RB2(Sched)":
            sched_load(task['sched_npz'])
            sched_reset()
            strat = sched_strategy
        elif task['strat'] == "RB2(Recale)":
            recale_reset()
            strat = recale_strategy
        else:
            strat = load_strategy(task['strat'])
        data = init_and_run_loop(
            strat, n_years=task['years'],
            replacement_accounting=task.get('replacement_accounting', REPLACEMENT_ACCOUNTING),
        )
        lpsp, deg, eens, uni = metrics(data)
        lb, lf, le = lifetimes(data)
        trace_digest = trajectory_digest(data) if task['draw'] == -1 else ""
        ok = True
    except Exception as e:
        lpsp = deg = eens = uni = lb = lf = le = None
        trace_digest = ""
        ok = False
        print("  [FAIL] %-11s draw=%s : %s" % (task['strat'], task['draw'], e), flush=True)
    return dict(strat=task['strat'], draw=task['draw'], world=task['world'],
                lpsp=lpsp, deg=deg, eens=eens, uni=uni,
                life_bat=lb, life_fc=lf, life_ely=le,
                trace_digest=trace_digest, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "%-11s draw=%-3s FAIL" % (r['strat'], r['draw'])
    return ("%-11s draw=%-3s LPSP %7.4f%%  deg %8.3f  unifie %8.3f kEUR"
            % (r['strat'], r['draw'], r['lpsp'], r['deg'], r['uni']))


def _detect_workers():
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm:
        return max(1, int(n_slurm))
    return max(1, (os.cpu_count() or 2) - 1)


def confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    """Ellipse de covariance (cf. sens_common.confidence_ellipse)."""
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


# --------------------- reutilisation d'un resultat precedent ---------------------
def _load_previous_legacy_forensics(out_txt, seed, lo, hi):
    """Lecteur historique conserve pour audit manuel, jamais appele par main.

    Relit un valeur_info_<tag>.txt et renvoie {(strat, draw): result}
    pour eviter de re-payer les runs deja faits. Renvoie {} si le fichier est
    absent ou si son header (seed / plage / VoLL) ne correspond pas. Les
    facteurs par tirage sont aussi renvoyes ({draw: [m...]}) pour verification
    croisee avec les mondes regeneres."""
    done, factors = {}, {}
    if not os.path.isfile(out_txt):
        return done, factors
    with open(out_txt, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    head = " ".join(lines[:6])
    m = re.search(r"mult U_log\[([0-9.]+),([0-9.]+)\] \| seed=(\d+) \| VoLL=([0-9.]+)", head)
    if not m:
        print("  (reuse) header illisible dans %s -> ignore" % out_txt, flush=True)
        return {}, {}
    if (abs(float(m.group(1)) - lo) > 1e-9 or abs(float(m.group(2)) - hi) > 1e-9
            or int(m.group(3)) != seed or abs(float(m.group(4)) - VOLL) > 1e-9):
        print("  (reuse) header de %s (lo/hi/seed/VoLL) different -> ignore" % out_txt, flush=True)
        return {}, {}
    section = None
    draw_strats = []
    for l in lines:
        if l.startswith("## Points nominaux"):
            section = "nom"; continue
        if l.startswith("## Tirages"):
            section = "draws"; continue
        if l.startswith("##"):
            section = None; continue
        if not l or l.startswith("#"):
            continue
        parts = l.split(";")
        if section == "nom":
            if parts[0] == "strat" or "FAIL" in l:
                continue
            s = parts[0]
            lb, lf, le = [None if v == "None" else float(v) for v in parts[5:8]]
            done[(s, -1)] = dict(strat=s, draw=-1, world=dict(NOMINAL_WORLD),
                                 lpsp=float(parts[1]), deg=float(parts[2]),
                                 eens=float(parts[3]), uni=float(parts[4]),
                                 life_bat=lb, life_fc=lf, life_ely=le, ok=True)
        elif section == "draws":
            if parts[0] == "draw":
                # colonnes : draw;7 facteurs;puis groupes <strat>_lpsp/_deg/_uni
                draw_strats = [c[:-5] for c in parts[8:] if c.endswith("_lpsp")]
                continue
            if not draw_strats:
                continue
            d = int(parts[0])
            factors[d] = [float(v) for v in parts[1:8]]
            for k, s in enumerate(draw_strats):
                lpsp, deg, uni = [float(v) for v in parts[8 + 3 * k: 11 + 3 * k]]
                done[(s, d)] = dict(strat=s, draw=d, world=None,
                                    lpsp=lpsp, deg=deg,
                                    eens=(uni - deg) / VOLL * 1000.0, uni=uni,
                                    life_bat=None, life_fc=None, life_ely=None, ok=True)
    return done, factors


# Le cache numerique nouveau est separe du rapport lisible et conserve la
# precision binaire utile aux contrastes proches de zero.
RAW_COLUMNS = (
    ["strat", "draw"] + FACTOR_KEYS
    + ["lpsp", "deg", "eens", "uni", "life_bat", "life_fc", "life_ely",
       "trace_digest"]
)


def _raw_value(value):
    return "" if value is None else format(float(value), ".17g")


def initialise_raw(path, provenance):
    with open(path, "w", encoding="utf-8") as stream:
        for line in provenance_header_lines(provenance):
            stream.write(line + "\n")
        stream.write(";".join(RAW_COLUMNS) + "\n")
    write_provenance_sidecar(str(path) + ".provenance.json", provenance, [path])


def append_raw(path, result, provenance):
    if not result["ok"]:
        return
    row = [result["strat"], str(int(result["draw"]))]
    row.extend(_raw_value(result["world"][key]) for key in FACTOR_KEYS)
    row.extend(_raw_value(result[key]) for key in (
        "lpsp", "deg", "eens", "uni", "life_bat", "life_fc", "life_ely"
    ))
    row.append(result.get("trace_digest", ""))
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(";".join(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_provenance_sidecar(str(path) + ".provenance.json", provenance, [path])


def load_raw(path, provenance):
    valid, reason = validate_cache(path, provenance, allow_legacy=False)
    if valid:
        valid, reason = validate_append_only_artifact(
            str(path) + ".provenance.json", path, provenance
        )
    if not valid:
        if os.path.isfile(path):
            print("  (reuse) cache brut ignore : %s" % reason, flush=True)
        return {}
    done = {}
    with open(path, encoding="utf-8") as stream:
        rows = [line.rstrip("\n") for line in stream if not line.startswith("#")]
    if not rows or rows[0].split(";") != RAW_COLUMNS:
        print("  (reuse) colonnes du cache brut invalides -> ignore", flush=True)
        return {}
    for line in rows[1:]:
        if not line:
            continue
        parts = line.split(";")
        if len(parts) != len(RAW_COLUMNS):
            raise ValueError("ligne incomplete dans %s" % path)
        values = dict(zip(RAW_COLUMNS, parts))
        world = {key: float(values[key]) for key in FACTOR_KEYS}
        def optional(key):
            return None if values[key] == "" else float(values[key])
        result = {
            "strat": values["strat"], "draw": int(values["draw"]),
            "world": world, "ok": True,
            "lpsp": float(values["lpsp"]), "deg": float(values["deg"]),
            "eens": float(values["eens"]), "uni": float(values["uni"]),
            "life_bat": optional("life_bat"), "life_fc": optional("life_fc"),
            "life_ely": optional("life_ely"),
            "trace_digest": values["trace_digest"],
        }
        key = (result["strat"], result["draw"])
        if key in done:
            raise ValueError("resultat brut duplique : %r" % (key,))
        done[key] = result
    write_provenance_sidecar(str(path) + ".provenance.json", provenance, [path])
    return done


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Valeur informationnelle du SoH sous incertitude de vieillissement")
    ap.add_argument("--quick", action="store_true", help="fumee locale : 2 ans, N_MC=4")
    ap.add_argument("--nmc", type=int, default=None)
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--seed", type=int, default=MC_SEED)
    ap.add_argument("--lo", type=float, default=MC_LO)
    ap.add_argument("--hi", type=float, default=MC_HI)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--output-root", default=HERE,
                    help="racine de runs/ et .science_cache/ (defaut : V9_4)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore le txt existant (re-paye tous les runs)")
    args = ap.parse_args()

    years = args.years if args.years is not None else (2 if args.quick else N_YEARS)
    n_mc  = args.nmc if args.nmc is not None else (4 if args.quick else N_MC)
    workers = args.workers if args.workers is not None else _detect_workers()
    if years <= 0 or n_mc <= 0 or workers <= 0:
        ap.error("years, nmc et workers doivent etre strictement positifs")
    if args.lo <= 0 or args.hi < args.lo:
        ap.error("exiger 0 < lo <= hi pour les multiplicateurs log-uniformes")

    tag = "%dy" % years
    repo_root = Path(HERE).parents[1]
    run_provenance = build_provenance(
        "p1_value_information_soh", provenance_files(),
        {
            "horizon_years": years, "n_worlds": n_mc, "seed": args.seed,
            "multipliers_log_uniform": [args.lo, args.hi],
            "factor_keys": FACTOR_KEYS, "strategies": STRATS,
            "schedule_semantics": "global_nominal_time_no_unit_age_reset",
            "voll_eur_per_kwh": VOLL,
            "replacement_accounting": REPLACEMENT_ACCOUNTING,
        }, repo_root=repo_root,
    )
    run_dir = fingerprinted_run_dir(args.output_root, "p1_value_info", run_provenance)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(run_dir)
    out_txt = str(run_dir / "results.txt")
    out_raw = str(run_dir / "results_raw.tsv")
    out_fig = str(run_dir / "figure")

    sched_provenance = build_provenance(
        "p1_nominal_soh_schedule", provenance_files(),
        {
            "horizon_years": years, "strategy": "RB2(SoH)",
            "schedule_semantics": "global_nominal_time",
            "replacement_accounting": REPLACEMENT_ACCOUNTING,
        }, repo_root=repo_root,
    )
    cache_dir = Path(args.output_root) / ".science_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sched_npz = str(cache_dir / (
        "p1_sched_%s.npz" % sched_provenance["run_fingerprint"][:12]
    ))
    sched_sidecar = sched_npz + ".provenance.json"

    print("=== VALEUR DE L'INFORMATION SoH -- boucle fermee vs boucle ouverte ===", flush=True)
    print("    horizon=%d ans | N_MC=%d | multiplicateurs U_log[%.2f, %.2f] | seed=%d | VoLL=%.1f"
          % (years, n_mc, args.lo, args.hi, args.seed, VOLL), flush=True)
    print("    empreinte=%s | sortie=%s"
          % (run_provenance["run_fingerprint"][:12], run_dir), flush=True)

    # --- 1) programme SoH nominal (design-time) : run RB2(SoH) monde nominal ---
    sched_valid, sched_reason = validate_sidecar_artifact(
        sched_sidecar, sched_npz, sched_provenance
    )
    if not sched_valid:
        print("\n--- Programme nominal a construire (%s) -> RB2(SoH), %d ans..."
              % (sched_reason, years), flush=True)
        t0 = time.time()
        apply_world(NOMINAL_WORLD)
        data = init_and_run_loop(
            load_strategy("RB2(SoH)"), n_years=years,
            replacement_accounting=REPLACEMENT_ACCOUNTING,
        )
        np.savez_compressed(
            sched_npz, SoH_fc=data["SoH_fc"], SoH_ely=data["SoH_ely"],
            run_fingerprint=sched_provenance["run_fingerprint"],
        )
        write_provenance_sidecar(sched_sidecar, sched_provenance, [sched_npz])
        print("    programme enregistre : %s (%.0fs)" % (sched_npz, time.time() - t0), flush=True)
    else:
        print("\n--- Programme nominal valide reutilise : %s" % sched_npz, flush=True)

    # --- 2) tirages du monde (CRN : memes mondes pour toutes les strategies) ---
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(args.lo), np.log(args.hi)
    worlds = []
    for d in range(n_mc):
        worlds.append({k: float(np.exp(rng.uniform(lo, hi))) for k in FACTOR_KEYS})

    # --- 2b) cache brut empreinte ; aucun TXT legacy n'est jamais relu ---
    done = {} if args.fresh else load_raw(out_raw, run_provenance)
    if done:
        bad = [d for (_, d), result in done.items() if d >= 0 and (
            d >= n_mc or any(result["world"][key] != worlds[d][key]
                             for key in FACTOR_KEYS)
        )]
        if bad:
            print("  (reuse) facteurs bruts incoherents (tirages %s) -> cache ignore"
                  % bad[:5], flush=True)
            done = {}
        else:
            print("  (reuse) %d resultats pleine precision repris de %s"
                  % (len(done), out_raw), flush=True)
    if args.fresh or not done:
        initialise_raw(out_raw, run_provenance)

    # Test nul fail-fast avant les Monte-Carlo couteux.
    for strategy in ("RB2(SoH)", "RB2(Sched)"):
        key = (strategy, -1)
        if key not in done:
            result = evaluate({
                "strat": strategy, "world": dict(NOMINAL_WORLD), "draw": -1,
                "years": years, "sched_npz": sched_npz,
                "replacement_accounting": REPLACEMENT_ACCOUNTING,
            })
            if not result["ok"]:
                raise RuntimeError("echec du test nul nominal : %s" % strategy)
            done[key] = result
            append_raw(out_raw, result, run_provenance)
    null_gaps = {
        metric: abs(done[("RB2(SoH)", -1)][metric]
                    - done[("RB2(Sched)", -1)][metric])
        for metric in ("lpsp", "deg", "eens")
    }
    trace_equal = (
        bool(done[("RB2(SoH)", -1)].get("trace_digest"))
        and done[("RB2(SoH)", -1)]["trace_digest"]
        == done[("RB2(Sched)", -1)].get("trace_digest")
    )
    null_ok = all(gap < 1e-9 for gap in null_gaps.values()) and trace_equal
    print("\nTEST NUL PRE-MC Sched==SoH : LPSP %.3e | deg %.3e kEUR | EENS %.3e kWh | trace=%s -> %s"
          % (null_gaps["lpsp"], null_gaps["deg"], null_gaps["eens"],
             "identique" if trace_equal else "differente",
             "OK" if null_ok else "ECHEC"), flush=True)
    if not null_ok:
        raise RuntimeError("test nul P1 invalide : aucun Monte-Carlo lance")

    tasks = []
    for s in STRATS:                                   # tirage -1 = monde nominal (test nul)
        if (s, -1) not in done:
            tasks.append(dict(strat=s, world=dict(NOMINAL_WORLD), draw=-1,
                              years=years, sched_npz=sched_npz,
                              replacement_accounting=REPLACEMENT_ACCOUNTING))
    for d, w in enumerate(worlds):
        for s in STRATS:
            if (s, d) not in done:
                tasks.append(dict(strat=s, world=w, draw=d, years=years,
                                  sched_npz=sched_npz,
                                  replacement_accounting=REPLACEMENT_ACCOUNTING))

    print("\n--- %d runs a lancer (%d repris, %d workers) ---"
          % (len(tasks), len(done), workers), flush=True)
    t0 = time.time(); res = list(done.values())
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(evaluate, tasks), 1):
                res.append(r)
                append_raw(out_raw, r, run_provenance)
                print("  [%3d/%d] %s" % (i, len(tasks), _fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)
    failed = [(r["strat"], r["draw"]) for r in res if not r["ok"]]
    expected = {(strategy, draw) for strategy in STRATS
                for draw in [-1] + list(range(n_mc))}
    completed = {(r["strat"], r["draw"]) for r in res if r["ok"]}
    missing = sorted(expected.difference(completed), key=lambda item: (item[1], item[0]))
    if failed or missing:
        raise RuntimeError(
            "P1 incomplet : echecs=%s ; manquants=%s (cache partiel conserve)"
            % (failed[:10], missing[:10])
        )

    # --- 3) tri + stats ---
    nom = {r['strat']: r for r in res if r['draw'] == -1 and r['ok']}
    by  = {s: {r['draw']: r for r in res if r['strat'] == s and r['draw'] >= 0 and r['ok']}
           for s in STRATS}
    draws_ok = sorted(set.intersection(*[set(by[s].keys()) for s in STRATS])) if all(by[s] for s in STRATS) else []

    def col(s, key):
        return np.array([by[s][d][key] for d in draws_ok], dtype=float)

    def cvar_hi(x, q=0.9):
        """ES empirique exacte : les ceil((1-q)N) plus grandes valeurs."""
        return cvar_high(x, q=q)

    # cascade : (etiquette, strat A, strat B) -> increment A - B
    CASCADE = [("re-calage", "RB2(Recale)", "RB2"),
               ("horloge",   "RB2(Sched)",  "RB2(Recale)"),
               ("capteur",   "RB2(SoH)",    "RB2(Sched)"),
               ("total",     "RB2(SoH)",    "RB2")]

    # severite du monde : moyenne geometrique des 7 multiplicateurs
    if draws_ok:
        sev = np.array([np.exp(np.mean([np.log(worlds[d][k]) for k in FACTOR_KEYS]))
                        for d in draws_ok])
        gain_capteur = -(col("RB2(SoH)", 'uni') - col("RB2(Sched)", 'uni'))   # >0 = gain
        gain_total   = -(col("RB2(SoH)", 'uni') - col("RB2", 'uni'))
        corr_sev = float(np.corrcoef(gain_capteur, sev)[0, 1]) if len(sev) > 2 else float('nan')
        q1, q2 = np.quantile(sev, [1.0 / 3, 2.0 / 3])
        terciles = [(lab, mask) for lab, mask in
                    [("doux",   sev <= q1),
                     ("moyen",  (sev > q1) & (sev < q2)),
                     ("severe", sev >= q2)] if mask.sum() > 0]   # petits N : terciles vides ignores

    # --- 4) sauvegarde TXT ---
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in provenance_header_lines(run_provenance):
            f.write(line + "\n")
        f.write("# Valeur informationnelle du SoH -- boucle fermee vs boucle ouverte\n")
        f.write("# horizon=%d ans | N_MC=%d | mult U_log[%.2f,%.2f] | seed=%d | VoLL=%.1f EUR/kWh\n"
                % (years, n_mc, args.lo, args.hi, args.seed, VOLL))
        f.write("# facteurs perturbes : FC(a_irr,b_rev,s) ELY(a2,b2,s) BAT(echelle table)\n")
        f.write("# unifie = deg + VoLL*EENS [kEUR] ; CRN : memes mondes pour les strategies\n")
        f.write("# cascade : re-calage = Recale-RB2 ; horloge = Sched-Recale ; capteur = SoH-Sched\n")
        f.write("# TEST NUL nominal Sched==SoH : %s\n" % ("OK" if null_ok else "ECHEC"))
        f.write("\n## Points nominaux (multiplicateurs = 1)\n")
        f.write("strat;LPSP_%;deg_kEUR;EENS_kWh;unifie_kEUR;vie_bat;vie_fc;vie_ely\n")
        for s in STRATS:
            r = nom.get(s)
            if r is None:
                f.write("%s;NOMINAL_FAIL\n" % s); continue
            f.write("%s;%.4f;%.3f;%.1f;%.3f;%s;%s;%s\n"
                    % (s, r['lpsp'], r['deg'], r['eens'], r['uni'],
                       r['life_bat'], r['life_fc'], r['life_ely']))
        f.write("\n## Tirages (mondes perturbes, %d complets)\n" % len(draws_ok))
        f.write("draw;" + ";".join(FACTOR_KEYS)
                + ";" + ";".join("%s_lpsp;%s_deg;%s_uni" % (s, s, s) for s in STRATS) + "\n")
        for d in draws_ok:
            w = worlds[d]
            row = [str(d)] + ["%.4f" % w[k] for k in FACTOR_KEYS]
            for s in STRATS:
                r = by[s][d]
                row += ["%.4f" % r['lpsp'], "%.3f" % r['deg'], "%.3f" % r['uni']]
            f.write(";".join(row) + "\n")

        if draws_ok:
            f.write("\n## Stats par strategie (cout unifie, kEUR)\n")
            f.write("strat;mean;std;min;P5;P50;P95;max;ES90;LPSP_mean;deg_mean\n")
            for s in STRATS:
                u = col(s, 'uni')
                f.write("%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.4f;%.3f\n"
                        % (s, u.mean(), u.std(), u.min(),
                           np.quantile(u, 0.05), np.quantile(u, 0.50), np.quantile(u, 0.95),
                           u.max(), cvar_hi(u), col(s, 'lpsp').mean(), col(s, 'deg').mean()))

            f.write("\n## Decomposition en cascade (differences appariees CRN, kEUR ; negatif = gain)\n")
            f.write("etape;paire;mean;std;P5;P95;pct_gagne\n")
            for lab, a, b in CASCADE:
                dif = col(a, 'uni') - col(b, 'uni')
                f.write("%s;%s - %s;%.3f;%.3f;%.3f;%.3f;%.1f%%\n"
                        % (lab, a, b, dif.mean(), dif.std(),
                           np.quantile(dif, 0.05), np.quantile(dif, 0.95),
                           100.0 * (dif < 0).mean()))

            f.write("\n## Valeur du capteur vs severite du monde (sev = moy. geometrique des 7 mult.)\n")
            f.write("# corr(gain_capteur, sev) = %.3f\n" % corr_sev)
            f.write("tercile;sev_min;sev_max;N;gain_capteur_mean_kEUR;gain_total_mean_kEUR;part_capteur_%\n")
            for lab, mask in terciles:
                gc, gt = gain_capteur[mask], gain_total[mask]
                part = 100.0 * gc.mean() / gt.mean() if len(gt) and gt.mean() != 0 else float('nan')
                f.write("%s;%.3f;%.3f;%d;%.3f;%.3f;%.1f\n"
                        % (lab, sev[mask].min(), sev[mask].max(), mask.sum(),
                           gc.mean(), gt.mean(), part))

            f.write("\n## Regret vs meilleur des %d par tirage [kEUR]\n" % len(STRATS))
            f.write("strat;mean;P95;max\n")
            best = np.min(np.vstack([col(s, 'uni') for s in STRATS]), axis=0)
            for s in STRATS:
                reg = col(s, 'uni') - best
                f.write("%s;%.3f;%.3f;%.3f\n" % (s, reg.mean(), np.quantile(reg, 0.95), reg.max()))

    # --- 5) figure ---
    if draws_ok:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
        colors = {"RB2": "tab:blue", "RB2(Recale)": "tab:purple",
                  "RB2(Sched)": "tab:orange", "RB2(SoH)": "tab:green"}
        ax = axes[0]
        for s in STRATS:
            x = col(s, 'lpsp'); y = col(s, 'deg'); c = colors[s]
            ax.scatter(x, y, s=14, color=c, alpha=0.3, zorder=2)
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=c, facecolor='none', lw=1.6, zorder=4)
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=c, facecolor='none', lw=0.9, ls='--', alpha=0.6, zorder=4)
            r = nom.get(s)
            if r is not None:
                ax.scatter([r['lpsp']], [r['deg']], marker='o', s=75, color=c,
                           edgecolor='k', linewidth=0.7, zorder=6, label=s)
        ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
        ax.set_title("Incertitude du vieillissement : nuages + ellipses 1s/2s")
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

        ax = axes[1]
        d_horl = col("RB2(Sched)", 'uni') - col("RB2(Recale)", 'uni')
        d_capt = col("RB2(SoH)", 'uni') - col("RB2(Sched)", 'uni')
        bins = np.histogram_bin_edges(np.concatenate([d_horl, d_capt]), bins=25)
        ax.hist(d_horl, bins=bins, alpha=0.55, color="tab:orange",
                label="horloge : Sched - Recale")
        ax.hist(d_capt, bins=bins, alpha=0.55, color="tab:green",
                label="capteur : SoH - Sched")
        ax.axvline(0.0, color='k', lw=0.9)
        ax.set_xlabel("Increment apparie de cout unifie [kEUR]  (<0 = gain)")
        ax.set_ylabel("Tirages")
        ax.set_title("Decomposition : horloge vs capteur (CRN)")
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

        ax = axes[2]
        ax.scatter(sev, gain_capteur, s=16, color="tab:green", alpha=0.45, zorder=3)
        if len(sev) > 2:
            a1, a0 = np.polyfit(sev, gain_capteur, 1)
            xs = np.linspace(sev.min(), sev.max(), 50)
            ax.plot(xs, a1 * xs + a0, color='k', lw=1.2, zorder=4,
                    label="tendance (corr=%.2f)" % corr_sev)
        for lab, mask in terciles:
            ax.plot([sev[mask].min(), sev[mask].max()],
                    [gain_capteur[mask].mean()] * 2,
                    color="tab:red", lw=2.2, zorder=5)
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.axvline(1.0, color='k', lw=0.8, ls=':', alpha=0.6)
        ax.set_xlabel("Severite du monde (moy. geometrique des multiplicateurs)")
        ax.set_ylabel("Gain du capteur [kEUR]  (SoH vs Sched, >0 = gain)")
        ax.set_title("Valeur de la boucle fermee vs ecart au modele")
        ax.grid(True, ls='--', alpha=0.5)
        if len(sev) > 2:
            ax.legend(fontsize=9)

        fig.suptitle("Valeur informationnelle du SoH sous incertitude de vieillissement (%d ans, N=%d)"
                     % (years, len(draws_ok)), fontsize=12)
        fig.tight_layout()
        fig.savefig(out_fig + ".pdf", bbox_inches="tight")
        fig.savefig(out_fig + ".png", dpi=160, bbox_inches="tight")
        plt.close()

    artifacts = [out_raw, out_raw + ".provenance.json", out_txt,
                 out_fig + ".pdf", out_fig + ".png"]
    write_provenance_sidecar(
        run_dir / "provenance.json", run_provenance,
        [path for path in artifacts if os.path.isfile(path)],
    )

    # --- 6) resume console ---
    print("\n" + "=" * 78)
    if draws_ok:
        print("COUT UNIFIE [kEUR] sous incertitude du vieillissement (N=%d)" % len(draws_ok))
        print("%-12s | %8s | %8s | %8s | %8s" % ("strat", "mean", "P95", "ES90", "regret"))
        best = np.min(np.vstack([col(s, 'uni') for s in STRATS]), axis=0)
        for s in STRATS:
            u = col(s, 'uni'); reg = u - best
            print("%-12s | %8.3f | %8.3f | %8.3f | %8.3f"
                  % (s, u.mean(), np.quantile(u, 0.95), cvar_hi(u), reg.mean()))
        print("-" * 78)
        print("CASCADE (apparie, negatif = gain) :")
        for lab, a, b in CASCADE:
            dif = col(a, 'uni') - col(b, 'uni')
            print("  %-9s %-26s : %+.3f +/- %.3f kEUR ; gagne %.0f%%"
                  % (lab, "%s - %s" % (a, b), dif.mean(), dif.std(), 100.0 * (dif < 0).mean()))
        print("-" * 78)
        print("CAPTEUR vs SEVERITE : corr=%.3f ; gain moyen par tercile : %s"
              % (corr_sev, "  ".join("%s %.3f" % (lab, gain_capteur[mask].mean())
                                     for lab, mask in terciles)))
    print("=" * 78)
    print("Resultats : %s" % out_txt)
    print("Cache brut: %s" % out_raw)
    print("Figure    : %s.pdf" % out_fig)
    run_lock.close()


if __name__ == "__main__":
    main()
