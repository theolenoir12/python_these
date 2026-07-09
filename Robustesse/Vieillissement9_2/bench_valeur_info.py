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

PROTOCOLE
---------
Trois competiteurs, memes tirages (common random numbers) :
  RB2         boucle ouverte STATIQUE   : setpoints constants (dossier RB2) ;
  RB2(Sched)  boucle ouverte PROGRAMMEE : clone exact de RB2(SoH) dont les
              signaux SoH_fc/SoH_ely sont REMPLACES par la trajectoire SoH
              enregistree lors d'un run NOMINAL de RB2(SoH) (design-time).
              Implemente comme WRAPPER de la vraie fonction RB2(SoH) : zero
              duplication, toute retouche des setpoints est heritee ;
  RB2(SoH)    boucle FERMEE             : la strategie voit le SoH VRAI du
              monde perturbe (estimation supposee parfaite ; le bruit
              d'estimation est traite a part par sens_soh_estimation.py).

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

TEST NUL INTEGRE : au tirage nominal (tous multiplicateurs = 1), RB2(Sched)
rejoue EXACTEMENT le run qui a produit son programme -> RB2(Sched) et RB2(SoH)
doivent etre IDENTIQUES (verifie et imprime). Tout ecart = bug.

LECTURE ATTENDUE
----------------
  - si RB2(SoH) ~ RB2(Sched) meme sous perturbation -> la valeur du SoH est
    essentiellement de la NON-STATIONNARITE (une horloge suffit) : resultat
    negatif honnete, a assumer dans le manuscrit ;
  - si RB2(SoH) < RB2(Sched) sur les tirages perturbes (P95, CVaR, regret)
    -> la boucle fermee S'AUTO-CORRIGE quand le modele de conception est faux :
    c'est la valeur de ROBUSTESSE du jumeau numerique attendue par l'AAPG 3.2.

SORTIES (a cote de ce script)
-----------------------------
  valeur_info_<Ny>y.txt         table par tirage + stats appariees + resume
  valeur_info_<Ny>y.pdf/.png    (1) nuages LPSP/deg + ellipses 1s/2s
                                (2) histogramme des differences appariees
  valeur_info_sched_<Ny>y.npz   programme SoH nominal (regenere si absent)

LANCER
------
  local (fumee, ~5 min)  : python bench_valeur_info.py --quick
  mesocentre (nominal)   : sbatch run_meso_valeur_info.slurm
                           (= python bench_valeur_info.py  -> N_MC=200, 25 ans,
                            ~600 runs, ~6-8 h sur 32 coeurs)
Options : --nmc N | --years N | --seed S | --lo x --hi x | --workers N
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from Common import Init_EMR_MG_v16_python as I            # noqa: E402
from Common.main_init_and_loop import init_and_run_loop   # noqa: E402
from Common import cost_fcn_total2 as C                   # noqa: E402

# ============================ CONFIGURATION ============================
N_MC    = 200          # tirages Monte-Carlo (mesocentre ; --quick le reduit)
MC_SEED = 2026         # graine des multiplicateurs (CRN entre strategies)
MC_LO   = 0.5          # borne basse des multiplicateurs (log-uniforme)
MC_HI   = 2.0          # borne haute
N_YEARS = 25           # horizon
VOLL    = 3.0          # EUR/kWh -- cout unifie = deg + VOLL * EENS (comme Fable)

STRATS  = ["RB2", "RB2(Sched)", "RB2(SoH)"]   # RB2(Sched) = controle open-loop

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
    """(LPSP %, deg kEUR, EENS kWh, unifie kEUR) -- LPSP/deg EXACTEMENT comme
    Analyse_sensibilite/sens_common.metrics (interpolation SoH_bat aux
    remplacements) ; cout unifie = deg + VOLL*EENS comme les bancs Fable.
    Evalue sous le monde COURANT de C (a appeler avant tout changement)."""
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
    Ts_h = I.LOAD['Ts'] / 3600.0
    P_planned = (P_dc_load - P_dc_pv) / 1000.0
    P_real    = (P_dc_load - P_dc_pv) * (1 - lol) / 1000.0
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / p.sum() * 100) if p.sum() > 0 else 0.0
    eens_kwh = float(np.clip(p - r, 0, None).sum() * Ts_h)
    deg = C.get_cost_total(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC,
                           I.LOAD, I.BAT, I.FC, I.ELY, SoH_bat) / 1000.0
    unified = float(deg) + VOLL * eens_kwh / 1000.0
    return float(lpsp), float(deg), eens_kwh, unified


def lifetimes(data):
    """Premier remplacement de chaque composant (annees) ; None si aucun."""
    yr = I.LOAD['Ts'] / 3600 / 24 / 365
    out = []
    for key in ("SoH_bat", "SoH_fc", "SoH_ely"):
        s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
        out.append(float(rep[0] * yr) if len(rep) > 0 else None)
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
        else:
            strat = load_strategy(task['strat'])
        data = init_and_run_loop(strat, n_years=task['years'])
        lpsp, deg, eens, uni = metrics(data)
        lb, lf, le = lifetimes(data)
        ok = True
    except Exception as e:
        lpsp = deg = eens = uni = lb = lf = le = None
        ok = False
        print("  [FAIL] %-10s draw=%s : %s" % (task['strat'], task['draw'], e), flush=True)
    return dict(strat=task['strat'], draw=task['draw'], world=task['world'],
                lpsp=lpsp, deg=deg, eens=eens, uni=uni,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "%-10s draw=%-3s FAIL" % (r['strat'], r['draw'])
    return ("%-10s draw=%-3s LPSP %7.4f%%  deg %8.3f  unifie %8.3f kEUR"
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
    args = ap.parse_args()

    years = args.years or (2 if args.quick else N_YEARS)
    n_mc  = args.nmc if args.nmc is not None else (4 if args.quick else N_MC)
    workers = args.workers or _detect_workers()

    tag = "%dy" % years
    out_txt   = os.path.join(HERE, "valeur_info_%s.txt" % tag)
    out_fig   = os.path.join(HERE, "valeur_info_%s" % tag)
    sched_npz = os.path.join(HERE, "valeur_info_sched_%s.npz" % tag)

    print("=== VALEUR DE L'INFORMATION SoH -- boucle fermee vs boucle ouverte ===", flush=True)
    print("    horizon=%d ans | N_MC=%d | multiplicateurs U_log[%.2f, %.2f] | seed=%d | VoLL=%.1f"
          % (years, n_mc, args.lo, args.hi, args.seed, VOLL), flush=True)

    # --- 1) programme SoH nominal (design-time) : run RB2(SoH) monde nominal ---
    if not os.path.isfile(sched_npz):
        print("\n--- Programme nominal absent -> run RB2(SoH) monde nominal (%d ans)..." % years, flush=True)
        t0 = time.time()
        apply_world(NOMINAL_WORLD)
        data = init_and_run_loop(load_strategy("RB2(SoH)"), n_years=years)
        np.savez_compressed(sched_npz, SoH_fc=data["SoH_fc"], SoH_ely=data["SoH_ely"])
        print("    programme enregistre : %s (%.0fs)" % (sched_npz, time.time() - t0), flush=True)
    else:
        print("\n--- Programme nominal reutilise : %s" % sched_npz, flush=True)

    # --- 2) tirages du monde (CRN : memes mondes pour les 3 strategies) ---
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(args.lo), np.log(args.hi)
    worlds = []
    for d in range(n_mc):
        worlds.append({k: float(np.exp(rng.uniform(lo, hi))) for k in FACTOR_KEYS})

    tasks = []
    for s in STRATS:                                   # tirage -1 = monde nominal (test nul)
        tasks.append(dict(strat=s, world=dict(NOMINAL_WORLD), draw=-1,
                          years=years, sched_npz=sched_npz))
    for d, w in enumerate(worlds):
        for s in STRATS:
            tasks.append(dict(strat=s, world=w, draw=d, years=years, sched_npz=sched_npz))

    print("\n--- %d runs (%d workers) ---" % (len(tasks), workers), flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(evaluate, tasks), 1):
            res.append(r)
            print("  [%3d/%d] %s" % (i, len(tasks), _fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)

    # --- 3) tri + stats ---
    nom = {r['strat']: r for r in res if r['draw'] == -1 and r['ok']}
    by  = {s: {r['draw']: r for r in res if r['strat'] == s and r['draw'] >= 0 and r['ok']}
           for s in STRATS}
    draws_ok = sorted(set.intersection(*[set(by[s].keys()) for s in STRATS])) if all(by[s] for s in STRATS) else []

    # test nul : RB2(Sched) doit REJOUER RB2(SoH) au monde nominal
    null_ok = None
    if "RB2(SoH)" in nom and "RB2(Sched)" in nom:
        null_gap = abs(nom["RB2(SoH)"]['uni'] - nom["RB2(Sched)"]['uni'])
        null_ok = null_gap < 1e-6
        print("\nTEST NUL (monde nominal) : |unifie RB2(SoH) - RB2(Sched)| = %.3e kEUR -> %s"
              % (null_gap, "OK" if null_ok else "ECHEC (bug a corriger avant toute lecture)"), flush=True)

    def col(s, key):
        return np.array([by[s][d][key] for d in draws_ok], dtype=float)

    def cvar_hi(x, q=0.9):
        """Moyenne de la queue haute (pire decile par defaut)."""
        if len(x) == 0:
            return float('nan')
        thr = np.quantile(x, q)
        tail = x[x >= thr]
        return float(tail.mean()) if len(tail) else float('nan')

    # --- 4) sauvegarde TXT ---
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Valeur informationnelle du SoH -- boucle fermee vs boucle ouverte\n")
        f.write("# horizon=%d ans | N_MC=%d | mult U_log[%.2f,%.2f] | seed=%d | VoLL=%.1f EUR/kWh\n"
                % (years, n_mc, args.lo, args.hi, args.seed, VOLL))
        f.write("# facteurs perturbes : FC(a_irr,b_rev,s) ELY(a2,b2,s) BAT(echelle table)\n")
        f.write("# unifie = deg + VoLL*EENS [kEUR] ; CRN : memes mondes pour les 3 strategies\n")
        if null_ok is not None:
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
            w = by[STRATS[0]][d]['world']
            row = [str(d)] + ["%.4f" % w[k] for k in FACTOR_KEYS]
            for s in STRATS:
                r = by[s][d]
                row += ["%.4f" % r['lpsp'], "%.3f" % r['deg'], "%.3f" % r['uni']]
            f.write(";".join(row) + "\n")

        if draws_ok:
            f.write("\n## Stats par strategie (cout unifie, kEUR)\n")
            f.write("strat;mean;std;min;P5;P50;P95;max;CVaR90;LPSP_mean;deg_mean\n")
            for s in STRATS:
                u = col(s, 'uni')
                f.write("%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.4f;%.3f\n"
                        % (s, u.mean(), u.std(), u.min(),
                           np.quantile(u, 0.05), np.quantile(u, 0.50), np.quantile(u, 0.95),
                           u.max(), cvar_hi(u), col(s, 'lpsp').mean(), col(s, 'deg').mean()))

            f.write("\n## Differences appariees (CRN) du cout unifie [kEUR] (negatif = 1er gagne)\n")
            f.write("paire;mean;std;P5;P95;pct_gagne\n")
            pairs = [("RB2(SoH)", "RB2"), ("RB2(SoH)", "RB2(Sched)"), ("RB2(Sched)", "RB2")]
            for a, b in pairs:
                dif = col(a, 'uni') - col(b, 'uni')
                f.write("%s - %s;%.3f;%.3f;%.3f;%.3f;%.1f%%\n"
                        % (a, b, dif.mean(), dif.std(),
                           np.quantile(dif, 0.05), np.quantile(dif, 0.95),
                           100.0 * (dif < 0).mean()))

            f.write("\n## Regret vs meilleur des 3 par tirage [kEUR]\n")
            f.write("strat;mean;P95;max\n")
            best = np.min(np.vstack([col(s, 'uni') for s in STRATS]), axis=0)
            for s in STRATS:
                reg = col(s, 'uni') - best
                f.write("%s;%.3f;%.3f;%.3f\n" % (s, reg.mean(), np.quantile(reg, 0.95), reg.max()))

    # --- 5) figure ---
    if draws_ok:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        colors = {"RB2": "tab:blue", "RB2(Sched)": "tab:orange", "RB2(SoH)": "tab:green"}
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
        d1 = col("RB2(SoH)", 'uni') - col("RB2", 'uni')
        d2 = col("RB2(SoH)", 'uni') - col("RB2(Sched)", 'uni')
        bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=25)
        ax.hist(d1, bins=bins, alpha=0.55, color="tab:blue",   label="RB2(SoH) - RB2")
        ax.hist(d2, bins=bins, alpha=0.55, color="tab:orange", label="RB2(SoH) - RB2(Sched)")
        ax.axvline(0.0, color='k', lw=0.9)
        ax.set_xlabel("Difference appariee de cout unifie [kEUR]  (<0 = SoH gagne)")
        ax.set_ylabel("Tirages")
        ax.set_title("Valeur de la boucle fermee, par tirage (CRN)")
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)
        fig.suptitle("Valeur informationnelle du SoH sous incertitude de vieillissement (%d ans, N=%d)"
                     % (years, len(draws_ok)), fontsize=12)
        fig.tight_layout()
        fig.savefig(out_fig + ".pdf", bbox_inches="tight")
        fig.savefig(out_fig + ".png", dpi=160, bbox_inches="tight")
        plt.close()

    # --- 6) resume console ---
    print("\n" + "=" * 78)
    if draws_ok:
        print("COUT UNIFIE [kEUR] sous incertitude du vieillissement (N=%d)" % len(draws_ok))
        print("%-11s | %8s | %8s | %8s | %8s" % ("strat", "mean", "P95", "CVaR90", "regret"))
        best = np.min(np.vstack([col(s, 'uni') for s in STRATS]), axis=0)
        for s in STRATS:
            u = col(s, 'uni'); reg = u - best
            print("%-11s | %8.3f | %8.3f | %8.3f | %8.3f"
                  % (s, u.mean(), np.quantile(u, 0.95), cvar_hi(u), reg.mean()))
        d2 = col("RB2(SoH)", 'uni') - col("RB2(Sched)", 'uni')
        print("-" * 78)
        print("RB2(SoH) - RB2(Sched) : %.3f +/- %.3f kEUR ; gagne %.0f%% des tirages"
              % (d2.mean(), d2.std(), 100.0 * (d2 < 0).mean()))
    print("=" * 78)
    print("Resultats : %s" % out_txt)
    print("Figure    : %s.pdf" % out_fig)


if __name__ == "__main__":
    main()
