"""
sens_timestep.py -- ETAPE 7 : sensibilite au PAS DE TEMPS et au BRUIT INTRA-HORAIRE.
====================================================================================
Reviewer APEN R3-major-4 (et R2-2) : avec un pas horaire, les evenements de
start-stop SOUS-HORAIRES sont agreges ; la domination du start-stop dans la
degradation (~96%% PEMFC, ~55%% PEMWE) pourrait donc etre un ARTEFACT de la
resolution. RR : "tu peux refaire avec un pas plus court, et aussi forcer un bruit
intra-horaire". On REJOUE donc la simulation a Ts=10min en injectant un bruit
gaussien intra-horaire de MOYENNE EGALE a la valeur horaire (preserve_mean) sur le
PV ET la charge, pour differentes amplitudes, et on mesure l'impact sur :
  - la part start-stop de la degradation PEMFC / PEMWE (le point du reviewer),
  - les indicateurs LPSP / cout,
  - l'ordre RB2 vs RB2(SoH).

Pourquoi c'est le bon test
--------------------------
Le modele de degradation (cost_fcn_total2) est INVARIANT en Ts PARTOUT sauf au
comptage des start-stop (np.diff(P_fc<1)) et au terme transient (somme |dP|) :
les couts haute/basse puissance et idle sont integres en temps (compteur*Ts), la
PEMWE est close-form invariante en Ts. Donc a Ts=10min SANS bruit, le profil
horaire est simplement maintenu 6x -> AUCUN nouveau franchissement de seuil ->
resultats ~identiques (verifie ici par la ligne sigma=0). C'est le BRUIT
intra-horaire qui cree des franchissements sous-horaires et fait l'experience.

Profil
------
On REGENERE le profil bruite a la volee A PARTIR DU PROFIL HORAIRE DE PRODUCTION
(I.sidelec_PV / I.sidelec_conso, celui reellement utilise par le modele), et NON
a partir du sidelec_roche_plate_10min.csv present sur disque (issu d'un script
brouillon, incoherent avec la chaine de prod : autre source de donnees, autre
echelle). add_intrahour_noise reprend la logique du prototype plot_noisy_profiles
(bruit gaussien relatif, recentrage par bloc horaire -> moyenne horaire preservee,
plancher a 0).

Methode (sans toucher Vieillissement8)
---------------------------------------
On MUTE EN PLACE I.LOAD['P_ref'], I.PV['P'], I.LOAD['Ts'] (memes objets dict que
ceux vus par main_init_and_loop et cost_fcn_total2 via 'import *') puis on restaure
les valeurs d'origine en fin de tache. La decomposition de degradation est relue
sur les tableaux COMPLETS via get_cost_fc / get_cost_ely (meme methodologie pour
1h et 10min -> comparaison equitable).

SOURCE 100%% ASCII (volontaire). NE MODIFIE RIEN d'autre dans Vieillissement8.

Sorties (dans ./results/) : sens_timestep.txt, sens_timestep.pdf.
Lancer :  python sens_timestep.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_common import (I, init_and_run_loop, load_strategy, metrics,
                         run_pool, RESULTS_DIR)
from Common import cost_fcn_total2 as CF   # get_cost_fc / get_cost_ely (decomposition)

# ============================ CONFIGURATION ============================
EMS_LIST = [
    ("RB2",      "RB2"),
    ("RB2(SoH)", "RB2(SoH)"),
]

TS_FINE   = 600           # pas fin [s] = 10 min
TS_COARSE = 3600          # pas horaire de reference [s]
REPEAT    = TS_COARSE // TS_FINE   # 6 sous-pas par heure

# Amplitudes de bruit intra-horaire (ecart-type RELATIF a la valeur horaire).
# 0.0 -> profil horaire simplement maintenu (sert de controle "10min sans bruit").
SIGMA_LIST = [0.0, 0.05, 0.10, 0.15]

N_YEARS_TILE = 51         # replication annuelle (comme la prod) ; couvre 25 ans a 10min
BASE_SEED    = 2026       # graine ; MEMES tirages pour les 2 EMS a sigma donne

OUT_TXT = os.path.join(RESULTS_DIR, "sens_timestep.txt")
OUT_PDF = os.path.join(RESULTS_DIR, "sens_timestep.pdf")
# ======================================================================


def add_intrahour_noise(hourly_values, repeat, rel_sigma, rng,
                        clip_min=0.0, preserve_mean=True):
    """Etend un profil horaire en sous-pas en ajoutant un bruit gaussien
    proportionnel autour de chaque moyenne horaire (cf. plot_noisy_profiles).
    rel_sigma=0 -> simple np.repeat (maintien). preserve_mean recentre chaque bloc
    de `repeat` points sur la valeur horaire -> moyenne horaire conservee."""
    base = np.repeat(np.asarray(hourly_values, dtype=float), repeat)
    if rel_sigma <= 0.0:
        return base
    sigma = rel_sigma * base                       # sigma nul quand valeur nulle (PV nuit)
    noisy = base + rng.normal(0.0, 1.0, size=base.shape) * sigma
    if clip_min is not None:
        noisy = np.clip(noisy, clip_min, None)
    if preserve_mean:
        blocks = noisy.reshape(-1, repeat)
        block_means = blocks.mean(axis=1, keepdims=True)
        target = base.reshape(-1, repeat)[:, :1]   # valeur horaire (constante par bloc)
        blocks = blocks - block_means + target
        noisy = blocks.reshape(-1)
        if clip_min is not None:
            noisy = np.clip(noisy, clip_min, None)
    return noisy


def _breakdown(data):
    """Decomposition de degradation sur les tableaux COMPLETS (meme methodo 1h/10min).
    Renvoie (ss_fc_share, ss_ely_share) en %% : part du start-stop dans la
    degradation totale PEMFC et PEMWE sur tout l'horizon."""
    alpha_fc  = data["alpha_fc"][:-1]
    alpha_ely = data["alpha_ely"][:-1]
    P_fc = data["P_fc"]; P_ely = data["P_ely"]
    # get_cost_fc -> (cost_eur, cost_on_off, cost_low, cost_shift, cost_high)
    _, c_ss, c_low, c_shift, c_high = CF.get_cost_fc(alpha_fc, P_fc)
    tot_fc = c_ss + c_low + c_shift + c_high
    ss_fc = 100.0 * c_ss / tot_fc if tot_fc > 0 else 0.0
    # get_cost_ely -> (cost_eur, deg_ss, deg_idle, deg_rev, deg_irr)
    _, e_ss, e_idle, e_rev, e_irr = CF.get_cost_ely(alpha_ely, P_ely)
    tot_ely = e_ss + e_idle + e_rev + e_irr
    ss_ely = 100.0 * e_ss / tot_ely if tot_ely > 0 else 0.0
    return float(ss_fc), float(ss_ely)


def evaluate(params):
    """Worker picklable. params = dict(folder, ems, ts, sigma, seed).
    Mute I.LOAD/I.PV/I.LOAD['Ts'] en place, lance, relit la decomposition, restaure."""
    # Sauvegarde des objets d'origine (restauration en finally)
    orig_pref = I.LOAD['P_ref']
    orig_pv   = I.PV['P']
    orig_ts   = I.LOAD['Ts']
    try:
        ts    = params['ts']
        sigma = params['sigma']
        repeat = TS_COARSE // ts                       # 1 si 3600, 6 si 600
        rng = np.random.default_rng(params['seed'])    # memes tirages -> meme bruit pour les 2 EMS
        # Profil horaire de PRODUCTION, replique sur N_YEARS_TILE annees
        h_load = np.tile(np.asarray(I.sidelec_conso, dtype=float), N_YEARS_TILE)
        h_pv   = np.tile(np.asarray(I.sidelec_PV,   dtype=float), N_YEARS_TILE)
        # NB : meme rng consomme pour charge puis PV -> reproductible et identique
        # entre EMS (memes seed et meme ordre d'appel).
        load_prof = add_intrahour_noise(h_load, repeat, sigma, rng)
        pv_prof   = add_intrahour_noise(h_pv,   repeat, sigma, rng)

        I.LOAD['P_ref'] = load_prof
        I.PV['P']       = pv_prof
        I.LOAD['Ts']    = ts

        strat = load_strategy(params['folder'])
        data  = init_and_run_loop(strat)
        lpsp, cost = metrics(data)
        ss_fc, ss_ely = _breakdown(data)
        lb = lf = le = None
        # vies (premier remplacement) en annees
        yr = ts / 3600.0 / 24.0 / 365.0
        for key, sl in (("SoH_bat", "b"), ("SoH_fc", "f"), ("SoH_ely", "e")):
            s = np.asarray(data[key]); rep = np.where((s[1:] == 1) & (s[:-1] != 1))[0]
            v = float(rep[0] * yr) if len(rep) > 0 else None
            if sl == "b": lb = v
            elif sl == "f": lf = v
            else: le = v
        ok = True
    except Exception as e:
        lpsp = cost = ss_fc = ss_ely = lb = lf = le = None
        ok = False
        print("  [FAIL] %-9s ts=%ss sigma=%.2f : %s"
              % (params['ems'], params['ts'], params['sigma'], e), flush=True)
    finally:
        I.LOAD['P_ref'] = orig_pref
        I.PV['P']       = orig_pv
        I.LOAD['Ts']    = orig_ts
    return dict(ems=params['ems'], ts=params['ts'], sigma=params['sigma'],
                lpsp=lpsp, cost=cost, ss_fc=ss_fc, ss_ely=ss_ely,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _yr(x):
    return "%.1f" % x if x is not None else ">hor"


def _fmt(r):
    if not r['ok']:
        return "%-9s ts=%ss sigma=%.2f -> FAIL" % (r['ems'], r['ts'], r['sigma'])
    return ("%-9s ts=%4ss sig=%.2f -> LPSP %6.4f%%  deg %7.2f kEUR  ss_FC %5.1f%%  "
            "ss_ELY %5.1f%%  vie_bat %s"
            % (r['ems'], r['ts'], r['sigma'], r['lpsp'], r['cost'],
               r['ss_fc'], r['ss_ely'], _yr(r['life_bat'])))


def main():
    print("=== ETAPE 7 -- Pas de temps & bruit intra-horaire (RB2, RB2(SoH), 25 ans) ===", flush=True)
    print("    Ref 1h + 10min sigma in %s (PV+charge, moyenne horaire preservee)"
          % SIGMA_LIST, flush=True)

    tasks = []
    # Reference 1h (profil de prod, sans bruit) -- doit reproduire le papier
    for folder, ems in EMS_LIST:
        tasks.append(dict(folder=folder, ems=ems, ts=TS_COARSE, sigma=0.0,
                          seed=BASE_SEED))
    # Balayage 10min x sigma (memes tirages pour les 2 EMS a sigma donne)
    for si, sigma in enumerate(SIGMA_LIST):
        for folder, ems in EMS_LIST:
            tasks.append(dict(folder=folder, ems=ems, ts=TS_FINE, sigma=sigma,
                              seed=BASE_SEED + si))

    res = run_pool(evaluate, tasks, "Pas de temps -- 1h ref + 10min x sigma", _fmt)
    ok = [r for r in res if r['ok']]

    # ===================== SAUVEGARDE TXT (ASCII) =====================
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Sensibilite pas de temps & bruit intra-horaire -- RB2, RB2(SoH), 25 ans\n")
        f.write("# Ts_fine=%ds (repeat=%d), sigma=%s, PV+charge, preserve_mean, seed_base=%d\n"
                % (TS_FINE, REPEAT, SIGMA_LIST, BASE_SEED))
        f.write("# ss_FC/ss_ELY = part start-stop dans la degradation totale (%)\n\n")
        f.write("ems;Ts_s;sigma;LPSP_%;deg_kEUR;ss_FC_%;ss_ELY_%;vie_bat;vie_fc;vie_ely\n")
        for r in res:
            if not r['ok']:
                f.write("%s;%s;%.2f;FAIL\n" % (r['ems'], r['ts'], r['sigma'])); continue
            f.write("%s;%d;%.2f;%.4f;%.3f;%.2f;%.2f;%s;%s;%s\n"
                    % (r['ems'], r['ts'], r['sigma'], r['lpsp'], r['cost'],
                       r['ss_fc'], r['ss_ely'], _yr(r['life_bat']),
                       _yr(r['life_fc']), _yr(r['life_ely'])))

    # ===================== FIGURE : parts start-stop & indicateurs vs sigma =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {'RB2': 'tab:blue', 'RB2(SoH)': 'tab:red'}
    for _, ems in EMS_LIST:
        sub = sorted([r for r in ok if r['ems'] == ems and r['ts'] == TS_FINE],
                     key=lambda r: r['sigma'])
        if not sub:
            continue
        sig = [r['sigma'] * 100 for r in sub]
        col = colors.get(ems, None)
        axes[0].plot(sig, [r['ss_fc'] for r in sub], 'o-', color=col, label=ems)
        axes[1].plot(sig, [r['ss_ely'] for r in sub], 'o-', color=col, label=ems)
        axes[2].plot(sig, [r['cost'] for r in sub], 'o-', color=col, label="%s deg" % ems)
        # reference 1h (sigma=0 horaire) en pointilles horizontaux
        ref = [r for r in ok if r['ems'] == ems and r['ts'] == TS_COARSE]
        if ref:
            axes[0].axhline(ref[0]['ss_fc'],  color=col, ls=':', lw=1, alpha=0.6)
            axes[1].axhline(ref[0]['ss_ely'], color=col, ls=':', lw=1, alpha=0.6)
            axes[2].axhline(ref[0]['cost'],   color=col, ls=':', lw=1, alpha=0.6)
    axes[0].set_ylabel("Part start-stop PEMFC [%]")
    axes[1].set_ylabel("Part start-stop PEMWE [%]")
    axes[2].set_ylabel("Cout de degradation [kEUR]")
    for ax in axes:
        ax.set_xlabel("Ecart-type du bruit intra-horaire [% de la valeur horaire]")
        ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)
    axes[0].set_title("Pointilles = reference 1h", fontsize=10)
    fig.suptitle("Robustesse au pas de temps : 10min + bruit intra-horaire vs 1h", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    # ===================== RESUME CONSOLE =====================
    print("\n" + "=" * 88)
    print("PART START-STOP & INDICATEURS : 1h ref -> 10min (sigma croissant)")
    print("-" * 88)
    print("%-9s | Ts   | sigma | LPSP%%   | deg kEUR | ss_FC%% | ss_ELY%%" % "EMS")
    for r in res:
        if not r['ok']:
            print("%-9s | %4ss | FAIL" % (r['ems'], r['ts'])); continue
        print("%-9s | %4ss | %.2f  | %6.4f | %8.2f | %6.1f | %6.1f"
              % (r['ems'], r['ts'], r['sigma'], r['lpsp'], r['cost'],
                 r['ss_fc'], r['ss_ely']))
    print("=" * 88)
    print("Resultats : %s" % OUT_TXT)
    print("Figure    : %s" % OUT_PDF)


if __name__ == "__main__":
    main()
