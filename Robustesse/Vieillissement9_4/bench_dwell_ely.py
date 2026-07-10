"""
bench_dwell_ely.py -- P4 : REDUIRE LES DEMARRAGES ELY PAR LA PREVISION
                      (+ diagnostic : la part reversible est-elle un gisement ?)
=============================================================================
SOURCE 100% ASCII (convention mesocentre).

MOTIVATION (proposition P4 de ../ANALYSE_CRITIQUE_integration_vieillissement.txt)
----------------------------------------------------------------------------------
Le modele V9 rend le start-stop ELY couteux : ~9 000-9 700 demarrages sur
25 ans x 11.7 uV/cycle ~ 105-113 mV/cellule, soit ~70 % d'UNE vie de stack
(seuil 150 mV) partie en demarrages. C'est le poste de la physique V9 que les
strategies n'exploitaient pas. Levier previsionnel : NE DEMARRER L'ELY QUE SI
LE SURPLUS PREVU DURE ASSEZ LONGTEMPS -- un demarrage pour une heure de
surplus coute plus (11.7 uV) qu'il ne stocke.
NB : la piste "reserve H2 saisonniere" (P5) est ecartee dans ce contexte
equatorial (La Reunion) : saisonnalite PV faible, pas de cycle annuel du
stock H2 a exploiter.

PROTOCOLE (attribution stricte, style A1-A4 / Fable)
----------------------------------------------------
Base = RB2(SoH) (constantes DWELL_C_ELY/DWELL_G_ELY ci-dessous, A MAINTENIR
ALIGNEES avec RB2(SoH)/get_optimal_action_RB.py). Trois familles, memes
graines de bruit (CRN) :
  base          RB2(SoH) nu (reference ; = omni a N=0, TEST NUL integre) ;
  minoff(N)     SANS INFORMATION : apres un arret ELY, redemarrage interdit
                pendant N heures (anti-cyclage temporel pur). Separe la
                valeur de l'ANTI-CYCLAGE de celle de la PREVISION ;
  omni(N)       PREVISIONNEL OMNISCIENT : demarrage autorise ssi les N
                prochaines heures ont TOUTES un surplus superieur au setpoint
                ELY courant (le vrai futur ; borne superieure) ;
  noisy(N)      idem avec prevision BRUITEE : bruit AR(1) horaire (rho=0.8)
                calibre pour retrouver le backtest LSTM a 18 h
                (sigma = 39.38 kWh, biais = -2.32 kWh), memes constantes de
                design que RB2(Pred)/MPC. N_SEEDS realisations par N.
Le filtre ne s'applique qu'aux DEMARRAGES (ELY OFF -> ON) : il ne peut que
RETARDER un demarrage, jamais en ajouter -> la reduction des starts est
garantie par construction ; le cout est du surplus non stocke (LPSP/H2).
Mise en oeuvre : wrapper de la vraie fonction RB2(SoH) -- le blocage passe
par SoH_ely_t = 0 (setpoint ELY = 0.300*Pmax*0^1.5 = 0), zero duplication de
la regle. Monde de vieillissement NOMINAL (le levier teste la physique
start-stop, pas l'incertitude des taux -- axes deja traites par P1/P3).

DIAGNOSTIC (ii) INCLUS : sur le run de base, decomposition ss/idle/rev/irr
de la degradation FC et ELY (get_cost_* standalone) + niveau moyen/max de la
part REVERSIBLE. Si le reversible residuel est negligeable (repos naturels
suffisants : tau_rest ELY ~ 0.3 min, FC ~ 0.5 h), le levier "repos
programmes" n'a PAS de gisement -> resultat negatif documente sans banc.

SORTIES (a cote de ce script)
-----------------------------
  dwell_ely_<Ny>y.txt      diagnostic + table par config + stats noisy
  dwell_ely_<Ny>y.pdf/.png (1) plan LPSP/deg parametre par N (3 familles)
                           (2) demarrages ELY vs N
                           (3) cout unifie vs N (base / minoff / omni / noisy)
LANCER
------
  local (fumee)  : python bench_dwell_ely.py --quick
  mesocentre     : sbatch run_meso_dwell.slurm     (~170 runs 25 ans, ~5 min)
Options : --nlist "2,4,6,8,12" | --seeds M | --years N | --workers N
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

import bench_valeur_info as VI                              # noqa: E402
from Common import Init_EMR_MG_v16_python as I              # noqa: E402
from Common import cost_fcn_total2 as C                     # noqa: E402
from Common.main_init_and_loop import init_and_run_loop     # noqa: E402

# ============================ CONFIGURATION ============================
N_YEARS  = 25
VOLL     = 3.0
N_LIST   = [2, 4, 6, 8, 12]     # duree minimale de surplus prevue [h] (--nlist)
N_SEEDS  = 32                   # realisations de bruit par N (--seeds)
SEED0    = 3026                 # graines CRN : SEED0+i pour toutes les configs

# Setpoint ELY de la base -- A MAINTENIR ALIGNE avec RB2(SoH)/get_optimal_action_RB.py
DWELL_C_ELY = 0.300
DWELL_G_ELY = 1.5

# Bruit de prevision : AR(1) horaire calibre sur le backtest LSTM a 18 h
# (memes constantes de design que RB2(Pred) / MPC : sigma_18h, biais_18h, rho).
SIGMA_18H_KWH = 39.38
BIAS_18H_KWH  = -2.32
NOISE_RHO     = 0.8
_H_CAL = 18
_f = _H_CAL + 2 * sum((_H_CAL - l) * NOISE_RHO ** l for l in range(1, _H_CAL))
SIGMA_H_W = SIGMA_18H_KWH * 1000.0 / np.sqrt(_f)   # ecart-type horaire [W]
BIAS_H_W  = BIAS_18H_KWH * 1000.0 / _H_CAL         # biais horaire [W]

STRATEGY = "RB2(SoH)"
# ======================================================================


# --------------------- wrapper "dwell" de RB2(SoH) ---------------------
# Filtre de demarrage ELY : quand l'ELY est OFF et qu'un surplus se presente,
# on n'autorise le demarrage que si la duree de surplus PREVUE >= N heures
# (omni : vrai futur ; noisy : + bruit AR(1) ; minoff : pas de prevision,
# simple delai apres le dernier arret). Blocage via SoH_ely_t = 0 (setpoint
# ELY nul dans la vraie RB2(SoH)) ; tout le reste de la regle est inchange.
_DW = {"base": None, "net": None, "j": 0, "prev_on": False,
       "N": 0, "mode": "omni", "rng": None, "last_stop": None}


def dwell_reset(N, mode, seed=0):
    if _DW["base"] is None:
        _DW["base"] = VI.load_strategy(STRATEGY)
    if _DW["net"] is None:
        # net "verite terrain" = meme formule que la boucle (P_dc_load - P_pv)
        _DW["net"] = (np.asarray(I.LOAD['P_ref'], dtype=float) / I.CONV['eta']
                      - np.asarray(I.PV['P'], dtype=float))
    _DW.update(j=0, prev_on=False, N=int(N), mode=mode,
               rng=np.random.default_rng(seed), last_stop=None)


def dwell_strategy(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                   alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                   P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
    j = _DW["j"]
    _DW["j"] = j + 1
    soh_ely_pass = SoH_ely_t
    N = _DW["N"]
    if N > 0 and P_tot_ref_t < 0 and not _DW["prev_on"]:
        if _DW["mode"] == "minoff":
            allow = (_DW["last_stop"] is None) or (j - _DW["last_stop"] >= N)
        else:
            P_set = DWELL_C_ELY * I.ELY['P_ely_max'] * SoH_ely_t ** DWELL_G_ELY
            seg = _DW["net"][j:j + N]
            if _DW["mode"] == "noisy" and len(seg):
                eps = np.empty(len(seg))
                e = 0.0
                for k in range(len(seg)):     # AR(1) stationnaire, re-tire a chaque appel
                    e = NOISE_RHO * e + np.sqrt(1 - NOISE_RHO ** 2) * _DW["rng"].standard_normal()
                    eps[k] = e
                seg = seg + SIGMA_H_W * eps + BIAS_H_W
            # demarrage autorise ssi surplus > setpoint sur TOUTES les N heures
            allow = len(seg) >= N and bool(np.all(seg < -P_set))
        if not allow:
            soh_ely_pass = 0.0
    action, lol = _DW["base"](SoC_t, P_tot_ref_t, defaillances, lol_tab,
                              alpha_fc_t, alpha_ely_t, SoH_bat_t, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t,
                              RUL_fc_t, RUL_ely_t, SoH_fc_t, soh_ely_pass)
    on = abs(action[2]) > 1e-9
    if _DW["prev_on"] and not on:
        _DW["last_stop"] = j
    _DW["prev_on"] = on
    return action, lol


# ------------------------------- metriques -------------------------------
def ely_starts(data):
    """Nombre de transitions OFF->ON de l'ELY (seuil = celui du modele de
    cout, 0.0005*Pmax, evalue au Pmax neuf -- compteur de reporting)."""
    P = np.abs(np.asarray(data['P_ely']))
    on = P >= 0.0005 * I.ELY['P_ely_max']
    return int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))


def deg_breakdown(data):
    """Decomposition standalone (%) ss/idle/rev/irr pour ELY et FC sur toute
    la trajectoire (get_cost_* integre depuis 0, conventions de la these)."""
    alpha_ely = data["alpha_ely"][:-1]; alpha_fc = data["alpha_fc"][:-1]
    _, ss_e, idle_e, rev_e, irr_e = C.get_cost_ely(alpha_ely, data["P_ely"])
    _, ss_f, idle_f, rev_f, irr_f = C.get_cost_fc(alpha_fc, data["P_fc"])
    return dict(ely=dict(ss=ss_e, idle=idle_e, rev=rev_e, irr=irr_e),
                fc=dict(ss=ss_f, idle=idle_f, rev=rev_f, irr=irr_f))


# ------------------------------- worker -------------------------------
def evaluate(task):
    """task = dict(family, N, seed, years)."""
    try:
        VI.apply_world(VI.NOMINAL_WORLD)
        if task['family'] == 'base':
            strat = VI.load_strategy(STRATEGY)
        else:
            dwell_reset(task['N'], {'minoff': 'minoff', 'omni': 'omni',
                                    'noisy': 'noisy'}[task['family']], task['seed'])
            strat = dwell_strategy
        data = init_and_run_loop(strat, n_years=task['years'])
        lpsp, deg, eens, uni = VI.metrics(data)
        lb, lf, le = VI.lifetimes(data)
        starts = ely_starts(data)
        ok = True
    except Exception as e:
        lpsp = deg = eens = uni = lb = lf = le = starts = None
        ok = False
        print("  [FAIL] %-7s N=%-3s seed=%s : %s" % (task['family'], task['N'], task['seed'], e), flush=True)
    return dict(family=task['family'], N=task['N'], seed=task['seed'],
                lpsp=lpsp, deg=deg, eens=eens, uni=uni, starts=starts,
                life_bat=lb, life_fc=lf, life_ely=le, ok=ok)


def _fmt(r):
    if not r['ok']:
        return "%-7s N=%-3s seed=%-3s FAIL" % (r['family'], r['N'], r['seed'])
    return ("%-7s N=%-3s seed=%-3s LPSP %7.4f%%  deg %7.3f  unifie %8.3f  starts %5d"
            % (r['family'], r['N'], r['seed'], r['lpsp'], r['deg'], r['uni'], r['starts']))


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="P4 : reduction des demarrages ELY par la prevision")
    ap.add_argument("--quick", action="store_true", help="fumee locale : 2 ans, N={2,4}, 2 graines")
    ap.add_argument("--nlist", type=str, default=None, help="ex. '2,4,6,8,12'")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    years   = args.years or (2 if args.quick else N_YEARS)
    n_list  = ([int(x) for x in args.nlist.split(",")] if args.nlist
               else ([2, 4] if args.quick else N_LIST))
    n_seeds = args.seeds if args.seeds is not None else (2 if args.quick else N_SEEDS)
    workers = args.workers or VI._detect_workers()

    tag = "%dy" % years
    out_txt = os.path.join(HERE, "dwell_ely_%s.txt" % tag)
    out_fig = os.path.join(HERE, "dwell_ely_%s" % tag)

    print("=== P4 -- DEMARRAGES ELY : filtre previsionnel de demarrage ===", flush=True)
    print("    horizon=%d ans | N=%s | %d graines/N | bruit AR(1) rho=%.1f sigma_h=%.0f W (18h: %.1f kWh)"
          % (years, n_list, n_seeds, NOISE_RHO, SIGMA_H_W, SIGMA_18H_KWH), flush=True)

    tasks = [dict(family='base', N=0, seed=0, years=years)]
    tasks += [dict(family='omni', N=0, seed=0, years=years)]        # test nul : == base
    for N in n_list:
        tasks.append(dict(family='minoff', N=N, seed=0, years=years))
        tasks.append(dict(family='omni',   N=N, seed=0, years=years))
        for s in range(n_seeds):
            tasks.append(dict(family='noisy', N=N, seed=SEED0 + s, years=years))

    print("\n--- %d runs (%d workers) ---" % (len(tasks), workers), flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(evaluate, tasks), 1):
            res.append(r)
            print("  [%3d/%d] %s" % (i, len(tasks), _fmt(r)), flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)

    base = next(r for r in res if r['family'] == 'base' and r['ok'])
    null = next((r for r in res if r['family'] == 'omni' and r['N'] == 0 and r['ok']), None)
    null_ok = None
    if null is not None:
        gap = abs(null['uni'] - base['uni'])
        null_ok = gap < 1e-6
        print("\nTEST NUL (omni N=0 == base) : ecart %.3e kEUR -> %s"
              % (gap, "OK" if null_ok else "ECHEC"), flush=True)

    # --- diagnostic (ii) : gisement reversible sur le run de base ---
    print("\n--- Diagnostic reversible/start-stop (run de base, re-simulation)...", flush=True)
    VI.apply_world(VI.NOMINAL_WORLD)
    data_b = init_and_run_loop(VI.load_strategy(STRATEGY), n_years=years)
    bd = deg_breakdown(data_b)
    vrev_ely = np.asarray(data_b['deg_ely']['reversible'])
    vrev_fc  = np.asarray(data_b['deg_fc']['reversible'])

    def one(d, vrev, name):
        tot = d['ss'] + d['idle'] + d['rev'] + d['irr']
        line = ("%s : total %.2f %% dont start-stop %.2f (%.0f %%) | irr %.2f | "
                "rev RESIDUEL %.3f (%.1f %%) | idle %.2f ; niveau rev moyen %.4f %%, max %.4f %%"
                % (name, tot, d['ss'], 100 * d['ss'] / tot if tot else 0, d['irr'],
                   d['rev'], 100 * d['rev'] / tot if tot else 0, d['idle'],
                   vrev.mean(), vrev.max()))
        print("  " + line, flush=True)
        return line
    diag_lines = [one(bd['ely'], vrev_ely, "ELY"), one(bd['fc'], vrev_fc, "FC ")]

    # --- agregats ---
    def rows(fam, N):
        return [r for r in res if r['family'] == fam and r['N'] == N and r['ok']]

    def agg(fam, N, key):
        v = np.array([r[key] for r in rows(fam, N)], dtype=float)
        return v.mean(), (v.std() if len(v) > 1 else 0.0), len(v)

    # --- txt ---
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# P4 -- filtre previsionnel de demarrage ELY (base %s, monde nominal)\n" % STRATEGY)
        f.write("# horizon=%d ans | N=%s | %d graines/N | AR(1) rho=%.1f sigma_h=%.0f W | VoLL=%.1f\n"
                % (years, n_list, n_seeds, NOISE_RHO, SIGMA_H_W, VOLL))
        f.write("# setpoint duplique : c_ely=%.3f gamma=%.1f (a maintenir aligne avec RB2(SoH))\n"
                % (DWELL_C_ELY, DWELL_G_ELY))
        if null_ok is not None:
            f.write("# TEST NUL omni(N=0)==base : %s\n" % ("OK" if null_ok else "ECHEC"))
        f.write("\n## Diagnostic (ii) -- gisement de la part reversible (run de base)\n")
        for l in diag_lines:
            f.write(l + "\n")
        f.write("\n## Reference\n")
        f.write("family;N;LPSP_%;deg_kEUR;unifie_kEUR;starts;vie_ely\n")
        f.write("base;0;%.4f;%.3f;%.3f;%d;%s\n"
                % (base['lpsp'], base['deg'], base['uni'], base['starts'], base['life_ely']))
        f.write("\n## Configurations (minoff/omni : deterministes ; noisy : moyenne +/- std sur %d graines)\n" % n_seeds)
        f.write("family;N;LPSP_mean;LPSP_std;deg_mean;deg_std;uni_mean;uni_std;starts_mean;d_uni_vs_base;N_ok\n")
        for fam in ("minoff", "omni", "noisy"):
            for N in n_list:
                if not rows(fam, N):
                    continue
                lm, ls, _ = agg(fam, N, 'lpsp')
                dm, ds, _ = agg(fam, N, 'deg')
                um, us, k = agg(fam, N, 'uni')
                sm, _, _  = agg(fam, N, 'starts')
                f.write("%s;%d;%.4f;%.4f;%.3f;%.3f;%.3f;%.3f;%.0f;%+.3f;%d\n"
                        % (fam, N, lm, ls, dm, ds, um, us, sm, um - base['uni'], k))

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    colors = {"minoff": "tab:gray", "omni": "tab:green", "noisy": "tab:orange"}
    labels = {"minoff": "minoff (sans info)", "omni": "prevision omnisciente",
              "noisy": "prevision bruitee (moy. %d graines)" % n_seeds}
    ax = axes[0]
    ax.scatter([base['lpsp']], [base['deg']], marker='*', s=160, color='k', zorder=6, label='base RB2(SoH)')
    for fam in ("minoff", "omni", "noisy"):
        xs = [agg(fam, N, 'lpsp')[0] for N in n_list if rows(fam, N)]
        ys = [agg(fam, N, 'deg')[0] for N in n_list if rows(fam, N)]
        ax.plot(xs, ys, 'o-', color=colors[fam], label=labels[fam])
        for N in n_list:
            if rows(fam, N):
                ax.annotate(str(N), (agg(fam, N, 'lpsp')[0], agg(fam, N, 'deg')[0]),
                            textcoords="offset points", xytext=(4, 3), fontsize=8, color=colors[fam])
    ax.set_xlabel("LPSP [%]"); ax.set_ylabel("Cout de degradation [kEUR]")
    ax.set_title("Plan LPSP-degradation parametre par N [h]")
    ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

    ax = axes[1]
    ax.axhline(base['starts'], color='k', ls=':', label='base (%d)' % base['starts'])
    for fam in ("minoff", "omni", "noisy"):
        ns = [N for N in n_list if rows(fam, N)]
        ax.plot(ns, [agg(fam, N, 'starts')[0] for N in ns], 'o-', color=colors[fam], label=labels[fam])
    ax.set_xlabel("N [h]"); ax.set_ylabel("Demarrages ELY (25 ans)")
    ax.set_title("Reduction des demarrages")
    ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

    ax = axes[2]
    ax.axhline(base['uni'], color='k', ls=':', label='base')
    for fam in ("minoff", "omni", "noisy"):
        ns = [N for N in n_list if rows(fam, N)]
        means = np.array([agg(fam, N, 'uni')[0] for N in ns])
        ax.plot(ns, means, 'o-', color=colors[fam], label=labels[fam])
        if fam == "noisy":
            stds = np.array([agg(fam, N, 'uni')[1] for N in ns])
            ax.fill_between(ns, means - stds, means + stds, color=colors[fam], alpha=0.2)
    ax.set_xlabel("N [h]"); ax.set_ylabel("Cout unifie [kEUR]")
    ax.set_title("Cout unifie vs N")
    ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

    fig.suptitle("P4 -- filtre previsionnel de demarrage ELY (%d ans, monde nominal)" % years, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_fig + ".pdf", bbox_inches="tight")
    fig.savefig(out_fig + ".png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- resume console ---
    print("\n" + "=" * 78)
    print("base : LPSP %.4f%%  deg %.3f  unifie %.3f  starts %d"
          % (base['lpsp'], base['deg'], base['uni'], base['starts']))
    for fam in ("minoff", "omni", "noisy"):
        for N in n_list:
            if not rows(fam, N):
                continue
            um, us, k = agg(fam, N, 'uni')
            sm, _, _ = agg(fam, N, 'starts')
            print("%-7s N=%-3d : unifie %8.3f%s  (d=%+.3f)  starts %5.0f"
                  % (fam, N, um, (" +/- %.3f" % us) if k > 1 else "", um - base['uni'], sm))
    print("=" * 78)
    print("Resultats : %s" % out_txt)
    print("Figure    : %s.pdf" % out_fig)


if __name__ == "__main__":
    main()
