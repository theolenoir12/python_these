"""
sweep_setpoints_rb1.py
======================
Recherche des MEILLEURS seuils de RB1 au sens du COUT UNIFIE (degradation +
valorisation VoLL de la LPSP), sur le MEME principe que le sweep des setpoints
RB2 (Vieillissement8/RB2/sweep_setpoints_rb2.py + rank_unified_cost.py).

RB1 (RB1/get_optimal_action_RB.py) est parametree par DEUX seuils de SoC qui
definissent la bande de melange lineaire batterie/H2 :
  - SOC_LOW  : en-dessous, la FC couvre TOUT le deficit (batterie protegee) ;
  - SOC_HIGH : au-dessus (deficit) la batterie couvre tout ; en surplus, au-dela
               de SOC_HIGH la batterie sature et l'ELY absorbe le reste ;
  - entre les deux (deficit), la fraction batterie monte lineairement.
Defaut actuel du code : (SOC_LOW, SOC_HIGH) = (0.40, 0.75).

On balaie une grille (SOC_LOW, SOC_HIGH). Chaque couple -> 1 simulation 25 ans
NOMINALE (init_and_run_loop, memes hypotheses que le sweep RB2) -> 1 point
(LPSP %, cout de degradation k€). On valorise ensuite la LPSP avec la VoLL
EXACTEMENT definie dans Analyse_sensibilite/voll_common.py (VoLL=3 EUR/kWh
constante, E_REF=273380.73 kWh sur 25 ans) pour obtenir le COUT UNIFIE, puis on
classe et on identifie le minimiseur.

Contrairement a Defaillances/sweep_rb1.py (qui optimise la ROBUSTESSE sous
defaillance via un harness Monte-Carlo), ce script optimise le cout unifie en
marche NOMINALE, exactement dans l'esprit du sweep des setpoints RB2.

Sorties (dans le dossier RB1/) :
  - sweep_setpoints_rb1.txt          : tous les points (LPSP, deg).
  - sweep_setpoints_rb1_unified.txt  : classement par cout unifie + minimiseur.
  - sweep_setpoints_rb1_unified.pdf/.png : nuage + minimiseur annote.

Lancement (env conda simu_env, depuis Vieillissement8/RB1/) :
    python sweep_setpoints_rb1.py            # sweep complet + classement + figure
    python sweep_setpoints_rb1.py --smoke    # smoke-test (2 sims, horizon court)
    python sweep_setpoints_rb1.py --replot    # refait classement+figure depuis le .txt
Regler via SLURM_CPUS_PER_TASK (sinon coeurs-1) le nb de workers.
"""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from concurrent.futures import ProcessPoolExecutor

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../Vieillissement8/RB1
PARENT = os.path.dirname(HERE)                               # .../Vieillissement8
sys.path.insert(0, PARENT)                                   # pour importer Common
from Common import Init_EMR_MG_v16_python as I
from Common.main_init_and_loop import init_and_run_loop
from Common.cost_fcn_total2 import get_cost_from_ledger
from Common.get_lol import get_lol

# voll_common vit dans Robustesse/Analyse_sensibilite/
SENS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Analyse_sensibilite"))
sys.path.insert(0, SENS_DIR)
import voll_common as V   # source de verite pour la VoLL / le cout unifie

# ======================= CONFIGURATION =======================
N_YEARS   = 25                       # horizon (defaut du coeur)
# Grille des deux seuils de SoC (fractions de capacite batterie).
# Etendue vers le BAS + affinee pres du coin gagnant : le 1er sweep plafonnait a
# (0.20, 0.60), sur le BORD de la grille (tendance monotone) -> le minimum n'etait
# pas encadre. On descend SOC_LOW jusqu'a 0.05 et SOC_HIGH jusqu'a 0.40 pour le
# borner (au risque, assume, de faire exploser la LPSP quand la batterie sature).
GRID_LOW  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # SOC_LOW  candidats
GRID_HIGH = [0.30, 0.35, 0.40, 0.45, 0.45, 0.50]   # SOC_HIGH candidats
MIN_GAP   = 0.15                          # contrainte SOC_HIGH - SOC_LOW >= MIN_GAP
RB1_DEFAULT = (0.40, 0.75)                # RB1 actuelle (reference, surlignee)
RB1_OLD     = (0.20, 0.60)                # ancien reglage (repere)
# Nb de coeurs : honore SLURM_CPUS_PER_TASK sur le mesocentre, sinon coeurs-1.
_N_AVAIL  = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", (os.cpu_count() or 2) - 1)))

OUT_TXT   = os.path.join(HERE, "sweep_setpoints_rb1.txt")
OUT_UTXT  = os.path.join(HERE, "sweep_setpoints_rb1_unified.txt")
OUT_PDF   = os.path.join(HERE, "sweep_setpoints_rb1_unified.pdf")
OUT_PNG   = os.path.join(HERE, "sweep_setpoints_rb1_unified.png")
# =============================================================


def make_rb1(soc_low, soc_high):
    """Fabrique une action RB1 IDENTIQUE a l'originale (RB1/get_optimal_action_RB)
    mais parametree par (soc_low, soc_high). Signature 15 args (compatible coeur)."""
    a = float(soc_low)
    b = float(soc_high)

    def get_optimal_action_RB(SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                              alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                              P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t):
        if P_tot_ref_t > 0:                         # --- DEFICIT (decharge) ---
            if SoC_t <= a:
                frac = 0.0                          # FC couvre tout
            elif SoC_t >= b:
                frac = 1.0                          # batterie couvre tout
            else:
                frac = (SoC_t - a) / (b - a)        # melange lineaire
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_fc_t  = P_tot_ref_t - P_dc_bat_t
            P_dc_ely_t = 0.0
        else:                                        # --- SURPLUS (charge) ---
            if SoC_t <= b:
                frac = 1.0                          # batterie absorbe tout, ELY off
            elif SoC_t >= 1.0:
                frac = 0.0
            else:
                frac = (1.0 - SoC_t) / (1.0 - b)    # au-dela de b : ELY prend le reste
            P_dc_bat_t = P_tot_ref_t * frac
            P_dc_ely_t = P_tot_ref_t - P_dc_bat_t
            P_dc_fc_t  = 0.0

        if 'FC' in defaillances and P_tot_ref_t > 0:
            P_dc_bat_t = P_tot_ref_t
        if 'ELY' in defaillances and P_tot_ref_t < 0:
            P_dc_bat_t = P_tot_ref_t

        action = P_dc_bat_t, P_dc_fc_t, P_dc_ely_t
        action, lol = get_lol(SoC_t, action, P_tot_ref_t, defaillances, E_h2_t,
                              E_h2_init, P_fc_max_t, P_ely_max_t, SoH_bat_t)
        return action, lol

    return get_optimal_action_RB


def _compute_metrics(data):
    """(LPSP %, cout k€) EXACTEMENT comme sweep_setpoints_rb2._compute_metrics
    (interpolation NaN de SoH_bat avant remplacement)."""
    P_bat = data["P_bat"]; P_fc = data["P_fc"]; P_ely = data["P_ely"]
    P_dc_load = data["P_dc_load"]; P_dc_pv = data["P_dc_pv"]; lol_tab = data["lol_tab"]
    SoC = data["SoC"]
    alpha_fc  = data["alpha_fc"][:-1]
    alpha_ely = data["alpha_ely"][:-1]
    SoH_bat   = data["SoH_bat"][:-1].copy()
    for k in range(1, len(SoH_bat)):
        if SoH_bat[k] == 1:
            SoH_bat[k - 1] = np.nan
    if np.isnan(SoH_bat).any():
        SoH_bat[np.isnan(SoH_bat)] = np.interp(
            np.flatnonzero(np.isnan(SoH_bat)),
            np.flatnonzero(~np.isnan(SoH_bat)),
            SoH_bat[~np.isnan(SoH_bat)])
    P_planned = np.array([(a - b) / 1000 for a, b in zip(P_dc_load, P_dc_pv)])
    P_real    = np.array([(a - b) * (1 - c) / 1000 for a, b, c in zip(P_dc_load, P_dc_pv, lol_tab)])
    p, r = np.clip(P_planned, 0, None), np.clip(P_real, 0, None)
    load = np.clip(np.asarray(P_dc_load, dtype=float) / 1000.0, 0, None)
    lpsp = (np.clip(p - r, 0, None).sum() / load.sum() * 100) if load.sum() > 0 else 0.0
    cost_keur = get_cost_from_ledger(data) / 1000.0
    return float(lpsp), float(cost_keur)


def _eval(args):
    a, b, n_years = args
    data = init_and_run_loop(make_rb1(a, b), n_years=n_years)
    lpsp, cost = _compute_metrics(data)
    return (a, b, lpsp, cost)


# ----------------------------- run -----------------------------
def build_grid():
    combos = [(a, b) for a in GRID_LOW for b in GRID_HIGH if b - a >= MIN_GAP - 1e-9]
    for ref in (RB1_DEFAULT, RB1_OLD):
        if ref not in combos:
            combos.append(ref)
    return sorted(set(combos))


def run_sweep(smoke=False):
    n_years = 2 if smoke else N_YEARS
    if smoke:
        combos = [(RB1_DEFAULT[0], RB1_DEFAULT[1], n_years),
                  (RB1_OLD[0], RB1_OLD[1], n_years)]
    else:
        combos = [(a, b, n_years) for (a, b) in build_grid()]
    n_workers = max(1, min(_N_AVAIL, len(combos)))
    print(f"--- Sweep RB1 : {len(combos)} couples (SoC_low,SoC_high) sur {n_years} ans "
          f"({n_workers} workers) ---", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_eval, combos), 1):
            results.append(res)
            print(f"  [{i:2d}/{len(combos)}] low={res[0]:.3f} high={res[1]:.3f} "
                  f"-> LPSP={res[2]:6.3f}%  cout={res[3]:8.3f} k€", flush=True)
    print(f"--- termine en {time.time()-t0:.0f}s ---", flush=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Sweep seuils RB1 (SoC_low, SoC_high), simu nominale {n_years} ans\n")
        f.write(f"# MIN_GAP={MIN_GAP}  RB1_DEFAULT={RB1_DEFAULT}  RB1_OLD={RB1_OLD}\n")
        f.write("kind;soc_low;soc_high;LPSP(%);Cost(kEUR)\n")
        for a, b, lpsp, cost in results:
            tag = "RB1_sweep"
            if (a, b) == RB1_DEFAULT:
                tag = "RB1_default"
            elif (a, b) == RB1_OLD:
                tag = "RB1_old"
            f.write(f"{tag};{a:.3f};{b:.3f};{lpsp:.4f};{cost:.4f}\n")
    print(f"Ecrit : {OUT_TXT}", flush=True)
    return OUT_TXT


# ----------------------------- rank + plot -----------------------------
def _read_txt():
    rows = []
    with open(OUT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            p = line.strip().split(";")
            if len(p) != 5:
                continue
            kind, a, b, lpsp, cost = p
            rows.append(dict(kind=kind, a=float(a), b=float(b),
                             lpsp=float(lpsp), deg=float(cost)))
    return rows


def rank_and_plot():
    rows = _read_txt()
    for r in rows:
        r["clpsp"] = V.cost_lpsp_keur(r["lpsp"])
        r["total"] = V.total_cost_keur(r["lpsp"], r["deg"])   # deg + VoLL*LPSP
    rows_sorted = sorted(rows, key=lambda r: r["total"])
    best = rows_sorted[0]
    default = next((r for r in rows if r["kind"] == "RB1_default"), None)

    # --- ecriture classement ---
    with open(OUT_UTXT, "w", encoding="utf-8") as f:
        f.write("# Cout unifie = deg + VoLL*(LPSP/100)*E_REF/1000  (voll_common.py)\n")
        f.write(f"# VoLL={V.VOLL_TIERS}  E_REF_KWH={V.E_REF_KWH:.3f}  "
                f"-> {V.cost_lpsp_keur(1.0):.4f} kEUR par point de LPSP\n")
        f.write("rang;kind;soc_low;soc_high;LPSP(%);deg(kEUR);coutLPSP(kEUR);total(kEUR)\n")
        for i, r in enumerate(rows_sorted, 1):
            f.write(f"{i};{r['kind']};{r['a']:.3f};{r['b']:.3f};{r['lpsp']:.4f};"
                    f"{r['deg']:.4f};{r['clpsp']:.4f};{r['total']:.4f}\n")
        f.write(f"\n# MIN cout unifie : SoC_low={best['a']:.3f} SoC_high={best['b']:.3f}"
                f" -> total={best['total']:.4f} kEUR\n")
        if default is not None:
            f.write(f"# RB1 actuelle ({RB1_DEFAULT[0]:.2f}/{RB1_DEFAULT[1]:.2f}) : "
                    f"total={default['total']:.4f} kEUR "
                    f"(ecart best {best['total']-default['total']:+.4f} kEUR)\n")

    # --- console ---
    print(f"\nVoLL={V.VOLL_TIERS}  ({V.cost_lpsp_keur(1.0):.4f} kEUR / point LPSP)\n")
    print("rang  kind          low    high   LPSP%    deg     +LPSP    = total kEUR")
    for i, r in enumerate(rows_sorted, 1):
        star = "  <--" if r is best else ""
        print(f"{i:>3}  {r['kind']:<12} {r['a']:.3f}  {r['b']:.3f}  "
              f"{r['lpsp']:6.3f}  {r['deg']:6.2f}  {r['clpsp']:6.2f}   "
              f"{r['total']:7.3f}{star}")
    print(f"\n>>> MIN cout unifie : SoC_low={best['a']:.3f} SoC_high={best['b']:.3f}"
          f"  -> {best['total']:.3f} kEUR")
    if default is not None:
        print(f"    RB1 actuelle ({RB1_DEFAULT[0]:.2f}/{RB1_DEFAULT[1]:.2f}) : "
              f"{default['total']:.3f} kEUR  (ecart {best['total']-default['total']:+.3f})")

    # --- figure ---
    LABEL_STROKE = [pe.withStroke(linewidth=2.0, foreground='white')]
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 18, "axes.titlesize": 15, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8, 6))

    # nuage des seuils balayes (gris)
    for r in rows:
        if r is best or (default is not None and r is default):
            continue
        ax.scatter(r["lpsp"], r["deg"], color="0.72", s=45, zorder=2)
        ax.annotate(f"{r['a']:.2f}/{r['b']:.2f}", (r["lpsp"], r["deg"]),
                    fontsize=7, color="0.5", xytext=(3, 3),
                    textcoords="offset points", zorder=2)

    # RB1 actuelle (reference)
    if default is not None:
        ax.scatter(default["lpsp"], default["deg"], marker="s", color="black", s=60,
                   zorder=5, label=(f"RB1 actuelle ({default['a']:.2f}/{default['b']:.2f})"
                                    f"  {default['total']:.1f} k€"))
        ax.annotate(f"{default['total']:.1f} k€", (default["lpsp"], default["deg"]),
                    fontsize=10, color="black", path_effects=LABEL_STROKE,
                    xytext=(6, -12), textcoords="offset points", zorder=5)

    # minimiseur du cout unifie
    ax.scatter(best["lpsp"], best["deg"], marker="*", color="crimson", s=300, zorder=6,
               edgecolors="white", linewidths=0.8,
               label=(f"RB1 optimal ({best['a']:.2f}/{best['b']:.2f})"
                      f"  {best['total']:.1f} k€"))
    ax.annotate(f"{best['total']:.1f} k€", (best["lpsp"], best["deg"]),
                fontsize=11, color="crimson", weight="bold", path_effects=LABEL_STROKE,
                xytext=(6, 6), textcoords="offset points", zorder=6)

    # droites d'iso-cout unifie (pente = -VoLL*E_REF/1e5)
    slope = -V.cost_lpsp_keur(1.0)
    xs = np.array([min(r["lpsp"] for r in rows) - 0.05,
                   max(r["lpsp"] for r in rows) + 0.05])
    pts = [(best, "crimson")]
    if default is not None:
        pts.append((default, "black"))
    for r, c in pts:
        ax.plot(xs, r["deg"] + slope * (xs - r["lpsp"]), ls=":", color=c, lw=1.0,
                alpha=0.6, zorder=1)

    ax.set_xlabel("LPSP [%]")
    ax.set_ylabel("Coût de dégradation [k€]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_title("Coût unifié (deg + VoLL·LPSP) : seuils RB1 optimaux (25 ans)")
    plt.tight_layout()
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEcrit : {OUT_UTXT}\n        {OUT_PDF}\n        {OUT_PNG}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 sims, horizon court")
    ap.add_argument("--replot", action="store_true",
                    help="refait classement+figure depuis le .txt")
    args = ap.parse_args()
    if args.replot:
        rank_and_plot()
    else:
        run_sweep(smoke=args.smoke)
        rank_and_plot()
