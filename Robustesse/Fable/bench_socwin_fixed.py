# -*- coding: utf-8 -*-
"""
bench_socwin_fixed.py -- CONTROLE du levier "plafond SoC vieillissant".
=======================================================================
Le levier RB2(SoH_bat) abaisse le plafond de SoC AVEC l'usure de la batterie :

    soc_max(t) = 0.995 - g * (1 - SoH_bat(t))          (Common/get_lol.py)

et le balayage bench_fable.py --sweep socwin conclut que g = 0.2 gagne
~-0.56 kEUR (net, indicateur unifie VoLL=3) contre g = 0.

QUESTION DE ROBUSTESSE traitee ici : ce gain vient-il vraiment de l'INFORMATION
SoH (adapter le plafond au vieillissement), ou simplement du fait d'ABAISSER le
plafond de SoC (confiner le cyclage sous ~0.6, ou la densite de degradation est
4-6x plus faible) -- ce qu'un plafond FIXE, independant du SoH, ferait aussi ?

On compare donc, sur le MEME moteur / la MEME metrique / les MEMES graines que
bench_fable :

    (1) socle                    soc_max = 0.995                       (g=0)
    (2) plafond FIXE             soc_max = 0.995 - g        (independant du SoH)
    (3) plafond VIEILLISSANT     soc_max = 0.995 - g*(1-SoH_bat)   (levier SoH_bat)

Un plafond fixe s'obtient SANS toucher get_lol.py : SOC_MAX = 0.995 - g avec
SOC_MAX_AGED_GAIN = 0. Le plafond vieillissant : SOC_MAX = 0.995 avec
SOC_MAX_AGED_GAIN = g. Les deux familles sont donc evaluees a egalite.

LECTURE :
  - si le MEILLEUR plafond fixe egale/bat le meilleur plafond vieillissant a
    LPSP comparable  -> l'info SoH n'apporte rien, seul compte "baisser le plafond".
  - si le plafond vieillissant reste devant a LPSP egale                -> l'info
    SoH a une valeur propre (il coupe la capacite utile SEULEMENT quand la
    batterie est deja abimee, en gardant la pleine plage tant qu'elle est neuve).

Runs DETERMINISTES sur le socle (levier prevision OFF), comme --sweep socwin.

USAGE (depuis Robustesse/Fable, meme env que bench_fable) :
    python bench_socwin_fixed.py            # 25 ans
    python bench_socwin_fixed.py 25         # n_years explicite
    python bench_socwin_fixed.py --quick    # fumee : 1 an (verifie que ca tourne)

Sorties (dans Fable/) : bench_socwin_fixed.txt (tableau) + _cloud.csv (brut).
"""
import os, sys, csv, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# On reutilise EXACTEMENT le moteur, la metrique, le chargeur de strategie et les
# constantes de bench_fable -> comparaison a isoperimetre, zero duplication de la
# physique. (bench_fable ajoute Predictions/ au sys.path et rend Common importable.)
import bench_fable as bf  # noqa: E402  (regle aussi sys.path pour Common)
from bench_fable import metrics, _load, VOLL, MC_SEED  # noqa: E402

SOC_MAX_NOMINAL = 0.995   # plafond de reference (== Common/get_lol.SOC_MAX)

# --- Grilles de balayage ------------------------------------------------------
# g = ampleur de l'abaissement du plafond. Pour le plafond FIXE, soc_max = 0.995-g
# EN PERMANENCE ; pour le plafond VIEILLISSANT, soc_max descend de 0 (neuf) a
# 0.3*g (fin de vie SoH=0.7). Grille fine encadrante cote fixe pour trouver le
# MEILLEUR plafond fixe, puis on le confronte au meilleur plafond vieillissant.
GRID_FIXED = [0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]
# Cote vieillissant : quelques points autour de l'optimum connu (socwin_fine : g*~0.2).
GRID_AGED  = [0.1, 0.2, 0.4]


def build_strats():
    """(label, dossier, overrides) -- socle, plafond fixe, plafond vieillissant.
    Tout sur le socle RB2(Prop) avec levier prevision OFF + bruit OFF (deterministe)."""
    prop = os.path.join(HERE, "RB2(Prop)")
    base_ov = {"ENABLE": False, "NOISE_ENABLE": False}
    strats = [("socle (g=0)", prop, dict(base_ov))]
    for g in GRID_FIXED:
        if g == 0.0:
            continue  # == socle, deja present
        strats.append((f"FIXE  soc_max={SOC_MAX_NOMINAL - g:.3f} (g={g:.3f})",
                       prop, dict(base_ov, **{"_lol:SOC_MAX": SOC_MAX_NOMINAL - g})))
    for g in GRID_AGED:
        strats.append((f"SoH   g={g:.2f} (soc_max 0.995->{SOC_MAX_NOMINAL - 0.3 * g:.3f})",
                       prop, dict(base_ov, **{"_lol:SOC_MAX_AGED_GAIN": g})))
    return strats


def evaluate(task):
    """Worker picklable : un run deterministe. Regle les DEUX flags de fenetre SoC
    de Common/get_lol (fixe via SOC_MAX, vieillissant via SOC_MAX_AGED_GAIN), tous
    deux remis a leur defaut historique a chaque tache (pool reutilise -> pas de
    fuite d'etat)."""
    from Common.main_init_and_loop_forecast import init_and_run_loop_forecast
    import Common.get_lol as _gl
    ov = task["overrides"]
    _gl.SOC_MAX           = ov.get("_lol:SOC_MAX", SOC_MAX_NOMINAL)
    _gl.SOC_MAX_AGED_GAIN = ov.get("_lol:SOC_MAX_AGED_GAIN", 0.0)
    _gl.LOL_COMBINED      = False
    s = _load(task["folder"])
    for k, v in ov.items():
        if not k.startswith("_lol:"):
            setattr(s, k, v)
    if hasattr(s, "reset"):
        s.reset()
    data = init_and_run_loop_forecast(s.get_optimal_action_RB, H_forecast=48,
                                      n_years=task["ny"])
    lpsp, deg, eens, ely_starts = metrics(data)
    total = deg + VOLL * eens / 1000.0
    out = dict(task)
    out.update(lpsp=lpsp, deg=deg, eens=eens, total=total, ely_starts=ely_starts)
    return out


def run(strats, ny, tag):
    tasks = [dict(label=lab, folder=fold, overrides=ov, seed=MC_SEED, ny=ny)
             for lab, fold, ov in strats]  # deterministe -> 1 graine
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    nw = min(nw, len(tasks))
    print(f"[bench_socwin_fixed] {len(tasks)} runs ({ny} ans), {nw} workers ...", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(evaluate, tasks)):
            results.append(r)
            print(f"  [{i+1:2d}/{len(tasks)}] {r['label']:<34s} "
                  f"LPSP={r['lpsp']:.4f}%  deg={r['deg']:.3f}  total={r['total']:.3f}"
                  f"  ELYstarts={r['ely_starts']}", flush=True)
    print(f"[bench_socwin_fixed] termine en {time.time()-t0:.0f} s", flush=True)

    # --- cloud CSV (brut) ---
    cloud = os.path.join(HERE, tag + "_cloud.csv")
    with open(cloud, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["label", "lpsp_pct", "deg_keur", "eens_kwh", "total_keur", "ely_starts"])
        for r in results:
            w.writerow([r["label"], f"{r['lpsp']:.6f}", f"{r['deg']:.6f}",
                        f"{r['eens']:.3f}", f"{r['total']:.6f}", r["ely_starts"]])

    # --- tableau (dtotal vs socle) ---
    base = results[0]["total"]
    lines = [f"# bench_socwin_fixed : {ny} ans ; VoLL={VOLL} ; deterministe (socle) ; "
             f"base=socle ; plafond FIXE (soc_max=0.995-g) vs VIEILLISSANT (0.995-g*(1-SoH))",
             "label ; LPSP(%) ; deg(kEUR) ; EENS(kWh) ; total(kEUR) ; dtotal_vs_socle(kEUR) ; ELY_starts"]
    for r in results:
        lines.append(f"{r['label']} ; {r['lpsp']:.4f} ; {r['deg']:.3f} ; {r['eens']:.1f} ; "
                     f"{r['total']:.3f} ; {r['total'] - base:+.3f} ; {r['ely_starts']}")
    txt = os.path.join(HERE, tag + ".txt")

    # --- verdict : comparaison A ISO-LPSP ---------------------------------
    # Comparer les "min total" bruts serait trompeur : un plafond fixe agressif
    # fait baisser total surtout en ECHANGEANT de la capacite utile (LPSP monte)
    # contre moins de degradation, via le taux VoLL. Le levier SoH, lui, gagne a
    # LPSP quasi-constant. La seule lecture juste : a LPSP EGAL, quel total ?
    # -> on interpole la courbe du plafond fixe (LPSP -> total, monotone en g)
    #    au LPSP de chaque reglage SoH, et on compare les total a ce point.
    fixed = [r for r in results if "FIXE" in r["label"]]
    aged  = [r for r in results if r["label"].startswith("SoH")]
    verdict = []
    if fixed and aged:
        # socle inclus dans la courbe fixe (g=0) pour ancrer le bas de la plage.
        fam = sorted([results[0]] + fixed, key=lambda r: r["lpsp"])
        lp_f  = np.array([r["lpsp"]  for r in fam])
        tot_f = np.array([r["total"] for r in fam])
        lo, hi = lp_f.min(), lp_f.max()
        verdict = ["",
                   "# === VERDICT : plafond FIXE vs VIEILLISSANT, A ISO-LPSP (VoLL=3) ===",
                   f"# socle : LPSP={results[0]['lpsp']:.4f}%  total={base:.3f} kEUR",
                   "# pour chaque reglage SoH : on lit le total d'un plafond FIXE amene au MEME LPSP.",
                   "# reglage SoH        LPSP_SoH   total_SoH   total_FIXE@memeLPSP   valeur_info_SoH"]
        for r in aged:
            extrap = "" if lo - 1e-9 <= r["lpsp"] <= hi + 1e-9 else "  (EXTRAPOLE!)"
            tot_fix_iso = float(np.interp(r["lpsp"], lp_f, tot_f))
            val = tot_fix_iso - r["total"]   # >0 : SoH strictement meilleur a iso-LPSP
            verdict.append(f"#   {r['label'][:16]:<16s}  {r['lpsp']:7.4f}%  {r['total']:8.3f}   "
                           f"{tot_fix_iso:8.3f}            {val:+.3f} kEUR{extrap}")
        verdict += [
            "# Lecture de 'valeur_info_SoH' = total(plafond fixe au meme LPSP) - total(SoH) :",
            "#   > 0  -> a LPSP egal le levier SoH fait mieux : l'info SoH a une valeur propre.",
            "#   ~ 0  -> un simple plafond fixe reproduit le gain : l'info SoH n'apporte rien.",
            "#   < 0  -> le plafond fixe fait meme mieux a LPSP egal : inclure le SoH est contre-productif.",
        ]
        if ny < 10:
            verdict += ["#",
                        f"# ATTENTION : horizon={ny} an(s). La batterie vieillit peu -> le plafond",
                        "#   vieillissant reste ~au plafond nominal et le levier SoH est SOUS-exerce.",
                        "#   Le verdict n'est fiable qu'a l'horizon complet (25 ans)."]

    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines + verdict) + "\n")
    print("\n".join(lines + verdict))
    print(f"\n[bench_socwin_fixed] -> {txt}\n[bench_socwin_fixed] -> {cloud}")
    return results


def main(argv):
    quick = "--quick" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    ny = 1 if quick else (int(args[0]) if args else 25)
    run(build_strats(), ny, "bench_socwin_fixed" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main(sys.argv)
