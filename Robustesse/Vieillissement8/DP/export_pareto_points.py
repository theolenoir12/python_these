# -*- coding: utf-8 -*-
"""
export_pareto_points.py -- ecrit les coordonnees de TOUS les points du front PD.
================================================================================
Relit le .npz du front (dp_pareto_25y_51x51.npz) et ecrit un .txt lisible avec,
pour chaque point : eps, LPSP %, cout de degradation kEUR, EENS, cout unifie, et
un FLAG DE DOMINANCE.

Motivation : le front est obtenu par epsilon-contrainte (un run DP par valeur de
eps). Rien ne garantit que chaque point soit Pareto-optimal : un point peut etre
GLOBALEMENT DOMINE par un autre (LPSP ET degradation au moins aussi bons, l'un
strictement meilleur). C'est typiquement le cas A GAUCHE (bas LPSP), ou plusieurs
points partagent un LPSP quasi identique mais une degradation plus elevee.

Convention : on MINIMISE les deux objectifs (LPSP, deg). Le point i est domine si
il existe j != i tel que  LPSP_j <= LPSP_i  ET  deg_j <= deg_i, avec au moins une
inegalite stricte.

Sortie -> meme dossier que le .npz : dp_pareto_points_25y.txt
Usage  : python export_pareto_points.py [chemin/dp_pareto_25y_51x51.npz]
"""
import os
import sys
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))


def find_npz():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for sub in ("results_meso2/results", "results_meso", "results"):
        p = os.path.join(_THIS, *sub.split("/"), "dp_pareto_25y_51x51.npz")
        if os.path.exists(p):
            return p
    sys.exit("dp_pareto_25y_51x51.npz introuvable.")


def dominance(lpsp, deg):
    """Renvoie (dominated[bool], by[idx ou -1]). Minimisation de (lpsp, deg)."""
    n = len(lpsp)
    dominated = np.zeros(n, dtype=bool)
    by = np.full(n, -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if lpsp[j] <= lpsp[i] and deg[j] <= deg[i] and (
                    lpsp[j] < lpsp[i] or deg[j] < deg[i]):
                dominated[i] = True
                by[i] = j
                break
    return dominated, by


def main():
    npz = find_npz()
    d = np.load(npz)
    eps, lpsp, deg = d["eps"], d["lpsp"], d["deg_keur"]
    eens = d["eens_kwh"] if "eens_kwh" in d else np.full_like(lpsp, np.nan)
    unif = d["unif3_keur"] if "unif3_keur" in d else np.full_like(lpsp, np.nan)

    order = np.argsort(lpsp)                       # tri par LPSP croissant
    eps, lpsp, deg = eps[order], lpsp[order], deg[order]
    eens, unif = eens[order], unif[order]
    dominated, by = dominance(lpsp, deg)
    # remappe l'index "domine par" vers la NUMEROTATION triee (lisible)
    pos = {old: new for new, old in enumerate(range(len(lpsp)))}

    out = os.path.join(os.path.dirname(npz), "dp_pareto_points_25y.txt")
    with open(out, "w") as f:
        f.write("# Points du front de Pareto PD (programmation dynamique), 25 ans.\n")
        f.write("# Source : %s\n" % os.path.relpath(npz, _THIS).replace("\\", "/"))
        f.write("# Objectifs MINIMISES : LPSP [%] et cout de degradation [kEUR].\n")
        f.write("# 'dom' = 1 si le point est GLOBALEMENT DOMINE (candidat a retrait) ;\n")
        f.write("#         'par' donne l'indice du point qui le domine.\n")
        f.write("# Tri par LPSP croissant.\n")
        f.write("#\n")
        f.write("idx;eps;LPSP_%;deg_kEUR;EENS_kWh;unif3_kEUR;dom;par\n")
        for i in range(len(lpsp)):
            f.write("%2d;%.5f;%.4f;%.4f;%.1f;%.4f;%d;%s\n"
                    % (i, eps[i], lpsp[i], deg[i], eens[i], unif[i],
                       1 if dominated[i] else 0,
                       ("%d" % by[i]) if dominated[i] else "-"))
        nd = ~dominated
        f.write("#\n# --- FRONT NETTOYE (points NON domines uniquement, %d/%d) ---\n"
                % (nd.sum(), len(lpsp)))
        f.write("idx;LPSP_%;deg_kEUR\n")
        for i in range(len(lpsp)):
            if nd[i]:
                f.write("%2d;%.4f;%.4f\n" % (i, lpsp[i], deg[i]))

    # --- .npz du FRONT NETTOYE (memes cles que l'original, filtrees) -----------
    n0 = len(d["lpsp"])
    keep_orig = np.zeros(n0, dtype=bool)        # masque non-domine, ordre d'origine
    keep_orig[order[~dominated]] = True          # order[i] = index d'origine du i-eme trie
    clean = {k: (v[keep_orig] if np.ndim(v) >= 1 and len(v) == n0 else v)
             for k, v in d.items()}
    out_npz = os.path.join(os.path.dirname(npz), "dp_pareto_25y_51x51_clean.npz")
    np.savez(out_npz, **clean)

    # recap console
    print("Ecrit ->", out)
    print("Ecrit ->", out_npz, "(%d points non domines)" % (~dominated).sum())
    print("%d points, dont %d DOMINES (a gauche/bas LPSP) :" % (len(lpsp), dominated.sum()))
    for i in range(len(lpsp)):
        tag = "  DOMINE par #%d" % by[i] if dominated[i] else "  (front)"
        print("  #%2d  LPSP=%.4f  deg=%.4f%s" % (i, lpsp[i], deg[i], tag))


if __name__ == "__main__":
    main()
