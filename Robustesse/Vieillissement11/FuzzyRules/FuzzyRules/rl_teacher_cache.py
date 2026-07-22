"""Chargeur audite du cache enseignant PD pour le rule-learning V11-p=2.

Ce module lit la trajectoire canonique ``PD_seq_v2`` du cache PD central
``DP/runs/dp_aging_v11_p2_25y_51x51.npz`` et en construit un jeu enseignant
horaire aligne etat -> commande signee de la chaine H2. Il n'ecrit ni ne
relance aucune simulation : il ne fait qu'auditer et exposer les tableaux.

Conventions (mesurees, pas supposees, cf. ``power_balance_diagnostics``) :

- puissance nette DC a servir : ``P_net = P_dc_load - P_dc_pv`` (identique a
  ``P_tot_ref_t`` de ``Common.main_init_and_loop``) ;
- commande signee de la chaine H2 : ``u_h2 = P_dc_fc + P_dc_ely`` (PEMFC > 0 ;
  ``P_dc_ely`` est deja stocke negatif sur le bus DC, donc PEMWE < 0) ; la
  batterie ferme le bilan ``P_dc_bat = P_net - u_h2`` ;
- l'etat au pas ``j`` (SoC, E_h2) est pris avant la transition (bornes
  ``[:-1]``), l'action au pas ``j`` est la commande realisee.

Le manifeste enregistre l'empreinte du cache source, le prefixe enseignant, la
liste des variables, les bornes de normalisation et les residus d'identite du
bilan de puissance (protocole PLAN_FUZZY_RULE_LEARNING_V11_P2 sections 4-5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V11 = HERE.parent
DEFAULT_CACHE = V11 / "DP" / "runs" / "dp_aging_v11_p2_25y_51x51.npz"

TEACHER_PREFIX = "PD_seq_v2"

# Bornes de normalisation, identiques a la FLC experte I0 (flc_policy_v11).
SOC_NORM = (0.20, 0.995)
E_H2_INIT = 200.0  # kWh, cf. Common.main_init_and_loop
H2_NORM = (0.0, E_H2_INIT)

# Champs bruts extraits du cache pour le prefixe enseignant.
_STATE_KEYS = ("SoC", "E_h2", "SoH_bat", "SoH_fc", "SoH_ely")
_ACTION_KEYS = ("P_fc", "P_ely", "P_bat", "P_dc_fc", "P_dc_ely", "P_dc_bat",
                "P_dc_load", "P_dc_pv", "lol_tab", "alpha_fc", "alpha_ely")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(cache, key):
    return np.asarray(cache[f"{TEACHER_PREFIX}__{key}"], dtype=float)


def load_teacher(cache_path: Path = DEFAULT_CACHE):
    """Charge la trajectoire enseignante et derive etats/commande alignes.

    Retourne un dictionnaire avec, tous de longueur ``n`` (nombre d'actions) :
    ``P_net``, ``u_h2`` (commande signee cible), ``SoC``, ``E_h2``,
    ``SoH_bat/fc/ely`` (etats pre-transition), et le residu de bilan.
    """
    cache_path = Path(cache_path)
    cache = np.load(cache_path, allow_pickle=True)

    P_dc_load = _field(cache, "P_dc_load")
    P_dc_pv = _field(cache, "P_dc_pv")
    P_dc_fc = _field(cache, "P_dc_fc")
    P_dc_ely = _field(cache, "P_dc_ely")
    P_dc_bat = _field(cache, "P_dc_bat")
    n = P_dc_load.shape[0]

    # Etats aux bornes -> etat avant transition au pas j.
    SoC = _field(cache, "SoC")[:n]
    E_h2 = _field(cache, "E_h2")[:n]
    SoH_bat = _field(cache, "SoH_bat")[:n]
    SoH_fc = _field(cache, "SoH_fc")[:n]
    SoH_ely = _field(cache, "SoH_ely")[:n]

    P_net = P_dc_load - P_dc_pv
    u_h2 = P_dc_fc + P_dc_ely  # commande signee DC (P_dc_ely deja negatif)

    return {
        "cache_path": str(cache_path),
        "teacher_prefix": TEACHER_PREFIX,
        "n_steps": int(n),
        "P_net": P_net,
        "u_h2": u_h2,
        "SoC": SoC,
        "E_h2": E_h2,
        "SoH_bat": SoH_bat,
        "SoH_fc": SoH_fc,
        "SoH_ely": SoH_ely,
        "P_dc_fc": P_dc_fc,
        "P_dc_ely": P_dc_ely,
        "P_dc_bat": P_dc_bat,
        "lol_tab": _field(cache, "lol_tab"),
        "model_id": str(cache["model_id"]),
        "ely_stress_exponent": float(cache["ely_stress_exponent"]),
    }


def power_balance_diagnostics(teacher) -> dict:
    """Mesure les residus des identites de bilan de puissance candidates.

    On ne suppose pas la convention de signe : on rapporte le residu
    ``P_net - (P_dc_bat + u_h2)`` qui doit tomber a la precision machine si
    ``u_h2 = P_dc_fc - P_dc_ely`` et si la batterie ferme le bilan.
    """
    P_net = teacher["P_net"]
    u_h2 = teacher["u_h2"]
    P_dc_bat = teacher["P_dc_bat"]

    residual = P_net - (P_dc_bat + u_h2)

    # Le residu non nul est physique : delestage (EENS) en deficit non couvert
    # et ecretage PV en surplus non absorbe. Il tombe a ~0 sur les pas servis a
    # SoC interieur, ce qui valide le decodage u_h2 = P_dc_fc + P_dc_ely.
    lol = teacher["lol_tab"]
    SoC = teacher["SoC"]
    served = lol <= 1e-9
    soc_interior = (SoC > 0.2002) & (SoC < 0.9948)
    interior = served & soc_interior

    def stats(r):
        r = np.abs(r)
        return {"max_abs_w": float(r.max()), "mean_abs_w": float(r.mean()),
                "p99_abs_w": float(np.percentile(r, 99))}

    return {
        "identity": "P_net = P_dc_bat + u_h2 (bus DC)",
        "residual_all": stats(residual),
        "residual_interior_served": stats(residual[interior]),
        "n_interior_served": int(interior.sum()),
        "note": ("residu = delestage + ecretage PV, gere par get_lol au replay ;"
                 " ~0 a l'interieur valide le decodage du signe"),
    }


def build_manifest(teacher, diagnostics) -> dict:
    P_net = teacher["P_net"]
    u_h2 = teacher["u_h2"]
    deficit = P_net > 0.0
    surplus = P_net < 0.0
    return {
        "cache_path": teacher["cache_path"],
        "cache_sha256": sha256_file(Path(teacher["cache_path"])),
        "teacher_prefix": teacher["teacher_prefix"],
        "model_id": teacher["model_id"],
        "ely_stress_exponent": teacher["ely_stress_exponent"],
        "n_steps": teacher["n_steps"],
        "target": "u_h2 = P_dc_fc - P_dc_ely (signed, PEMFC>0, PEMWE<0)",
        "features_i0": ["P_net_w", "SoC_norm", "E_h2_norm"],
        "normalization": {
            "SoC": {"lower": SOC_NORM[0], "upper": SOC_NORM[1]},
            "E_h2": {"lower": H2_NORM[0], "upper": H2_NORM[1]},
            "P_net": "raw watts (arbre invariant aux transformations monotones)",
        },
        "ranges": {
            "P_net_w": [float(P_net.min()), float(P_net.max())],
            "u_h2_w": [float(u_h2.min()), float(u_h2.max())],
            "SoC": [float(teacher["SoC"].min()), float(teacher["SoC"].max())],
            "E_h2_kwh": [float(teacher["E_h2"].min()), float(teacher["E_h2"].max())],
        },
        "class_balance": {
            "deficit_frac": float(deficit.mean()),
            "surplus_frac": float(surplus.mean()),
            "idle_frac": float((~deficit & ~surplus).mean()),
            "fc_active_frac": float((teacher["P_dc_fc"] > 1e-9).mean()),
            "ely_active_frac": float((np.abs(teacher["P_dc_ely"]) > 1e-9).mean()),
            "lol_event_frac": float((teacher["lol_tab"] > 1e-9).mean()),
        },
        "power_balance": diagnostics,
    }


def normalize_features(P_net, SoC, E_h2):
    """Features I0 alignees sur la FLC : P_net brut, SoC et H2 clampes [0,1]."""
    soc_n = np.clip((SoC - SOC_NORM[0]) / (SOC_NORM[1] - SOC_NORM[0]), 0.0, 1.0)
    h2_n = np.clip((E_h2 - H2_NORM[0]) / (H2_NORM[1] - H2_NORM[0]), 0.0, 1.0)
    return np.column_stack([np.asarray(P_net, float), soc_n, h2_n])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--write", action="store_true",
                        help="ecrire le manifeste dans runs/")
    args = parser.parse_args()

    teacher = load_teacher(Path(args.cache))
    diagnostics = power_balance_diagnostics(teacher)
    manifest = build_manifest(teacher, diagnostics)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.write:
        out_dir = HERE / "runs"
        out_dir.mkdir(exist_ok=True)
        tag = manifest["cache_sha256"][:12]
        out_path = out_dir / f"rl_teacher_manifest_{tag}.json"
        out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
