"""Construction des jeux d'apprentissage rule-learning V11-p=2.

A partir du chargeur audite ``rl_teacher_cache``, ce module produit deux
matrices strictement appariees, I0 et IS, qui ne different que par les
variables de sante (protocole PLAN_FUZZY_RULE_LEARNING_V11_P2 sections 3 et 5).

Choix methodologiques :

- cible unique = commande signee ``u_h2`` (PEMFC>0, PEMWE<0), la batterie
  fermant le bilan ; l'imitation porte sur cette commande DC ;
- I0 = ``[P_net_w, SoC_norm, E_h2_norm]`` (aucun calendrier, aucun SoH) ;
- IS = I0 + ``[wear_bat, wear_fc, wear_ely]`` avec usure normalisee
  ``(1 - SoH) / (1 - SoH_EoL)`` ; a l'etat neuf ces trois colonnes sont nulles ;
- decoupage par **blocs temporels contigus** (annees), jamais aleatoire par
  heure ; les indices de partition sont enregistres ;
- chaque observation recoit une strate (signe et amplitude de P_net, zone SoC,
  zone H2, evenement de delestage, proximite de saturation H2) pour le controle
  d'equilibre et une surponderation optionnelle des etats rares critiques.

Mise en garde enregistree dans le manifeste : le profil meteorologique est
repete a l'identique chaque annee dans ce cache. Une bonne performance sur les
annees tenues a l'ecart ne demontre donc pas a elle seule la generalisation a
d'autres profils ; celle-ci se teste avec des profils hors apprentissage
(Etape C du plan).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from .rl_teacher_cache import (
    DEFAULT_CACHE,
    HERE,
    SOC_NORM,
    H2_NORM,
    build_manifest,
    load_teacher,
    power_balance_diagnostics,
    sha256_file,
)

STEPS_PER_YEAR = int(round(8760 * 3600 / I.LOAD["Ts"]))  # 8760

SOH_EOL = {
    "bat": float(I.BAT["SoH_EoL"]),
    "fc": float(I.FC["SoH_EoL"]),
    "ely": float(I.ELY["SoH_EoL"]),
}

I0_FEATURES = ("P_net_w", "SoC_norm", "E_h2_norm")
IS_EXTRA = ("wear_bat", "wear_fc", "wear_ely")
IS_FEATURES = I0_FEATURES + IS_EXTRA

# Horizon de prevision IF, aligne sur la FLC-IF promue (H18, energie nette).
HORIZON_STEPS_IF = 18
IF_FEATURES = I0_FEATURES + ("E_net_h18_kwh",)
_DT_H = float(I.LOAD["Ts"]) / 3600.0


def net_energy_forecast_kwh(P_net, horizon=HORIZON_STEPS_IF):
    """Energie nette cumulee sur l'horizon a venir, en kWh (queue tronquee).

    ``E[t] = sum_{k=t}^{min(t+H, n)-1} P_net[k] * dt_h / 1000``. La troncature de
    fin reproduit ``profile_net[j:min(j+H, n)]`` de ``main_init_and_loop``.
    """
    P_net = np.asarray(P_net, float)
    n = len(P_net)
    c = np.concatenate([[0.0], np.cumsum(P_net)])
    hi = np.minimum(np.arange(n) + horizon, n)
    return (c[hi] - c[:n]) * _DT_H / 1000.0

# Bornes de partition par defaut, en annees (blocs contigus).
DEFAULT_SPLIT_YEARS = {"train": (0, 15), "val": (15, 20), "test": (20, 25)}

# Zones pour la stratification.
_SOC_EDGES = (0.35, 0.70)
_H2_EDGES = (0.15, 0.60)  # fraction du reservoir
_PNET_ABS_EDGES = (500.0, 3000.0)  # W


def _wear(soh, eol):
    return np.clip((1.0 - np.asarray(soh, float)) / (1.0 - eol), 0.0, None)


def _normalize_features(P_net, SoC, E_h2):
    soc_n = np.clip((SoC - SOC_NORM[0]) / (SOC_NORM[1] - SOC_NORM[0]), 0.0, 1.0)
    h2_n = np.clip((E_h2 - H2_NORM[0]) / (H2_NORM[1] - H2_NORM[0]), 0.0, 1.0)
    return np.asarray(P_net, float), soc_n, h2_n


def _zone(x, edges):
    return np.digitize(x, edges)  # 0=low, 1=mid, 2=high


def build_strata(teacher, h2_n):
    """Etiquette de strate par observation (chaine lisible)."""
    P_net = teacher["P_net"]
    SoC = teacher["SoC"]
    lol = teacher["lol_tab"]

    sign = np.where(P_net > 0, "def", np.where(P_net < 0, "sur", "idle"))
    amp = _zone(np.abs(P_net), _PNET_ABS_EDGES)
    soc_z = _zone(SoC, _SOC_EDGES)
    h2_z = _zone(h2_n, _H2_EDGES)
    lol_flag = (lol > 1e-9).astype(int)

    strata = np.array([
        f"{sign[i]}|p{amp[i]}|s{soc_z[i]}|h{h2_z[i]}|l{lol_flag[i]}"
        for i in range(len(P_net))
    ])
    return strata


def build_split_mask(n, split_years=DEFAULT_SPLIT_YEARS):
    """Masque de partition par blocs d'annees contigus."""
    split = np.empty(n, dtype="<U6")  # "unused" fait 6 caracteres
    split[:] = "unused"
    for name, (y0, y1) in split_years.items():
        lo = y0 * STEPS_PER_YEAR
        hi = min(y1 * STEPS_PER_YEAR, n)
        split[lo:hi] = name
    return split


def inverse_frequency_weights(strata, split, subset="train"):
    """Poids par frequence inverse de strate, sur le sous-ensemble donne."""
    weights = np.zeros(len(strata))
    mask = split == subset
    labels, counts = np.unique(strata[mask], return_counts=True)
    freq = dict(zip(labels, counts))
    n_sub = mask.sum()
    for i in np.where(mask)[0]:
        weights[i] = n_sub / (len(freq) * freq[strata[i]])
    return weights


def build_dataset(cache_path=DEFAULT_CACHE, split_years=DEFAULT_SPLIT_YEARS):
    teacher = load_teacher(Path(cache_path))
    n = teacher["n_steps"]

    P_net, soc_n, h2_n = _normalize_features(
        teacher["P_net"], teacher["SoC"], teacher["E_h2"]
    )
    X_i0 = np.column_stack([P_net, soc_n, h2_n])

    wear_bat = _wear(teacher["SoH_bat"], SOH_EOL["bat"])
    wear_fc = _wear(teacher["SoH_fc"], SOH_EOL["fc"])
    wear_ely = _wear(teacher["SoH_ely"], SOH_EOL["ely"])
    X_is = np.column_stack([X_i0, wear_bat, wear_fc, wear_ely])

    # Feature IF : energie nette prevue oracle sur H18 (vrai futur du maitre).
    e_net_h18 = net_energy_forecast_kwh(teacher["P_net"], HORIZON_STEPS_IF)
    X_if = np.column_stack([X_i0, e_net_h18])

    y = teacher["u_h2"].copy()

    strata = build_strata(teacher, h2_n)
    split = build_split_mask(n, split_years)
    w_train = inverse_frequency_weights(strata, split, "train")

    return {
        "teacher": teacher,
        "X_i0": X_i0,
        "X_is": X_is,
        "X_if": X_if,
        "y": y,
        "strata": strata,
        "split": split,
        "inv_freq_weight": w_train,
        "split_years": split_years,
    }


def dataset_manifest(ds):
    teacher = ds["teacher"]
    split = ds["split"]
    strata = ds["strata"]
    base = build_manifest(teacher, power_balance_diagnostics(teacher))

    counts = {name: int((split == name).sum())
              for name in ("train", "val", "test", "unused")}
    # equilibre des strates rares sur le train
    train_mask = split == "train"
    labels, cnts = np.unique(strata[train_mask], return_counts=True)
    order = np.argsort(cnts)
    rarest = {labels[i]: int(cnts[i]) for i in order[:8]}

    base.update({
        "features_i0": list(I0_FEATURES),
        "features_is": list(IS_FEATURES),
        "soh_eol": SOH_EOL,
        "split_years": {k: list(v) for k, v in ds["split_years"].items()},
        "steps_per_year": STEPS_PER_YEAR,
        "split_counts": counts,
        "n_strata_train": int(len(labels)),
        "rarest_train_strata": rarest,
        "strata_definition": {
            "sign": "def(P_net>0)/sur(P_net<0)/idle",
            "pnet_abs_edges_w": list(_PNET_ABS_EDGES),
            "soc_edges": list(_SOC_EDGES),
            "h2_edges_frac": list(_H2_EDGES),
            "lol_flag": "lol_tab>0",
        },
        "weather_caveat": ("profil meteo repete chaque annee ; holdout annuel"
                           " != generalisation profil (cf. Etape C)"),
    })
    return base


def save_dataset(ds, out_dir=None):
    teacher = ds["teacher"]
    tag = sha256_file(Path(teacher["cache_path"]))[:12]
    out_dir = Path(out_dir) if out_dir else HERE / "runs" / f"rl_dataset_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "dataset.npz",
        X_i0=ds["X_i0"], X_is=ds["X_is"], y=ds["y"],
        strata=ds["strata"], split=ds["split"],
        inv_freq_weight=ds["inv_freq_weight"],
    )
    manifest = dataset_manifest(ds)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return out_dir, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    ds = build_dataset(Path(args.cache))
    manifest = dataset_manifest(ds)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.write:
        out_dir, _ = save_dataset(ds)
        print(f"\n[written] {out_dir}")


if __name__ == "__main__":
    main()
