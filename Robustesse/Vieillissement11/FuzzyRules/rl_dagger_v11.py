"""DAgger (Dataset Aggregation) pour la distillation PD -> arbre, V11-p=2.

Direction B du recap : l'arbre I0 distille est domine par manque de COHERENCE
(derive de distribution du behavioral cloning, section 2.2), et non par manque
d'information (feature ignoree section 2.5, regle penalisante section 8). DAgger
attaque directement cette cause : on deroule le disciple, on RE-ETIQUETTE les
etats qu'il visite avec l'action de la PD (l'oracle), on agrege ces paires au jeu
d'apprentissage, et on re-ajuste l'arbre ; on itere. Le disciple apprend ainsi
a se rattraper sur sa PROPRE distribution d'etats, pas seulement sur celle de la
PD omnisciente.

Oracle de re-etiquetage : la grille de politique 1 an SoH=1 de ``DP.dp_core``
(``backward(store_policy=True)`` -> ``policy[t, i_SoC, j_H2, fc_on, ely_on]``).
Elle est SANS ETAT : le re-etiquetage se fait donc hors-ligne, par plus-proche-
voisin sur ``(SoC, E_h2)`` et lookup exact sur ``(t, fc_on, ely_on)`` -- exactement
la regle que ``dp_core.make_dp_policy`` applique a la PD elle-meme.

Ce module isole la logique PURE et testable (logger d'etats visites, extraction
de features, re-etiquetage contre une grille, agregation). Le calcul de la vraie
grille PD et la boucle refit/evaluation en boucle fermee vivent dans le runner
mesocentre ``run_dagger_rl_v11`` (non testable hors cache/simu).
"""

from __future__ import annotations

import numpy as np

from .rl_dataset_v11 import _normalize_features
from .rl_teacher_cache import E_H2_INIT

# Seuils marche/arret alignes sur la boucle physique (cf. dp_aging.forward :
# fc_on quand P_dc_fc depasse ~1 W ; ely_on quand |P_dc_ely| depasse un epsilon).
_FC_ON_W = 1.0
_ELY_ON_W = 1.0


def onoff_from_action(action, fc_on_w=_FC_ON_W, ely_on_w=_ELY_ON_W):
    """(fc_on, ely_on) a partir de l'action realisee (P_dc_bat, P_dc_fc, P_dc_ely)."""
    return int(action[1] > fc_on_w), int(abs(action[2]) > ely_on_w)


class DiscipleStateLogger:
    """Enveloppe une policy disciple : delegue et journalise l'etat visite.

    Signature de boucle identique a la policy enveloppee. A chaque pas on
    enregistre l'etat AVANT transition (P_net, SoC, E_h2) et les indicateurs
    marche/arret DERIVES DE L'ACTION realisee (ce que l'oracle grille attend en
    entree de son lookup au pas suivant). Un compteur de pas fournit l'index
    temporel ``t`` du template PD.
    """

    def __init__(self, disciple):
        self.disciple = disciple
        self.reset()

    def reset(self):
        reset = getattr(self.disciple, "reset", None)
        if callable(reset):
            reset()
        self.t = 0
        self.P_net, self.SoC, self.E_h2 = [], [], []
        self.E_h2_init, self.fc_on, self.ely_on = [], [], []

    def __call__(self, SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t,
                 alpha_ely_t, SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t,
                 P_ely_max_t, RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t,
                 **kwargs):
        action, lol = self.disciple(
            SoC_t, P_tot_ref_t, defaillances, lol_tab, alpha_fc_t, alpha_ely_t,
            SoH_bat_t, E_h2_t, E_h2_init, P_fc_max_t, P_ely_max_t,
            RUL_fc_t, RUL_ely_t, SoH_fc_t, SoH_ely_t, **kwargs)
        fc_on, ely_on = onoff_from_action(action)
        self.P_net.append(float(P_tot_ref_t))
        self.SoC.append(float(SoC_t))
        self.E_h2.append(float(E_h2_t))
        self.E_h2_init.append(float(E_h2_init))
        self.fc_on.append(fc_on)
        self.ely_on.append(ely_on)
        self.t += 1
        return action, lol

    def visited(self):
        """Etats visites journalises, en tableaux numpy alignes."""
        return {
            "P_net": np.asarray(self.P_net, float),
            "SoC": np.asarray(self.SoC, float),
            "E_h2": np.asarray(self.E_h2, float),
            "E_h2_init": np.asarray(self.E_h2_init, float),
            "fc_on": np.asarray(self.fc_on, int),
            "ely_on": np.asarray(self.ely_on, int),
            "t": np.arange(len(self.P_net), dtype=int),
        }


def build_i0_features(visited):
    """Matrice de features I0 ``[P_net_w, SoC_norm, E_h2_norm]`` des etats visites.

    Normalisation STRICTEMENT identique au jeu d'apprentissage
    (``rl_dataset_v11._normalize_features``) pour que les paires re-etiquetees
    vivent dans le meme espace que les labels enseignants.
    """
    P_net, soc_n, h2_n = _normalize_features(
        visited["P_net"], visited["SoC"], visited["E_h2"])
    return np.column_stack([P_net, soc_n, h2_n])


def _nearest_index(grid, values):
    """Indice du plus proche voisin de ``values`` dans ``grid`` (croissant)."""
    grid = np.asarray(grid, float)
    idx = np.clip(np.searchsorted(grid, values), 0, len(grid) - 1)
    lo = np.clip(idx - 1, 0, len(grid) - 1)
    take_lo = np.abs(values - grid[lo]) < np.abs(values - grid[idx])
    return np.where(take_lo, lo, idx)


def relabel_with_policy_grid(visited, policy_grid, u, soc_grid, h2_grid):
    """Re-etiquette les etats visites par l'action u_h2 de la grille PD.

    ``policy_grid`` a la forme ``(T, Ns, Nh, 2, 2)`` (int) et contient l'indice
    du controle optimal dans ``u`` ; ``soc_grid`` (Ns) et ``h2_grid`` (Nh) sont
    les noeuds de la PD. Le lookup est celui de ``dp_core.make_dp_policy`` :
    plus-proche-voisin sur SoC et E_h2, exact sur ``(t, fc_on, ely_on)``. Un pas
    au-dela de l'horizon du template est ramene au dernier pas disponible.
    """
    policy_grid = np.asarray(policy_grid)
    u = np.asarray(u, float)
    T = policy_grid.shape[0]
    t = np.clip(visited["t"], 0, T - 1)
    i = _nearest_index(soc_grid, visited["SoC"])
    j = _nearest_index(h2_grid, visited["E_h2"])
    fc_on = np.clip(visited["fc_on"], 0, 1)
    ely_on = np.clip(visited["ely_on"], 0, 1)
    ctrl_idx = policy_grid[t, i, j, fc_on, ely_on]
    return u[ctrl_idx]


def dagger_aggregate(base_X, base_y, relabel_batches, base_weight=1.0,
                     visited_weight=1.0):
    """Agrege le jeu enseignant et les lots re-etiquetes (union DAgger).

    Retourne ``(X, y, sample_weight)``. ``relabel_batches`` est une liste de
    couples ``(X_k, y_k)`` collectes aux iterations successives. Les poids
    permettent, si besoin, de sur/sous-ponderer les etats visites face au socle
    enseignant (par defaut : poids uniformes).
    """
    Xs, ys, ws = [np.asarray(base_X, float)], [np.asarray(base_y, float)], [
        np.full(len(base_y), float(base_weight))]
    for X_k, y_k in relabel_batches:
        X_k = np.asarray(X_k, float)
        y_k = np.asarray(y_k, float)
        Xs.append(X_k)
        ys.append(y_k)
        ws.append(np.full(len(y_k), float(visited_weight)))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


if __name__ == "__main__":
    # Demonstration hors boucle fermee : re-etiquetage d'etats synthetiques
    # contre une grille jouet (aucun cache requis).
    T, Ns, Nh, Nu = 4, 5, 5, 3
    soc_grid = np.linspace(0.2, 0.995, Ns)
    h2_grid = np.linspace(0.0, E_H2_INIT, Nh)
    u = np.array([-3000.0, 0.0, 1000.0])
    rng = np.random.default_rng(0)
    policy_grid = rng.integers(0, Nu, size=(T, Ns, Nh, 2, 2))
    visited = {
        "P_net": np.array([2000.0, -5000.0]),
        "SoC": np.array([0.6, 0.9]),
        "E_h2": np.array([100.0, 20.0]),
        "E_h2_init": np.array([E_H2_INIT, E_H2_INIT]),
        "fc_on": np.array([1, 0]),
        "ely_on": np.array([0, 1]),
        "t": np.array([0, 1]),
    }
    y = relabel_with_policy_grid(visited, policy_grid, u, soc_grid, h2_grid)
    X = build_i0_features(visited)
    print("features visites :\n", X)
    print("labels PD re-etiquetes :", y)
