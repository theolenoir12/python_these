"""Moteur ANFIS Takagi-Sugeno d'ordre 1, NumPy pur (V11-p=2).

Système flou adaptatif de type Sugeno (Jang 1993) : partition en grille de
fonctions d'appartenance gaussiennes par entrée, prémisses fixées à
l'initialisation (couvrant les mêmes partitions que la FLC), et conséquences
LINÉAIRES apprises par moindres carrés (la couche 1 de l'apprentissage hybride
de Jang ; l'affinage par gradient des prémisses est une extension ultérieure).

Une règle r a une force w_r(x) = produit des appartenances sélectionnées, et une
conséquence d'ordre 1 f_r(x) = coef_r · x + bias_r. La sortie est la moyenne
pondérée normalisée y(x) = Σ_r w̄_r(x) f_r(x). Comme y est LINÉAIRE en les
paramètres de conséquence à prémisses figées, l'ajustement est un moindres carrés
(ridge) exact, rapide et déterministe -- pas de descente de gradient ici.

Toutes les entrées sont standardisées (moyenne/écart-type appris) pour placer les
appartenances et conditionner la régression ; les conséquences opèrent dans cet
espace standardisé et la sortie est directement en unités de la cible (u_h2 en W).
"""

from __future__ import annotations

import itertools

import numpy as np

_EPS = 1e-12


def gaussian_mf(x, center, sigma):
    """Appartenance gaussienne exp(-0.5 ((x-c)/sigma)^2)."""
    return np.exp(-0.5 * ((np.asarray(x, float) - center) / sigma) ** 2)


class AnfisTS1:
    """ANFIS Takagi-Sugeno ordre 1 à partition en grille (prémisses figées)."""

    def __init__(self, scaler_center, scaler_scale, mf_centers, mf_sigmas,
                 coef=None, bias=None):
        self.scaler_center = np.asarray(scaler_center, float)   # (n_inputs,)
        self.scaler_scale = np.asarray(scaler_scale, float)     # (n_inputs,)
        self.mf_centers = np.asarray(mf_centers, float)         # (n_inputs, n_mf)
        self.mf_sigmas = np.asarray(mf_sigmas, float)           # (n_inputs, n_mf)
        self.n_inputs, self.n_mf = self.mf_centers.shape
        # Règles = produit cartésien des indices d'appartenance (grille).
        self.rules = np.array(
            list(itertools.product(range(self.n_mf), repeat=self.n_inputs)),
            dtype=int)                                          # (R, n_inputs)
        self.n_rules = len(self.rules)
        self.coef = None if coef is None else np.asarray(coef, float)  # (R, n_inputs)
        self.bias = None if bias is None else np.asarray(bias, float)  # (R,)

    # -- initialisation -----------------------------------------------------
    @classmethod
    def init_uniform(cls, X, n_mf=3, sigma_scale=1.0):
        """Prémisses uniformes sur l'étendue standardisée de chaque entrée.

        ``sigma_scale`` élargit (ou resserre) les appartenances relativement à
        l'espacement des centres (1.0 = recouvrement standard).
        """
        X = np.asarray(X, float)
        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < _EPS, 1.0, scale)
        Xs = (X - center) / scale
        n_inputs = X.shape[1]
        mf_centers = np.zeros((n_inputs, n_mf))
        mf_sigmas = np.zeros((n_inputs, n_mf))
        for d in range(n_inputs):
            lo, hi = float(Xs[:, d].min()), float(Xs[:, d].max())
            if n_mf == 1:
                mf_centers[d] = 0.5 * (lo + hi)
                mf_sigmas[d] = max((hi - lo), 1.0) * sigma_scale
            else:
                mf_centers[d] = np.linspace(lo, hi, n_mf)
                spacing = (hi - lo) / (n_mf - 1)
                mf_sigmas[d] = max(spacing, _EPS) * sigma_scale
        return cls(center, scale, mf_centers, mf_sigmas)

    # -- couches ANFIS ------------------------------------------------------
    def _scale(self, X):
        return (np.asarray(X, float) - self.scaler_center) / self.scaler_scale

    def normalized_firing(self, X, scaled=False):
        """Forces de règles normalisées w̄ (n, R) (couches 1-3 de Jang)."""
        Xs = X if scaled else self._scale(X)
        Xs = np.atleast_2d(Xs)
        # appartenances (n, n_inputs, n_mf)
        mem = gaussian_mf(Xs[:, :, None], self.mf_centers[None, :, :],
                          self.mf_sigmas[None, :, :])
        n = Xs.shape[0]
        w = np.ones((n, self.n_rules))
        for d in range(self.n_inputs):
            w *= mem[:, d, self.rules[:, d]]     # produit sur les entrées
        wsum = w.sum(axis=1, keepdims=True)
        return w / (wsum + _EPS)

    def _consequent_design(self, Xs, wbar):
        """Matrice de conception du moindres carrés : (n, R*(n_inputs+1))."""
        n = Xs.shape[0]
        aug = np.concatenate([Xs, np.ones((n, 1))], axis=1)      # (n, n_inputs+1)
        # bloc règle r = wbar[:, r, None] * aug
        return (wbar[:, :, None] * aug[:, None, :]).reshape(n, -1)

    def fit_consequents(self, X, y, ridge=1e-6):
        """Ajuste les conséquences par moindres carrés régularisés (ridge)."""
        Xs = self._scale(X)
        y = np.asarray(y, float)
        wbar = self.normalized_firing(Xs, scaled=True)
        phi = self._consequent_design(Xs, wbar)                  # (n, R*(K+1))
        k = self.n_inputs + 1
        A = phi.T @ phi + ridge * np.eye(phi.shape[1])
        theta = np.linalg.solve(A, phi.T @ y).reshape(self.n_rules, k)
        self.coef = theta[:, :self.n_inputs]
        self.bias = theta[:, self.n_inputs]
        return self

    def predict(self, X):
        if self.coef is None:
            raise RuntimeError("conséquences non ajustées (fit_consequents)")
        Xs = self._scale(X)
        Xs = np.atleast_2d(Xs)
        wbar = self.normalized_firing(Xs, scaled=True)
        f = Xs @ self.coef.T + self.bias[None, :]                # (n, R)
        return np.sum(wbar * f, axis=1)

    # -- diagnostics --------------------------------------------------------
    def spec(self):
        """Résumé structurel (nb de règles, forme des paramètres)."""
        return {
            "kind": "anfis-ts1-grid",
            "n_inputs": int(self.n_inputs),
            "n_mf_per_input": int(self.n_mf),
            "n_rules": int(self.n_rules),
            "n_consequent_params": int(self.n_rules * (self.n_inputs + 1)),
            "fitted": self.coef is not None,
        }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 2] + 4.0        # cible linéaire
    model = AnfisTS1.init_uniform(X, n_mf=3).fit_consequents(X, y)
    pred = model.predict(X)
    print("spec:", model.spec())
    print(f"RMSE sur cible linéaire : {np.sqrt(np.mean((pred - y) ** 2)):.3e}")
