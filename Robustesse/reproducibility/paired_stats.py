"""Statistiques appariees sans dependance SciPy.

Les experiences P1/P3 emploient des common random numbers. L'unite
statistique pertinente est donc la difference par monde, pas la dispersion
des deux niveaux prise separement.
"""

from __future__ import annotations

import math

import numpy as np


def cvar_high(values, q=0.90):
    """Expected shortfall empirique : moyenne exacte des pires ``1-q``.

    A N=200 et q=0.90, exactement les 20 plus grandes observations sont
    retenues. Cette convention evite qu'une masse d'egalites au quantile fasse
    varier arbitrairement la taille de la queue.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan")
    n_tail = max(1, int(math.ceil((1.0 - q) * x.size - 1e-12)))
    return float(np.partition(x, x.size - n_tail)[-n_tail:].mean())


def exact_sign_test(differences, zero_tol=5e-12):
    """Test bilateral exact des signes, egalites exclues."""
    d = np.asarray(differences, dtype=float)
    positive = int(np.sum(d > zero_tol))
    negative = int(np.sum(d < -zero_tol))
    ties = int(d.size - positive - negative)
    n = positive + negative
    if n == 0:
        return {"positive": positive, "negative": negative, "ties": ties, "pvalue": 1.0}
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "pvalue": min(1.0, 2.0 * tail),
    }


def bootstrap_mean_ci(differences, confidence=0.95, n_resamples=30000, seed=20260711):
    """IC percentile de la moyenne d'une difference appariee."""
    d = np.asarray(differences, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    if np.all(d == d[0]):
        return float(d[0]), float(d[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    batch = 1000
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        indices = rng.integers(0, d.size, size=(stop - start, d.size))
        means[start:stop] = d[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_cvar_difference_ci(
    values_a,
    values_b,
    q=0.90,
    confidence=0.95,
    n_resamples=30000,
    seed=20260711,
):
    """IC apparie de ``ES_q(A) - ES_q(B)`` par bootstrap des mondes."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("values_a et values_b doivent etre deux vecteurs appariees")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    n_tail = max(1, int(math.ceil((1.0 - q) * a.size - 1e-12)))
    batch = 500
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        indices = rng.integers(0, a.size, size=(stop - start, a.size))
        sample_a = a[indices]
        sample_b = b[indices]
        tail_a = np.partition(sample_a, a.size - n_tail, axis=1)[:, -n_tail:]
        tail_b = np.partition(sample_b, b.size - n_tail, axis=1)[:, -n_tail:]
        estimates[start:stop] = tail_a.mean(axis=1) - tail_b.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_difference(differences, bootstrap_seed=20260711):
    """Resume descriptif et inferentiel d'une difference A-B.

    Une valeur negative signifie que A a un cout inferieur a B.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("differences doit etre un vecteur fini non vide")
    q1, median, q3 = np.quantile(d, [0.25, 0.50, 0.75])
    ci_low, ci_high = bootstrap_mean_ci(d, seed=bootstrap_seed)
    signs = exact_sign_test(d)
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "sample_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p05": float(np.quantile(d, 0.05)),
        "p95": float(np.quantile(d, 0.95)),
        "cvar90_difference": cvar_high(d),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
        "wins": signs["negative"],
        "ties": signs["ties"],
        "losses": signs["positive"],
        "sign_pvalue": signs["pvalue"],
    }
