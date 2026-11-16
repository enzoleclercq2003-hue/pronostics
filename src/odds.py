"""Conversion des cotes en probabilités implicites (marge retirée)."""
import numpy as np


def implied_proportional(odds: np.ndarray) -> np.ndarray:
    """Normalisation proportionnelle : p_i = (1/cote_i) / somme(1/cote)."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def implied_shin(odds: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """Méthode de Shin (1993) : la marge est attribuée à une part z de parieurs initiés.

    p_i(z) = (sqrt(z^2 + 4 (1 - z) pi_i^2 / B) - z) / (2 (1 - z)), z résolu par dichotomie
    pour que la somme des p_i vaille 1. Corrige le biais favori / outsider.
    """
    inv = 1.0 / np.asarray(odds, dtype=float)
    booksum = inv.sum(axis=1, keepdims=True)

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / booksum) - z) / (2 * (1 - z))

    lo = np.zeros((len(inv), 1))
    hi = np.full((len(inv), 1), 0.5)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        too_big = probs(mid).sum(axis=1, keepdims=True) > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    p = probs((lo + hi) / 2)
    return p / p.sum(axis=1, keepdims=True)
