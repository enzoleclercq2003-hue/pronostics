"""Métriques probabilistes, intervalles bootstrap et tests appariés."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, f1_score

import config as cfg

EPS = 1e-12


def per_match_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Pertes par match (permet le bootstrap et les tests appariés)."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    return {
        "accuracy": (p.argmax(axis=1) == y).astype(float),
        "logloss": -np.log(p[np.arange(len(y)), y]),
        "brier": ((p - onehot) ** 2).sum(axis=1),               # Brier multiclasse (somme sur 3 classes)
        "rps": ((cum_p - cum_o) ** 2).sum(axis=1) / 2.0,        # Ranked Probability Score (H < D < A)
    }


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    losses = per_match_losses(y, p)
    out = {k: float(v.mean()) for k, v in losses.items()}
    pred = np.asarray(p).argmax(axis=1)
    f1 = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    out.update({f"f1_{c}": float(f) for c, f in zip(cfg.CLASSES, f1)})
    out["n"] = int(len(y))
    return out


def confusion(y, p) -> np.ndarray:
    return confusion_matrix(y, np.asarray(p).argmax(axis=1), labels=[0, 1, 2])


def bootstrap_ci(values: np.ndarray, n_boot: int = cfg.N_BOOTSTRAP, seed: int = cfg.SEED, alpha: float = 0.05):
    """Intervalle de confiance bootstrap (percentiles) de la moyenne d'une perte par match."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = cfg.N_BOOTSTRAP, seed: int = cfg.SEED):
    """Bootstrap apparié de la différence moyenne (a - b) sur les mêmes matchs.

    Renvoie (différence, borne basse, borne haute, p-valeur bilatérale).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(min(p, 1.0))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray):
    """Test de Diebold-Mariano (horizon 1) sur le différentiel de perte. Renvoie (stat, p-valeur)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))
