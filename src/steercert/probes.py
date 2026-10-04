"""Read-side directions: linear probes fitted on activations.

All probes return a unit direction in the original activation coordinates, so
they can be audited with :func:`steercert.audit` exactly like a steering
vector. They depend only on NumPy; use any external probe (scikit-learn,
a trained classifier head) by passing its weight vector instead.
"""

from __future__ import annotations

import numpy as np

from .geometry import EPS, unit

__all__ = ["auc", "difference_of_means", "direction_auc", "fisher_probe", "logistic_probe"]


def _split(states: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    labels = np.asarray(labels).astype(int).ravel()
    if states.ndim != 2 or states.shape[0] != labels.shape[0]:
        raise ValueError("states must be (n, width) and labels must have length n")
    positive, negative = states[labels == 1], states[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("both classes must be present")
    return positive, negative


def difference_of_means(states: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Class-mean difference ``mu_1 - mu_0``, the ActAdd / CAA steering vector."""
    positive, negative = _split(states, labels)
    return unit(positive.mean(axis=0) - negative.mean(axis=0))


def fisher_probe(states: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3) -> np.ndarray:
    """Shrinkage linear discriminant ``(Sigma + gamma I)^{-1} (mu_1 - mu_0)``.

    ``shrinkage`` is relative to the mean eigenvalue of the pooled within-class
    covariance, so it is invariant to the overall activation scale.
    """
    positive, negative = _split(states, labels)
    delta = positive.mean(axis=0) - negative.mean(axis=0)
    centred = np.concatenate([positive - positive.mean(axis=0), negative - negative.mean(axis=0)])
    covariance = centred.T @ centred / max(len(centred) - 2, 1)
    ridge = shrinkage * (np.trace(covariance) / covariance.shape[0] + EPS)
    return unit(np.linalg.solve(covariance + ridge * np.eye(covariance.shape[0]), delta))


def logistic_probe(
    states: np.ndarray,
    labels: np.ndarray,
    l2: float = 1e-3,
    steps: int = 500,
    learning_rate: float = 0.5,
) -> np.ndarray:
    """L2-regularised logistic regression on standardised features.

    Full-batch gradient descent; deterministic. The weight vector is mapped
    back to raw activation coordinates before normalisation.
    """
    positive, negative = _split(states, labels)
    x = np.concatenate([positive, negative])
    y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    mean, scale = x.mean(axis=0), x.std(axis=0) + EPS
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1])
    bias = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ weights + bias, -60, 60)))
        weights -= learning_rate * (z.T @ (p - y) / len(y) + l2 * weights)
        bias -= learning_rate * float((p - y).mean())
    return unit(weights / scale)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based ROC AUC with tie correction."""
    from .stats import rankdata

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def direction_auc(states: np.ndarray, labels: np.ndarray, direction: np.ndarray) -> float:
    """AUC of reading the label out along ``direction``."""
    return auc(np.asarray(states, dtype=np.float64) @ unit(direction), labels)
