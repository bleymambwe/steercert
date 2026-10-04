"""Synthetic sites where read quality and write alignment are set independently.

Used by the examples, the tests and ``steercert demo``. The construction shows
that a probe can read a label perfectly while pointing almost orthogonally to
the direction that moves behaviour, and lets each quantity be checked against
its known population value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SyntheticSite", "decoupled_site"]


@dataclass(frozen=True)
class SyntheticSite:
    states: np.ndarray
    labels: np.ndarray
    gradients: np.ndarray
    read_direction: np.ndarray
    write_direction: np.ndarray
    overlap: float


def decoupled_site(
    n: int = 512,
    width: int = 64,
    separation: float = 3.0,
    overlap: float = 0.1,
    gradient_rank: int = 4,
    gradient_spread: float = 0.3,
    seed: int = 0,
) -> SyntheticSite:
    """Generate one site with a chosen read/write overlap.

    Parameters
    ----------
    separation:
        Distance between class means along the read direction, in noise units.
        The population AUC of the optimal probe is ``Phi(separation / sqrt(2))``.
    overlap:
        Cosine between the read direction and the dominant write direction.
        The probe's population RMS efficiency is close to this value.
    gradient_rank:
        Number of directions the per-example gradients spread over; the first
        dominates, so the stable rank stays small, as in trained models.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0, 1]")
    if gradient_rank + 2 > width:
        raise ValueError("width must exceed gradient_rank + 2")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(width, width)))
    read = basis[:, 0]
    write = overlap * read + np.sqrt(1.0 - overlap**2) * basis[:, 1]
    labels = rng.integers(0, 2, size=n)
    signs = 2.0 * labels - 1.0
    states = rng.normal(size=(n, width)) + 0.5 * separation * signs[:, None] * read
    coefficients = 1.0 + gradient_spread * rng.normal(size=n)
    gradients = coefficients[:, None] * write
    for j in range(gradient_rank - 1):
        scale = gradient_spread * 0.5 ** (j + 1)
        gradients += scale * rng.normal(size=n)[:, None] * basis[:, 2 + j]
    return SyntheticSite(states, labels, gradients, read, write, overlap)
