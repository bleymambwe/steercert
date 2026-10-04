"""Two-clock analysis for diffusion (and autoregressive) language models.

In an absorbing-state (masked) diffusion LM every intervention site has two
time coordinates: depth ``l`` and denoising step ``t``. A committed token is
frozen, so a probe at step ``t`` reads evidence that has already been
committed, while a write at step ``t`` can only act through positions that are
still masked. Read information therefore tends to rise along the denoising
clock while write leverage falls. Autoregressive generation is the special case
where positions are unmasked left to right.

This module provides:

* :class:`Lattice`, a container for read and write measurements on the
  ``(depth, step)`` grid with the falsifier checks F1 and F2;
* :func:`shape_gap` and :func:`counting_baseline`, which test whether a
  controllability curve carries attribute-specific structure or is explained
  by the shrinking set of writable positions alone (falsifier F5);
* the exact closed-form construction (:func:`make_schedule`,
  :func:`analytic_read_auc`, :func:`analytic_write_norm`) in which the read and
  write directions are orthogonal at every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Lattice",
    "Schedule",
    "analytic_read_auc",
    "analytic_write_norm",
    "counting_baseline",
    "make_schedule",
    "normalise_curve",
    "remask_recoverable",
    "shape_gap",
]


def normalise_curve(values: np.ndarray) -> np.ndarray:
    """Divide a non-negative curve by its maximum (zero curves are returned unchanged)."""
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(values)) if values.size else 0.0
    return values / peak if peak > 0 else values


def shape_gap(curve: np.ndarray, baseline: np.ndarray) -> float:
    """Largest difference between two curves after each is peak-normalised.

    A small gap means the measured controllability curve has the same shape as
    the content-free baseline, so its decay is explained by counting alone.
    """
    curve, baseline = np.asarray(curve, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if curve.shape != baseline.shape:
        raise ValueError("curve and baseline must have the same length")
    return float(np.max(np.abs(normalise_curve(curve) - normalise_curve(baseline))))


def counting_baseline(writable_counts: np.ndarray) -> np.ndarray:
    """Content-free write-leverage curve from the number of writable positions.

    If each still-masked payload position contributes an equal, independent
    gradient component, the mean-gradient norm scales as the square root of
    the number of writable positions. Schedule-aware steering methods should
    beat this curve, which costs one count per step and no training.
    """
    counts = np.asarray(writable_counts, dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("writable counts must be non-negative")
    return normalise_curve(np.sqrt(counts))


@dataclass(frozen=True)
class Lattice:
    """Read and write measurements on a ``(depth, denoising step)`` grid.

    Attributes
    ----------
    read:
        ``(n_depths, n_steps)`` decodability, for example probe AUC.
    write:
        ``(n_depths, n_steps)`` absolute write leverage, for example ``|g_bar|``.
    steps:
        Denoising progress for each column, increasing (0 = fully masked).
    depths:
        Labels for each row.
    """

    read: np.ndarray
    write: np.ndarray
    steps: np.ndarray
    depths: tuple

    def __post_init__(self) -> None:
        read = np.asarray(self.read, dtype=np.float64)
        write = np.asarray(self.write, dtype=np.float64)
        if read.shape != write.shape or read.ndim != 2:
            raise ValueError("read and write must be matrices of equal shape")
        if len(self.steps) != read.shape[1] or len(self.depths) != read.shape[0]:
            raise ValueError("steps/depths do not match the matrix shape")
        object.__setattr__(self, "read", read)
        object.__setattr__(self, "write", write)
        object.__setattr__(self, "steps", np.asarray(self.steps, dtype=np.float64))

    def write_decreasing(self, tolerance: float = 0.0) -> list[bool]:
        """F1 check per depth: write leverage never rises along the denoising clock."""
        return [bool(np.all(np.diff(row) <= tolerance)) for row in self.write]

    def clocks_dissociated(self) -> list[bool]:
        """F2 check per depth: decodability and leverage peak at different steps."""
        return [int(np.argmax(r)) != int(np.argmax(w)) for r, w in zip(self.read, self.write, strict=True)]

    def joint_optimum(self, read_weight: float = 0.5) -> list[float]:
        """Step that maximises a weighted geometric mean of normalised read and write."""
        result = []
        for r, w in zip(self.read, self.write, strict=True):
            score = normalise_curve(r) ** read_weight * normalise_curve(w) ** (1.0 - read_weight)
            result.append(float(self.steps[int(np.argmax(score))]))
        return result

    def summary(self) -> dict:
        return {
            "depths": list(self.depths),
            "write_decreasing": self.write_decreasing(),
            "clocks_dissociated": self.clocks_dissociated(),
            "read_argmax_step": [float(self.steps[int(np.argmax(r))]) for r in self.read],
            "write_argmax_step": [float(self.steps[int(np.argmax(w))]) for w in self.write],
        }


@dataclass(frozen=True)
class Schedule:
    """An absorbing-diffusion unmasking order over read and write positions."""

    length: int
    read_positions: np.ndarray
    write_positions: np.ndarray
    order: np.ndarray

    def committed(self, step: int) -> np.ndarray:
        return self.order[:step]

    def masked(self, step: int) -> np.ndarray:
        return self.order[step:]

    def n_committed_read(self, step: int) -> int:
        return int(np.isin(self.committed(step), self.read_positions).sum())

    def n_masked_write(self, step: int) -> int:
        return int(np.isin(self.masked(step), self.write_positions).sum())


def make_schedule(n_read: int, n_write: int, seed: int = 0, order: str = "random") -> Schedule:
    """Build a schedule; ``order='identity'`` is the autoregressive case."""
    length = n_read + n_write
    if order == "identity":
        permutation = np.arange(length)
    elif order == "random":
        permutation = np.random.default_rng(seed).permutation(length)
    else:
        raise ValueError(f"unknown order {order!r}")
    return Schedule(length, np.arange(n_read), np.arange(n_read, length), permutation)


def analytic_read_auc(schedule: Schedule, step: int, kappa: float, sigma: float) -> float:
    """Population AUC of the optimal linear readout of committed evidence.

    With ``m`` committed read positions the pooled statistic has separation
    ``d' = kappa sqrt(m) / sigma`` and ``AUC = Phi(sqrt(2) d')``.
    """
    m = schedule.n_committed_read(step)
    if m == 0 or kappa == 0.0:
        return 0.5
    z = math.sqrt(2.0) * kappa * math.sqrt(m) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def analytic_write_norm(schedule: Schedule, step: int, gamma: float) -> float:
    """Mean-gradient norm at ``step``; proportional to sqrt(masked write positions)."""
    k = schedule.n_masked_write(step)
    return (gamma * 0.25 / max(len(schedule.write_positions), 1)) * math.sqrt(k)


def remask_recoverable(schedule: Schedule, step: int, n_remask: int) -> int:
    """Write positions made writable again by remasking ``n_remask`` recent write commits.

    Read commitments are untouched, so remasking raises controllability at
    unchanged decodability. The same operation underlies remask-based safety
    defences and remask-based attacks.
    """
    committed = schedule.committed(step)
    return int(min(n_remask, int(np.isin(committed, schedule.write_positions).sum())))
