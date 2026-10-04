"""Control directions for steering experiments.

A steering result is only as informative as the comparator it beats. Three
families are provided, matching current practice (SteerCheck, arXiv:2608.24335):

``isotropic``
    Uniform random unit directions. In high dimensions they concentrate near
    orthogonality to everything, so beating them is a weak test.
``shuffled_label_probe``
    The same probe recipe fitted on permuted labels. Matches the construction
    while breaking the label link.
``sign_randomized_gradient``
    The mean of per-example gradients with random signs. Matches the gradient
    construction while destroying its shared sign. When the effect Gramian has
    stable rank near 1, per-example gradients are nearly collinear and this
    control collapses back onto the mean gradient, so it must be reported with
    the stable rank (see :func:`sign_control_is_degenerate`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .geometry import cosine_abs, mean_gradient, unit
from .probes import difference_of_means

__all__ = [
    "ControlSet",
    "isotropic",
    "make_controls",
    "shuffled_label_probes",
    "sign_control_is_degenerate",
    "sign_randomized_gradients",
]

FAMILIES = ("isotropic", "shuffled_label_probe", "sign_randomized_gradient")


def isotropic(width: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` uniformly random unit directions of dimension ``width``."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(n, width))
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def sign_randomized_gradients(gradients: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Means of per-example gradients after flipping each row's sign at random."""
    gradients = np.asarray(gradients, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    attempts = 0
    while len(rows) < n and attempts < 20 * max(n, 1):
        attempts += 1
        signs = rng.choice(np.array((-1.0, 1.0)), size=len(gradients))
        try:
            rows.append(unit((gradients * signs[:, None]).mean(axis=0)))
        except ValueError:
            continue
    return np.array(rows).reshape(len(rows), gradients.shape[1])


def shuffled_label_probes(
    states: np.ndarray,
    labels: np.ndarray,
    n: int,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray] = difference_of_means,
    seed: int = 0,
) -> np.ndarray:
    """Probes fitted with ``fit`` on randomly permuted labels."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    return np.stack([unit(fit(states, rng.permutation(labels))) for _ in range(n)])


@dataclass(frozen=True)
class ControlSet:
    """Named control directions grouped by family."""

    directions: dict[str, np.ndarray]
    family_of: dict[str, str]

    def cosines_to(self, reference: np.ndarray) -> dict[str, list[float]]:
        """Absolute cosine of every control to ``reference``, grouped by family."""
        result: dict[str, list[float]] = {}
        for name, direction in self.directions.items():
            result.setdefault(self.family_of[name], []).append(cosine_abs(direction, reference))
        return result


def make_controls(
    gradients: np.ndarray,
    states: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    n_isotropic: int = 8,
    n_shuffled: int = 4,
    n_sign_randomized: int = 8,
    probe_fit: Callable[[np.ndarray, np.ndarray], np.ndarray] = difference_of_means,
    seed: int = 0,
) -> ControlSet:
    """Build all three control families.

    Shuffled-label probes are skipped when ``states`` and ``labels`` are not given.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    width = gradients.shape[1]
    directions: dict[str, np.ndarray] = {}
    family_of: dict[str, str] = {}

    def add(family: str, rows: np.ndarray) -> None:
        for index, row in enumerate(rows):
            name = f"control/{family}/{index:03d}"
            directions[name] = row
            family_of[name] = family

    add("isotropic", isotropic(width, n_isotropic, seed))
    if states is not None and labels is not None and n_shuffled > 0:
        add("shuffled_label_probe", shuffled_label_probes(states, labels, n_shuffled, probe_fit, seed + 1))
    add("sign_randomized_gradient", sign_randomized_gradients(gradients, n_sign_randomized, seed + 2))
    return ControlSet(directions, family_of)


def sign_control_is_degenerate(gradients: np.ndarray, stable_rank: float, threshold: float = 0.5) -> bool:
    """Whether sign-randomised controls are uninformative for these gradients.

    Returns ``True`` when the stable rank is below 2 or the median absolute
    cosine of a fresh sign-randomised family to the mean gradient exceeds
    ``threshold``.
    """
    if stable_rank < 2.0:
        return True
    probes = sign_randomized_gradients(gradients, 16, seed=12345)
    reference = mean_gradient(gradients)
    cosines = [cosine_abs(row, reference) for row in probes]
    return bool(np.median(cosines) > threshold) if cosines else True
