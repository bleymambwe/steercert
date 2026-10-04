"""Write-side geometry: the effect Gramian and certificates built from it.

Every function here operates on NumPy arrays and needs no deep-learning
framework. Model instrumentation lives in :mod:`steercert.hf`.

Notation
--------
``gradients`` has shape ``(n_examples, width)``. Row ``i`` is the derivative of
a scalar behavioural functional ``phi_i`` with respect to the activation at one
intervention site. ``directions`` has shape ``(n_directions, width)``; rows are
normalised internally, so only their orientation matters.

The effect Gramian is ``W = E[g g^T]``. It is never formed as a dense
``width x width`` matrix: all spectral quantities come from a thin SVD of the
gradient matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12

__all__ = [
    "EffectGeometry",
    "certificate_scores",
    "cosine_abs",
    "effect_geometry",
    "mean_gradient",
    "project_orthogonal",
    "rho",
    "rms_dark_direction",
    "second_order_prediction",
    "signed_dark_direction",
    "stacked_mean_gradient_direction",
    "subspace_dark_direction",
    "unit",
    "usable_radius",
]


def _as_matrix(array: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have shape (rows, width) with both > 0")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def unit(vector: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return a unit-norm copy of ``vector``.

    Raises
    ------
    ValueError
        If the vector is zero or not finite, because its direction is undefined.
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("cannot normalise a zero or non-finite direction")
    return vector / norm


def cosine_abs(left: np.ndarray, right: np.ndarray, eps: float = EPS) -> float:
    """Absolute cosine between two vectors, or ``0.0`` if either is null."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= eps:
        return 0.0
    return float(abs(np.dot(left, right)) / denom)


def project_orthogonal(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project ``vector`` onto the orthogonal complement of ``normal``."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    normal = unit(normal)
    return vector - np.dot(vector, normal) * normal


def mean_gradient(gradients: np.ndarray) -> np.ndarray:
    """Mean behavioural gradient ``g_bar``; its norm is the absolute write leverage."""
    return _as_matrix(gradients, "gradients").mean(axis=0)


def rho(gradients: np.ndarray, probe: np.ndarray) -> float:
    """Alignment ``|cos(probe, g_bar)|`` between a read direction and the mean gradient.

    ``rho = 1`` means the probe points exactly where a signed write is most
    effective; ``rho = 0`` means writing along the probe has no first-order
    mean effect.
    """
    return cosine_abs(mean_gradient(gradients), probe)


@dataclass(frozen=True)
class EffectGeometry:
    """Spectral summary of the effect Gramian ``W = E[g g^T]``.

    Attributes
    ----------
    trace:
        ``tr(W)``, the mean squared gradient norm.
    leading_eigenvalue:
        ``lambda_max(W)``; ``sqrt`` of it is the RMS effect of the best direction.
    stable_rank:
        ``tr(W) / lambda_max(W)``. Near 1, per-example gradients are nearly
        collinear and sign-randomised controls become degenerate.
    participation_ratio:
        ``tr(W)^2 / ||W||_F^2``.
    entropy_effective_rank:
        ``exp`` of the Shannon entropy of the normalised spectrum.
    entropy_rank_over_width:
        Effective rank divided by the activation width.
    mean_gradient_norm:
        ``||g_bar||``, the absolute signed write leverage. Report it next to
        every relative efficiency.
    top_eigenvalues:
        Leading eigenvalues of ``W``.
    top_direction:
        Leading eigenvector of ``W``, the RMS-optimal write direction.
    """

    trace: float
    leading_eigenvalue: float
    stable_rank: float
    participation_ratio: float
    entropy_effective_rank: float
    entropy_rank_over_width: float
    mean_gradient_norm: float
    top_eigenvalues: tuple[float, ...]
    top_direction: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (the direction is omitted)."""
        return {
            "trace": self.trace,
            "leading_eigenvalue": self.leading_eigenvalue,
            "stable_rank": self.stable_rank,
            "participation_ratio": self.participation_ratio,
            "entropy_effective_rank": self.entropy_effective_rank,
            "entropy_rank_over_width": self.entropy_rank_over_width,
            "mean_gradient_norm": self.mean_gradient_norm,
            "top_eigenvalues": list(self.top_eigenvalues),
        }


def effect_geometry(gradients: np.ndarray, top_k: int = 10) -> EffectGeometry:
    """Compute the effect-Gramian spectrum and rank diagnostics.

    The leading direction is the first right singular vector of the gradient
    matrix, oriented so that it has a non-negative inner product with the mean
    gradient.
    """
    gradients = _as_matrix(gradients, "gradients")
    _, singular_values, vt = np.linalg.svd(gradients, full_matrices=False)
    eigenvalues = np.square(singular_values) / gradients.shape[0]
    trace = float(eigenvalues.sum())
    leading = float(eigenvalues[0])
    squared_sum = float(np.square(eigenvalues).sum())
    stable_rank = trace / leading if leading > EPS else 0.0
    participation = trace * trace / squared_sum if squared_sum > EPS else 0.0
    if trace > EPS:
        probabilities = eigenvalues[eigenvalues > EPS] / trace
        entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        entropy_rank = 0.0
    g_bar = gradients.mean(axis=0)
    top = vt[0].copy()
    if float(np.dot(top, g_bar)) < 0.0:
        top = -top
    return EffectGeometry(
        trace=trace,
        leading_eigenvalue=leading,
        stable_rank=stable_rank,
        participation_ratio=participation,
        entropy_effective_rank=entropy_rank,
        entropy_rank_over_width=entropy_rank / gradients.shape[1],
        mean_gradient_norm=float(np.linalg.norm(g_bar)),
        top_eigenvalues=tuple(float(x) for x in eigenvalues[:top_k]),
        top_direction=top,
    )


def certificate_scores(
    gradients: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order write certificates for each candidate direction.

    Returns
    -------
    signed, mean_absolute, rms:
        ``|E[g^T w]|``, ``E[|g^T w|]`` and ``sqrt(E[(g^T w)^2])`` for every
        unit-normalised row ``w`` of ``directions``. The signed score predicts
        the mean effect of a small step; the other two keep per-example effects
        that cancel in the mean.
    """
    gradients = _as_matrix(gradients, "gradients")
    directions = _as_matrix(np.atleast_2d(directions), "directions")
    if gradients.shape[1] != directions.shape[1]:
        raise ValueError(f"gradient width {gradients.shape[1]} != direction width {directions.shape[1]}")
    normalised = np.stack([unit(row) for row in directions])
    projections = gradients @ normalised.T
    signed = np.abs(projections.mean(axis=0))
    mean_absolute = np.abs(projections).mean(axis=0)
    rms = np.sqrt(np.square(projections).mean(axis=0))
    return signed, mean_absolute, rms


def signed_dark_direction(gradients: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Best signed-effect direction constrained to be orthogonal to ``probe``.

    This is ``(I - p p^T) g_bar`` normalised: a direction the probe cannot see
    that still moves the behaviour on average.
    """
    return unit(project_orthogonal(mean_gradient(gradients), probe))


def rms_dark_direction(gradients: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Best RMS-effect direction constrained to be orthogonal to ``probe``.

    Solves the Rayleigh problem for ``P W P`` with ``P = I - p p^T`` through
    the SVD of the projected gradients.
    """
    return subspace_dark_direction(gradients, np.atleast_2d(probe))


def subspace_dark_direction(gradients: np.ndarray, read_basis: np.ndarray) -> np.ndarray:
    """Best RMS-effect direction orthogonal to a whole read subspace.

    Parameters
    ----------
    read_basis:
        Array of shape ``(k, width)`` whose rows span the subspace a monitor
        reads (for example several probes, or SAE features used for detection).
    """
    gradients = _as_matrix(gradients, "gradients")
    basis = _as_matrix(np.atleast_2d(read_basis), "read_basis")
    if basis.shape[1] != gradients.shape[1]:
        raise ValueError("read_basis width differs from gradient width")
    q, r = np.linalg.qr(basis.T)
    keep = np.abs(np.diag(r)) > EPS * max(basis.shape)
    q = q[:, keep]
    projected = gradients - (gradients @ q) @ q.T
    if float(np.linalg.norm(projected)) <= EPS:
        raise ValueError("every gradient lies inside the read subspace; no dark direction exists")
    _, _, vt = np.linalg.svd(projected, full_matrices=False)
    direction = vt[0]
    if float(np.dot(direction, gradients.mean(axis=0))) < 0.0:
        direction = -direction
    return unit(direction)


def usable_radius(gradients: np.ndarray, direction: np.ndarray, curvature: float) -> float:
    """Step size up to which the first-order certificate is trustworthy.

    For a unit direction with signed certificate ``s`` and a bound ``M`` on the
    directional second derivative, the linear term dominates the quadratic
    remainder while ``alpha < 2 s / M``.
    """
    signed, _, _ = certificate_scores(gradients, np.atleast_2d(direction))
    if curvature <= EPS:
        return float("inf")
    return float(2.0 * signed[0] / curvature)


def second_order_prediction(
    alpha: float, first_directional: np.ndarray, second_directional: np.ndarray
) -> np.ndarray:
    """Second-order Taylor prediction ``alpha * d1 + alpha^2 / 2 * d2`` per example."""
    first = np.asarray(first_directional, dtype=np.float64)
    second = np.asarray(second_directional, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("first- and second-order arrays must have equal shape")
    return alpha * first + 0.5 * alpha * alpha * second


def stacked_mean_gradient_direction(gradient_blocks: list[np.ndarray]) -> np.ndarray:
    """Optimal unit-energy direction for writing at several sites at once.

    Each block has shape ``(n_examples, width_at_site)`` and rows must refer to
    the same examples. The result concatenates the per-site mean gradients and
    normalises them under one total-energy constraint.
    """
    if not gradient_blocks:
        raise ValueError("at least one gradient block is required")
    arrays = [_as_matrix(block, "gradient block") for block in gradient_blocks]
    n_examples = arrays[0].shape[0]
    if any(block.shape[0] != n_examples for block in arrays):
        raise ValueError("gradient blocks must have the same number of examples")
    return unit(np.concatenate([block.mean(axis=0) for block in arrays]))
