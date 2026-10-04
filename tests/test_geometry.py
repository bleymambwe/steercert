import numpy as np
import pytest

from steercert.geometry import (
    certificate_scores,
    cosine_abs,
    effect_geometry,
    rho,
    rms_dark_direction,
    second_order_prediction,
    signed_dark_direction,
    stacked_mean_gradient_direction,
    subspace_dark_direction,
    unit,
    usable_radius,
)


def test_exact_read_write_decoupling():
    gradients = np.tile(np.array([[0.0, 2.0, 0.0]]), (20, 1))
    probe = np.array([1.0, 0.0, 0.0])
    dark = signed_dark_direction(gradients, probe)
    signed, mean_abs, rms = certificate_scores(gradients, np.stack([probe, dark]))
    np.testing.assert_allclose(signed, [0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(mean_abs, [0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(rms, [0.0, 2.0], atol=1e-12)
    assert abs(float(probe @ dark)) < 1e-12
    assert rho(gradients, probe) == pytest.approx(0.0, abs=1e-12)


def test_gramian_rank_and_top_direction():
    gradients = np.array([[3.0, 0.0], [-3.0, 0.0], [3.0, 0.0]])
    geometry = effect_geometry(gradients)
    assert geometry.trace == pytest.approx(9.0)
    assert geometry.leading_eigenvalue == pytest.approx(9.0)
    assert geometry.stable_rank == pytest.approx(1.0)
    assert geometry.participation_ratio == pytest.approx(1.0)
    assert geometry.entropy_effective_rank == pytest.approx(1.0)
    assert cosine_abs(geometry.top_direction, [1.0, 0.0]) == pytest.approx(1.0)
    assert float(geometry.top_direction @ gradients.mean(axis=0)) >= 0.0


def test_rank_summaries_are_not_conflated():
    # eigenvalues 2 and 1: tr/lambda_max = 1.5 but tr^2/||W||_F^2 = 1.8
    gradients = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, np.sqrt(2.0)], [0.0, -np.sqrt(2.0)]])
    geometry = effect_geometry(gradients)
    assert geometry.trace == pytest.approx(3.0)
    assert geometry.stable_rank == pytest.approx(1.5)
    assert geometry.participation_ratio == pytest.approx(1.8)


def test_second_order_quadratic_is_exact():
    alpha, first, second = 0.4, np.array([2.0, -1.0]), np.array([6.0, 4.0])
    np.testing.assert_allclose(
        second_order_prediction(alpha, first, second), alpha * first + 0.5 * alpha**2 * second
    )
    with pytest.raises(ValueError):
        second_order_prediction(alpha, first, second[:1])


def test_rms_dark_survives_signed_cancellation():
    gradients = np.array([[0.0, 3.0], [0.0, -3.0]])
    probe = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        signed_dark_direction(gradients, probe)
    dark = rms_dark_direction(gradients, probe)
    assert abs(float(probe @ dark)) < 1e-12
    _, _, rms = certificate_scores(gradients, np.stack([probe, dark]))
    np.testing.assert_allclose(rms, [0.0, 3.0], atol=1e-12)


def test_stacked_direction_uses_one_energy_budget():
    left = np.tile(np.array([[3.0, 0.0]]), (5, 1))
    right = np.tile(np.array([[0.0, 4.0]]), (5, 1))
    np.testing.assert_allclose(stacked_mean_gradient_direction([left, right]), [0.6, 0.0, 0.0, 0.8])
    with pytest.raises(ValueError):
        stacked_mean_gradient_direction([left, right[:3]])


def test_rms_dark_is_optimal_among_orthogonal_directions():
    rng = np.random.default_rng(0)
    gradients = rng.normal(size=(200, 6)) @ np.diag([3, 2, 1, 0.5, 0.2, 0.1])
    probe = unit(rng.normal(size=6))
    dark = rms_dark_direction(gradients, probe)
    best = certificate_scores(gradients, dark[None])[2][0]
    for _ in range(2000):
        candidate = rng.normal(size=6)
        candidate -= (candidate @ probe) * probe
        assert certificate_scores(gradients, candidate[None])[2][0] <= best + 1e-9


def test_subspace_dark_is_orthogonal_to_every_monitor():
    rng = np.random.default_rng(1)
    gradients = rng.normal(size=(100, 8))
    monitors = rng.normal(size=(3, 8))
    dark = subspace_dark_direction(gradients, monitors)
    np.testing.assert_allclose(monitors @ dark, 0.0, atol=1e-10)
    with pytest.raises(ValueError):
        subspace_dark_direction(gradients, np.eye(8))


def test_scores_invariant_under_orthogonal_reparameterisation():
    """Certificates must not depend on the basis of the residual stream."""
    rng = np.random.default_rng(2)
    gradients = rng.normal(size=(64, 10))
    directions = rng.normal(size=(4, 10))
    rotation, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    before = certificate_scores(gradients, directions)
    after = certificate_scores(gradients @ rotation, directions @ rotation)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=1e-10)
    g1, g2 = effect_geometry(gradients), effect_geometry(gradients @ rotation)
    assert g1.stable_rank == pytest.approx(g2.stable_rank)
    assert g1.entropy_effective_rank == pytest.approx(g2.entropy_effective_rank)
    assert rho(gradients, directions[0]) == pytest.approx(rho(gradients @ rotation, directions[0] @ rotation))


def test_usable_radius():
    gradients = np.tile([[2.0, 0.0]], (4, 1))
    assert usable_radius(gradients, np.array([1.0, 0.0]), curvature=4.0) == pytest.approx(1.0)
    assert usable_radius(gradients, np.array([1.0, 0.0]), curvature=0.0) == float("inf")


@pytest.mark.parametrize("bad", [np.zeros(3), np.array([np.nan, 1.0, 0.0])])
def test_unit_rejects_undefined_directions(bad):
    with pytest.raises(ValueError):
        unit(bad)


def test_shape_validation():
    with pytest.raises(ValueError):
        certificate_scores(np.ones((3, 4)), np.ones((2, 5)))
    with pytest.raises(ValueError):
        effect_geometry(np.array([[np.inf, 1.0]]))
