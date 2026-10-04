import numpy as np
import pytest

from steercert.controls import (
    isotropic,
    make_controls,
    shuffled_label_probes,
    sign_control_is_degenerate,
    sign_randomized_gradients,
)
from steercert.geometry import cosine_abs, effect_geometry
from steercert.probes import auc, difference_of_means, direction_auc, fisher_probe, logistic_probe
from steercert.stats import bootstrap_interval, rankdata, spearman
from steercert.synthetic import decoupled_site


def test_auc_matches_known_values():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auc([1, 2], [1, 1]))


def test_rankdata_and_spearman_handle_ties():
    np.testing.assert_allclose(rankdata([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 2], [1, 2]))


@pytest.mark.parametrize("fit", [difference_of_means, fisher_probe, logistic_probe])
def test_probes_recover_the_read_direction(fit):
    site = decoupled_site(n=3000, width=16, separation=3.0, seed=0)
    probe = fit(site.states, site.labels)
    assert cosine_abs(probe, site.read_direction) > 0.9
    assert direction_auc(site.states, site.labels, probe) > 0.9


def test_probe_needs_both_classes():
    with pytest.raises(ValueError):
        difference_of_means(np.ones((4, 3)), np.ones(4))


def test_bootstrap_interval_brackets_mean():
    values = np.random.default_rng(0).normal(size=500)
    low, high = bootstrap_interval(lambda idx: float(values[idx].mean()), len(values), repeats=300)
    assert low < values.mean() < high


def test_control_families_shapes_and_norms():
    site = decoupled_site(seed=1)
    assert np.allclose(np.linalg.norm(isotropic(8, 5), axis=1), 1.0)
    assert sign_randomized_gradients(site.gradients, 6).shape == (6, site.gradients.shape[1])
    assert shuffled_label_probes(site.states, site.labels, 3).shape == (3, site.gradients.shape[1])
    controls = make_controls(site.gradients, site.states, site.labels)
    families = set(controls.family_of.values())
    assert families == {"isotropic", "shuffled_label_probe", "sign_randomized_gradient"}
    cosines = controls.cosines_to(site.gradients.mean(axis=0))
    assert np.median(cosines["sign_randomized_gradient"]) > np.median(cosines["isotropic"])


def test_sign_degeneracy_tracks_stable_rank():
    near_rank_one = decoupled_site(gradient_spread=0.05, seed=2).gradients
    assert sign_control_is_degenerate(near_rank_one, effect_geometry(near_rank_one).stable_rank)
    spread = np.random.default_rng(3).normal(size=(500, 32))
    assert not sign_control_is_degenerate(spread, effect_geometry(spread).stable_rank)
