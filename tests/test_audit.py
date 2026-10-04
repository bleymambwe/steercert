import json

import numpy as np
import pytest

import steercert
from steercert.audit import EffectMeasurement, check_report
from steercert.synthetic import decoupled_site


def codes(report):
    return {finding.code for finding in report.findings}


def test_true_probe_efficiency_matches_designed_overlap():
    site = decoupled_site(n=4000, overlap=0.2, gradient_spread=0.1, seed=3)
    report = steercert.audit(site.gradients, probe=site.read_direction, controls=False)
    assert report.row("probe").rms_efficiency == pytest.approx(0.2, abs=0.02)
    assert report.row("reference/gramian_top").rms_efficiency == pytest.approx(1.0)
    assert report.row("reference/dark_rms").rms_efficiency > 0.97


def test_read_write_gap_finding():
    site = decoupled_site(overlap=0.05, seed=4)
    report = steercert.audit(
        site.gradients, probe=site.read_direction, states=site.states, labels=site.labels
    )
    assert "SC201" in codes(report)
    assert report.row("probe").auc > 0.95


def test_aligned_probe_finding():
    site = decoupled_site(overlap=0.95, gradient_spread=0.1, seed=5)
    report = steercert.audit(site.gradients, probe=site.read_direction, controls=False)
    assert "SC202" in codes(report)
    assert "SC201" not in codes(report)


def test_sign_control_degeneracy_flagged_near_rank_one():
    site = decoupled_site(gradient_spread=0.05, seed=6)
    report = steercert.audit(
        site.gradients, probe=site.read_direction, states=site.states, labels=site.labels
    )
    assert report.geometry.stable_rank < 1.2
    assert "SC102" in codes(report)


def test_sign_control_not_flagged_for_isotropic_gradients():
    rng = np.random.default_rng(7)
    gradients = rng.normal(size=(400, 16)) + 0.3
    report = steercert.audit(gradients, states=rng.normal(size=(400, 16)), labels=rng.integers(0, 2, 400))
    assert "SC102" not in codes(report)


def test_isotropic_only_controls_warn():
    site = decoupled_site(seed=8)
    report = steercert.audit(site.gradients, probe=site.read_direction)
    assert "SC103" in codes(report)


def test_candidate_below_matched_controls_warns():
    site = decoupled_site(overlap=0.0, gradient_spread=0.05, seed=9)
    weak = site.read_direction
    report = steercert.audit(site.gradients, {"weak": weak}, states=site.states, labels=site.labels)
    assert "SC203" in codes(report)


def test_small_sample_warning():
    site = decoupled_site(n=20, seed=10)
    report = steercert.audit(site.gradients, controls=False)
    assert "SC101" in codes(report)


def test_linear_model_measurements_rank_perfectly():
    site = decoupled_site(seed=11)
    report = steercert.audit(site.gradients, {"probe": site.read_direction}, probe="probe")
    names, matrix = report.direction_matrix()
    # for a linear functional the finite slope equals g . w exactly
    slopes = {name: site.gradients @ row for name, row in zip(names, matrix)}
    measured = report.with_measurement(EffectMeasurement(slopes, alpha=1.0, alpha_mode="residual-median"))
    assert measured.ranking_spearman == pytest.approx(1.0)
    for row in measured.rows:
        assert row.measured_efficiency == pytest.approx(row.rms_efficiency, rel=1e-9)
    assert "SC301" not in codes(measured)


def test_bad_ranking_and_absolute_alpha_warn():
    site = decoupled_site(seed=12)
    report = steercert.audit(site.gradients, {"probe": site.read_direction}, probe="probe")
    names, matrix = report.direction_matrix()
    rng = np.random.default_rng(0)
    slopes = {name: rng.normal(size=len(site.gradients)) * rng.uniform(0.1, 1) for name in names}
    measured = report.with_measurement(EffectMeasurement(slopes, alpha=0.25, alpha_mode="absolute"))
    assert {"SC301", "SC302"} <= codes(measured)


def test_measurement_for_unknown_direction_raises():
    site = decoupled_site(seed=13)
    report = steercert.audit(site.gradients, controls=False)
    with pytest.raises(KeyError):
        report.with_measurement({"nope": np.zeros(len(site.gradients))})


def test_json_roundtrip_and_check(tmp_path):
    site = decoupled_site(overlap=0.1, seed=14)
    report = steercert.audit(
        site.gradients, {"vec": site.write_direction, "probe": site.read_direction}, probe="probe"
    )
    path = tmp_path / "report.json"
    report.to_json(path, include_directions=True)
    loaded = json.loads(path.read_text())
    assert loaded["probe"] == "probe"
    assert len(loaded["directions"]["vec"]) == site.gradients.shape[1]
    assert check_report(loaded, min_efficiency={"vec": 0.9}) == []
    failures = check_report(loaded, min_efficiency={"probe": 0.5}, min_ranking=0.7)
    assert len(failures) == 2
    assert check_report(loaded, max_efficiency={"missing": 0.1}) == [
        "direction 'missing' missing from report"
    ]
    assert report.check(min_stable_rank=100.0)


def test_input_validation():
    site = decoupled_site(seed=15)
    with pytest.raises(KeyError):
        steercert.audit(site.gradients, {"a": site.read_direction}, probe="b")
    with pytest.raises(ValueError):
        steercert.audit(site.gradients, {"reference/mean_gradient": site.read_direction})


def test_read_basis_adds_subspace_dark_direction():
    site = decoupled_site(seed=16)
    rng = np.random.default_rng(1)
    basis = np.stack([site.read_direction, rng.normal(size=site.gradients.shape[1])])
    report = steercert.audit(site.gradients, read_basis=basis, controls=False)
    dark = report.directions["reference/dark_subspace"]
    np.testing.assert_allclose(basis @ dark, 0.0, atol=1e-10)


def test_summary_mentions_absolute_leverage():
    site = decoupled_site(seed=17)
    text = steercert.audit(site.gradients, probe=site.read_direction).summary()
    assert "absolute leverage" in text and "stable rank" in text
