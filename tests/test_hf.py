import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import steercert  # noqa: E402
from steercert.hf import (  # noqa: E402
    Site,
    collect,
    directional_derivatives,
    find_blocks,
    measure_effects,
    module_name,
    residual_norm_median,
    token_log_probability,
    token_logit_difference,
)

pytestmark = pytest.mark.hf

PHI = token_logit_difference([5, 6, 7], [10, 11])


def _directions(width, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return {f"d{i}": rng.normal(size=width) for i in range(n)}


@pytest.mark.parametrize("site", [Site.embeddings(), Site.block(0), Site.block(1), Site.final_norm()])
@pytest.mark.parametrize("positions", ["last", "all"])
def test_gradients_match_finite_differences(tiny_lm, prompts, site, positions):
    model, tok = tiny_lm
    data = collect(model, tok, prompts, site, PHI, positions=positions, batch_size=3)
    gradients = data.gradients[site]
    directions = _directions(gradients.shape[1])
    measured = measure_effects(
        model,
        tok,
        prompts,
        site,
        directions,
        PHI,
        alpha=1e-4,
        alpha_mode="absolute",
        positions=positions,
        batch_size=3,
    )
    # Llama's RMSNorm casts to float32 even in a float64 model, so its finite
    # differences carry ~1e-7 / alpha round-off; a wrong site or position is off by O(1)
    atol = 2e-4 if model.config.model_type == "llama" else 1e-6
    for name, vector in directions.items():
        predicted = gradients @ (vector / np.linalg.norm(vector))
        np.testing.assert_allclose(measured.slopes[name], predicted, rtol=1e-3, atol=atol)


def test_per_example_gradients_do_not_depend_on_batching(tiny_lm, prompts):
    model, tok = tiny_lm
    site = Site.block(1)
    one = collect(model, tok, prompts, site, PHI, batch_size=1).gradients[site]
    many = collect(model, tok, prompts, site, PHI, batch_size=8).gradients[site]
    np.testing.assert_allclose(one, many, rtol=1e-9, atol=1e-12)


def test_sites_match_hidden_states(tiny_lm, prompts):
    model, tok = tiny_lm
    n_blocks = len(find_blocks(model))
    sites = [Site.from_hidden_state_index(h, n_blocks) for h in range(n_blocks + 1)]
    data = collect(model, tok, prompts[:3], sites, gradients=False)
    encoded = tok(prompts[:3])
    with torch.no_grad():
        hidden = model(**encoded, output_hidden_states=True).hidden_states
    last = encoded["attention_mask"].sum(1) - 1
    rows = torch.arange(3)
    for h, site in enumerate(sites):
        np.testing.assert_allclose(data.states[site], hidden[h][rows, last].numpy(), rtol=1e-10, atol=1e-10)


def test_site_parsing_and_validation():
    assert Site.parse("block.3") == Site.block(3) == Site.parse("block:3")
    assert Site.parse("final_norm") == Site.final_norm()
    assert str(Site.block(2)) == "block.2"
    assert Site.from_hidden_state_index(-1, 6) == Site.final_norm()
    with pytest.raises(ValueError):
        Site("block")
    with pytest.raises(ValueError):
        Site.parse("layer7")
    with pytest.raises(IndexError):
        Site.from_hidden_state_index(9, 6)


def test_second_order_matches_finite_curvature(gpt2_lm, prompts):
    model, tok = gpt2_lm
    site = Site.block(1)
    directions = _directions(32, n=2, seed=1)
    first, second = directional_derivatives(model, tok, prompts, site, directions, PHI)
    small = measure_effects(model, tok, prompts, site, directions, PHI, alpha=1e-4, alpha_mode="absolute")
    large = measure_effects(model, tok, prompts, site, directions, PHI, alpha=1e-2, alpha_mode="absolute")
    for name in directions:
        np.testing.assert_allclose(small.slopes[name], first[name], rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(large.curvature[name], second[name], rtol=2e-2, atol=1e-3)


def test_residual_median_alpha_and_end_to_end_audit(gpt2_lm, prompts):
    model, tok = gpt2_lm
    site = Site.block(1)
    scale = residual_norm_median(model, tok, prompts, site)
    assert scale > 0
    data = collect(model, tok, prompts, site, PHI)
    report = steercert.audit(data.gradients[site], _directions(32), controls=False, site=str(site))
    measured = measure_effects(model, tok, prompts, site, report.direction_matrix(), PHI, alpha=1e-3)
    assert measured.alpha == pytest.approx(1e-3 * scale)
    report = report.with_measurement(measured)
    assert report.ranking_spearman > 0.9
    assert report.measurement["alpha_mode"] == "residual-median"


def test_log_probability_functional_runs(gpt2_lm, prompts):
    model, tok = gpt2_lm
    data = collect(model, tok, prompts, Site.block(0), token_log_probability([3, 4]))
    assert np.all(data.values <= 0.0)


def test_module_names(tiny_lm):
    model, _ = tiny_lm
    name = module_name(model, Site.block(1), root=model.base_model)
    assert name.endswith("1")
    with pytest.raises(ValueError):
        module_name(model, Site.embeddings())


def test_collect_requires_functional_for_gradients(gpt2_lm, prompts):
    model, tok = gpt2_lm
    with pytest.raises(ValueError):
        collect(model, tok, prompts, Site.block(0))
    with pytest.raises(ValueError):
        collect(model, tok, prompts, Site.block(0), PHI, positions="first")
