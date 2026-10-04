"""Integration tests.

The lm-eval test reproduces lm-eval-harness's own hook resolution and ``add``
semantics, so it runs without lm-eval installed. The other tests skip when
their target library is absent.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from steercert.hf import Site, _edit  # noqa: E402
from steercert.integrations import as_site_vectors  # noqa: E402
from steercert.integrations.lm_eval import load_steer_config, save_steer_config, to_steer_config  # noqa: E402

pytestmark = pytest.mark.hf


def _logits(model, tok, prompts):
    with torch.no_grad():
        return model(**tok(prompts)).logits


def _lm_eval_style_add(model, config):
    """Replicates lm_eval.models.hf_steered.steer for action='add'."""
    handles = []
    modules = dict(model.base_model.named_modules())
    for hookpoint, info in config.items():
        vector = info["steering_vector"].to(torch.float64) * info["steering_coefficient"]

        def hook(_m, _i, output, vector=vector):
            if isinstance(output, tuple):
                return (output[0] + vector, *output[1:])
            return output + vector

        handles.append(modules[hookpoint].register_forward_hook(hook))
    return handles


def test_lm_eval_export_matches_steercert_edit(tiny_lm, prompts, tmp_path):
    model, tok = tiny_lm
    rng = np.random.default_rng(0)
    vectors = {Site.block(1): rng.normal(size=32), Site.final_norm(): rng.normal(size=32)}
    config = to_steer_config(model, vectors, coefficient=0.7)
    assert all(info["action"] == "add" for info in config.values())

    handles = _lm_eval_style_add(model, config)
    try:
        theirs = _logits(model, tok, prompts)
    finally:
        for handle in handles:
            handle.remove()

    from contextlib import ExitStack

    with ExitStack() as stack:
        for site, vector in vectors.items():
            v = torch.as_tensor(0.7 * vector / np.linalg.norm(vector), dtype=torch.float64)
            stack.enter_context(_edit(model, site, lambda h, v=v: h + v))
        ours = _logits(model, tok, prompts)
    torch.testing.assert_close(theirs, ours)

    path = save_steer_config(
        tmp_path / "steer.pt", model, vectors, coefficient={1: 0.7, Site.final_norm(): 0.2}
    )
    loaded = load_steer_config(path, model)
    assert set(loaded) == set(vectors)
    assert loaded[Site.final_norm()][1] == pytest.approx(0.2)
    np.testing.assert_allclose(
        loaded[Site.block(1)][0], vectors[Site.block(1)] / np.linalg.norm(vectors[Site.block(1)]), rtol=1e-6
    )


def test_lm_eval_rejects_embeddings_site(gpt2_lm):
    model, _ = gpt2_lm
    with pytest.raises(ValueError):
        to_steer_config(model, {Site.embeddings(): np.ones(32)})


def test_as_site_vectors_accepts_ints_and_tensors():
    converted = as_site_vectors({2: torch.ones(4), Site.final_norm(): np.zeros(4)})
    assert set(converted) == {Site.block(2), Site.final_norm()}
    assert converted[Site.block(2)].dtype == np.float64


def test_steering_vectors_roundtrip_and_semantics(gpt2_lm, prompts):
    pytest.importorskip("steering_vectors")
    from steercert.integrations.steering_vectors import from_steering_vector, to_steering_vector

    model, tok = gpt2_lm
    vector = np.random.default_rng(1).normal(size=32)
    sv = to_steering_vector({1: vector}, dtype=torch.float64)
    assert np.allclose(from_steering_vector(sv)[Site.block(1)], vector)
    with sv.apply(model):
        theirs = _logits(model, tok, prompts)
    v = torch.as_tensor(vector, dtype=torch.float64)
    with _edit(model, Site.block(1), lambda h: h + v):
        ours = _logits(model, tok, prompts)
    torch.testing.assert_close(theirs, ours)


def test_repeng_roundtrip():
    pytest.importorskip("repeng")
    from steercert.integrations.repeng import from_control_vector, to_control_vector

    vector = np.arange(8, dtype=np.float64)
    cv = to_control_vector({3: vector}, model_type="gpt2")
    assert np.allclose(from_control_vector(cv)[Site.block(3)], vector)


NNSIGHT_CHECK = """
import numpy as np, torch, transformers, sys
sys.path.insert(0, {tests!r})
from conftest import PROMPTS, _tiny
from nnsight import NNsight
from steercert.hf import Site, collect, token_logit_difference
from steercert.integrations.nnsight import collect_nnsight

model, tok = _tiny("gpt2")
phi = token_logit_difference([5, 6], [9])
reference = collect(model, tok, PROMPTS, [Site.block(0), Site.block(2)], phi)
wrapped = NNsight(model)
wrapped.tokenizer = tok
result = collect_nnsight(wrapped, PROMPTS, [0, 2], phi)
for site in (Site.block(0), Site.block(2)):
    np.testing.assert_allclose(result.gradients[site], reference.gradients[site], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.states[site], reference.states[site], rtol=1e-6, atol=1e-9)
np.testing.assert_allclose(result.values, reference.values, rtol=1e-6, atol=1e-9)
print("nnsight-ok")
"""


def test_nnsight_matches_hf_collect():
    """Run in a subprocess: nnsight's greenlets disable coverage tracing for later tests."""
    pytest.importorskip("nnsight")
    import subprocess
    import sys
    from pathlib import Path

    script = NNSIGHT_CHECK.format(tests=str(Path(__file__).parent))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "nnsight-ok" in result.stdout, result.stderr[-3000:]
