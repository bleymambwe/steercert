# Integrations

All integrations import their target library lazily. Every one of them uses
the **block-output** convention: layer `k` means the output of decoder block
`k`, which is `steercert.hf.Site.block(k)`.

## steering-vectors

```python
from steercert.integrations.steering_vectors import from_steering_vector, to_steering_vector

vectors = from_steering_vector(sv)          # {Site.block(k): np.ndarray}
better = to_steering_vector({k: direction})  # back to a SteeringVector
```

`SteeringVector.apply` adds the vector at every selected token. Audit it with
`collect(..., positions="all")`, which sums gradients over positions and is
the exact derivative for that edit. `tests/test_integrations.py` checks that
the logits steering-vectors produces equal those of steercert's own edit.

## repeng

```python
from steercert.integrations.repeng import from_control_vector, to_control_vector
```

repeng trains on `hidden_states[k + 1]` and controls the output of block `k`.
These match except at the last block, where Hugging Face applies the final norm first.

## lm-evaluation-harness

```python
from steercert.integrations.lm_eval import save_steer_config, load_steer_config

save_steer_config("dark.pt", model, {Site.block(6): report.directions["reference/dark_rms"]},
                  coefficient=0.03 * residual_norm)
```

```bash
lm_eval run --model steered --model_args pretrained=EleutherAI/pythia-160m,steer_path=dark.pt --tasks hellaswag
```

This measures the capability cost of any audited direction on standard
benchmarks. `load_steer_config` reads existing configs, including ones derived
from SAE features, so they can be audited. The tests replicate lm-eval's hook
resolution and `add` semantics and check them against steercert's edit.

## nnsight and NDIF (experimental)

```python
from nnsight import LanguageModel
from steercert.integrations.nnsight import collect_nnsight

model = LanguageModel("meta-llama/Llama-3.1-8B")
data = collect_nnsight(model, prompts, layers=[8, 16, 24], functional=phi, remote=True)
report = steercert.audit(data.gradients[Site.block(16)])
```

With `remote=True` the trace runs on the National Deep Inference Fabric, which
serves open models far larger than a laptop can hold. Locally, on nnsight 0.7,
gradients and states match `steercert.hf.collect` to 1e-6. Remote behaviour
depends on NDIF's deployed models and policies. Validate on a small model first.

## Any other source

`steercert.audit` takes plain arrays. Gradients from TransformerLens hooks,
pyvene interventions or a custom JAX model work unchanged, as does any
direction such as an SAE decoder row, a classifier head or a CAST condition
vector.
