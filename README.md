# steercert

**Does the direction your safety monitor reads also control the behaviour?**

A linear probe that detects a behaviour in a language model's activations is a
*sensor*. A steering vector that changes the behaviour is an *actuator*. They
are often assumed to be the same direction. `steercert` measures whether they
are, using the geometry of per-example behavioural gradients (the *effect
Gramian*), and reports the answer under explicit reporting rules.

For any candidate direction at an intervention site (a probe, a steering
vector, an SAE feature), `steercert` reports:

- **write efficiency**: the share of the best achievable steering effect the
  direction captures, with the *absolute* leverage `‖ḡ‖` always reported next
  to it;
- **dark directions**: the strongest direction a probe (or a set of monitors)
  cannot see, and how much leverage it keeps;
- **controls**: isotropic, shuffled-label and sign-randomised families, with a
  check for the stable-rank degeneracy that makes sign-randomised controls
  uninformative;
- **validation**: optional held-out finite interventions and the Spearman
  ranking between the certificate and measured effects;
- **findings**: machine-readable warnings when an audit cannot support a claim.

The certificate costs one backward pass per batch, with no interventions.

## Install

```bash
pip install steercert            # NumPy core only
pip install "steercert[hf]"      # + PyTorch / Hugging Face adapter
pip install "steercert[all]"     # + steering-vectors, repeng and nnsight integrations
```

## Thirty-second example (no model download)

```bash
steercert demo
```

```python
import steercert
from steercert.synthetic import decoupled_site

site = decoupled_site(overlap=0.1)   # a probe that reads perfectly but barely writes
report = steercert.audit(
    site.gradients,
    probe=site.read_direction,
    states=site.states,
    labels=site.labels,
)
print(report.summary())
```

## Auditing a Hugging Face model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import steercert
from steercert.hf import Site, collect, measure_effects, token_ids, token_logit_difference

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").eval()
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
phi = token_logit_difference(token_ids(tok, ["hate", "stupid", "idiot"]),
                             token_ids(tok, ["love", "kind", "great"]))
site = Site.block(6)

# three disjoint prompt splits: fit the probe, build the certificate, validate
fit = collect(model, tok, fit_prompts, site, phi)
cert = collect(model, tok, cert_prompts, site, phi)
probe = steercert.logistic_probe(fit.states[site], fit_labels)

report = steercert.audit(cert.gradients[site], probe=probe,
                         states=cert.states[site], labels=cert_labels, site=str(site))
measured = measure_effects(model, tok, test_prompts, site,
                           report.direction_matrix(include_controls=False), phi)
report = report.with_measurement(measured)
print(report.summary())
report.to_json("audit.json")
```

## Works with the tools you already use

| Library | What steercert adds |
|---|---|
| [steering-vectors](https://github.com/steering-vectors/steering-vectors) | `from_steering_vector` / `to_steering_vector`: audit a trained vector, or export a dark direction as one |
| [repeng](https://github.com/vgel/repeng) | `from_control_vector` / `to_control_vector` |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `save_steer_config`: run any audited direction through `lm_eval --model steered` to measure capability cost; `load_steer_config` audits existing configs, including SAE-derived ones |
| [nnsight](https://nnsight.net) / NDIF | `collect_nnsight(..., remote=True)`: certificates on 8B–405B open models without a local GPU (experimental) |

All integrations live in `steercert.integrations` and import their target
library only when used.

## Diffusion language models

`steercert.two_clock` covers masked-diffusion LMs, where every site has a depth
and a denoising step. It provides the counting baseline that schedule-aware
steering methods should beat, the normalised shape-gap test, the exact
closed-form construction, and a `Lattice` container with falsifier checks.

## Reporting rules built in

These rules came from errors found while auditing real experiments:

1. Report absolute leverage `‖ḡ‖` next to every relative efficiency. A ratio can
   rise while the leverage it is a ratio of collapses.
2. Use residual-norm-comparable step sizes (`alpha_mode="residual-median"`)
   before comparing sites or models.
3. Report the effect-Gramian stable rank whenever sign-randomised controls are
   used; near rank one they collapse onto the mean gradient (finding `SC102`).
4. Beat construction-matched controls, not just isotropic random directions
   (`SC103`, `SC203`).
5. Validate the certificate on held-out prompts before using it to select
   directions (`SC301`).

## CI regression checks

```bash
steercert check audit.json --min-efficiency my_vector=0.5 --min-ranking 0.7
```

exits with status 1 if a steering vector has lost leverage after a fine-tune or
quantisation.

## Scope

The bundled functionals are next-token proxies. They are not measurements of
generated-text behaviour. Treat a certificate as a cheap, local, first-order
screen and validate conclusions with generation-level evaluation.

## Citing

See [`CITATION.cff`](CITATION.cff). A JOSS paper draft is in [`paper/`](paper/).

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT. See [`LICENSE`](LICENSE).
