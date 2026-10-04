# steercert

**Does the direction your safety monitor reads also control the behaviour?**

Linear probes on activations are now used as low-cost safety monitors, and
activation steering is used to change model behaviour at inference time. Both
work with a direction in a model's residual stream, and it is common to assume
that a direction that *detects* a behaviour can also *change* it, or that a
monitor reading one direction sees every way the behaviour can be moved.

`steercert` tests that assumption. Given per-example gradients of a
behavioural functional at an intervention site, it builds the **effect
Gramian** $W = \mathbb{E}[g g^\top]$ and reports, for any direction:

| Question | Quantity |
|---|---|
| How much of the achievable steering effect does this direction capture? | RMS write efficiency $\sqrt{w^\top W w / \lambda_{\max}(W)}$ |
| How much leverage is there at this site at all? | absolute leverage $\lVert \bar g \rVert$ |
| What is the strongest direction a probe cannot see? | dark direction, and its efficiency |
| Does the direction beat construction-matched controls? | control families with degeneracy checks |
| Does the certificate predict real interventions here? | held-out Spearman ranking |

The certificate needs one backward pass per batch and no interventions.
Finite interventions are used only to validate it.

```python
import steercert
from steercert.synthetic import decoupled_site

site = decoupled_site(overlap=0.1)
report = steercert.audit(site.gradients, probe=site.read_direction,
                         states=site.states, labels=site.labels)
print(report.summary())
```

!!! note "Scope"
    The bundled behavioural functionals are next-token proxies. A certificate
    is a cheap, local, first-order screen. Validate conclusions about
    deployed behaviour with generation-level evaluation.

## Where to go next

- [Quick start](quickstart.md): audit a probe in a Hugging Face model with three disjoint splits.
- [Concepts](concepts.md): the definitions and what each number means.
- [Integrations](integrations.md): steering-vectors, repeng, lm-evaluation-harness, nnsight/NDIF.
- [Validation](validation.md): how the package is tested and what it reproduces.
