---
title: 'steercert: sensor/actuator audits for language-model probes and steering vectors'
tags:
  - Python
  - interpretability
  - activation steering
  - linear probes
  - AI safety
  - language models
authors:
  - name: Blessings Mambwe
    orcid: 0000-0000-0000-0000  # TODO: replace with the author's ORCID
    affiliation: 1
affiliations:
  - name: TODO, author affiliation
    index: 1
date: 4 October 2026
bibliography: paper.bib
---

# Summary

Researchers and safety teams increasingly work with *directions* in the
activation space of large language models. A linear probe [@alain2016;
@belinkov2022] fitted to activations can detect a behaviour such as toxicity
or a harmful request, and production systems now screen user traffic with such
probes [@kramar2026gemini]. Activation steering adds a direction to the
residual stream to change a behaviour at inference time [@turner2023actadd;
@panickssery2023caa; @zou2023repe]. The two uses are often treated as
interchangeable: a direction that detects a behaviour is assumed to control it,
and a monitor that reads one direction is assumed to see the ways the behaviour
can change.

`steercert` is a Python package that tests that assumption at any intervention
site. From per-example gradients of a behavioural functional, it builds the
*effect Gramian* $W = \mathbb{E}[g g^\top]$. For any candidate direction it
then reports the share of the achievable steering effect the direction
captures, alongside the absolute leverage available at the site. It also
reports the strongest direction a probe (or a set of monitors) cannot see,
how the candidate compares with construction-matched controls, and, when
held-out finite interventions are supplied, whether the gradient certificate
ranks real effects correctly. The certificate costs one backward pass per
batch and no interventions. Each audit returns machine-readable findings
that flag the conditions under which its numbers cannot support a claim.

# Statement of need

Decodability and causal control are different properties, and recent work
finds them dissociated in practice. Probes can read a behaviour almost
perfectly while steering along them is weak or harmful
[@basu2026actionability], and sparse-autoencoder features that are active on a
concept are not necessarily the ones that move outputs [@arad2025sae].
Comparing steering directions is also error-prone. Isotropic random
baselines are a weak comparator, and construction-matched controls can be
degenerate [@luo2026steercheck]. Raw-unit step sizes can produce apparent
scaling trends that disappear under comparable units [@wu2026steerable].

Steering libraries make it easy to *build* and *apply* directions, but none
measure whether a direction is a good actuator or what a monitor misses. Each
research group therefore re-implements gradient collection, intervention hooks
and controls, often with subtle errors. Examples are editing a different
tensor than the one differentiated (Hugging Face returns the final-norm output
as the last hidden state) and reporting relative efficiencies without the
absolute leverage they are relative to. `steercert` supplies these pieces
once, tested against finite differences and against the hooks of the
libraries it integrates with. It encodes the reporting rules as checks that
can run in continuous integration, for example to confirm that a deployed
steering vector keeps its leverage after a fine-tune.

The target users are interpretability and AI-safety researchers evaluating
probes, steering vectors and sparse-autoencoder features; safety engineers who
deploy probe-based monitors and need to know what those monitors cannot see;
and researchers studying masked-diffusion language models
[@sahoo2024mdlm], where decodability and controllability move along a
denoising clock as well as depth.

# State of the field

Gradient-based attribution such as AtP [@kramar2024atp] approximates the
effect of patching model components. Steering libraries
([steering-vectors](https://github.com/steering-vectors/steering-vectors),
[repeng](https://github.com/vgel/repeng), and IBM's activation-steering
[@lee2024cast]) train and apply vectors. Intervention frameworks such as
pyvene [@wu2024pyvene] and nnsight [@fiottokaufman2024nnsight] provide general
access to model internals. Benchmarks such as AxBench [@wu2025axbench] compare
steering methods by their effect on generations. The Linear Accessibility
Profile [@billa2026lap] predicts where steering vectors succeed from logit-lens
readouts, and SteerCheck [@luo2026steercheck] audits attribution specificity
in steering experiments.

`steercert` is complementary to these tools. It does not train vectors or
replace general intervention frameworks. It consumes their outputs and adds
a quantity none of them report: the geometry of the write side, meaning the
effect Gramian, efficiencies relative to its leading direction, and
probe-orthogonal dark directions. These come with reporting rules and
held-out validation, built so that a direction from any of these libraries
can be audited in a few lines and the result fed back to them.

# Software design

The package has three layers. The core (`geometry`, `audit`, `controls`,
`probes`, `stats`, `two_clock`) depends only on NumPy, operates on plain
arrays, and can audit gradients produced by any framework. It never forms the
dense Gramian. All spectral quantities come from a thin singular value
decomposition of the gradient matrix, so cost scales with the number of
examples rather than with the square of the hidden width. Every reported
quantity is invariant under orthogonal changes of basis of the activation
space, and the test suite checks this.

The `hf` adapter collects gradients by inserting an additive zero tensor at
the site, $h \mapsto h + \delta$, and differentiating with respect to
$\delta$. The same hook mechanism applies finite interventions, so gradients
and edits always refer to the same tensor. This works with frozen and
half-precision weights. Sites are defined by module rather than by
hidden-state index, matching the block-output convention of steering-vectors,
repeng and lm-evaluation-harness. Step sizes default to a fraction of the
median activation norm so they are comparable across model widths, and
second derivatives come from Hessian-vector products.

Integrations import their target library lazily. They convert vectors to and
from steering-vectors and repeng, write lm-evaluation-harness `steer_path`
files so audited directions can be checked for capability cost on standard
benchmarks, and collect gradients through nnsight. With nnsight's remote
execution on the National Deep Inference Fabric, audits can run on models
too large for local hardware. The `two_clock` module provides a counting
baseline and shape-gap test for masked-diffusion models, and the exact
closed-form construction behind them.

Tests use tiny randomly initialised GPT-2, GPT-NeoX and Llama models, so
continuous integration needs no downloads. They compare gradients with finite
differences at every site type and check batch independence of per-example
gradients. They also confirm that states match Hugging Face hidden states, and
that steering-vectors' and lm-evaluation-harness's `add` hooks produce
logits identical to `steercert`'s edit.

# Research impact statement

`steercert` was extracted from a research codebase that audited toxicity
probes in the Pythia suite [@biderman2023pythia] on RealToxicityPrompts
[@gehman2020rtp]. Run through the package's public API, one stored cell of
that study reproduces to within $5 \times 10^{-4}$ for every reported
quantity. In that cell, Pythia-70M's middle layer, a probe that reads toxicity
at AUC 0.85 captures 18% of the measured steering leverage, while a direction
orthogonal to it keeps 99%. The package will accompany a forthcoming preprint
on reading and writing in autoregressive and diffusion language models. It is
designed for direct reuse through integrations with widely used steering,
evaluation and intervention libraries.

# Acknowledgements

TODO: funding, collaborators and compute providers.

# References
