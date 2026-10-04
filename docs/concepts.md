# Concepts

## Sites, functionals and gradients

An **intervention site** is a tensor in the forward pass, for example the output
of decoder block $\ell$ at the last token. A **behavioural functional**
$\phi(x)$ is a scalar computed from the model's output, for example the
log-mean-exp logit of a set of toxic tokens minus that of a reference set.

For example $i$ let $h_i$ be the activation at the site and
$g_i = \partial \phi_i / \partial h_i$ its gradient. Adding $\alpha w$ to $h_i$
changes the behaviour by

$$
\Delta\phi_i(\alpha w) = \alpha\, g_i^\top w + O(\alpha^2).
$$

`steercert.hf.collect` obtains every $g_i$ in one backward pass per batch,
because examples in a causal-LM batch do not interact.

## The effect Gramian and certificates

Over a set of examples define the mean gradient $\bar g$ and the **effect
Gramian**

$$
W = \mathbb{E}[g g^\top].
$$

For a unit direction $w$ the three first-order **certificates** are

| certificate | formula | predicts |
|---|---|---|
| signed | $\lvert \bar g^\top w \rvert$ | the mean effect of a small step |
| mean absolute | $\mathbb{E}\lvert g^\top w \rvert$ | the typical per-example effect |
| RMS | $\sqrt{w^\top W w}$ | the root-mean-square per-example effect |

The best RMS direction is the leading eigenvector of $W$, with value
$\sqrt{\lambda_{\max}}$. The best signed direction is $\bar g / \lVert\bar g\rVert$.

**Efficiency** divides a certificate by its maximum:
$\text{RMS eff}(w) = \sqrt{w^\top W w / \lambda_{\max}}$ and
$\text{signed eff}(w) = \lvert\cos(w, \bar g)\rvert$. The latter equals the
alignment $\rho$ when $w$ is a probe.

!!! warning "Always report absolute leverage"
    Efficiency is a ratio. It can rise while the leverage it is a ratio of
    collapses, as happens late in depth and late in denoising. The report
    always prints $\lVert\bar g\rVert$ and $\sqrt{\lambda_{\max}}$ next to it.

## Rank diagnostics

- **stable rank** $\operatorname{tr} W / \lambda_{\max}$: near 1, per-example
  gradients are nearly collinear.
- **participation ratio** $(\operatorname{tr} W)^2 / \lVert W\rVert_F^2$.
- **entropy effective rank**: $\exp$ of the entropy of the normalised spectrum.

All of these, and every certificate, are invariant under orthogonal changes of
basis of the residual stream (tested in `tests/test_geometry.py`).

## Dark directions

Given a probe $p$ (what a monitor reads), the **RMS dark direction** maximises
$w^\top W w$ subject to $w \perp p$. It is the strongest way to move the
behaviour that the monitor cannot see. `subspace_dark_direction` does the
same for a whole set of monitors. A high dark-direction efficiency with a low
probe efficiency is a **read/write gap** (finding `SC201`).

## Controls

A candidate should beat construction-matched controls, not just random
directions. Three families are provided:

- **isotropic**: random unit vectors. In high dimension they are nearly orthogonal
  to everything, so beating them is a weak test.
- **shuffled-label probe**: the same probe recipe fitted on permuted labels.
- **sign-randomised gradient**: the mean of per-example gradients with random
  signs. If the stable rank is near 1 this control collapses onto $\bar g$
  (median cosine near 1). It then cannot discriminate, which raises finding `SC102`.

## Step sizes

Activation norms grow with model width, so a fixed absolute step is not
comparable across models. `measure_effects(alpha_mode="residual-median")`
scales $\alpha$ by the median activation norm at the site.

## Second order

`steercert.hf.directional_derivatives` returns exact first and second
directional derivatives through Hessian-vector products. With a curvature
bound $M$, the first-order certificate dominates while
$\alpha < 2\lvert\bar g^\top w\rvert / M$ (`geometry.usable_radius`).
