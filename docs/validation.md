# Validation

## Test suite

`pytest` runs about 90 tests. Those that need PyTorch use tiny randomly
initialised GPT-2, GPT-NeoX and Llama models with a byte-level tokenizer, so
CI downloads nothing. They check:

- **Gradients against finite differences** at the embedding, block-output and
  final-norm sites, for last-token and all-token edits, on all three architectures.
- **Batch independence**: per-example gradients from batch size 1 equal those
  from batch size 8.
- **Site mapping**: states at every `Site` equal the corresponding Hugging Face
  `output_hidden_states` tensor to 1e-10.
- **Second order**: Hessian-vector-product curvature against finite second differences.
- **Integrations**: steering-vectors' and lm-evaluation-harness's `add` hooks
  produce the same logits as steercert's edit. nnsight collection matches the
  direct Hugging Face path.
- **Geometry**: closed-form cases, optimality of dark directions, and
  invariance under orthogonal reparameterisation.
- **Reporting rules**: each finding fires, and only, under its condition.

## Reproduction of a research artifact

steercert was extracted from a research codebase that audited toxicity probes
in Pythia models. Running one stored cell (Pythia-70M, split seed 17, output of
block 2, step 3% of the median residual norm) through the package's public API
on CPU reproduces the stored GPU result:

| quantity | steercert | stored | abs. diff |
|---|---:|---:|---:|
| probe AUC (held-out) | 0.8477 | 0.8477 | 0 |
| alignment $\rho$ | 0.1064 | 0.1064 | 2e-5 |
| absolute leverage $\lVert\bar g\rVert$ | 1.3158 | 1.3158 | 1e-5 |
| stable rank | 4.3186 | 4.3191 | 5e-4 |
| probe certificate RMS efficiency | 0.1659 | 0.1659 | 1e-5 |
| step size $\alpha$ | 0.3741 | 0.3741 | 6e-8 |
| probe measured RMS efficiency | 0.1841 | 0.1840 | 9e-5 |
| dark direction measured RMS efficiency | 0.9941 | 0.9944 | 4e-4 |

The ranking Spearman (0.868 vs 0.824) depends on the random control directions
included in the ranked set, which were drawn from a different generator.

## Known limits

- The bundled functionals are next-token proxies, not generated-text measures.
- The certificate is first order. Check `usable_radius` or the measured
  curvature before trusting it at large step sizes.
- Some architectures (Llama's RMSNorm) compute internally in float32 even in
  float64 models. This bounds finite-difference agreement at about 1e-4.
