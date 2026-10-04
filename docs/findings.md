# Findings and reporting rules

Every report carries a list of findings. `info` findings state a substantive
result. `warning` findings mark a condition that limits what the audit can
support. Use `steercert check --fail-on-warnings` to block on them in CI.

| Code | Severity | Raised when | What to do |
|---|---|---|---|
| `SC101` | warning | fewer than 32 examples | collect more certificate examples |
| `SC102` | warning | stable rank < 2, or sign-randomised controls align with $\bar g$ | report the stable rank; rely on shuffled-label controls |
| `SC103` | warning | controls built without states/labels (isotropic only) | pass `states` and `labels` to build shuffled-label probes |
| `SC201` | info | probe RMS efficiency < 0.3 while the dark direction keeps > 0.9 | a monitor on this probe misses most ways the behaviour can move |
| `SC202` | info | probe RMS efficiency > 0.7 | the probe is also an effective actuator at this site |
| `SC203` | warning | a candidate does not beat the best shuffled-label control | treat the candidate as no better than chance construction |
| `SC204` | info | no leverage exists outside the read direction(s) | every effect lies inside what the monitor reads |
| `SC301` | warning | certificate ranks measured effects at Spearman < 0.7 | do not use the certificate to select directions at this site |
| `SC302` | warning | measurements used absolute step sizes | use `alpha_mode="residual-median"` before comparing models |
| `SC303` | info | `reference/gramian_top` was not measured | include it so measured efficiencies are defined |

## Where the rules come from

These rules came from errors found while auditing real experiments:

- Reporting relative efficiency alone hid a 16-fold fall in absolute leverage
  between the embedding and final layer of Pythia-70M.
- Sign-randomised controls reached a median cosine of 0.99 to the mean
  gradient in a model whose effect Gramian had stable rank 1.1–1.3. Steering
  audits have observed the symptom (arXiv:2608.24335); stable rank is the cause.
- Pooling absolute step sizes across model widths can manufacture apparent
  scaling trends (arXiv:2608.08159).
