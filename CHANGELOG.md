# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

First public release.

### Added

- `steercert.audit`: sensor/actuator audit with signed, mean-absolute and RMS
  certificates, write efficiencies, absolute leverage, dark directions
  (probe-orthogonal and subspace-orthogonal), control families and
  machine-readable findings `SC101`–`SC303`.
- Held-out validation: `AuditReport.with_measurement` and the certificate-to-
  measurement Spearman ranking.
- `steercert.hf`: Hugging Face adapter with exact block-output, embedding and
  final-norm sites, last-token and all-token gradients, symmetric finite
  interventions with residual-norm-comparable steps, and Hessian-vector-product
  second derivatives.
- Integrations: steering-vectors, repeng, lm-evaluation-harness steer configs,
  and experimental nnsight/NDIF collection.
- `steercert.two_clock`: counting baseline, shape gap, `Lattice` falsifier
  checks and the exact absorbing-diffusion construction.
- `steercert demo | summary | check` command-line interface for CI regression checks.
- Synthetic sites with independently set read quality and write alignment.
