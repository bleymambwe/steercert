# Diffusion language models

In an absorbing-state (masked) diffusion language model every site has two time
coordinates: depth and denoising step. Committed tokens are frozen. A probe at
step $t$ therefore reads evidence that is already committed, while a write at
step $t$ can only act through positions that are still masked. Decodability
tends to rise along the denoising clock while write leverage falls.
Autoregressive generation is the special case that unmasks left to right.

`steercert.two_clock` provides:

## The counting baseline and the shape gap

Part of the fall in leverage is simple counting: fewer positions remain
writable. `counting_baseline(writable_counts)` is that content-free curve.
`shape_gap(curve, baseline)` is the largest difference between the measured
and baseline curves after peak normalisation. A small gap means the measured
decay is explained by counting alone. Schedule-aware steering methods should
show a gap before claiming a mechanism.

```python
from steercert.two_clock import counting_baseline, shape_gap

gap = shape_gap(measured_write_norms, counting_baseline(masked_payload_counts))
```

## The lattice

```python
from steercert.two_clock import Lattice

lattice = Lattice(read=auc_matrix, write=gbar_norm_matrix, steps=progress, depths=(1, 2, 3, 4))
lattice.write_decreasing()     # F1: leverage never rises along the clock
lattice.clocks_dissociated()   # F2: read and write peak at different steps
```

Fill each cell by calling `steercert.audit` on gradients collected at that
depth and denoising step.

## The exact construction

`make_schedule`, `analytic_read_auc`, `analytic_write_norm` and
`remask_recoverable` implement a closed-form model in which the population
probe and the mean gradient are orthogonal at every step. Remasking committed
write positions restores leverage without changing decodability. That one
operation underlies remask-based defences and remask-based attacks.
