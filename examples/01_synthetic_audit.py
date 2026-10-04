"""Audit a synthetic site where a probe reads perfectly but barely writes.

Runs in a second with NumPy only:

    python examples/01_synthetic_audit.py
"""

import steercert
from steercert.synthetic import decoupled_site

# A site whose read direction and dominant write direction have cosine 0.1.
site = decoupled_site(n=4000, overlap=0.1, seed=0)
fit, held_out = slice(0, 2000), slice(2000, None)

# 1. Fit a probe on one split (the monitor's read direction).
probe = steercert.fisher_probe(site.states[fit], site.labels[fit])

# 2. Audit it on a disjoint split, next to a steering vector built the usual way.
report = steercert.audit(
    site.gradients[held_out],
    {
        "probe": probe,
        "difference_of_means": steercert.difference_of_means(site.states[fit], site.labels[fit]),
    },
    probe="probe",
    states=site.states[held_out],
    labels=site.labels[held_out],
    site="synthetic",
)
print(report.summary())

# The probe reads the label almost perfectly ...
assert report.row("probe").auc > 0.95
# ... yet captures a small share of the leverage, while a direction it cannot see keeps nearly all of it.
assert report.row("probe").rms_efficiency < 0.2
assert report.row("reference/dark_rms").rms_efficiency > 0.95
