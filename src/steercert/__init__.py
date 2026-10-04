"""steercert: sensor/actuator audits for probes and steering vectors.

A linear probe *reads* a behaviour from activations; a steering vector
*writes* to it. steercert measures how far apart those two roles are at an
intervention site, using per-example gradients of a behavioural functional
(the effect Gramian), and reports it under explicit reporting rules.

Quick start (no model needed)::

    import steercert
    from steercert.synthetic import decoupled_site

    site = decoupled_site(overlap=0.1)
    report = steercert.audit(site.gradients, probe=site.read_direction,
                             states=site.states, labels=site.labels)
    print(report.summary())

For Hugging Face models see :mod:`steercert.hf`; for other libraries see
:mod:`steercert.integrations`.
"""

from ._version import __version__
from .audit import AuditReport, DirectionRow, EffectMeasurement, Finding, audit, check_report
from .controls import make_controls
from .geometry import (
    EffectGeometry,
    certificate_scores,
    effect_geometry,
    rho,
    rms_dark_direction,
    signed_dark_direction,
    subspace_dark_direction,
)
from .probes import difference_of_means, direction_auc, fisher_probe, logistic_probe

__all__ = [
    "AuditReport",
    "DirectionRow",
    "EffectGeometry",
    "EffectMeasurement",
    "Finding",
    "__version__",
    "audit",
    "certificate_scores",
    "check_report",
    "difference_of_means",
    "direction_auc",
    "effect_geometry",
    "fisher_probe",
    "logistic_probe",
    "make_controls",
    "rho",
    "rms_dark_direction",
    "signed_dark_direction",
    "subspace_dark_direction",
]
