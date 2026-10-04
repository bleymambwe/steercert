"""The sensor/actuator audit: one call that turns gradients into a report.

:func:`audit` answers, for every candidate direction at one intervention site:

* how much of the available write leverage it captures (certificate efficiency),
* how that compares with the best direction, a probe-orthogonal "dark"
  direction, and construction-matched controls, and
* optionally, how well the certificate predicts held-out finite interventions.

The result is an :class:`AuditReport` that serialises to JSON and can be
checked against thresholds in CI with :func:`check_report` or the
``steercert check`` command.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .controls import ControlSet, make_controls, sign_control_is_degenerate
from .geometry import (
    EffectGeometry,
    certificate_scores,
    cosine_abs,
    effect_geometry,
    mean_gradient,
    signed_dark_direction,
    subspace_dark_direction,
    unit,
)
from .probes import direction_auc
from .stats import spearman

__all__ = ["AuditReport", "DirectionRow", "EffectMeasurement", "Finding", "audit", "check_report"]

RANKING_GATE = 0.7
MIN_EXAMPLES = 32


@dataclass(frozen=True)
class Finding:
    """A reporting-rule result attached to an audit.

    ``severity`` is ``"info"`` for a substantive result worth stating and
    ``"warning"`` for a condition that limits what the audit can support.
    """

    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class EffectMeasurement:
    """Measured finite-intervention effects for named directions.

    ``slopes[name]`` holds one symmetric finite-difference slope
    ``(phi(+alpha) - phi(-alpha)) / (2 alpha)`` per held-out example.
    """

    slopes: dict[str, np.ndarray]
    alpha: float
    alpha_mode: str
    residual_norm_median: float | None = None
    curvature: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "alpha_mode": self.alpha_mode,
            "residual_norm_median": self.residual_norm_median,
            "n_examples": int(next(iter(self.slopes.values())).shape[0]) if self.slopes else 0,
        }


@dataclass(frozen=True)
class DirectionRow:
    """Certificate and (optional) measured statistics for one direction."""

    name: str
    role: str
    family: str | None
    signed: float
    mean_absolute: float
    rms: float
    rms_efficiency: float
    signed_efficiency: float
    cosine_to_mean_gradient: float
    auc: float | None = None
    measured_signed: float | None = None
    measured_rms: float | None = None
    measured_efficiency: float | None = None


@dataclass
class AuditReport:
    """Result of :func:`audit`. Serialise with :meth:`to_dict` or :meth:`to_json`."""

    geometry: EffectGeometry
    rows: list[DirectionRow]
    findings: list[Finding]
    n_examples: int
    width: int
    probe: str | None = None
    site: str | None = None
    ranking_spearman: float | None = None
    measurement: dict | None = None
    metadata: dict = field(default_factory=dict)
    directions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def row(self, name: str) -> DirectionRow:
        """Return the row for direction ``name``."""
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(name)

    @property
    def candidates(self) -> list[DirectionRow]:
        return [row for row in self.rows if row.role in ("candidate", "probe")]

    def direction_matrix(self, include_controls: bool = True) -> tuple[list[str], np.ndarray]:
        """Names and stacked unit directions, ready for finite-intervention measurement."""
        names = [row.name for row in self.rows if include_controls or row.role != "control"]
        return names, np.stack([self.directions[name] for name in names])

    def to_dict(self, include_directions: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "steercert_version": __version__,
            "site": self.site,
            "n_examples": self.n_examples,
            "width": self.width,
            "probe": self.probe,
            "geometry": self.geometry.to_dict(),
            "rows": [asdict(row) for row in self.rows],
            "ranking_spearman": self.ranking_spearman,
            "measurement": self.measurement,
            "findings": [asdict(finding) for finding in self.findings],
            "metadata": self.metadata,
        }
        if include_directions:
            payload["directions"] = {k: v.tolist() for k, v in self.directions.items()}
        return _jsonable(payload)

    def to_json(self, path: str | Path | None = None, include_directions: bool = False) -> str:
        text = json.dumps(self.to_dict(include_directions), indent=2)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def check(self, **thresholds: Any) -> list[str]:
        """Return threshold failures; see :func:`check_report` for arguments."""
        return check_report(self.to_dict(), **thresholds)

    def summary(self, include_controls: bool = False) -> str:
        """Plain-text table for terminals and logs."""
        g = self.geometry
        lines = [
            f"steercert audit  site={self.site or '-'}  n={self.n_examples}  width={self.width}",
            f"  absolute leverage |g_bar| = {g.mean_gradient_norm:.4g}   "
            f"best RMS effect sqrt(lambda_max) = {math.sqrt(g.leading_eigenvalue):.4g}",
            f"  stable rank = {g.stable_rank:.2f}   entropy rank = {g.entropy_effective_rank:.2f}",
        ]
        if self.ranking_spearman is not None:
            lines.append(f"  certificate -> measured ranking (Spearman) = {self.ranking_spearman:.3f}")
        header = f"  {'direction':<38}{'role':<11}{'RMS eff':>9}{'signed eff':>11}{'cos(g)':>8}{'AUC':>7}{'meas eff':>10}"
        lines += ["", header, "  " + "-" * (len(header) - 2)]
        for row in self.rows:
            if row.role == "control" and not include_controls:
                continue
            lines.append(
                f"  {row.name[:37]:<38}{row.role:<11}{row.rms_efficiency:>9.3f}"
                f"{row.signed_efficiency:>11.3f}{row.cosine_to_mean_gradient:>8.3f}"
                f"{_fmt(row.auc):>7}{_fmt(row.measured_efficiency):>10}"
            )
        if not include_controls:
            families: dict[str, list[float]] = {}
            for row in self.rows:
                if row.role == "control":
                    families.setdefault(row.family or "control", []).append(row.rms_efficiency)
            for family, values in families.items():
                lines.append(
                    f"  {'controls: ' + family:<38}{'control':<11}"
                    f"{float(np.median(values)):>9.3f}  (median of {len(values)})"
                )
        if self.findings:
            lines.append("")
            for finding in self.findings:
                lines.append(f"  [{finding.severity}] {finding.code}: {finding.message}")
        return "\n".join(lines)

    def with_measurement(self, measured: EffectMeasurement | Mapping[str, np.ndarray]) -> AuditReport:
        """Return a copy with held-out measured effects attached and ranking computed."""
        return _attach_measurement(self, measured)


def _fmt(value: float | None) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, np.integer):
        return int(value)
    return value


def audit(
    gradients: np.ndarray,
    directions: Mapping[str, np.ndarray] | None = None,
    *,
    probe: str | np.ndarray | None = None,
    states: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    read_basis: np.ndarray | None = None,
    controls: bool | ControlSet = True,
    measured: EffectMeasurement | Mapping[str, np.ndarray] | None = None,
    site: str | None = None,
    seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> AuditReport:
    """Audit read and write directions at one intervention site.

    Parameters
    ----------
    gradients:
        ``(n_examples, width)`` per-example gradients of the behavioural
        functional at the site. Use a split disjoint from the one used to fit
        probes and from the one used for ``measured``.
    directions:
        Named candidate directions, for example steering vectors to compare.
    probe:
        The read direction a monitor uses: either the name of an entry in
        ``directions`` or a vector (stored as ``"probe"``). Enables the
        probe-orthogonal dark directions and read/write-gap findings.
    states, labels:
        Optional activations and binary labels from a held-out split. Adds an
        AUC column and enables shuffled-label probe controls.
    read_basis:
        Optional ``(k, width)`` rows spanning everything a monitor reads. Adds
        ``reference/dark_subspace``, the best direction invisible to all of them.
    controls:
        ``True`` builds the default families, ``False`` disables them, or pass a
        :class:`~steercert.controls.ControlSet`.
    measured:
        Held-out finite-intervention slopes (see :func:`steercert.hf.measure_effects`).
    site:
        Free-text label of the intervention site, stored in the report.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    geometry = effect_geometry(gradients)
    g_bar = mean_gradient(gradients)

    named: dict[str, np.ndarray] = {}
    role: dict[str, str] = {}
    family: dict[str, str | None] = {}

    def add(name: str, vector: np.ndarray, kind: str, fam: str | None = None) -> None:
        if name in named:
            raise ValueError(f"duplicate direction name {name!r}")
        named[name] = unit(vector)
        role[name] = kind
        family[name] = fam

    for name, vector in (directions or {}).items():
        add(name, vector, "candidate")

    probe_name: str | None = None
    if isinstance(probe, str):
        if probe not in named:
            raise KeyError(f"probe {probe!r} is not among the supplied directions")
        probe_name = probe
        role[probe] = "probe"
    elif probe is not None:
        probe_name = "probe"
        add(probe_name, probe, "probe")

    add("reference/mean_gradient", g_bar, "reference")
    add("reference/gramian_top", geometry.top_direction, "reference")
    unavailable: list[str] = []
    if probe_name is not None:
        for label, build in (
            ("reference/dark_signed", signed_dark_direction),
            ("reference/dark_rms", lambda g, p: subspace_dark_direction(g, np.atleast_2d(p))),
        ):
            try:
                add(label, build(gradients, named[probe_name]), "reference")
            except ValueError:
                unavailable.append(label)
    if read_basis is not None:
        try:
            add("reference/dark_subspace", subspace_dark_direction(gradients, read_basis), "reference")
        except ValueError:
            unavailable.append("reference/dark_subspace")

    control_set: ControlSet | None
    if controls is True:
        control_set = make_controls(gradients, states, labels, seed=seed)
    elif controls is False:
        control_set = None
    else:
        control_set = controls
    if control_set is not None:
        for name, vector in control_set.directions.items():
            add(name, vector, "control", control_set.family_of[name])

    names = list(named)
    signed, mean_abs, rms = certificate_scores(gradients, np.stack([named[n] for n in names]))
    best_rms = math.sqrt(max(geometry.leading_eigenvalue, 0.0))
    norm = geometry.mean_gradient_norm
    rows = []
    for index, name in enumerate(names):
        auc_value = None
        if states is not None and labels is not None:
            auc_value = direction_auc(states, labels, named[name])
        rows.append(
            DirectionRow(
                name=name,
                role=role[name],
                family=family[name],
                signed=float(signed[index]),
                mean_absolute=float(mean_abs[index]),
                rms=float(rms[index]),
                rms_efficiency=float(rms[index] / best_rms) if best_rms > 0 else float("nan"),
                signed_efficiency=float(signed[index] / norm) if norm > 0 else float("nan"),
                cosine_to_mean_gradient=cosine_abs(named[name], g_bar),
                auc=auc_value,
            )
        )

    report = AuditReport(
        geometry=geometry,
        rows=rows,
        findings=[],
        n_examples=int(gradients.shape[0]),
        width=int(gradients.shape[1]),
        probe=probe_name,
        site=site,
        metadata=dict(metadata or {}),
        directions=named,
    )
    report.findings = _rules(report, gradients, control_set, states is not None and labels is not None)
    if unavailable:
        report.findings.append(
            Finding(
                "SC204",
                "info",
                f"no write leverage exists outside the read direction(s) for {', '.join(unavailable)}: "
                "every effect lies inside what the monitor reads",
            )
        )
    if measured is not None:
        report = _attach_measurement(report, measured)
    return report


def _attach_measurement(
    report: AuditReport, measured: EffectMeasurement | Mapping[str, np.ndarray]
) -> AuditReport:
    if isinstance(measured, EffectMeasurement):
        slopes = measured.slopes
        meta = measured.to_dict()
    else:
        slopes = {k: np.asarray(v, dtype=np.float64) for k, v in measured.items()}
        meta = {"alpha_mode": "unspecified"}
    unknown = set(slopes) - set(report.directions)
    if unknown:
        raise KeyError(f"measurements for unknown directions: {sorted(unknown)}")
    rms_measured = {name: float(np.sqrt(np.mean(np.square(v)))) for name, v in slopes.items()}
    top = rms_measured.get("reference/gramian_top")
    rows = []
    for row in report.rows:
        if row.name in slopes:
            rows.append(
                replace(
                    row,
                    measured_signed=float(abs(np.mean(slopes[row.name]))),
                    measured_rms=rms_measured[row.name],
                    measured_efficiency=(rms_measured[row.name] / top) if top else None,
                )
            )
        else:
            rows.append(row)
    measured_rows = [row for row in rows if row.measured_rms is not None]
    ranking = None
    if len(measured_rows) >= 3:
        ranking = spearman(
            np.array([r.rms for r in measured_rows]),
            np.array([r.measured_rms for r in measured_rows], dtype=np.float64),
        )
    findings = [f for f in report.findings if not f.code.startswith("SC3")]
    if ranking is not None and not math.isnan(ranking) and ranking < RANKING_GATE:
        findings.append(
            Finding(
                "SC301",
                "warning",
                f"certificate ranks measured effects at Spearman {ranking:.2f} < {RANKING_GATE}; "
                "do not use it to select directions at this site without further validation",
            )
        )
    if meta.get("alpha_mode") == "absolute":
        findings.append(
            Finding(
                "SC302",
                "warning",
                "absolute step sizes are not comparable across models of different width; "
                "use alpha_mode='residual-median' before comparing sites or models (arXiv:2608.08159)",
            )
        )
    if top is None:
        findings.append(
            Finding(
                "SC303",
                "info",
                "reference/gramian_top was not measured, so measured efficiencies are unavailable",
            )
        )
    return replace(report, rows=rows, ranking_spearman=ranking, measurement=meta, findings=findings)


def _rules(
    report: AuditReport, gradients: np.ndarray, control_set: ControlSet | None, has_labels: bool
) -> list[Finding]:
    findings: list[Finding] = []
    if report.n_examples < MIN_EXAMPLES:
        findings.append(
            Finding(
                "SC101",
                "warning",
                f"only {report.n_examples} examples; certificate estimates are unstable below {MIN_EXAMPLES}",
            )
        )
    if control_set is not None:
        families = set(control_set.family_of.values())
        if "sign_randomized_gradient" in families and sign_control_is_degenerate(
            gradients, report.geometry.stable_rank
        ):
            findings.append(
                Finding(
                    "SC102",
                    "warning",
                    f"effect Gramian stable rank is {report.geometry.stable_rank:.2f}; per-example gradients "
                    "are nearly collinear, so sign-randomized controls collapse onto the mean gradient and "
                    "cannot discriminate. Report the stable rank and rely on shuffled-label controls",
                )
            )
        if not has_labels:
            findings.append(
                Finding(
                    "SC103",
                    "warning",
                    "no states/labels supplied, so no construction-matched probe controls were built; "
                    "isotropic random directions alone are a weak comparator (arXiv:2608.24335)",
                )
            )
        matched = [
            row.rms_efficiency
            for row in report.rows
            if row.role == "control" and row.family == "shuffled_label_probe"
        ]
        if matched:
            ceiling = float(np.max(matched))
            for row in report.rows:
                if row.role == "candidate" and row.rms_efficiency <= ceiling:
                    findings.append(
                        Finding(
                            "SC203",
                            "warning",
                            f"{row.name!r} has RMS efficiency {row.rms_efficiency:.3f}, not above the best "
                            f"shuffled-label control ({ceiling:.3f})",
                        )
                    )
    if report.probe is not None:
        probe_row = report.row(report.probe)
        try:
            dark_row = report.row("reference/dark_rms")
        except KeyError:
            dark_row = None
        if dark_row is not None and probe_row.rms_efficiency < 0.3 and dark_row.rms_efficiency > 0.9:
            findings.append(
                Finding(
                    "SC201",
                    "info",
                    f"read/write gap: the probe captures {probe_row.rms_efficiency:.0%} of the best RMS "
                    f"leverage while a probe-orthogonal direction keeps {dark_row.rms_efficiency:.0%}. "
                    "A monitor reading only this probe cannot see most of the ways this behaviour can be moved",
                )
            )
        elif probe_row.rms_efficiency > 0.7:
            findings.append(
                Finding(
                    "SC202",
                    "info",
                    f"the probe is also an effective actuator here (RMS efficiency {probe_row.rms_efficiency:.0%})",
                )
            )
    return findings


def check_report(
    report: Mapping[str, Any],
    *,
    min_ranking: float | None = None,
    min_efficiency: Mapping[str, float] | None = None,
    max_efficiency: Mapping[str, float] | None = None,
    min_mean_gradient_norm: float | None = None,
    min_stable_rank: float | None = None,
    fail_on_warnings: bool = False,
) -> list[str]:
    """Compare a serialised report with thresholds and list every failure.

    Intended for regression tests: for example, require a deployed steering
    vector to keep at least half of the available leverage after a fine-tune.
    ``min_efficiency`` and ``max_efficiency`` map direction names to bounds on
    certificate RMS efficiency.
    """
    failures: list[str] = []
    rows = {row["name"]: row for row in report["rows"]}
    ranking = report.get("ranking_spearman")
    if min_ranking is not None and (ranking is None or ranking < min_ranking):
        failures.append(f"ranking Spearman {ranking} < {min_ranking}")
    for bound, comparison, label in (
        (min_efficiency, lambda v, t: v >= t, ">="),
        (max_efficiency, lambda v, t: v <= t, "<="),
    ):
        for name, threshold in (bound or {}).items():
            if name not in rows:
                failures.append(f"direction {name!r} missing from report")
                continue
            value = rows[name]["rms_efficiency"]
            if value is None or not comparison(value, threshold):
                failures.append(f"{name}: RMS efficiency {value} not {label} {threshold}")
    geometry = report["geometry"]
    if min_mean_gradient_norm is not None and geometry["mean_gradient_norm"] < min_mean_gradient_norm:
        failures.append(f"absolute leverage {geometry['mean_gradient_norm']:.4g} < {min_mean_gradient_norm}")
    if min_stable_rank is not None and geometry["stable_rank"] < min_stable_rank:
        failures.append(f"stable rank {geometry['stable_rank']:.3g} < {min_stable_rank}")
    if fail_on_warnings:
        failures += [
            f"{f['code']}: {f['message']}" for f in report.get("findings", []) if f["severity"] == "warning"
        ]
    return failures
