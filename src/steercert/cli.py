"""Command-line interface: ``steercert demo | summary | check``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def _pairs(values: list[str] | None) -> dict[str, float]:
    result = {}
    for item in values or []:
        name, _, threshold = item.rpartition("=")
        if not name:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        result[name] = float(threshold)
    return result


def _demo(args: argparse.Namespace) -> int:
    from .audit import audit
    from .probes import fisher_probe
    from .synthetic import decoupled_site

    site = decoupled_site(n=args.n, overlap=args.overlap, seed=args.seed)
    half = args.n // 2
    probe = fisher_probe(site.states[:half], site.labels[:half])
    report = audit(
        site.gradients[half:],
        {"probe": probe},
        probe="probe",
        states=site.states[half:],
        labels=site.labels[half:],
        site="synthetic",
        seed=args.seed,
    )
    print(report.summary())
    if args.out:
        report.to_json(args.out)
        print(f"\nwrote {args.out}")
    return 0


def _summary(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    geometry = report["geometry"]
    print(f"site={report.get('site')}  n={report['n_examples']}  width={report['width']}")
    print(f"|g_bar|={geometry['mean_gradient_norm']:.4g}  stable_rank={geometry['stable_rank']:.2f}")
    for row in report["rows"]:
        if row["role"] != "control":
            print(f"  {row['name']:<34}{row['role']:<11}RMS eff {row['rms_efficiency']:.3f}")
    for finding in report.get("findings", []):
        print(f"  [{finding['severity']}] {finding['code']}: {finding['message']}")
    return 0


def _check(args: argparse.Namespace) -> int:
    from .audit import check_report

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    failures = check_report(
        report,
        min_ranking=args.min_ranking,
        min_efficiency=_pairs(args.min_efficiency),
        max_efficiency=_pairs(args.max_efficiency),
        min_mean_gradient_norm=args.min_leverage,
        min_stable_rank=args.min_stable_rank,
        fail_on_warnings=args.fail_on_warnings,
    )
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print("OK all thresholds met")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steercert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="audit a synthetic site; no model download")
    demo.add_argument("--n", type=int, default=4000)
    demo.add_argument("--overlap", type=float, default=0.1)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out", type=Path)
    demo.set_defaults(func=_demo)

    summary = sub.add_parser("summary", help="print a saved report")
    summary.add_argument("report", type=Path)
    summary.set_defaults(func=_summary)

    check = sub.add_parser("check", help="exit 1 if a saved report fails thresholds (for CI)")
    check.add_argument("report", type=Path)
    check.add_argument("--min-ranking", type=float)
    check.add_argument("--min-efficiency", nargs="*", metavar="NAME=VALUE")
    check.add_argument("--max-efficiency", nargs="*", metavar="NAME=VALUE")
    check.add_argument("--min-leverage", type=float, help="minimum |g_bar|")
    check.add_argument("--min-stable-rank", type=float)
    check.add_argument("--fail-on-warnings", action="store_true")
    check.set_defaults(func=_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
