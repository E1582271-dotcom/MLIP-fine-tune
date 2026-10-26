"""Command line.

Rebuild layer (needs only data/): ``mlipft tables | figures | docs | check-repo | all``.
Ingest layer (needs the raw backup in MLIPFT_RAW and the ingest dependencies):
``mlipft ingest published --legacy <repo> | dft | models | trajectories [--workers N] | collect | manifest``.
"""
from __future__ import annotations

import argparse
import sys


def _ingest(args) -> int:
    from . import paths

    if args.step == "published":
        from .ingest import published

        published.published(args.legacy)
        published.run_logs()
    elif args.step == "dft":
        from .ingest import qe

        qe.ingest()
    elif args.step == "models":
        from .ingest import models

        models.ingest()
    elif args.step == "trajectories":
        from .ingest import trajectories

        trajectories.analyse_all(paths.raw(), paths.work(), workers=args.workers)
    elif args.step == "collect":
        from .ingest import trajectories

        trajectories.collect(paths.work(), paths.MD)
    elif args.step == "manifest":
        from .ingest import manifest

        manifest.build()
    return 0


def _tables(_args) -> int:
    from . import tables

    for path in tables.build():
        print("wrote", path)
    return 0


def _figures(args) -> int:
    from .figures import registry

    for path in registry.build(args.only):
        print("wrote", path)
    return 0


def _docs(_args) -> int:
    from . import report

    for path in report.build():
        print("wrote", path)
    return 0


def _check(_args) -> int:
    from . import hygiene

    problems = hygiene.check()
    for problem in problems:
        print(problem)
    print("repository checks:", "FAIL" if problems else "pass")
    return 1 if problems else 0


def _all(args) -> int:
    for step in (_tables, _figures, _docs):
        step(args)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlipft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="raw backup -> data/ (local only)")
    ingest.add_argument("step", choices=("published", "dft", "models", "trajectories", "collect", "manifest"))
    ingest.add_argument("--legacy", help="working repository (for 'published')")
    ingest.add_argument("--workers", type=int, default=4)
    ingest.set_defaults(func=_ingest)
    sub.add_parser("tables", help="data/ -> results/").set_defaults(func=_tables)
    figures = sub.add_parser("figures", help="draw every figure")
    figures.add_argument("--only", nargs="*", default=None)
    figures.set_defaults(func=_figures)
    sub.add_parser("docs", help="write the generated documents").set_defaults(func=_docs)
    sub.add_parser("check-repo", help="no CJK text, no private identifiers, size limits").set_defaults(func=_check)
    everything = sub.add_parser("all", help="tables, figures and docs")
    everything.add_argument("--only", nargs="*", default=None)
    everything.set_defaults(func=_all)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
