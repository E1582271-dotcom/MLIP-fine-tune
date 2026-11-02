"""Command line.

Rebuild layer (needs only data/): ``mlipft tables | figures | docs | check-repo | all``;
``mlipft figures [--only fig2 aimd] [--strict] [--audit DIR]`` adds the figure QA gate and crop audit.
Ingest layer (needs the raw backup in MLIPFT_RAW and the ingest dependencies):
``mlipft ingest published --legacy <repo> | dft | models | trajectories [--workers N] | collect | manifest``;
``mlipft ingest aimd --source <folder>`` (the AIMD reference, amendment A3);
``mlipft ingest mattersim --source <json>`` (MatterSim against the labels, amendment A4).
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
    elif args.step == "aimd":
        from .ingest import aimd

        aimd.ingest(args.source)
    elif args.step == "mattersim":
        from .ingest import mattersim

        mattersim.ingest(args.source)
    return 0


def _tables(_args) -> int:
    from . import tables

    for path in tables.build():
        print("wrote", path)
    return 0


def _figures(args) -> int:
    from .figures import qa, registry

    strict = True if getattr(args, "strict", False) else None
    audit = getattr(args, "audit", None)
    if audit:
        try:
            audit = qa.outside_repository(audit)
        except ValueError as error:
            print(f"mlipft figures: {error}", file=sys.stderr)
            return 2
    try:
        written = registry.build(args.only, strict=strict, audit=audit)
    except qa.FigureQAError as error:
        print(error, file=sys.stderr)
        return 1
    for path in written:
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
        if step(args):
            return 1
    return 0


def _figure_options(parser) -> None:
    parser.add_argument("--only", nargs="*", default=None,
                        help="figure numbers or keys (fig2, ed08, forces, aimd); default: every figure, after "
                             "deleting all exports and source data")
    parser.add_argument("--strict", action="store_true",
                        help="fail on any figure QA error (local check; default: MLIPFT_QA, else warn)")
    parser.add_argument("--audit", metavar="DIR",
                        help="write zoomed 600-dpi crops, contact sheets and index.json per figure to DIR "
                             "(outside the repository)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlipft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="raw backup -> data/ (local only)")
    ingest.add_argument("step", choices=("published", "dft", "models", "trajectories", "collect", "manifest", "aimd", "mattersim"))
    ingest.add_argument("--legacy", help="working repository (for 'published')")
    ingest.add_argument("--source", help="AIMD folder ('aimd') or MatterSim output ('mattersim')")
    ingest.add_argument("--workers", type=int, default=4)
    ingest.set_defaults(func=_ingest)
    sub.add_parser("tables", help="data/ -> results/").set_defaults(func=_tables)
    figures = sub.add_parser("figures", help="draw every figure")
    _figure_options(figures)
    figures.set_defaults(func=_figures)
    sub.add_parser("docs", help="write the generated documents").set_defaults(func=_docs)
    sub.add_parser("check-repo", help="no CJK text, no private identifiers, size limits").set_defaults(func=_check)
    everything = sub.add_parser("all", help="tables, figures and docs")
    _figure_options(everything)
    everything.set_defaults(func=_all)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
