"""Repository layout, and the locations of the raw backup and the work directory.

The raw backup (July trajectories, QE runs, model files) and the work directory are outside the
repository; their locations come from the environment variables MLIPFT_RAW and MLIPFT_WORK and
are never written into any file.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
DOCS = REPO / "docs"
PROTOCOL = REPO / "protocol"

PUBLISHED = DATA / "published"
STRUCTURES = DATA / "structures"
MD = DATA / "md"
DFT = DATA / "dft"
MODELS = DATA / "models"
REFERENCES = DATA / "references"
ENVIRONMENT = DATA / "environment"
RAW_MANIFEST = DATA / "raw_manifest.csv"
SANITISATION = DATA / "sanitisation.csv"


def raw() -> Path:
    """Root of the raw backup (same layout as the July working folder on the cluster)."""
    value = os.environ.get("MLIPFT_RAW")
    if not value:
        raise SystemExit("set MLIPFT_RAW to the raw backup directory (see docs/data.md)")
    return Path(value).expanduser()


def work() -> Path:
    """Work directory for intermediate ingest results (default: <raw>/../work)."""
    value = os.environ.get("MLIPFT_WORK")
    return Path(value).expanduser() if value else raw().parent / "work"


def relpath(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name
