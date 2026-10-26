"""``data/raw_manifest.csv``: every file of the raw backup with its size, sha256 and role, and
``data/sanitisation.csv``: every committed file that is a sanitised copy of a raw file.

The raw backup holds the July files copied from the cluster (checked against sha256 sums computed
there) plus the foundation model from the local MACE cache. Nothing in the manifest is a path
outside the backup.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import re
from pathlib import Path

from .. import io, paths

ROLES = (("data/traj", "MD trajectory"), ("data/md_runs", "MD run log"), ("data/", "structure or July record"),
         ("finetune/qe_work", "QE input or output"), ("finetune/data", "DFT labels or snapshots"),
         ("finetune/models", "MACE model"), ("finetune/checkpoints", "MACE checkpoint"),
         ("finetune/logs", "MACE training log"), ("finetune/results", "MACE training record"),
         ("finetune/pseudo", "pseudopotential"), ("finetune/", "July labelling or fine-tuning script or log"),
         ("foundation/", "foundation model (MACE-MP-0 small)"), ("", "July script"))
JOB_OUTPUT = re.compile(r"\.o\d+$")  # batch job output files are named after the job id, which is masked


def role(path: str) -> str:
    return next(name for prefix, name in ROLES if path.startswith(prefix))


def build(raw: Path | None = None) -> list[Path]:
    raw = Path(raw or paths.raw())
    rows = []
    for path in sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        rel = path.relative_to(raw).as_posix()
        kind = "July batch job output" if JOB_OUTPUT.search(rel) else role(rel)
        rows.append({"path": JOB_OUTPUT.sub(".o<job>", rel), "role": kind, "bytes": path.stat().st_size,
                     "sha256": io.sha256_file(path)})
    written = [io.write_csv(paths.RAW_MANIFEST, ("path", "role", "bytes", "sha256"), rows)]
    by_sha = {row["sha256"]: row["path"] for row in rows}
    ledger = []
    for target in sorted((paths.DFT / "qe").rglob("*.gz")):
        clean = gzip.decompress(target.read_bytes())
        source = _source_of(target, raw)
        ledger.append({"file": paths.relpath(target), "source": source, "source_sha256": io.sha256_file(raw / source),
                       "sha256_uncompressed": hashlib.sha256(clean).hexdigest(),
                       "rule": "absolute paths replaced by ./pseudo/<file> or ./<basename>; gzip without time stamp"})
    for target in sorted(paths.DFT.rglob("*.xyz")) + sorted((paths.MODELS / "training").glob("*.jsonl")) + \
            sorted(paths.PUBLISHED.glob("*.json")):
        digest = io.sha256_file(target)
        ledger.append({"file": paths.relpath(target), "source": by_sha.get(digest, "working repository (published)"),
                       "source_sha256": digest, "sha256_uncompressed": digest, "rule": "byte-for-byte copy"})
    written.append(io.write_csv(paths.SANITISATION, ("file", "source", "source_sha256", "sha256_uncompressed", "rule"),
                                ledger))
    return written


def _source_of(target: Path, raw: Path) -> str:
    """The raw QE file a sanitised copy came from (via data/dft/configs.csv shard numbers)."""
    label_set = target.parent.name
    config, suffix = target.name.split(".")[0], target.name.split(".")[1]
    with open(paths.DFT / "configs.csv", newline="", encoding="utf-8") as handle:
        shard = next(int(row["snapshot_index"]) for row in csv.DictReader(handle) if row["config_id"] == config)
    folder = f"qe_work_s{shard}" if label_set == "gamma" else f"qe_work_27k_s{shard}"
    return f"finetune/{folder}/cfg{shard:04d}.{suffix}"
