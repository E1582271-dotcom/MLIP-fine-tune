"""The July 2026 record: published analysis files and MD run logs.

* ``published(legacy)`` copies the analysis files published by the working repository at commit
  ``LEGACY_COMMIT`` (read with ``git show``, never from its working tree) into ``data/published/``
  byte for byte, with ``SHA256SUMS``. These are the values the re-analysis is compared with.
* ``run_logs(raw)`` parses the run logs written by the MD driver (``data/md_runs*.json`` in the raw
  backup) into ``data/md/run_logs.csv``; trajectory paths are reduced to ``<folder>/<file>``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .. import io, paths

LEGACY_COMMIT = "e2b7416c4af887fb9101aa444836164f689999a9"
PUBLISHED_FILES = ("data/metrics.json", "data/metrics_long.json", "data/metrics_prod.json", "data/metrics_ft.json",
                   "data/metrics_dope_cl100.json", "data/metrics_dope_cl125.json", "data/metrics_dope_cl150.json",
                   "data/metrics_dope_cl175.json", "data/structures.json", "data/doped/doped.json")
RUN_LOG_FILES = ("md_runs.json", "md_runs_100ps.json", "md_runs_long.json", "md_runs_prod.json", "md_runs_prod_ms.json",
                 "md_runs_ft27k.json", "md_runs_dope_cl100.json", "md_runs_dope_cl125.json", "md_runs_dope_cl150.json",
                 "md_runs_dope_cl175.json", "md_runs_mace_baseline.json")
RUN_LOG_COLUMNS = ("log_file", "trajectory", "mlip", "temperature_K", "n_steps", "timestep_fs", "production_ps",
                   "log_every", "n_frames", "mean_T", "std_T", "mean_E_per_atom", "drift_meV_atom_ps", "wall_seconds")


def published(legacy: Path, out: Path | None = None) -> list[Path]:
    out = Path(out or paths.PUBLISHED)
    out.mkdir(parents=True, exist_ok=True)
    written, sums = [], []
    for name in PUBLISHED_FILES:
        payload = subprocess.run(["git", "-C", str(legacy), "show", f"{LEGACY_COMMIT}:{name}"], check=True,
                                 capture_output=True).stdout
        target = out / Path(name).name
        target.write_bytes(payload)
        sums.append(f"{hashlib.sha256(payload).hexdigest()}  {target.name}")
        written.append(target)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    (out / "SOURCE.md").write_text(
        "# Source\n\nThe analysis files published in July 2026 by the working repository of this study, copied byte "
        f"for byte from its commit `{LEGACY_COMMIT}` by `mlipft ingest published`. They are never edited; the "
        "re-analysis is compared with them in `results/reproduction_*.csv`.\n", encoding="utf-8")
    return written


def run_logs(raw: Path | None = None, out: Path | None = None) -> Path:
    raw = Path(raw or paths.raw())
    rows = []
    for name in RUN_LOG_FILES:
        record = json.loads((raw / "data" / name).read_text(encoding="utf-8"))
        for run in record["runs"]:
            path = run.get("traj_path", "")
            row = {key: run.get(key) for key in RUN_LOG_COLUMNS}
            row["log_file"] = name
            row["trajectory"] = "/".join(Path(path).parts[-2:]) if path else ""
            row["mlip"] = run.get("mlip")
            rows.append(row)
    return io.write_csv(Path(out or paths.MD / "run_logs.csv"), RUN_LOG_COLUMNS, rows)
