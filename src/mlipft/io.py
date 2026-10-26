"""File helpers with no heavy dependencies: extended-XYZ and CIF readers, deterministic writers."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

_INFO = re.compile(r'(\w+)=("[^"]*"|\S+)')
DIGITS = 10  # significant figures written for derived floats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_extxyz(path: Path) -> list[dict]:
    """Extended-XYZ frames: ``info`` (strings), ``species``, ``lattice`` and one array per property."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    frames, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        info = {key: value.strip('"') for key, value in _INFO.findall(lines[i + 1])}
        fields = info["Properties"].split(":")
        columns, start = {}, 0
        for name, _, width in zip(fields[::3], fields[1::3], fields[2::3]):
            columns[name] = (start, start + int(width))
            start += int(width)
        body = [line.split() for line in lines[i + 2:i + 2 + n]]
        numbers = np.array([[float(v) for v in row[1:]] for row in body])
        frame = {"info": info, "species": [row[0] for row in body],
                 "lattice": np.array([float(v) for v in info["Lattice"].split()]).reshape(3, 3)}
        for name, (lo, hi) in columns.items():
            if name != "species":
                frame[name] = numbers[:, lo - 1:hi - 1]
        frames.append(frame)
        i += n + 2
    return frames


def read_cif(path: Path) -> dict:
    """Cell length (cubic P1 cells written by pymatgen) and fractional sites of a structure file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    a = float(next(line.split()[1] for line in lines if line.startswith("_cell_length_a")))
    start = max(i for i, line in enumerate(lines) if line.strip() == "loop_")
    header = [line.strip() for line in lines[start + 1:] if line.strip().startswith("_atom_site")]
    tokens = shlex.split(" ".join(lines[start + 1 + len(header):]))
    sites = []
    for i in range(0, len(tokens), len(header)):
        values = dict(zip(header, tokens[i:i + len(header)]))
        sites.append((values["_atom_site_type_symbol"],
                      *(float(values[f"_atom_site_fract_{axis}"]) for axis in "xyz")))
    return {"a": a, "species": [site[0] for site in sites], "fractional": np.array([site[1:] for site in sites])}


def cell(value, digits: int = DIGITS) -> str:
    """CSV text of one value: '' for None/NaN, true/false, floats to ``digits`` significant figures."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, float | np.floating):
        return f"{float(value):.{digits}g}"
    return str(value)


def csv_text(columns: Sequence[str], rows: Iterable[Mapping], digits: int = DIGITS) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([cell(row.get(column), digits) for column in columns])
    return buffer.getvalue()


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping], digits: int = DIGITS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = csv_text(columns, rows, digits)
    if path.suffix == ".gz":
        write_gzip(path, text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_gzip(path: Path, payload: bytes) -> Path:
    """gzip without file name or time stamp, so equal payloads give equal files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(payload)
    return path
