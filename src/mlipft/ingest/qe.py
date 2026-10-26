"""DFT labels: configuration ids, the two label sets, the training splits and the QE runs.

Reads the raw backup (``finetune/data`` and ``finetune/qe_work*_s<i>``) and writes ``data/dft/``:

* ``raw_inputs/labelled_{gamma,k222}.xyz`` and ``training_sets/{ftG,ftK}_{train,valid}.xyz``: the
  July files, byte for byte;
* ``configs.csv``: one row per configuration with its id, source frame, position in each file,
  subset (T22, H3, H2) and DFT energy and pressure of both label sets;
* ``qe/{gamma,k222}/<config>.{in,out}.gz``: QE input and output with absolute paths replaced;
* ``qe_runs.csv`` and ``pseudopotentials.csv``: settings, cost and checksums parsed from the outputs.

Shard i of a labelling campaign is snapshot i of ``snapshots.xyz``; this is checked atom by atom.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from .. import io, paths

# Ry -> eV as used when the labels were written (ASE units based on CODATA 2006); the label energies
# equal the QE totals times this factor to 1e-6 eV.
RY_EV = 13.605691932782346
LABEL_SETS = {"gamma": {"work": "qe_work_s{i}", "labelled": "labelled.xyz", "model": "ftG",
                        "train": "train.xyz", "valid": "valid.xyz"},
              "k222": {"work": "qe_work_27k_s{i}", "labelled": "labelled_27k.xyz", "model": "ftK",
                       "train": "train_27k.xyz", "valid": "valid_27k.xyz"}}
N_CONFIGS = 27
POSITION_TOLERANCE = 1e-6
_PATH = re.compile(r"(['\"]?)(/[^\s'\"]+)")


def config_id(frame: dict) -> str:
    temperature = int(frame["info"]["source"].split("_")[1].rstrip("K"))
    return f"T{temperature}_f{int(frame['info']['frame_in_traj']):03d}"


def sanitise_text(text: str) -> str:
    """Replace every absolute path by ./pseudo/<name> (pseudopotentials) or ./<basename>."""
    def replace(match: re.Match) -> str:
        quote, path = match.groups()
        name = path.rstrip("/").split("/")[-1]
        if path.rstrip("/").endswith("pseudo"):
            return f"{quote}./pseudo"
        if name.upper().endswith(".UPF"):
            return f"{quote}./pseudo/{name}"
        return f"{quote}./{name}"
    lines = []
    for line in text.splitlines(keepends=True):
        if "/" in line and re.search(r"(^|[\s'\"=])/(nfs|home|scratch|Users|app|tmp|var)", line):
            line = _PATH.sub(replace, line)
        lines.append(line)
    return "".join(lines)


def _wall_seconds(text: str) -> float:
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)([hms])", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def parse_output(text: str) -> dict:
    """Settings, convergence, cost, energy and pressure of one pw.x SCF output."""
    def find(pattern, cast=str, group=1):
        match = re.search(pattern, text, re.M)
        return cast(match.group(group)) if match else None
    wall = re.search(r"PWSCF\s*:\s*(.+?)\s+CPU\s+(.+?)\s+WALL", text)
    pseudos = re.findall(r"PseudoPot\. #\s*\d+ for (\w+)\s+read from file:\s*\n\s*(\S+).*?MD5 check sum:\s*(\w+)",
                         text, re.S)
    return {
        "qe_version": find(r"Program PWSCF (v\.[\d.]+)"),
        "started": find(r"starts on\s+(\w+)\s+at", str),
        "mpi_processes": find(r"Number of MPI processes:\s+(\d+)", int),
        "k_points": find(r"number of k points=\s+(\d+)", int),
        "ecutwfc_Ry": find(r"kinetic-energy cutoff\s+=\s+([\d.]+)", float),
        "ecutrho_Ry": find(r"charge density cutoff\s+=\s+([\d.]+)", float),
        "conv_thr_Ry": find(r"convergence threshold\s+=\s+([\d.Ee+-]+)", float),
        "smearing_Ry": find(r"width \(Ry\)=\s+([\d.]+)", float),
        "scf_iterations": find(r"convergence has been achieved in\s+(\d+) iterations", int),
        "energy_eV": find(r"^!\s+total energy\s+=\s+([-\d.]+) Ry", float) * RY_EV,
        "pressure_kbar": find(r"P=\s*([-\d.]+)", float),
        "total_force_Ry_bohr": find(r"Total force =\s+([\d.]+)", float),
        "cpu_s": _wall_seconds(wall.group(1)) if wall else None,
        "wall_s": _wall_seconds(wall.group(2)) if wall else None,
        "pseudopotentials": [(element, Path(path).name, md5) for element, path, md5 in pseudos],
    }


def _input_positions(text: str) -> np.ndarray:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ATOMIC_POSITIONS"))
    rows = []
    for line in lines[start + 1:]:
        parts = line.split()
        if len(parts) < 4:
            break
        rows.append([float(v) for v in parts[1:4]])
    return np.array(rows)


def ingest(raw: Path | None = None, out: Path | None = None) -> dict:
    raw = Path(raw or paths.raw()) / "finetune"
    out = Path(out or paths.DFT)
    snapshots = io.read_extxyz(raw / "data" / "snapshots.xyz")
    assert len(snapshots) == N_CONFIGS
    snapshot_ids = [config_id(frame) for frame in snapshots]

    labelled, rows = {}, {cid: {"config_id": cid, "snapshot_index": i} for i, cid in enumerate(snapshot_ids)}
    for name, spec in LABEL_SETS.items():
        frames = io.read_extxyz(raw / "data" / spec["labelled"])
        assert sorted(config_id(f) for f in frames) == sorted(snapshot_ids), name
        labelled[name] = {config_id(f): f for f in frames}
        for index, frame in enumerate(frames):
            cid = config_id(frame)
            snapshot = snapshots[snapshot_ids.index(cid)]
            assert np.abs(frame["pos"] - snapshot["pos"]).max() < POSITION_TOLERANCE, (name, cid)
            assert frame["species"] == snapshot["species"]
            rows[cid][f"index_{name}"] = index
            rows[cid][f"dft_energy_{name}_eV"] = float(frame["info"]["dft_energy"])
        target = out / "raw_inputs" / f"labelled_{name}.xyz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw / "data" / spec["labelled"], target)
        for part in ("train", "valid"):
            target = out / "training_sets" / f"{spec['model']}_{part}.xyz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw / "data" / spec[part], target)
            for frame in io.read_extxyz(raw / "data" / spec[part]):
                rows[config_id(frame)][f"{spec['model']}_split"] = part

    # subsets
    for row in rows.values():
        g, k = row["ftG_split"], row["ftK_split"]
        row["subset"] = {("train", "train"): "T22", ("valid", "valid"): "H3", ("valid", "train"): "H2"}[(g, k)]
    counts = {name: sum(1 for r in rows.values() if r["subset"] == name) for name in ("T22", "H3", "H2")}
    assert counts == {"T22": 22, "H3": 3, "H2": 2}, counts

    # QE runs
    qe_rows, pseudo_rows, sanitised = [], {}, []
    for name, spec in LABEL_SETS.items():
        for i, cid in enumerate(snapshot_ids):
            folder = raw / spec["work"].format(i=i)
            stem = f"cfg{i:04d}"
            text_in = (folder / f"{stem}.in").read_text(encoding="utf-8")
            text_out = (folder / f"{stem}.out").read_text(encoding="utf-8")
            assert np.abs(_input_positions(text_in) - snapshots[i]["pos"]).max() < POSITION_TOLERANCE, (name, cid)
            parsed = parse_output(text_out)
            assert abs(parsed["energy_eV"] - rows[cid][f"dft_energy_{name}_eV"]) < 1e-6, (name, cid)
            for suffix, text in (("in", text_in), ("out", text_out)):
                clean = sanitise_text(text)
                target = io.write_gzip(out / "qe" / name / f"{cid}.{suffix}.gz", clean.encode("utf-8"))
                sanitised.append({"file": paths.relpath(target),
                                  "source": f"finetune/{spec['work'].format(i=i)}/{stem}.{suffix}",
                                  "source_sha256": io.sha256_file(folder / f"{stem}.{suffix}"),
                                  "sha256": io.sha256_file(target),
                                  "rule": "absolute paths -> ./pseudo/<file> or ./<basename>; gzip without time stamp"})
            for element, filename, md5 in parsed.pop("pseudopotentials"):
                pseudo_rows[(element, filename, md5)] = True
            rows[cid][f"pressure_{name}_kbar"] = parsed["pressure_kbar"]
            qe_rows.append({"label_set": name, "config_id": cid, "shard": i, **parsed})

    pseudo_table = []
    for element, filename, md5 in sorted(pseudo_rows):
        upf = raw / "pseudo" / filename
        pseudo_table.append({"element": element, "file": filename, "md5_reported_by_qe": md5,
                             "sha256": io.sha256_file(upf),
                             "library": "GBRV ultrasoft PBE; identical to the file in all_pbe_UPF_v1.5.tar.gz"})
    assert len(pseudo_table) == 4, pseudo_table

    order = sorted(rows.values(), key=lambda r: r["snapshot_index"])
    io.write_csv(out / "configs.csv", ("config_id", "snapshot_index", "index_gamma", "index_k222", "ftG_split",
                                       "ftK_split", "subset", "dft_energy_gamma_eV", "dft_energy_k222_eV",
                                       "pressure_gamma_kbar", "pressure_k222_kbar"), order, digits=17)
    io.write_csv(out / "qe_runs.csv", ("label_set", "config_id", "shard", "qe_version", "started", "mpi_processes",
                                       "k_points", "ecutwfc_Ry", "ecutrho_Ry", "conv_thr_Ry", "smearing_Ry",
                                       "scf_iterations", "energy_eV", "pressure_kbar", "total_force_Ry_bohr",
                                       "cpu_s", "wall_s"), qe_rows, digits=17)
    io.write_csv(out / "pseudopotentials.csv", ("element", "file", "md5_reported_by_qe", "sha256", "library"),
                 pseudo_table)
    return {"configs": len(order), "qe_runs": len(qe_rows), "subsets": counts, "sanitised": sanitised}
