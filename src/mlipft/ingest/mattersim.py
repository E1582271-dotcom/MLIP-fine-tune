"""MatterSim against the DFT labels (amendment A4): the output of ``ingest/mattersim_predict.py`` -> data/models/.

``ingest/mattersim_predict.py`` runs where MatterSim is installed and evaluates the model the July
MatterSim runs loaded on the 27 labelled configurations and on four frames of each 416-atom MatterSim
trajectory. ``mlipft ingest mattersim --source <json>`` checks that output and writes

* ``predictions_mattersim_energy.csv`` and ``predictions_mattersim_forces.csv.gz``, with the columns of
  the MACE prediction files (``mattersim_energy_eV``, ``mattersim_pressure_kbar``, ``mattersim_f{x,y,z}``);
* ``mattersim_identity.csv``: stored against recomputed forces of the MD frames; the model is the one
  that drove the MatterSim runs only if every RMS difference is below 1 meV/A;
* ``mattersim_evaluation.json``: package versions, checkpoint and its sha256, device and precision.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .. import io, paths

MATCH_meV_A = 1.0


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ingest(source: Path, out: Path | None = None) -> list[Path]:
    out = Path(out or paths.MODELS)
    record = json.loads(Path(source).read_text(encoding="utf-8"))
    configs = _rows(paths.DFT / "configs.csv")
    order = [row["config_id"] for row in sorted(configs, key=lambda row: int(row["snapshot_index"]))]
    predicted = record["configurations"]
    if [row["config_id"] for row in predicted] != order:
        raise ValueError("MatterSim predictions are not in the order of data/dft/configs.csv")
    identity = [{**row, "match": row["rms_meV_A"] < MATCH_meV_A} for row in record["fingerprints"]]
    if not identity or not all(row["match"] for row in identity):
        raise ValueError("the evaluated MatterSim model does not reproduce the forces stored in the MatterSim runs")
    energy = [{"config_id": row["config_id"], "n_atoms": row["n_atoms"], "mattersim_energy_eV": row["energy_eV"],
               "mattersim_pressure_kbar": row["pressure_kbar"]} for row in predicted]
    forces = [{"config_id": row["config_id"], "atom": index, "species": species,
               **{f"mattersim_f{axis}": value for axis, value in zip("xyz", force)}}
              for row in predicted for index, (species, force) in enumerate(zip(row["species"], row["forces_eV_A"]))]
    written = [
        io.write_csv(out / "predictions_mattersim_energy.csv",
                     ("config_id", "n_atoms", "mattersim_energy_eV", "mattersim_pressure_kbar"), energy, digits=17),
        io.write_csv(out / "predictions_mattersim_forces.csv.gz",
                     ("config_id", "atom", "species", "mattersim_fx", "mattersim_fy", "mattersim_fz"), forces,
                     digits=10),
        io.write_csv(out / "mattersim_identity.csv", ("temperature_K", "frames", "n_frames", "rms_meV_A", "match"),
                     identity),
    ]
    meta = {key: record[key] for key in ("packages", "checkpoint", "checkpoint_sha256", "device")}
    meta["precision"] = "float32"
    path = out / "mattersim_evaluation.json"
    path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return [*written, path]
