"""MatterSim on the 27 labelled configurations and on four frames of each MatterSim MD run (amendment A4).

    python3 ingest/mattersim_predict.py <repository> <raw MD folder with mattersim_{600,800,1000}K.traj> <out.json>

Runs where MatterSim is installed (the July runs used mattersim with the checkpoint
mattersim-v1.0.0-1M); needs only numpy, ase and mattersim. The configurations are read from the
repository's label file in the order of data/dft/configs.csv; ``mlipft ingest mattersim`` turns the
output into data/models/. Predictions are float32 on the CPU, the precision of the MatterSim MD runs.
"""
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

TEMPERATURES = (600, 800, 1000)
EV_A3_TO_KBAR = 1602.1766208


def _configurations(repository: Path):
    from ase.io import read

    with open(repository / "data" / "dft" / "configs.csv", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["snapshot_index"]))
    frames = read(str(repository / "data" / "dft" / "raw_inputs" / "labelled_gamma.xyz"), index=":")
    return [(row["config_id"], frames[int(row["index_gamma"])]) for row in rows]


def main() -> None:
    from ase.io.trajectory import Trajectory
    from mattersim.forcefield import MatterSimCalculator

    repository, md_folder, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    calc = MatterSimCalculator(device="cpu")
    checkpoints = sorted(Path.home().glob(".local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth"))
    record = {
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ("mattersim", "torch", "ase", "numpy")},
        "checkpoint": checkpoints[0].name if checkpoints else "",
        "checkpoint_sha256": hashlib.sha256(checkpoints[0].read_bytes()).hexdigest() if checkpoints else "",
        "device": "cpu", "configurations": [], "fingerprints": [],
    }
    for config_id, atoms in _configurations(repository):
        atoms = atoms.copy()
        atoms.calc = calc
        stress = atoms.get_stress(voigt=True)
        record["configurations"].append({
            "config_id": config_id, "n_atoms": len(atoms), "species": atoms.get_chemical_symbols(),
            "energy_eV": float(atoms.get_potential_energy()),
            "pressure_kbar": float(-stress[:3].mean() * EV_A3_TO_KBAR),
            "forces_eV_A": atoms.get_forces().tolist()})
        print(config_id, record["configurations"][-1]["energy_eV"], flush=True)
    for temperature in TEMPERATURES:
        trajectory = Trajectory(str(md_folder / f"mattersim_{temperature}K.traj"))
        n = len(trajectory)
        squared = []
        for index in (0, n // 3, 2 * n // 3, n - 1):
            stored = trajectory[index].get_forces()
            atoms = trajectory[index].copy()
            atoms.calc = calc
            squared.append(float(np.mean((atoms.get_forces() - stored) ** 2)))
        rms = float(np.sqrt(np.mean(squared)) * 1000.0)
        record["fingerprints"].append({"temperature_K": temperature, "frames": "0;n/3;2n/3;n-1",
                                       "n_frames": n, "rms_meV_A": rms})
        print("fingerprint", temperature, rms, flush=True)
    out.write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
