"""The transport reference of amendment A3: the Li6PS5Cl AIMD of Deng et al. (2017).

The four files are pymatgen ``DiffusionAnalyzer`` records of VASP AIMD runs (52 atoms, time step
2 fs, every 10th step stored, 100 ps per temperature), published with the notebooks of the paper in
materialsvirtuallab/matgenb under the BSD-3-Clause license. They are read as plain JSON: each
trajectory is the stored first structure plus the stored displacements. Every temperature is
analysed with the estimators and settings of the MLIP runs (``transport.estimators``,
``transport.msd``); only the time step and the stored-frame interval are those of the AIMD.

``ingest/fetch_aimd.sh <folder>`` downloads the files at the pinned commit and checks them;
``mlipft ingest aimd --source <folder>`` writes ``data/references/aimd_deng2017/``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import constants as C
from .. import io, paths
from ..transport import estimators, msd

REPOSITORY = "materialsvirtuallab/matgenb"
COMMIT = "2e77d515f23c507887856c265cdab84fa1a30082"
URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/notebooks/aimd_data/{{temperature}}.json"
SHA256 = {600: "9157f2128ee88a508ca0560989d9b0c401f612d5c59d61a277ae89ca4a5deabe",
          800: "fb31dce75edb856ae03a3fb0138c61dd7b8bb111501c2af2d661de0cf87c093a",
          1000: "bf4ea1d1104279d422fa460e5d4c50dedeb163e846ea8cbbead285f2378a0701",
          1200: "a5d283ca17110e17d1350bc1593ad43f45974ab54047959598d3964861f8ca49"}
TEMPERATURES = (600, 800, 1000, 1200)
OUT = paths.REFERENCES / "aimd_deng2017"
CITATION = ("Z. Deng, Z. Zhu, I.-H. Chu and S. P. Ong, Data-driven first-principles methods for the study and "
            "design of alkali superionic conductors, Chem. Mater. 29, 281-288 (2017), doi:10.1021/acs.chemmater.6b02648")
TRANSPORT_COLUMNS = ("temperature_K", "n_atoms", "n_Li", "a_A", "volume_A3", "li_density_per_cm3", "time_step_fs",
                     "step_skip", "n_frames", "length_ps", "D_cm2_s", "sigma_mS_cm", "kinisi_D_cm2_s",
                     "kinisi_D_std_cm2_s", "kinisi_sigma_mS_cm", "kinisi_sigma_std_mS_cm", "seed_scatter_D_rel",
                     "msd_loglog_slope_fit_window", "file", "sha256")


def load(source: Path, temperature: int) -> dict:
    """One AIMD record, checked against its sha256: lattice, symbols, unwrapped positions (frames, atoms, 3)."""
    path = Path(source) / f"{temperature}.json"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256[temperature]:
        raise ValueError(f"{path.name}: sha256 differs from the pinned file")
    record = json.loads(payload)
    if int(record["temperature"]) != temperature or record["specie"] != "Li":
        raise ValueError(f"{path.name}: unexpected record")
    sites = record["structure"]["sites"]
    lattice = np.array(record["structure"]["lattice"]["matrix"], float)
    fractional = np.array([site["abc"] for site in sites], float)
    displacements = np.array(record["displacements"], float)  # (atoms, frames, 3), Cartesian A, from frame 0
    if not np.allclose(displacements[:, 0, :], 0.0):
        raise ValueError(f"{path.name}: displacements do not start at the stored structure")
    return {"lattice": lattice, "symbols": [site["species"][0]["element"] for site in sites],
            "fractional": fractional, "displacements": displacements,
            "positions": (fractional @ lattice)[None, :, :] + displacements.transpose(1, 0, 2),
            "time_step_fs": float(record["time_step"]), "step_skip": int(record["step_skip"]),
            "sha256": SHA256[temperature], "file": path.name}


def analyse(record: dict, temperature: int) -> tuple[dict, list[dict], list[dict]]:
    """Transport row, kinisi seed rows and MSD rows of one temperature."""
    from ase import Atoms
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
    from pymatgen.core import Lattice, Structure

    lattice, symbols, positions = record["lattice"], record["symbols"], record["positions"]
    time_step, skip = record["time_step_fs"], record["step_skip"]
    frame_ps = time_step * skip / 1000.0
    n_frames = positions.shape[0]

    structure = Structure(Lattice(lattice), symbols, record["fractional"])
    pymatgen = DiffusionAnalyzer(structure, record["displacements"], "Li", temperature, time_step, skip,
                                 smoothed="max")
    frames = [Atoms(symbols, positions=frame, cell=lattice, pbc=True) for frame in positions]
    for frame in frames:
        frame.wrap()
    first = frames[0]
    density = estimators.li_density_per_cm3(first)

    analyzer = estimators.kinisi_analyzer(frames, time_step, skip)
    start_fs = C.KINISI_START_FRACTION * n_frames * frame_ps * 1000.0
    seeds = []
    for seed in C.KINISI_SEEDS:
        mean, std = estimators.kinisi_fit(analyzer, start_fs, seed)
        seeds.append({"temperature_K": temperature, "seed": seed, "D_cm2_s": mean, "D_std_cm2_s": std})
    D, D_std = seeds[0]["D_cm2_s"], seeds[0]["D_std_cm2_s"]
    sigma, sigma_std = estimators.kinisi_sigma(D, D_std, density, temperature)
    scatter = np.array([row["D_cm2_s"] for row in seeds[1:]])
    dt, li_msd, li_sd = estimators.kinisi_msd(analyzer)

    lags = msd.log_lags(n_frames)
    curves = msd.species_msd(positions, symbols, first.get_masses(), lags)
    keep = msd.log_lags(len(dt) + 1) - 1
    keep = keep[keep < len(dt)]
    msd_rows = []
    for i, lag in enumerate(lags * frame_ps):
        row = {"temperature_K": temperature, "lag_ps": lag}
        row.update({f"msd_{s}_A2": curves[s][i] for s in ("Li", "P", "S", "Cl")})
        if i < len(keep):
            row.update({"kinisi_msd_Li_A2": li_msd[keep[i]], "kinisi_msd_Li_sd_A2": li_sd[keep[i]]})
        msd_rows.append(row)

    row = {"temperature_K": temperature, "n_atoms": len(symbols), "n_Li": symbols.count("Li"),
           "a_A": float(np.linalg.norm(lattice[0])), "volume_A3": float(abs(np.linalg.det(lattice))),
           "li_density_per_cm3": density, "time_step_fs": time_step, "step_skip": skip, "n_frames": n_frames,
           "length_ps": (n_frames - 1) * frame_ps, "D_cm2_s": float(pymatgen.diffusivity),
           "sigma_mS_cm": float(pymatgen.conductivity), "kinisi_D_cm2_s": D, "kinisi_D_std_cm2_s": D_std,
           "kinisi_sigma_mS_cm": sigma, "kinisi_sigma_std_mS_cm": sigma_std,
           "seed_scatter_D_rel": float(scatter.std(ddof=1) / scatter.mean()),
           "msd_loglog_slope_fit_window": msd.loglog_slope(dt, li_msd, C.KINISI_START_FRACTION * dt.max()),
           "file": record["file"], "sha256": record["sha256"]}
    return row, seeds, msd_rows


def ingest(source: Path, out: Path | None = None) -> list[Path]:
    out = Path(out or OUT)
    transport, seeds, msd_rows = [], [], []
    for temperature in TEMPERATURES:
        row, seed_rows, curve = analyse(load(source, temperature), temperature)
        transport.append(row)
        seeds.extend(seed_rows)
        msd_rows.extend(curve)
        print(f"AIMD {temperature} K: kinisi D {row['kinisi_D_cm2_s']:.3e} cm2/s", flush=True)
    written = [
        io.write_csv(out / "transport.csv", TRANSPORT_COLUMNS, transport, digits=17),
        io.write_csv(out / "kinisi_seeds.csv", ("temperature_K", "seed", "D_cm2_s", "D_std_cm2_s"), seeds, digits=17),
        io.write_csv(out / "msd.csv.gz", ("temperature_K", "lag_ps", "msd_Li_A2", "msd_P_A2", "msd_S_A2", "msd_Cl_A2",
                                          "kinisi_msd_Li_A2", "kinisi_msd_Li_sd_A2"), msd_rows, digits=6),
    ]
    files = "\n".join(f"| `{t}.json` | `{SHA256[t]}` |" for t in TEMPERATURES)
    source_md = f"""# AIMD reference: Li6PS5Cl, Deng et al. (2017)

The transport reference of [amendment A3](../../../protocol/amendments.md). Derived here from the AIMD
records published with the notebooks of

> {CITATION}.

## Source files

Repository `{REPOSITORY}`, commit `{COMMIT}`, folder `notebooks/aimd_data/`. They are downloaded and
checked by [ingest/fetch_aimd.sh](../../../ingest/fetch_aimd.sh) and are not stored here.

| File | sha256 |
|---|---|
{files}

What the files record: VASP AIMD of Li24P4S20Cl4, the ordered S/Cl arrangement of this repository's
cell (Cl on 4a, S on 4c), 52 atoms, a = 10.10 A, time step 2 fs, every 10th step stored, 100 ps at
600, 800, 1000 and 1200 K. The exchange-correlation functional and the thermostat are not recorded in
the files.

## Files here

- `transport.csv`: per temperature, pymatgen D (`smoothed="max"`) and its conductivity, kinisi D
  (seed 0, fit from 30 % of the run) with its standard deviation, the scatter over kinisi seeds 1-10,
  and the log-log slope of the Li MSD in the fit window.
- `kinisi_seeds.csv`: kinisi D for seeds 0-10.
- `msd.csv.gz`: MSD per species (all time origins, centre of mass removed) and the kinisi Li MSD.

Made by `mlipft ingest aimd` ([src/mlipft/ingest/aimd.py](../../../src/mlipft/ingest/aimd.py)).

## License of the source data

The source files are part of matgenb, distributed under the BSD 3-Clause License, Copyright (c) 2017,
Materials Virtual Lab. The values here are derived from them.
"""
    path = out / "SOURCE.md"
    path.write_text(source_md, encoding="utf-8")
    written.append(path)
    return written
