"""Re-analysis of the July MD trajectories (raw backup -> work directory -> data/md/).

``analyse_run`` reads one trajectory and writes ``<work>/md/<run_id>.json`` (scalars) and
``<run_id>.npz`` (curves and series). ``collect`` turns the 42 results into the committed tables
of ``data/md/``. Settings follow ``protocol/reproduction_rule.md``; nothing here depends on
the published values.

Run with OMP_NUM_THREADS set (the worker count times the thread count should not exceed the
cores) and KMP_DUPLICATE_LIB_OK=TRUE in the analysis environment.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .. import constants as C
from ..transport import estimators, msd, thermostat

DIVERGENCE_FRAMES = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)


def trajectory_path(raw: Path, series: dict, temperature: int) -> Path:
    return Path(raw) / "data" / series["folder"] / f"{series['prefix']}_{temperature}K.traj"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kinisi_block(frames, n_li_density: float, temperature: float, start_fraction: float, seeds) -> list[dict]:
    analyzer = estimators.kinisi_analyzer(frames)
    start_fs = start_fraction * len(frames) * C.FRAME_PS * 1000.0
    rows = []
    for seed in seeds:
        mean, std = estimators.kinisi_fit(analyzer, start_fs, seed)
        sigma, sigma_std = estimators.kinisi_sigma(mean, std, n_li_density, temperature)
        rows.append({"seed": seed, "start_fraction": start_fraction, "D_cm2_s": mean, "D_std_cm2_s": std,
                     "sigma_mS_cm": sigma, "sigma_std_mS_cm": sigma_std})
    return rows, analyzer


def analyse_run(raw: Path, work: Path, series: dict, temperature: int) -> Path:
    """Analyse one run; returns the JSON path. Skips runs whose JSON exists."""
    from ase.io import read

    run = C.run_id(series["id"], temperature)
    out_json = Path(work) / "md" / f"{run}.json"
    if out_json.exists():
        return out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    path = trajectory_path(raw, series, temperature)
    frames = read(str(path), index=":")
    first = frames[0]
    symbols = first.get_chemical_symbols()
    density = estimators.li_density_per_cm3(first)
    record = {"run_id": run, "series": series["id"], "temperature_K": temperature,
              "trajectory": f"data/{series['folder']}/{path.name}", "trajectory_sha256": sha256(path),
              "n_frames": len(frames), "n_atoms": len(first), "n_Li": symbols.count("Li"),
              "volume_A3": float(first.get_volume()), "li_density_per_cm3": density}
    arrays = {}

    # thermostat statistics and block-averaged series
    series_full = thermostat.frame_series(frames)
    record["thermostat"] = thermostat.statistics(series_full)
    for key, values in thermostat.blocks(series_full).items():
        arrays[f"ts_{key}"] = values

    # MSD per species over all time origins (COM-corrected), at log-spaced lags
    positions = np.array([frame.get_positions() for frame in frames])
    lags = msd.log_lags(len(frames))
    arrays["msd_lag_ps"] = lags * C.FRAME_PS
    for species, values in msd.species_msd(positions, symbols, first.get_masses(), lags).items():
        arrays[f"msd_{species}"] = values
    arrays["divergence_frames"] = np.array([f for f in DIVERGENCE_FRAMES if f < len(frames)])
    arrays["divergence_positions"] = positions[arrays["divergence_frames"]]
    del positions

    # pymatgen (the fitted values), full run plus windows and blocks for the production runs
    structures = estimators.pymatgen_structures(frames)
    record["pymatgen"] = estimators.pymatgen_diffusion(structures, temperature)
    production = series["id"] in C.PRODUCTION_SERIES
    windows, blocks = [], []
    if production:
        frames_per_ps = int(round(1.0 / C.FRAME_PS))
        for window in C.WINDOW_PS[:-1]:
            n = window * frames_per_ps
            windows.append({"window_ps": window, **estimators.pymatgen_diffusion(structures[:n], temperature)})
        size = C.BLOCK_PS * frames_per_ps
        for index in range(len(frames) // size):
            part = estimators.pymatgen_diffusion(structures[index * size:(index + 1) * size], temperature)
            blocks.append({"block": index, "start_ps": index * C.BLOCK_PS, **part})
    del structures

    # kinisi: primary (seed 0) and scatter (seeds 1-10) at the July start fraction
    rows, analyzer = _kinisi_block(frames, density, temperature, C.KINISI_START_FRACTION, C.KINISI_SEEDS)
    record["kinisi"] = rows
    dt, li_msd, li_sd = estimators.kinisi_msd(analyzer)
    keep = msd.log_lags(len(dt) + 1) - 1
    keep = keep[keep < len(dt)]
    arrays["kinisi_dt_ps"], arrays["kinisi_msd_Li"], arrays["kinisi_msd_Li_sd"] = dt[keep], li_msd[keep], li_sd[keep]
    record["kinisi_fit_window_loglog_slope"] = msd.loglog_slope(dt, li_msd, C.KINISI_START_FRACTION * dt.max())
    if production:
        sensitivity = []
        for fraction in C.START_FRACTIONS:
            if fraction == C.KINISI_START_FRACTION:
                continue
            start_fs = fraction * len(frames) * C.FRAME_PS * 1000.0
            mean, std = estimators.kinisi_fit(analyzer, start_fs, 0)
            sensitivity.append({"start_fraction": fraction, "D_cm2_s": mean, "D_std_cm2_s": std})
        record["kinisi_start_sensitivity"] = sensitivity
        del analyzer
        frames_per_ps = int(round(1.0 / C.FRAME_PS))
        for row in windows:
            part, _ = _kinisi_block(frames[:row["window_ps"] * frames_per_ps], density, temperature,
                                    C.KINISI_START_FRACTION, (0,))
            row.update({f"kinisi_{key}": value for key, value in part[0].items() if key not in ("seed",)})
    record["windows"] = windows
    record["blocks"] = blocks
    record["seconds"] = round(time.time() - started, 1)
    np.savez_compressed(out_json.with_suffix(".npz"), **arrays)
    out_json.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return out_json


def _worker(arguments):
    raw, work, series_id, temperature = arguments
    return str(analyse_run(Path(raw), Path(work), C.SERIES_BY_ID[series_id], temperature))


def analyse_all(raw: Path, work: Path, workers: int = 4, only=None) -> list[str]:
    """Analyse every run (largest first) in ``workers`` processes."""
    from multiprocessing import get_context

    jobs = [(str(raw), str(work), series["id"], t) for _, series, t in C.runs()
            if not only or series["id"] in only]
    jobs.sort(key=lambda job: -(C.SERIES_BY_ID[job[2]]["n_atoms"] * C.SERIES_BY_ID[job[2]]["production_ps"]))
    with get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(_worker, jobs))


# --- collect: work directory -> data/md -----------------------------------------------------------
DIVERGENCE_PAIRS = (("mace_52_050", "mace_52_100"), ("mace_52_050", "mace_52_150"), ("mace_52_100", "mace_52_150"),
                    ("mattersim_52_050", "mattersim_52_100"), ("mattersim_52_050", "mattersim_52_150"),
                    ("mattersim_52_100", "mattersim_52_150"), ("mace_416_200", "clx000_416_150"))


def collect(work: Path, out: Path) -> list[Path]:
    """Write data/md/{transport,kinisi_seeds,thermostat,windows,divergence}.csv and {msd,timeseries}.csv.gz."""
    from .. import io

    work, out = Path(work) / "md", Path(out)
    records, arrays = {}, {}
    for run, _, _ in C.runs():
        records[run] = json.loads((work / f"{run}.json").read_text(encoding="utf-8"))
        arrays[run] = dict(np.load(work / f"{run}.npz"))
    transport, seeds, thermo, windows, msd_rows, series_rows = [], [], [], [], [], []
    for run, record in records.items():
        primary = next(row for row in record["kinisi"] if row["seed"] == 0)
        scatter = [row for row in record["kinisi"] if row["seed"] != 0]
        mean_d = np.mean([row["D_cm2_s"] for row in scatter])
        mean_s = np.mean([row["D_std_cm2_s"] for row in scatter])
        transport.append({
            "run_id": run, "series": record["series"], "temperature_K": record["temperature_K"],
            "n_frames": record["n_frames"], "n_atoms": record["n_atoms"], "n_Li": record["n_Li"],
            "volume_A3": record["volume_A3"], "li_density_per_cm3": record["li_density_per_cm3"],
            "D_cm2_s": record["pymatgen"]["D_cm2_s"], "sigma_mS_cm": record["pymatgen"]["sigma_mS_cm"],
            "kinisi_D_cm2_s": primary["D_cm2_s"], "kinisi_D_std_cm2_s": primary["D_std_cm2_s"],
            "kinisi_sigma_mS_cm": primary["sigma_mS_cm"], "kinisi_sigma_std_mS_cm": primary["sigma_std_mS_cm"],
            "seed_scatter_D_rel": float(np.std([row["D_cm2_s"] for row in scatter], ddof=1) / mean_d),
            "seed_scatter_D_std_rel": float(np.std([row["D_std_cm2_s"] for row in scatter], ddof=1) / mean_s),
            "msd_loglog_slope_fit_window": record["kinisi_fit_window_loglog_slope"],
            "trajectory": record["trajectory"], "trajectory_sha256": record["trajectory_sha256"]})
        for row in record["kinisi"]:
            seeds.append({"run_id": run, **{k: row[k] for k in ("seed", "D_cm2_s", "D_std_cm2_s")}})
        thermo.append({"run_id": run, **record["thermostat"]})
        for row in record["windows"]:
            windows.append({"run_id": run, "kind": "window", "value": row["window_ps"], "D_cm2_s": row["D_cm2_s"],
                            "sigma_mS_cm": row["sigma_mS_cm"], "kinisi_D_cm2_s": row["kinisi_D_cm2_s"],
                            "kinisi_D_std_cm2_s": row["kinisi_D_std_cm2_s"],
                            "kinisi_sigma_std_mS_cm": row["kinisi_sigma_std_mS_cm"]})
        for row in record["blocks"]:
            windows.append({"run_id": run, "kind": "block_start_ps", "value": row["start_ps"],
                            "D_cm2_s": row["D_cm2_s"], "sigma_mS_cm": row["sigma_mS_cm"]})
        for row in record.get("kinisi_start_sensitivity", []):
            windows.append({"run_id": run, "kind": "kinisi_start_fraction", "value": row["start_fraction"],
                            "kinisi_D_cm2_s": row["D_cm2_s"], "kinisi_D_std_cm2_s": row["D_std_cm2_s"]})
        a = arrays[run]
        for i, lag in enumerate(a["msd_lag_ps"]):
            row = {"run_id": run, "lag_ps": lag, "kinisi_msd_Li_A2": a["kinisi_msd_Li"][i],
                   "kinisi_msd_Li_sd_A2": a["kinisi_msd_Li_sd"][i]}
            row.update({f"msd_{s}_A2": a[f"msd_{s}"][i] for s in ("Li", "P", "S", "Cl") if f"msd_{s}" in a})
            msd_rows.append(row)
        for i, t in enumerate(a["ts_time_ps"]):
            series_rows.append({"run_id": run, "time_ps": t, "T_K": a["ts_T_K"][i], "E_eV_atom": a["ts_E_eV_atom"][i],
                                "P_GPa": a["ts_P_GPa"][i]})
    divergence = []
    for first, second in DIVERGENCE_PAIRS:
        for t in C.TEMPERATURES:
            a, b = arrays[C.run_id(first, t)], arrays[C.run_id(second, t)]
            for k, frame in enumerate(a["divergence_frames"]):
                if frame not in b["divergence_frames"]:
                    continue
                j = int(np.where(b["divergence_frames"] == frame)[0][0])
                rms = float(np.sqrt(np.mean(np.sum((a["divergence_positions"][k] - b["divergence_positions"][j]) ** 2,
                                                   axis=1))))
                divergence.append({"run_a": C.run_id(first, t), "run_b": C.run_id(second, t), "frame": int(frame),
                                   "time_ps": frame * C.FRAME_PS, "rms_position_difference_A": rms})
    written = [
        io.write_csv(out / "transport.csv", list(transport[0]), transport, digits=17),
        io.write_csv(out / "kinisi_seeds.csv", ("run_id", "seed", "D_cm2_s", "D_std_cm2_s"), seeds, digits=17),
        io.write_csv(out / "thermostat.csv", list(thermo[0]), thermo),
        io.write_csv(out / "windows.csv", ("run_id", "kind", "value", "D_cm2_s", "sigma_mS_cm", "kinisi_D_cm2_s",
                                           "kinisi_D_std_cm2_s", "kinisi_sigma_std_mS_cm"), windows, digits=17),
        io.write_csv(out / "divergence.csv", ("run_a", "run_b", "frame", "time_ps", "rms_position_difference_A"),
                     divergence),
        io.write_csv(out / "msd.csv.gz", ("run_id", "lag_ps", "msd_Li_A2", "msd_P_A2", "msd_S_A2", "msd_Cl_A2",
                                          "kinisi_msd_Li_A2", "kinisi_msd_Li_sd_A2"), msd_rows, digits=6),
        io.write_csv(out / "timeseries.csv.gz", ("run_id", "time_ps", "T_K", "E_eV_atom", "P_GPa"), series_rows,
                     digits=8),
    ]
    return written
