# Data

## What is in the repository

Everything the tables, figures and documents are built from is in `data/`, in plain text (CSV,
JSON, extended XYZ), gzip-compressed where large. `python -m mlipft tables` turns it into `results/`.

| Path | Rows | Content |
|---|---|---|
| `published/` | 10 files | The analysis files published in July 2026 (per-run transport values and Arrhenius fits of the 11 published series, structure summaries), byte for byte, with `SHA256SUMS` |
| `structures/` | 8 files | The 52-atom cell, its S/Cl orderings, the 416-atom supercell and the four Cl-excess cells (CIF) |
| `md/transport.csv` | 42 | Per MD run: pymatgen D and conductivity, kinisi D (seed 0) and its standard deviation, kinisi seed scatter, MSD slope in the kinisi fit window, trajectory sha256 |
| `md/kinisi_seeds.csv` | 462 | kinisi D and standard deviation for seeds 0-10 of every run |
| `md/thermostat.csv` | 42 | Mean and standard deviation of T, mean potential energy, drift, mean and standard deviation of pressure |
| `md/windows.csv` | 132 | 416-atom, 200-ps runs: fits of the first 50/100/150 ps, pymatgen D of 50-ps blocks, kinisi with other fit starts |
| `md/divergence.csv` | 210 | RMS position difference between runs that share settings (52-atom tiers; production and Cl x = 0) at 10 frames |
| `md/msd.csv.gz` | 4698 | MSD per species at about 150 log-spaced lags, and the Li MSD fitted by kinisi with its standard deviation |
| `md/timeseries.csv.gz` | 12000 | Temperature, potential energy and pressure in 0.5-ps blocks |
| `md/run_logs.csv` | 19 | The run logs written by the MD driver in July (only these survive) |
| `dft/configs.csv` | 27 | The labelled configurations: id, frame, position in each file, split of each fine-tune, subset (T22/H3/H2), DFT energy and pressure of both label sets |
| `dft/raw_inputs/labelled_{gamma,k222}.xyz` | 27 each | The July label files (Gamma point; 2x2x2 k-point mesh), byte for byte |
| `dft/training_sets/` | | The July training and validation files of both fine-tunes, byte for byte |
| `dft/qe/{gamma,k222}/*.{in,out}.gz` | 108 | Quantum ESPRESSO inputs and outputs, with absolute paths replaced |
| `dft/qe_runs.csv` | 54 | Settings, cost, energy and pressure parsed from the QE outputs |
| `dft/pseudopotentials.csv` | 4 | The GBRV pseudopotentials: names, MD5 reported by QE, sha256 |
| `models/registry.csv` | 13 | Every model file of the two fine-tunes and the foundation, with sha256 and the model it is |
| `models/log_reproduction.csv` | 5 | Errors printed in the training logs against the re-evaluation |
| `models/predictions_energy.csv` | 27 | Energy and pressure of the five models and both DFT label sets per configuration |
| `models/predictions_forces.csv.gz` | 1404 | Forces of both DFT label sets and of the five models per atom (eV/A) |
| `models/fingerprints.csv` | 9 | Stored MD forces against each candidate model (which model drove each run) |
| `models/training/*_record.jsonl` | | The MACE training records of both fine-tunes |
| `models/evaluation.json` | | Package versions of the evaluation and the settings parsed from both training logs |
| `references/literature.csv` | 10 | Experimental and simulated (MLIP-MD and AIMD) reference values with DOIs |
| `references/aimd_deng2017/` | | The AIMD reference (amendment A3): pymatgen and kinisi D per temperature, kinisi seeds, MSD curves, `SOURCE.md` with the source commit, checksums and license |
| `raw_manifest.csv` | 407 | Every file of the raw backup: path, role, size, sha256 (batch job ids masked) |
| `sanitisation.csv` | | Every committed file that copies a raw file, with both checksums and the rule applied |

Units are in the column names (`_cm2_s`, `_mS_cm`, `_meV_A`, `_kbar`, ...). Run ids are
`<series>_<T>K`; the series are listed in `src/mlipft/constants.py`. Configuration ids are
`T<source temperature>_f<frame>`.

## What is not in the repository, and why

| Item | Size | Where it is |
|---|---|---|
| 42 MD trajectories (ASE) | 3.3 GB | Raw backup (two copies); sha256 in `md/transport.csv` and `raw_manifest.csv` |
| Model weights (foundation, eight fine-tuned files) | 70 MB | Raw backup; sha256 in `models/registry.csv`. MACE-MP-0 small is downloaded by `mace_mp(model="small")` |
| GBRV pseudopotentials | 2 MB | Public archive; `generation/labels/fetch_pseudos.sh` downloads and checks them |
| AIMD reference trajectories (Deng et al. 2017) | 66 MB | Public (materialsvirtuallab/matgenb, BSD-3-Clause); `ingest/fetch_aimd.sh` downloads them at a pinned commit and checks them |
| July batch job logs and scripts | small | Raw backup; facts taken from them are in `docs/provenance.md` |

To rerun the ingest from the raw backup:

```bash
pip install -r requirements-ingest.lock.txt && pip install -e . --no-deps
export MLIPFT_RAW=<raw backup>   # layout: data/traj*/, finetune/, foundation/ (see raw_manifest.csv)
make ingest                      # DFT, models, then the 42 trajectories (about 19 h run by run, mostly kinisi)
```

The AIMD reference is re-derived with `ingest/fetch_aimd.sh <folder>` and `python -m mlipft ingest aimd --source <folder>`.

The values in `data/md/` were made on a cluster instead (amendment A2): one job per MD series with
[ingest/reanalyse_series.pbs](../ingest/reanalyse_series.pbs), which runs
[ingest/run_series.py](../ingest/run_series.py) and writes `work/md/<run>.json`; then, on the
workstation, `MLIPFT_WORK=<that work folder> python -m mlipft ingest collect` writes `data/md/`. Each
job analyses the three runs of a series in parallel on 12 cores; the runs took 1.5-73 minutes each.

The ingest writes only `data/`. On one platform it is deterministic: a second cluster run of one
series (`mace_52_050`) gave identical results, apart from the recorded wall time. Between platforms,
kinisi's Monte Carlo sampling makes its values differ slightly ([provenance.md](provenance.md)).
