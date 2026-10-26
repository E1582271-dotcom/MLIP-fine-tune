# Generation: how the July 2026 data were made

This folder holds the code that produced the simulation data of this study between 30 June and
8 July 2026: the Li6PS5Cl cells, the 42 MD trajectories (14 series, each at 600, 800 and
1000 K), two sets of DFT labels for the same 27 configurations (Gamma point and 2x2x2 k-point
mesh) and the two MACE models fine-tuned on them. None of it is run to rebuild the tables or
figures: trajectories and model weights are not part of the repository, and the analysis starts
from the July trajectories (step 8).

The library code is in `src/mlipft/simulation/`: `md.py` (the MD code that ran; the bytes after
its module docstring are those of the file that ran) and `structure.py` (structure building).
The scripts here are thin entry points around it, ported from the working repository of the
study; each docstring lists its changes, and the section "Differences from the July scripts"
below sums them up. `tests/test_generation.py` checks this folder (section "Tests").

Conventions:

- Commands are run from the repository root, with the package importable (`pip install -e .` or
  `export PYTHONPATH=src`).
- Every script writes below `generation/output/` (ignored by git): `structures/`,
  `data/traj<tag>/`, `labels/`, `pseudo/` and `finetune/`. `generation/output/data/` has the
  layout of the July raw backup (`data/traj<tag>/<potential>_<T>K.traj`).
- Cluster settings (container image, Python package trees, modules) come from
  `generation/hpc/site.env`: copy `generation/hpc/site.env.example` and edit it. The PBS job
  scripts carry the resource requests of the July runs; add site-specific `qsub` options (project,
  queue) yourself. MD and fine-tuning ran on an NVIDIA A40 GPU, the DFT single points on CPU nodes.

## 1. Structures

July produced the eight cells in `data/structures/` (pymatgen; CPU, seconds):

| File | Role | Atoms | Ewald energy of the 52-atom ordering (eV) |
|---|---|---|---|
| `Li6PS5Cl_mp-985592.cif` | Materials Project mp-985592, conventional cell (cached input) | 52 | |
| `config0.cif` | S/Cl ordering, Ewald rank 0 (52-atom MD) | 52 | -1010.681 |
| `config1.cif` | S/Cl ordering, Ewald rank 1 (no run used it) | 52 | -1000.657 |
| `config0_sc222.cif` | 2x2x2 supercell of config0 (416-atom MD) | 416 | -1010.681 |
| `doped_cl100_sc222.cif` | Cl-excess series, x = 0: Li6PS5Cl | 416 | -1010.681 |
| `doped_cl125_sc222.cif` | x = 0.25: Li5.75PS4.75Cl1.25 | 408 | -986.640 |
| `doped_cl150_sc222.cif` | x = 0.50: Li5.5PS4.5Cl1.5 | 400 | -963.141 |
| `doped_cl175_sc222.cif` | x = 0.75: Li5.25PS4.25Cl1.75 | 392 | -939.374 |

```bash
python generation/structures/build_structure.py                     # config0.cif, config1.cif
python generation/structures/build_structure.py --supercell 2,2,2   # config0_sc222.cif (and the unused config1_sc222.cif)
python generation/structures/build_doped.py                         # doped_cl{100,125,150,175}_sc222.cif
```

The cells are written to `generation/output/structures/`, so the committed files are never
overwritten; with pymatgen 2026.5 the three commands rebuild all seven built CIFs byte for byte
(`cmp generation/output/structures/config0.cif data/structures/config0.cif`). The methods (free
anions found by their distance from P, 50/50 S/Cl disorder ordered by Ewald energy, Cl excess
compensated by Li vacancies) are described in the docstring of `src/mlipft/simulation/structure.py`.
The base cell is read from the cached `Li6PS5Cl_mp-985592.cif`; it is downloaded from the
Materials Project (`$MP_API_KEY`) only if that file is missing.

## 2. Molecular dynamics

Every run is `md.run_nvt`: Langevin NVT at the fixed input cell (the PBE volume of mp-985592,
no NPT), 1 fs time step, friction 0.01 1/fs, Maxwell-Boltzmann velocities with seed 0 and zero
net momentum and rotation, an equilibration and then a production run with one frame every
50 steps. The potentials are MACE-MP-0 small (float32) and the default MatterSim model
(v1.0.0-1M). Each run writes `<potential>_<T>K.traj` (1000, 2000, 3000 or 4000 frames for 50,
100, 150 or 200 ps) and `<potential>_<T>K.json` (settings, thermostat statistics and the
temperature/energy series, which has one more sample, at the start of production).

| Series (`mlipft.constants.SERIES`) | Potential | Cell | Equilibration + production | Directory (`TRAJ_TAG`) | Jobs |
|---|---|---|---|---|---|
| `mace_52_050`, `mattersim_52_050` | MACE-MP-0, MatterSim | `config0.cif` | 5 + 50 ps | `traj/` (none) | one per potential |
| `mace_52_100`, `mattersim_52_100` | MACE-MP-0, MatterSim | `config0.cif` | 5 + 100 ps | `traj_100ps/` | one per potential |
| `mace_52_150`, `mattersim_52_150` | MACE-MP-0, MatterSim | `config0.cif` | 5 + 150 ps | `traj_long/` | MACE one, MatterSim one per T |
| `mace_416_200` | MACE-MP-0 | `config0_sc222.cif` | 10 + 200 ps | `traj_prod/` | one per T |
| `mattersim_416_200` | MatterSim | `config0_sc222.cif` | 10 + 200 ps | `traj_prod_ms/` | one per T |
| `ftG_416_200` | MACE fine-tuned on Gamma labels (step 7) | `config0_sc222.cif` | 10 + 200 ps | `traj_ft/` | one per T |
| `ftK_416_200` | MACE fine-tuned on 2x2x2 labels (step 7) | `config0_sc222.cif` | 10 + 200 ps | `traj_ft27k/` | one per T |
| `clx000_416_150` ... `clx075_392_150` | MACE-MP-0 | `doped_cl{100,...,175}_sc222.cif` | 10 + 150 ps | `traj_dope_cl100/` ... `traj_dope_cl175/` | one per composition and T |

The equilibration lengths are not in the July run records; they are the ones printed in the
July job log of every run. The commands, through `generation/hpc/run_md.pbs` (which runs
`generation/hpc/run_in_container.sh`, which runs `generation/md/run_md.py --device cuda`; the
docstring of `run_md.py` gives the equivalent direct calls):

```bash
# 52 atoms, 50 ps (MLIP=mattersim likewise)
qsub -v MLIP=mace,TEMPS=600_800_1000,STEPS=50000,EQUILIB=5000 generation/hpc/run_md.pbs
# 52 atoms, 100 ps (MLIP=mattersim likewise)
qsub -v MLIP=mace,TEMPS=600_800_1000,STEPS=100000,EQUILIB=5000,TRAJ_TAG=_100ps generation/hpc/run_md.pbs
# 52 atoms, 150 ps: MACE-MP-0 in one job, MatterSim in one job per temperature
qsub -l walltime=24:00:00 -v MLIP=mace,TEMPS=600_800_1000,STEPS=150000,EQUILIB=5000,TRAJ_TAG=_long generation/hpc/run_md.pbs
for T in 600 800 1000; do
  qsub -l walltime=24:00:00 -v MLIP=mattersim,TEMPS=$T,STEPS=150000,EQUILIB=5000,TRAJ_TAG=_long generation/hpc/run_md.pbs
done
# 416 atoms, 200 ps
for T in 600 800 1000; do
  qsub -l walltime=48:00:00 -v MLIP=mace,TEMPS=$T,STEPS=200000,EQUILIB=10000,SUPERCELL_TAG=_sc222,TRAJ_TAG=_prod generation/hpc/run_md.pbs
  qsub -l walltime=48:00:00 -v MLIP=mattersim,TEMPS=$T,STEPS=200000,EQUILIB=10000,SUPERCELL_TAG=_sc222,TRAJ_TAG=_prod_ms generation/hpc/run_md.pbs
done
# Cl-excess series, MACE-MP-0 small, 150 ps
for D in cl100 cl125 cl150 cl175; do for T in 600 800 1000; do
  qsub -l walltime=24:00:00 -v MLIP=mace,TEMPS=$T,STEPS=150000,EQUILIB=10000,CIF=data/structures/doped_${D}_sc222.cif,TRAJ_TAG=_dope_$D generation/hpc/run_md.pbs
done; done
```

The walltimes of 48 h (416 atoms, 200 ps) and 24 h (Cl-excess) are the recorded ones; 24 h for
the 150-ps runs is only a sufficient value (about 1.5 h per temperature on an A40). The 416-atom,
200-ps runs took 2.2 to 2.8 h per temperature (MD wall time in the job logs).

`md.py` is unchanged from the file that ran (sha256 `a773248a...` for the whole file; the test
checks the bytes after the docstring). The runs of 30 June to 1 July used the same code without
the Lennard-Jones test branch, which was added on 2 July and does not touch the MACE or
MatterSim paths; the test rebuilds that earlier version and checks its hash too.

## 3. Snapshots for the DFT labels

```bash
python generation/labels/sample_snapshots.py      # -> generation/output/labels/snapshots.xyz
```

The 27 configurations of both label sets come from the three 50-ps, 52-atom MACE-MP-0 runs
(`traj/mace_<T>K.traj`). The script drops the first 10 % of each 1000-frame trajectory and keeps
every 100th frame after that (frames 100, 200, ..., 900), 9 per temperature. The trajectories are
read in file-name order (`mace_1000K`, `mace_600K`, `mace_800K`), which fixes the snapshot index
0-26 used by the shards of step 4.

## 4. DFT labels

Each label is one Quantum ESPRESSO pw.x SCF (QE 7.1, Intel MPI) per snapshot, run by
`generation/labels/label_qe.py` inside `generation/labels/label_qe.pbs`, one job per snapshot
("shard"). The pseudopotentials go to `generation/output/pseudo/` first (downloaded from the
public GBRV archive and checked against the July checksums):

```bash
bash generation/labels/fetch_pseudos.sh
```

**Gamma-point labels** (`data/dft/raw_inputs/labelled_gamma.xyz`; July name `labelled.xyz`), 1 July:

```bash
bash generation/labels/submit_shards.sh             # -> generation/output/labels/labelled_s<i>.xyz
for s in 0 $(seq 10 19) 1 $(seq 20 26) $(seq 2 9); do cat generation/output/labels/labelled_s$s.xyz; done > generation/output/labels/labelled.xyz
```

Each job used 16 MPI ranks with npool 1 (a single k-point cannot be split into pools); each SCF
took 24-26 min, the 27 jobs ran concurrently and finished in about 1.8 h without a failure.

**2x2x2 k-point labels** (`data/dft/raw_inputs/labelled_k222.xyz`; July name `labelled_27k.xyz`),
2 to 5 July. The same snapshots and settings except the k-point mesh (`--kpts 2x2x2`, the form
that survives `qsub -v`), with 32 MPI ranks in 4 pools per job:

```bash
for s in $(seq 0 26); do
  qsub -l walltime=08:00:00 -l select=1:ncpus=32:mpiprocs=32:mem=48gb -N li6_k222_s$s \
       -v "NCPUS=32,NPOOL=4,PWARGS=--start $s --limit 1 --kpts 2x2x2 --conv-thr 1e-7 --workdir generation/output/labels/qe_work_k222_s$s --out generation/output/labels/labelled_k222_s$s.xyz" \
       generation/labels/label_qe.pbs
done
for s in 0 $(seq 10 19) 1 $(seq 20 26) $(seq 2 9); do cat generation/output/labels/labelled_k222_s$s.xyz; done > generation/output/labels/labelled_k222.xyz
```

Each job took 4.7-5.7 h; the queue accepted at most 7 of these jobs at a time, and all 27 SCF
runs converged.

**Merge order.** Both July files hold the shards in the order in which the shell glob
`labelled_s*.xyz` (and `labelled_27k_s*.xyz`) returned them on the cluster: shard 0, 10-19, 1,
20-26, 2-9. The merge commands above give that order explicitly; applied to the July shard files
they reproduce both July files byte for byte. The order matters because the split of step 5 is
a seeded shuffle of positions in the file.

**SCF settings** (the defaults of `label_qe.py`): PBE, ultrasoft pseudopotentials, ecutwfc 60 Ry
and ecutrho 480 Ry (fixed values; the 135^3 FFT grid recorded for these runs corresponds to
480 Ry), Gaussian smearing 0.01 Ry, conv_thr 1e-7 Ry, mixing_beta 0.4, forces and stress on,
Gamma point only with real wavefunctions (`--kpts gamma`) or the 2x2x2 Monkhorst-Pack mesh
without offset. `qe_scf_template.in` is a reference copy of the input the driver writes.

**Pseudopotentials.** The job log of every SCF of both label sets names the four files:
`li_pbe_v1.4.uspp.F.UPF`, `p_pbe_v1.5.uspp.F.UPF`, `s_pbe_v1.4.uspp.F.UPF` and
`cl_pbe_v1.4.uspp.F.UPF`, the GBRV ultrasoft PBE potentials (version 1.5 for P, 1.4 for the
others), as the working repository's report states ("GBRV all-PBE ultrasoft (Li/P/S/Cl)");
`data/dft/pseudopotentials.csv` gives their checksums. These four files are byte for byte the
ones in GBRV's public archive `all_pbe_UPF_v1.5.tar.gz`
(https://www.physics.rutgers.edu/gbrv/), which `fetch_pseudos.sh` downloads and checks. The
labels carry the fixed string `labelled_by=QE-pw.x-PBE-SSSP`, written by the July driver
(whose fetch script targeted the SSSP set); it does not describe the files used.

## 5. Training and validation split

```bash
# ftG_train.xyz (22) and ftG_valid.xyz (5), from the Gamma-point labels (the default input)
python generation/labels/split_train_valid.py --seed 0 --frac-valid 0.2
# ftK_train.xyz (24) and ftK_valid.xyz (3), from the 2x2x2 labels
python generation/labels/split_train_valid.py --labelled data/dft/raw_inputs/labelled_k222.xyz \
    --seed 0 --frac-valid 0.1 --out-dir generation/output/labels/k222
```

Both reproduce the July files (`data/dft/training_sets/`; July names `train.xyz`, `valid.xyz`,
`train_27k.xyz`, `valid_27k.xyz`) byte for byte. The script shuffles the indices with
`random.Random(seed)` and takes the first `round(frac_valid * 27)` as the validation set, so the
fraction sets only the number of validation configurations:

- 2x2x2 split: made in July by the script with its default of the time, `--frac-valid 0.1`,
  which gives 3 of 27.
- Gamma split: has 5 of 27. The value passed in July was not recorded; any value between 0.167
  and 0.203 gives these 5, and 0.2 is the default of the ported script.

Both label files have the same order and both splits use seed 0, so the 3 validation
configurations of the 2x2x2 split are the first 3 of the 5 of the Gamma split; no 1000 K
configuration is held out by either.

## 6. Fine-tuning

```bash
qsub generation/finetune/finetune_mace.pbs        # Gamma labels -> generation/output/finetune/models/li6ps5cl_ft{,_stagetwo}.model
qsub -v FT_NAME=li6ps5cl_ft27k,TRAIN_FILE=data/dft/training_sets/ftK_train.xyz,VALID_FILE=data/dft/training_sets/ftK_valid.xyz \
     generation/finetune/finetune_mace.pbs        # 2x2x2 labels -> li6ps5cl_ft27k{,_stagetwo}.model
```

`finetune_mace.sh` runs `mace_run_train` (mace-torch 0.3.16) from MACE-MP-0 small with the
settings in the table below, identical for both fine-tunes. `--E0s=average` fits the
per-element reference energies to the data by least squares: QE total energies (about
-269 eV/atom with these pseudopotentials) are on a different scale from the references of
MACE-MP-0. Every snapshot has the same composition, so only the composition-weighted sum of the
four E0s is determined (the logs show practically identical values for P and Cl); forces do not
depend on E0. `--swa` without `--start_swa` starts stage two at MACE's default epoch, 3/4 of
120 = 90, with MACE's default stage-two weights (energy 1000, forces 100) and learning rate
1e-3, as the training logs record. MACE writes two models per run: stage one (best epoch of
stage one) and stage two (stochastic weight averaging).

| Fine-tune | Training / validation | Stage one | Stage two | Validation force RMSE (meV/A) |
|---|---|---|---|---|
| Gamma labels, `li6ps5cl_ft` | 22 / 5 | epoch 89 | epoch 116 | foundation 234.43, stage one 78.9, stage two 62.4 |
| 2x2x2 labels, `li6ps5cl_ft27k` | 24 / 3 | epoch 89 | epoch 119 | foundation 216.23, stage one 70.3, stage two 57.1 |

Each run took about 2 minutes on one A40. The errors that the re-analysis must reproduce are
listed in `protocol/reproduction_rule.md` (item 8).

## 7. MD with the fine-tuned models

The same protocol as the 416-atom production runs:

```bash
for T in 600 800 1000; do
  qsub -l walltime=48:00:00 -v MLIP=mace,TEMPS=$T,STEPS=200000,EQUILIB=10000,SUPERCELL_TAG=_sc222,TRAJ_TAG=_ft,MACE_MODEL=generation/output/finetune/models/li6ps5cl_ft_stagetwo.model generation/hpc/run_md.pbs
  qsub -l walltime=48:00:00 -v MLIP=mace,TEMPS=$T,STEPS=200000,EQUILIB=10000,SUPERCELL_TAG=_sc222,TRAJ_TAG=_ft27k,MACE_MODEL=generation/output/finetune/models/li6ps5cl_ft27k.model generation/hpc/run_md.pbs
done
```

The model files are the ones the July records name: the stage-two model for the Gamma
fine-tune (named in the working repository's report; a comment written there before the
training names `li6ps5cl_ft.model`, the stage-one model), and `li6ps5cl_ft27k.model`, the
stage-one model, in the July command that submitted the 2x2x2 runs. Which model drove each
fine-tuned trajectory is established from the trajectories themselves (force fingerprints,
`protocol/analysis_plan.md`), not from these records.

## 8. Analysis

The July trajectory analysis (pymatgen `DiffusionAnalyzer` for D and sigma, kinisi for the
uncertainties, weighted Arrhenius fits) is reproduced by `mlipft ingest trajectories` (module
`mlipft.ingest.trajectories`), which uses the same pymatgen/kinisi call path with the settings of
`protocol/reproduction_rule.md`. The working repository's `03_analyze_transport.py` is therefore
not ported here. It reads the trajectories from the raw backup named by `$MLIPFT_RAW`
(`<raw>/data/traj<tag>/<potential>_<T>K.traj`), the layout that `generation/output/` reproduces.

## Settings

The values that produced the July data, as options of the scripts (paths relative to
`generation/`). They are also the defaults of those scripts; `tests/test_generation.py` checks
this table against them. Options passed explicitly in the commands above (MD lengths and tags,
the 2x2x2 k-point mesh, the 2x2x2 split fraction, the file names of the 2x2x2 fine-tune) are not
listed.

| Step | File | Option | Value |
|---|---|---|---|
| 1 | `structures/build_structure.py` | `--n-configs` | `2` |
| 1 | `structures/build_doped.py` | `--supercell` | `2,2,2` |
| 2 | `md/run_md.py` | `--timestep` | `1.0` |
| 2 | `md/run_md.py` | `--friction` | `0.01` |
| 2 | `md/run_md.py` | `--log-every` | `50` |
| 2 | `md/run_md.py` | `--config` | `0` |
| 2 | `md/run_md.py` | `--mace-model` | `small` |
| 3 | `labels/sample_snapshots.py` | `--stride` | `100` |
| 3 | `labels/sample_snapshots.py` | `--skip-first` | `0.1` |
| 4 | `labels/label_qe.py` | `--kpts` | `gamma` |
| 4 | `labels/label_qe.py` | `--ecutwfc` | `60` |
| 4 | `labels/label_qe.py` | `--ecutrho` | `480` |
| 4 | `labels/label_qe.py` | `--degauss` | `0.01` |
| 4 | `labels/label_qe.py` | `--conv-thr` | `1e-7` |
| 5 | `labels/split_train_valid.py` | `--frac-valid` | `0.2` |
| 5 | `labels/split_train_valid.py` | `--seed` | `0` |
| 6 | `finetune/finetune_mace.sh` | `--foundation_model` | `small` |
| 6 | `finetune/finetune_mace.sh` | `--E0s` | `average` |
| 6 | `finetune/finetune_mace.sh` | `--loss` | `weighted` |
| 6 | `finetune/finetune_mace.sh` | `--energy_weight` | `1.0` |
| 6 | `finetune/finetune_mace.sh` | `--forces_weight` | `10` |
| 6 | `finetune/finetune_mace.sh` | `--lr` | `0.0001` |
| 6 | `finetune/finetune_mace.sh` | `--batch_size` | `4` |
| 6 | `finetune/finetune_mace.sh` | `--max_num_epochs` | `120` |
| 6 | `finetune/finetune_mace.sh` | `--swa` | `on` |
| 6 | `finetune/finetune_mace.sh` | `--default_dtype` | `float32` |
| 6 | `finetune/finetune_mace.sh` | `--seed` | `0` |

In words: QE pw.x SCF, PBE, ultrasoft pseudopotentials, ecutwfc 60 Ry / ecutrho 480 Ry, Gaussian
smearing 0.01 Ry, conv_thr 1e-7 Ry, Gamma point only (or the 2x2x2 mesh); 22 training / 5
validation configurations (Gamma) and 24 / 3 (2x2x2); `mace_run_train` (mace-torch 0.3.16) from
MACE-MP-0 small with E0s = average, weighted energy/force loss 1/10, Adam, learning rate 1e-4,
batch size 4, 120 epochs, float32, seed 0.

## Differences from the July scripts

- Paths: every output goes below `generation/output/`; structures are read from
  `data/structures/`, labels from `data/dft/`. Site-specific settings moved to
  `generation/hpc/site.env`; figure code was removed from the structure and MD drivers.
- `label_qe.py`: the defaults of `--kpts` (was `2,2,2`) and `--conv-thr` (was `1e-8`) are the
  values passed for the Gamma labels (`gamma`, `1e-7`); the automatic cutoffs from an SSSP json
  file were removed (both label sets used the fixed 60/480 Ry).
- `split_train_valid.py`: the default `--frac-valid` is 0.2 (was 0.1); the `--tag` option is
  not kept (use `--out-dir`).
- `run_md.py`: `--friction` is exposed (the value is the one `run_nvt` used by default); one run
  record per trajectory is written next to it instead of a shared `md_runs<tag>.json`, which
  per-temperature jobs overwrote.
- The relay script that chained the 2x2x2 labelling, split, fine-tune and MD jobs in July is not
  ported; the commands of steps 4 to 7 do the same steps one by one.

## Files

| File | Role |
|---|---|
| `structures/build_structure.py` | step 1: Ewald-ranked S/Cl orderings and supercells |
| `structures/build_doped.py` | step 1: Cl-excess series |
| `md/run_md.py` | steps 2 and 7: NVT MD driver around `mlipft.simulation.md` |
| `hpc/run_md.pbs` | steps 2 and 7: GPU job around `hpc/run_in_container.sh` |
| `hpc/run_in_container.sh` | runs `md/run_md.py` inside the container |
| `hpc/site.env.example` | site settings for all job scripts (copy to `hpc/site.env`) |
| `labels/sample_snapshots.py` | step 3: decorrelated snapshots from MD trajectories |
| `labels/fetch_pseudos.sh` | step 4: download PBE UPF files into `output/pseudo/` |
| `labels/label_qe.py` | step 4: one pw.x SCF per snapshot -> `dft_energy`, `dft_forces`, `dft_stress` |
| `labels/label_qe.pbs` | step 4: CPU job around `label_qe.py` (`mpirun pw.x`) |
| `labels/submit_shards.sh` | step 4: one `label_qe.pbs` job per snapshot (Gamma labels) |
| `labels/qe_scf_template.in` | reference copy of the pw.x input the driver writes |
| `labels/split_train_valid.py` | step 5: seeded training/validation split |
| `finetune/finetune_mace.sh` | step 6: `mace_run_train` command |
| `finetune/finetune_mace.pbs` | step 6: GPU job around `finetune_mace.sh` |

## Tests

`tests/test_generation.py` needs none of torch, mace or mattersim:

- the bytes of `md.py` after its docstring have the sha256 of the code that ran, and removing the
  Lennard-Jones branch gives the sha256 of the earlier version;
- every Python file here and in `src/mlipft/simulation/` compiles, and every shell and PBS file
  passes `bash -n`;
- the settings table above matches the script defaults, and the MD defaults match
  `mlipft.constants` (time step, frame spacing, temperatures);
- `mlipft.simulation.md` imports without torch, mace or mattersim, and `run_md.py` runs a short
  Lennard-Jones smoke test (needs ase);
- the merge commands of step 4 give the configuration order of both label files, and
  `split_train_valid.py` reproduces the four training-set files byte for byte (skipped until
  `data/dft/` exists; needs ase);
- the structure builder reproduces the cells of `data/structures/` (skipped without pymatgen).
