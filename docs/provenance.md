# Provenance

## Where every number comes from

The simulations, DFT labels and fine-tunes were made between 30 June and 8 July 2026 on a GPU
cluster (MD and fine-tuning on NVIDIA A40 GPUs, DFT on CPU nodes). A working repository analysed
part of them and published its values on 11 July 2026. This repository was rebuilt in October
2026 from two sources:

1. **The raw files.** Everything relevant on the cluster was copied, with sha256 sums computed
   on the cluster and checked after each copy. That includes the 42 trajectories, the QE inputs
   and outputs of both label sets, the label files, the model files, the training logs, the MD
   run logs and the batch job logs. Each file is listed in [data/raw_manifest.csv](../data/raw_manifest.csv).
   Two verified copies are kept outside the repository.
2. **The published record.** The analysis files published on 11 July, read with `git show` from
   commit `e2b7416c4af887fb9101aa444836164f689999a9` of the working repository, are in
   [data/published/](../data/published/), byte for byte.

`mlipft ingest` turns the raw files into `data/`. The DFT labels and the model predictions were
ingested on the workstation. All 42 trajectories were re-analysed in one environment, a Linux CPU
node of the cluster, with the package versions of
[requirements-ingest.lock.txt](../requirements-ingest.lock.txt) (amendment A2; the job is in
[ingest/](../ingest/)). The rules for comparing that re-analysis with the published values were
committed first ([protocol/](../protocol/), commit `951b94c`), and the outcome is in
[reproduction.md](reproduction.md).

## Data analysed here for the first time

| Data | Made | Published in July? |
|---|---|---|
| 2x2x2 k-point labels of the 27 configurations | 2-5 July | No |
| Fine-tune on those labels (`ftK`) and its 416-atom, 200-ps MD | 5 July | No |
| 52-atom, 100-ps MD of both universal potentials | 7-8 July | No |
| Thermostat statistics of the 23 runs without a surviving July run log | from the trajectories | No |
| Mean-squared displacement curves, T/E/P time series, pressure | from the trajectories | No |
| Per-atom predictions of all models on both label sets | 26 October | No |

## External data

| Data | Source | Use |
|---|---|---|
| Li6PS5Cl AIMD, 600-1200 K | Deng et al., Chem. Mater. 29, 281 (2017); files in materialsvirtuallab/matgenb at commit `2e77d515f23c507887856c265cdab84fa1a30082`, BSD-3-Clause | Transport reference of every model (amendment A3); derived values in `data/references/aimd_deng2017/` |
| GBRV pseudopotentials | `all_pbe_UPF_v1.5.tar.gz` | Identified as the files used in July (see below) |

## Facts established in the rebuild

- **Pseudopotentials.** Every QE output names the four GBRV files: `li_pbe_v1.4`, `p_pbe_v1.5`,
  `s_pbe_v1.4` and `cl_pbe_v1.4`. These files are byte-identical to those in GBRV's public
  `all_pbe_UPF_v1.5.tar.gz`. The label files carry the string `labelled_by=QE-pw.x-PBE-SSSP`,
  written by the July driver, which does not describe the files used.
- **Energy units.** The label energies equal the QE totals multiplied by 13.605691933 eV/Ry
  (CODATA 2006, the ASE version of July), not today's ASE value.
- **Which models drove the MD.** The fingerprint of the stored forces matches a single model for
  each run:
  - the published fine-tuned run (`ftG`) used the stage-two model (epoch 116), as the July report
    said;
  - the 2x2x2 run (`ftK`) used the stage-one model (epoch 89), not stage two, because its job
    loaded `models/li6ps5cl_ft27k.model`;
  - the MACE-MP-0 production run matches the foundation model.
- **Training logs.** For the Gamma fine-tune, MACE's logged training errors cover 20 of the 22
  training configurations (amendment A1). All logged validation errors are reproduced exactly.
- **Equilibration lengths** are not in the run records; the July batch job logs give them:
  - 5 ps for all 52-atom runs;
  - 10 ps for the 416-atom and Cl-excess runs.
- **The 52-atom tiers and the Cl x = 0 run are independent trajectories.** They share their
  starting structure and random seed with other runs, but their atoms are already 1.5-5 Å apart
  (RMS) in the first stored production frame (`data/md/divergence.csv`, Extended Data Fig. 4a).
  The spread of their σ(300 K) values is therefore a spread between independent runs.
- **Platform dependence of the re-analysis.** 26 runs had first been analysed on the workstation
  (macOS, arm64) before the move to the cluster (amendment A2). With the same code, settings and
  package versions, the two platforms give pymatgen D within 8e-16 (relative), identical thermostat
  statistics, and kinisi D (seed 0) within 0.9 %, at most 1.8 times the kinisi seed scatter of
  that run. kinisi samples its posterior by Markov-chain Monte Carlo, and a different platform acts
  like a different random seed. The workstation values are not used anywhere else.
- **Thermostat logs.** For a 52-atom, 50-ps run the July log's energy drift includes the
  production starting frame, which the trajectory does not store; this explains why four of these
  runs miss the drift check of the reproduction rule by 2-5 × 10⁻⁴ meV/atom/ps beyond the logged
  rounding ([reproduction.md](reproduction.md)).

## Timeline (UTC+8)

| Date | Event |
|---|---|
| 30 June | Structures; 52-atom, 50-ps MACE-MP-0 runs |
| 1 July | 52-atom, 150-ps MACE-MP-0 and 50-ps MatterSim runs; 27 Gamma-point labels; fine-tune `ftG`; 416-atom MACE-MP-0 and `ftG` runs |
| 2 July | 52-atom, 150-ps MatterSim runs; 2x2x2 labelling starts |
| 5 July | 2x2x2 labels complete; fine-tune `ftK`; its 416-atom runs |
| 6 July | 416-atom MatterSim runs; Cl-excess series |
| 7-8 July | 52-atom, 100-ps runs |
| 9 July | The working repository switches to weighted Arrhenius fits |
| 11 July | Values published (the files in `data/published/`) |
| 26 October | Raw files copied and checked; protocol committed; DFT and model ingest |
| 27 October | Trajectory re-analysis on a cluster CPU node (amendment A2) |
| 27 October | AIMD reference added (amendment A3) |

Before 9 July the working repository fitted without weights, and its commit history quotes those
values (for example 5.57 mS/cm for MACE-MP-0). Only the weighted fits are used here.

## Code

| Here | Origin | Change |
|---|---|---|
| [src/mlipft/simulation/md.py](../src/mlipft/simulation/md.py) | the MD driver that ran | Docstring only; the bytes below it are tested against the original sha256 |
| [src/mlipft/simulation/structure.py](../src/mlipft/simulation/structure.py), [generation/](../generation/) | the July scripts | Paths and cluster settings moved to `generation/hpc/site.env`; algorithms unchanged (byte-identical structures and splits) |
| [src/mlipft/transport/arrhenius.py](../src/mlipft/transport/arrhenius.py) | the July weighted Arrhenius fit | Names only |
| [src/mlipft/transport/estimators.py](../src/mlipft/transport/estimators.py) | the July pymatgen and kinisi calls | Explicit kinisi random seed |

## Corrections to the working repository's documents

The working repository's documents contain these statements, which are not carried over:

- **A "mechanochemical" Li6PS5Cl conductivity of 1.33 mS/cm.** No traceable source.
- **A Cl-excess value of 6.4 mS/cm at Cl 1.3.** No source.
- **Experimental conductivities of other syntheses and an experimental activation-energy range of
  0.22-0.38 eV.** Not re-checked against their sources, so omitted. The experimental spread used
  instead is the 0.06-1.1 mS/cm compiled by Li et al. (2025).
- **"Single-crystal upper bound".** The simulated conductivity is not a bound, because potential,
  ordering and sampling errors act in either direction.
- **The fine-tuned model "0.14 times experiment".** The fine-tuned models are compared with their
  DFT reference and with the foundation model, not with experiment.
- **"SSSP" pseudopotentials, and "UPF files not recorded".** See "Facts established in the
  rebuild".

## Not included

- A July pilot that ran four other candidate structures with the 150-ps protocol (superseded by
  the companion repository Solid-Electrolyte-MLIP-MD).
- Portfolio overview figures and exploratory notebooks of the working repository.
- `snapshots_v2.xyz`: 300 further configurations that were prepared but never labelled.
