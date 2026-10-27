# Amendments

Dated changes to `reproduction_rule.md` and `analysis_plan.md` made after they were committed,
with the reason and whether results had been seen.

## A1 (2026-10-26): training-set errors in the MACE logs

**Rule affected:** `reproduction_rule.md`, item 8.

**What was found.** Item 8 assumed that the training-set errors printed in the MACE logs cover all
training configurations. mace-torch 0.3.16 evaluates the training set with its training data
loader, which shuffles and drops the last incomplete batch (`drop_last=True`). With a batch size
of 4:

- the Gamma fine-tune has 22 training configurations, so each logged training error covers a
  random 20 of them;
- the 2x2x2 fine-tune has 24, so all of them are used.

The validation loader drops nothing.

**Change.**

- Validation errors must equal the logged values, as before.
- Training errors must equal the logged values when the training set is a multiple of the batch
  size.
- Otherwise the logged value must be produced by at least one subset of the evaluated size
  (here 20 of 22). Model files are assigned to a training stage by their validation error.

**Results seen before the change.** Yes:

- Gamma fine-tune, stage one: the re-evaluated training error over all 22 configurations is
  72.18 meV/A, against 72.4 logged. 11 of the 231 subsets of 20 configurations give 72.4.
- Gamma fine-tune, stage two: 47.93 over all 22, against 47.9 logged. 21 of the 231 subsets give
  47.9.
- All validation errors and both 2x2x2 training errors matched as originally stated.

## A2 (2026-10-27): where the trajectories are re-analysed

**Rule affected:** `reproduction_rule.md`, item 5 (all values of one comparison come from one
environment). The rule itself is unchanged; this records the environment that satisfies it.

**What happened.** The re-analysis started on the local workstation (macOS, arm64) and was stopped
after 26 of the 42 runs because the machine was overloaded.

**Change.** All 42 runs are re-analysed again, from the beginning, in one environment: a cluster CPU
node (Linux, x86_64), with CPython 3.11.15 and the same package versions as the local ingest
environment (numpy 2.4.6, ase 3.29.0, pymatgen 2026.5.4, pymatgen-analysis-diffusion 2025.11.15,
kinisi 2.0.5, scipp 26.3.1, scipy 1.17.1, emcee 3.1.6), with the same code and settings. Every
transport value in `data/md/` comes from that run. The 26 local results are not used for any
result; they are compared with the cluster values as a check of platform dependence. The DFT and
model ingest (`data/dft/`, `data/models/`) stays local; it enters different comparisons.

**Results seen before the change.** Yes: the run-level reproduction classes of 10 of the 26 local
runs had been computed (pymatgen R1 and kinisi R2 for all 10).

## A3 (2026-10-27): AIMD as the reference for Li transport

**Rules affected:** `analysis_plan.md`, Part A (benchmark) and Part C (references). `reproduction_rule.md`
is unchanged.

**What was found.** The plan compared the universal potentials with experiment, and the fine-tuned
models only with the foundation model, so no transport value was checked against DFT. Two facts make
experiment the wrong yardstick:

- Every run simulates the fully ordered S/Cl approximant (Cl on 4a, S on 4c), whereas the measured
  conductivities are of S/Cl-disordered polycrystals. In AIMD the two differ by about two orders of
  magnitude (Baktash et al. 2020).
- Published AIMD extrapolations of ordered Li6PS5Cl to 300 K span 2e-6 to 0.29 S/cm (Table 2 of
  Baktash et al. 2020), so an extrapolated value is not a usable reference either.

**Change.**

- The transport reference for every model, universal or fine-tuned, is AIMD of the same ordered
  structure: the Li6PS5Cl trajectories of Deng et al. (2017), published with the paper's notebooks
  (materialsvirtuallab/matgenb, commit 2e77d515f23c507887856c265cdab84fa1a30082, BSD-3-Clause).
  - From the files: VASP, 52 atoms, a = 10.10 A, time step 2 fs, every 10th step stored, 100 ps at
    each of 600, 800, 1000 and 1200 K.
  - The exchange-correlation functional and thermostat are not recorded in the files.
- The comparison is D at the simulated temperatures 600, 800 and 1000 K, without extrapolation:
  - the ratio of the kinisi D (seed 0, fit from 30 % of the run) of each MLIP run to that of AIMD,
    with the two relative standard deviations added in quadrature;
  - the pymatgen D ratio and the log-log slope of the Li MSD in the fit window are reported with it.
- The AIMD trajectories are analysed with the estimators and settings of the MLIP runs. The only
  differences are their own time step and stored-frame interval.
- The analysis runs on the workstation (macOS, arm64) with the ingest environment, not on the
  cluster node of A2. The platform check of A2 bounds the difference at 0.9 % in kinisi D, far below
  the ratios compared here.
- Known differences between the AIMD and the MLIP runs, stated with every result:
  - lattice parameter 10.10 A against 10.28 A (5 % smaller volume);
  - 52 atoms against 416 atoms for the production runs (the 52-atom MLIP runs match the AIMD cell);
  - one trajectory per temperature in both.
- Experimental values stay only as context for the universal potentials. They are no longer a
  headline comparison, and the fine-tuned models are still never compared with experiment.
- The Part C reading f is unchanged and still reported.

**Results seen before the change.** Yes. All MLIP results were known. The AIMD values were first
computed in a scratch analysis on 2026-10-27, with the settings stated here, before this entry was
written. Relative to AIMD, the kinisi D of the 416-atom runs was:

| Model | 600 K | 800 K | 1000 K |
|---|---|---|---|
| MACE-MP-0 | 8.8x | 5.1x | 3.2x |
| MatterSim | 1.8x | 1.9x | 1.9x |
| Fine-tuned on Gamma-point labels | 1.4x | 2.1x | 1.8x |
| Fine-tuned on 2x2x2 labels | 1.4x | 2.6x | 1.9x |

At 600 K the AIMD Li MSD is almost flat in the fit window (log-log slope 0.07), so the 600 K
ratio is not a ratio of diffusion coefficients.
