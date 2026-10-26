# Analysis plan

Committed on 2026-10-26, before the analyses it describes were run (see the disclosure in
`reproduction_rule.md`). Changes after that date are recorded in `amendments.md`.

## Question and comparison rule

How far is the Li-ion conductivity of Li6PS5Cl predicted by universal machine-learning
interatomic potentials from experiment, what does fine-tuning one of them to DFT change, and how
much does the quality of the DFT labels matter?

- Universal potentials (MACE-MP-0 small, MatterSim) are compared with experiment.
- Fine-tuned potentials are compared with DFT labels and, under the same MD protocol, with the
  foundation model they were fine-tuned from. **No fine-tuned model is compared with
  experiment** in any table, figure or text. A test enforces this on the result tables and the
  figure source data.

No new MD, DFT or training is done. Everything below uses data produced in July 2026.

## Definitions

**Configurations.** The 27 DFT-labelled configurations are frames 100, 200, ..., 900 of the
52-atom, 50-ps MACE-MP-0 trajectories at 600, 800 and 1000 K. Id: `T{600|800|1000}_f{frame:03d}`.
Both label sets must hold the same configurations: positions agree to 1e-6 A atom by atom, and
agree with the frames of the snapshot file.

**Label sets.**
- `gamma`: Quantum ESPRESSO, PBE, GBRV ultrasoft pseudopotentials, 60/480 Ry, Gaussian smearing
  0.01 Ry, convergence 1e-7 Ry, Gamma point only.
- `k222`: the same settings with a 2x2x2 Monkhorst-Pack mesh. Taken as the better reference.

**Subsets.** Verified against the original training and validation files, not just stated:
- V5: validation set of the Gamma fine-tune (5 configurations);
- H3: held out by both fine-tunes (the validation set of the 2x2x2 fine-tune, a subset of V5);
- H2: V5 minus H3, held out by the Gamma fine-tune only;
- T22: used in training by both fine-tunes;
- All27.

No 1000 K configuration is held out by either fine-tune.

**Models.**

| Id | Model |
|---|---|
| `foundation` | MACE-MP-0 small, 2023-12-10, 128 channels, L = 0 |
| `ftG_s1`, `ftG_s2` | fine-tuned on `gamma` (22 training configurations): stage one and stage two (stochastic weight averaging) |
| `ftK_s1`, `ftK_s2` | fine-tuned on `k222` (24 training configurations): stage one and stage two |
| `mattersim` | MatterSim (MD only) |

Model files are identified by sha256. Files with identical predictions on all configurations
are treated as one model.

**Fingerprints.** The fine-tuned MD runs store the forces of the model that drove them. For each
of the six fine-tuned trajectories, four frames (first, 1/3, 2/3, last) are evaluated with every
fine-tuned model in float32 on a CPU. A trajectory is assigned to a model if the RMS force
residual is below 1 meV/A and every other candidate is at least 10 times worse; otherwise it is
reported as unassigned. Part B reports the assigned models as the primary rows, and all four
fine-tuned models as secondary rows.

## Part A: universal potentials against experiment

- Benchmark: sintered Li6PS5Cl, 3.15 mS/cm (Wang et al. 2018), and the range of experimental
  values compiled by Li et al. (2025), 0.06-1.1 mS/cm.
- Headline: the production fits (416 atoms, 200 ps) quoted according to `reproduction_rule.md`.
- Convergence, from re-analysis values:
  - 52-atom runs at 50, 100 and 150 ps (whether these are independent runs or extensions of one
    run is tested from the trajectories and reported);
  - the 416-atom production runs truncated to their first 50, 100, 150 and 200 ps (nested
    windows of the same data);
  - the 416-atom, 150-ps MACE-MP-0 run of the Cl-excess series at x = 0, the same structure and
    protocol as the production run, as a second trajectory.
- Diagnostics (log-log slope of the Li MSD in the fit window outside 0.8-1.2; framework P, S and
  Cl MSD) are reported and never used to exclude a run.

## Part B: fine-tuned models against DFT

Predictions in float64 on a CPU, with mace-torch 0.3.16.

- **Forces**: error per Cartesian component (the MACE convention): RMSE and MAE overall, per
  element (Li, P, S, Cl) and per configuration. Softening slope: least-squares slope through the
  origin of model force against DFT force, overall and per element.
- **Energies**: every configuration has the composition Li24P4S20Cl4, so reference energies can
  only add a constant per (model, label set). The offset c = mean over T22 of
  (E_model - E_DFT) / N is subtracted and the centred error per atom is reported for every
  configuration; the offset taken over All27 is a sensitivity check. The absolute energy error
  of the foundation model is never reported.
- **Stress**: pressure error per configuration, as a diagnostic only (no model was trained on
  stress).
- **Label-error floor** (All27 and H3): RMSE of F_gamma - F_k222 overall and per element;
  centred energy difference; pressure difference. Decomposition
  RMSE(m vs k222)^2 = RMSE(m vs gamma)^2 + floor^2 + 2 cov, whose cross term shows whether a
  model trained on Gamma labels learned their error.
- **Primary estimand**: force RMSE against `k222` on H3, the only configurations outside the
  training data of all three models.
- **Secondary views**: each model's own validation set against its own labels (must reproduce
  the training logs, `reproduction_rule.md` item 8); H2; All27; errors by source temperature
  (descriptive).
- **Statistics**: with three held-out configurations there are no confidence intervals and no
  significance statements; every figure shows the individual H3 configurations. Leave-one-
  temperature-out validation would need retraining and is not done.

## Part C: transport, model against model

- Series: foundation (production run), and the MD runs of the two fine-tuned models, all at
  416 atoms, 200 ps, 600/800/1000 K. Re-analysis values for all three.
- Quantities: D ratio to the foundation at each temperature (error: kinisi relative standard
  deviations added in quadrature); sigma(300 K) and Ea from the weighted fit; Li MSD at 600 K;
  mean pressure at the fixed cell.
- Reading, fixed in advance: f = ln(sigma_k / sigma_G) / ln(sigma_foundation / sigma_G), the
  fraction of the fine-tuning slowdown recovered by the 2x2x2 labels.
  - f < 0.25: the k-point sampling of the labels does not explain the slowdown;
  - f > 0.75: it explains most of it;
  - otherwise partial;
  - if abs(ln(sigma_k / sigma_G)) is below its 1-sigma uncertainty, the two fine-tunes are
    reported as indistinguishable.
  If the two MD runs used models from different training stages, that confound is stated with
  the result. The pseudopotentials and the size of the data set, the other candidate causes,
  are not varied here and cannot be tested.

## Part D (secondary): Cl-excess series

Li6-xPS5-xCl1+x, x = 0, 0.25, 0.5, 0.75, MACE-MP-0, 150 ps. sigma(300 K) and Ea against Cl
content; enhancement over x = 0 compared with Adeli et al. (2019) at Cl 1.5. Extended Data and
documentation only.

## Thermostat statistics

From the trajectories for all runs: instantaneous temperature from the kinetic energy (3N
degrees of freedom, as in the July runs); mean and standard deviation over the production run;
slope of a straight-line fit of the potential energy per atom against time (meV/atom/ps);
pressure P = -tr(stress)/3 + N k_B T / V from the stored stress. Compared with the July run logs
as in `reproduction_rule.md` item 7.
