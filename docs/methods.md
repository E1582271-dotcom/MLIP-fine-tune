# Methods

All simulations were run between 30 June and 8 July 2026; this repository re-analyses them (no new
MD, DFT or training). The commands that produced the data are in
[generation/README.md](../generation/README.md); the comparisons and the reproduction rule were
fixed in [protocol/](../protocol/) before the re-analysis.

## Structure model

- **Starting cell.** Materials Project entry mp-985592: an ordered approximant of Li6PS5Cl (F-43m) in its conventional cubic cell, a = 10.2795 Å, 52 atoms. The PBE lattice parameter is 4.3 % larger than the measured 9.859 Å, and every MD run keeps it (constant volume).
- **Free anions.** Sulfur atoms more than 2.6 Å from every P are free anions; the other 16 S atoms form the PS4 tetrahedra.
- **S/Cl ordering.** The 8 free-anion sites were set to 50 % S and 50 % Cl. Their orderings were ranked by Ewald energy (pymatgen `OrderDisorderedStructureTransformation`; formal charges Li+1, P+5, S-2, Cl-1).
- **Cells used.** The lowest ordering (config0) is the 52-atom cell; its 2×2×2 supercell is the 416-atom cell (192 Li, a = 20.559 Å). Li positions are those of the Materials Project cell.
- **Cl-excess series.** Li6-xPS5-xCl1+x (x = 0, 0.25, 0.5, 0.75):
  - the free-anion sites take the target S/Cl ratio;
  - the Li sites take the occupancy that balances the charge (one vacancy per added Cl);
  - the lowest-Ewald ordering is kept and expanded to 2×2×2 (416, 408, 400, 392 atoms).

Code: [src/mlipft/simulation/structure.py](../src/mlipft/simulation/structure.py). The builders rebuild the committed CIFs byte for byte.

## Molecular dynamics

Every run uses the same setup ([src/mlipft/simulation/md.py](../src/mlipft/simulation/md.py)):

- ASE Langevin dynamics at constant volume, 1 fs time step, friction 0.01 fs-1.
- Maxwell-Boltzmann initial velocities with random seed 0; net momentum and rotation removed.
- Equilibration, then production with one frame every 50 fs.
- One trajectory per temperature (600, 800 and 1000 K).
- Precision and hardware: float32 on an NVIDIA A40 GPU.

| Series | Potential | Atoms | Equilibration + production | Part |
|---|---|---|---|---|
| 52 atoms, 50 / 100 / 150 ps | MACE-MP-0 small, MatterSim | 52 | 5 ps + 50 / 100 / 150 ps | A (sampling check) |
| Production | MACE-MP-0 small, MatterSim | 416 | 10 ps + 200 ps | A |
| Fine-tuned, Gamma labels | MACE fine-tuned on Gamma-point labels (stage two) | 416 | 10 ps + 200 ps | C |
| Fine-tuned, 2x2x2 labels | MACE fine-tuned on 2x2x2 labels (stage one) | 416 | 10 ps + 200 ps | C |
| Cl-excess series | MACE-MP-0 small | 392-416 | 10 ps + 150 ps | D |

Potentials:

- MACE-MP-0 small: the model of 2023-12-10 with 128 channels and L = 0, used through `mace_mp` without a dispersion correction.
- MatterSim: the default model of the `mattersim` package (v1.0.0-1M).

Which fine-tuned model drove each run was established from the forces stored in the trajectories (see "Model identity" below).

## Transport analysis

For every trajectory ([src/mlipft/transport/estimators.py](../src/mlipft/transport/estimators.py)), with the settings of July 2026. All 42 trajectories were analysed in one environment, a Linux CPU node, with the package versions of [requirements-ingest.lock.txt](../requirements-ingest.lock.txt) ([protocol/amendments.md](../protocol/amendments.md) A2).

- **pymatgen.**
  - `DiffusionAnalyzer.from_structures(smoothed="max")` gives the Li tracer diffusion coefficient D.
  - It also gives the Nernst-Einstein conductivity σ = n q² D / (k_B T), with the Haven ratio taken as 1. This σ is the value that is fitted.
- **kinisi.**
  - `DiffusionAnalyzer.from_ase` models the correlations of the mean-squared displacement and samples a posterior of D. The fit starts at 30 % of the trajectory.
  - The mean and standard deviation of the posterior are kept.
  - Seed 0 gives the value used; seeds 1-10 measure the Monte Carlo scatter of kinisi itself.
- **Arrhenius fit** ([src/mlipft/transport/arrhenius.py](../src/mlipft/transport/arrhenius.py)).
  - Weighted least squares of ln(σT) against 1/T. The error of each point is (kinisi standard deviation of σ) / (pymatgen σ).
  - Eₐ = -slope · k_B, and σ(300 K) = exp(slope/300 + intercept) / 300.
  - The unscaled covariance gives the standard error of Eₐ and a 1-sigma interval of σ(300 K).
  - With three temperatures per series, the interval reflects the per-point uncertainties of these trajectories, not the scatter between independent runs.
- **Mean-squared displacement curves.**
  - Averaged over all time origins (FFT algorithm), per species, after removing the drift of the cell's centre of mass.
  - Stored at about 150 log-spaced lags, together with the Li MSD that kinisi fits ([src/mlipft/transport/msd.py](../src/mlipft/transport/msd.py)).
- **Thermostat statistics** ([src/mlipft/transport/thermostat.py](../src/mlipft/transport/thermostat.py)).
  - Temperature from the kinetic energy (3N degrees of freedom); mean and standard deviation over the production run.
  - Drift: the slope of a straight-line fit of the potential energy per atom against time.
  - Pressure: -tr(stress)/3 from the stored virial stress plus N k_B T / V.
  - Stored as 0.5-ps block averages.
- **Sampling checks for the 416-atom, 200-ps runs.**
  - Fits of the first 50, 100 and 150 ps.
  - pymatgen D in four consecutive 50-ps blocks.
  - kinisi with the fit starting at 10-50 % of the trajectory.

## Reproduction of the July values

[protocol/reproduction_rule.md](../protocol/reproduction_rule.md) fixes how the re-analysis is compared with the published values and which value is quoted. The outcome is in [docs/reproduction.md](reproduction.md), and the thermostat statistics are compared with the July run logs where these exist.

## DFT labels

- **Configurations.** The 27 configurations are frames 100, 200, ..., 900 of the three 52-atom, 50-ps MACE-MP-0 runs: 9 per temperature, 5 ps apart, after dropping the first 10 %.
- **Two label sets.** Both are Quantum ESPRESSO 7.1 pw.x SCF calculations with the same settings, differing only in k-point sampling:

| Setting | Value |
|---|---|
| Functional | PBE |
| Pseudopotentials | GBRV ultrasoft: li_pbe_v1.4, p_pbe_v1.5, s_pbe_v1.4, cl_pbe_v1.4 (byte-identical to GBRV's public `all_pbe_UPF_v1.5.tar.gz`) |
| Cutoffs | 60 Ry (wavefunctions), 480 Ry (charge density) |
| Smearing | Gaussian, 0.01 Ry |
| Convergence | 1e-7 Ry, mixing 0.4 |
| k-points | `gamma`: Γ point only, real wavefunctions (about 26 min per configuration on 16 cores); `k222`: 2×2×2 Monkhorst-Pack mesh (about 5 h on 32 cores) |

- **Reference set.** The 2×2×2 labels are taken as the better reference.
- **Label-error floor.** The difference between the two label sets is the error floor of the Γ-point labels.
- **Energy units.** Label energies were converted from Ry with the CODATA 2006 constant of the ASE version of July.

## Fine-tuning

There are two independent fine-tunes of MACE-MP-0 small, with `mace_run_train` (mace-torch 0.3.16) on an NVIDIA A40. Settings common to both:

| Setting | Value |
|---|---|
| Reference energies | `E0s=average` (least squares over the data set) |
| Optimiser | Adam, learning rate 1e-4, weight decay 5e-7, batch size 4 |
| Stage one | Weighted loss: energy 1, forces 10 |
| Stage two | From epoch 90: stochastic weight averaging; energy 1000, forces 100, learning rate 1e-3 |
| Other | 120 epochs, float32, seed 0 |

| Fine-tune | Labels | Training / validation | Stage-one model | Stage-two model |
|---|---|---|---|---|
| `ftG` | Γ point | 22 / 5 | epoch 89 | epoch 116 |
| `ftK` | 2×2×2 | 24 / 3 | epoch 89 | epoch 119 |

Splits and subsets:

- Both splits come from one seeded shuffle of the same file order, so the validation set of `ftK` is the first three configurations of the validation set of `ftG`. This gives three subsets:
  - **H3**: held out by both fine-tunes;
  - **H2**: held out by `ftG` only;
  - **T22**: used in training by both.
- No 1000 K configuration is held out by either fine-tune.
- Every configuration has the composition Li24P4S20Cl4, so the fitted reference energies only shift the energy by a constant.

## Model identity

- **Files.** The training runs left eight model files and four TorchScript exports. Files that give identical predictions on all 27 configurations are one model, so there are four fine-tuned models.
- **Assigning stages.** Each model was assigned to its training stage by reproducing the validation errors printed in its training log ([protocol/amendments.md](../protocol/amendments.md) A1 explains the training errors).
- **Fingerprints.** For each fine-tuned MD run, four frames (first, 1/3, 2/3, last) were evaluated with every candidate model in float32. A run is assigned to a model when the RMS difference between the stored and the recomputed forces is below 1 meV/Å and every other candidate is at least 10 times further off.

## Fine-tuned models against DFT (part B)

Predictions in float64 on a CPU ([src/mlipft/ingest/models.py](../src/mlipft/ingest/models.py)). Metrics ([src/mlipft/dft/agreement.py](../src/mlipft/dft/agreement.py)):

- **Forces.** Error per Cartesian component, as RMSE and MAE, overall and per element. The softening slope is the least-squares slope through the origin of model force against DFT force.
- **Energies.** The error per atom after subtracting the mean error over T22 for that model and label set (the offset over all 27 is a sensitivity check). The foundation's absolute energy error is not used, because its reference energies differ from those of the DFT data.
- **Pressures.** Diagnostic only; no model was trained on stress.
- **Label-error floor.** The same measures for the Γ labels against the 2×2×2 labels, and the decomposition RMSE(m vs k222)² = RMSE(m vs Γ)² + floor² + cross term.
- **Primary estimand.** The force RMSE against the 2×2×2 labels on H3. With three configurations, no confidence intervals or significance statements are given.

## Transport, model against model (part C)

- **Runs compared.** The foundation (MACE-MP-0 production run) and the two fine-tuned MD runs, all at 416 atoms and 200 ps and all re-analysed in one environment.
- **Measures.** D ratios to the foundation at each temperature, and σ(300 K) and Eₐ of the weighted fits.
- **Reading fixed in the analysis plan.** f = ln(σ_K/σ_G) / ln(σ_foundation/σ_G), the fraction of the fine-tuning slowdown recovered with the 2×2×2 labels:
  - f < 0.25: the k-point sampling of the labels does not explain the slowdown;
  - f > 0.75: it explains most of it;
  - otherwise partial;
  - if the two fine-tuned values differ by less than their combined 1-sigma uncertainty, the two runs are indistinguishable.
- **Training-stage confound.** The Gamma-label run used the stage-two model and the 2x2x2-label run the stage-one model (see "Model identity"). On the three configurations held out by both fine-tunes, these two models differ in force error by about as much as the two stages of either fine-tune, so the comparison mixes label quality with training stage.
- **Not tested here.** Two other possible causes of the slowdown cannot be tested with these data: both fine-tunes use the same pseudopotentials, and their training sets differ by only two configurations (22 and 24).
- **Not compared with experiment.** The fine-tuned models are not compared with experiment.

## Cl-excess series (part D)

- Weighted Arrhenius fits of the four compositions.
- The enhancement over x = 0, with an interval that combines the two fits' log half-widths in quadrature.
- Comparison with the measured enhancement of Li5.5PS4.5Cl1.5 over Li6PS5Cl (Adeli et al. 2019). This is allowed because the series uses a universal potential.
