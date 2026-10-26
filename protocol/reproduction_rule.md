# Reproduction rule

Committed on 2026-10-26, before any trajectory of this study was re-analysed. It fixes how the
re-analysis is compared with the values published in July 2026 and which value is quoted when
the two differ. The thresholds below are also constants in `src/mlipft/transport/reproduction.py`;
a test checks that the two agree.

## What is compared

The 11 series published in July 2026 (`data/published/`): MACE-MP-0 and MatterSim in the 52-atom
cell at 50 ps and 150 ps; MACE-MP-0, MatterSim and the fine-tuned MACE (Gamma-point labels) at
416 atoms and 200 ps; MACE-MP-0 for the four Cl-excess compositions at 150 ps. Each series has
three runs (600, 800 and 1000 K), 33 runs in all. The other 9 runs (52 atoms at 100 ps for both
universal potentials, and the model fine-tuned on 2x2x2 k-point labels) were never analysed in
July and are labelled "new".

## Fixed settings (the July call path)

- Diffusion coefficient D: `pymatgen.analysis.diffusion.analyzer.DiffusionAnalyzer.from_structures`
  with `smoothed="max"`, time step 1 fs, frame spacing 50 steps, species Li.
- Conductivity: Nernst-Einstein, sigma = n q^2 D / (k_B T), with the Li number density of the
  first frame and no Haven ratio.
- Uncertainty: `kinisi.analyze.DiffusionAnalyzer.from_ase`, same time step and frame spacing,
  `diffusion()` with the fit starting at 30 % of the trajectory length and kinisi's default
  sampling settings. The July runs used `random_state=None`; here the primary value uses
  `random_state=numpy.random.RandomState(0)` and seeds 1 to 10 are used only to measure the
  Monte-Carlo scatter s_seed of each kinisi quantity.
- Arrhenius fit: ln(sigma T) against 1/T, weighted least squares with the error of each point
  equal to (kinisi standard deviation of sigma) / (pymatgen sigma), unscaled covariance,
  interval of sigma(300 K) at 1 sigma (`src/mlipft/transport/arrhenius.py`, the July function).

## Outcome classes

| Quantity | R1 exact | R2 reproduced | R3 discrepant |
|---|---|---|---|
| pymatgen D and sigma | relative difference <= 1e-9 | <= 1e-6 | > 1e-6 |
| kinisi D | - | relative difference <= max(3 s_seed, 2 %) | larger |
| kinisi standard deviation | - | relative difference <= max(3 s_seed, 10 %) | larger |
| Fitted sigma(300 K) and Ea | relative difference <= 1e-6 | abs(d ln sigma300) <= 0.1 x published log half-width and abs(d Ea) <= 0.1 x published Ea error | beyond 1 x the published uncertainty |

A fit between 0.1 and 1 times the published uncertainty is classed **R2-** ("consistent within
the published uncertainty"). s_seed comes only from today's ten-seed scatter, never from the
comparison with July.

## Which value is quoted

1. The published values are the record of what was published. They are never edited; the
   companion repository Solid-Electrolyte-MLIP-MD keeps citing them.
2. If all 11 fits are R1 or R2, the text quotes the published values.
3. If a fit is R2- or R3 only because the kinisi weights changed (pymatgen D is R1), the
   re-analysis value is quoted in the text and figures, because a number that cannot be
   regenerated should not be the primary one. Tables show both values and the reason.
4. If pymatgen D of any run is R3, the work stops at that point: the trajectory, its frame count
   or the pymatgen version is suspect. No figures are made until the cause is found and written
   into `amendments.md`.
5. Comparisons between models (Part C of the analysis plan) and the convergence checks of Part A
   always use re-analysis values for every series, so that all values in one comparison come
   from one environment.
6. No setting changes after results are seen. Sensitivity analyses never replace primary values.
   Any change that alters a derived number needs a dated entry in `amendments.md`, and the
   first-pass outputs are kept (`results/reproduction_first_pass.csv` and git history).
7. Thermostat statistics recomputed from the trajectories are checked against the July run logs
   where those exist. The logs hold one extra sample at step 0 and rounded values, so the
   tolerance is abs(d mean T) <= 0.2 K for 416 atoms and <= 1 K for 52 atoms, and the energy
   and drift must agree to the logged rounding.
8. Training logs: errors of the fine-tuned models re-evaluated on their own training and
   validation sets, rounded to the printed precision of the MACE log, must equal the logged
   values (Gamma: 62.4 / 47.9 meV/A stage two, 78.9 / 72.4 stage one, foundation 234.43;
   2x2x2: 57.1 / 43.9 stage two, 70.3 / 67.6 stage one).

## Disclosure

Before this rule was written, the foundation model and the stage-two Gamma model had already
been evaluated against the Gamma-point labels (2026-10-26), and the training log of the 2x2x2
model (validation force RMSE 57.1 meV/A) had been read. The 2x2x2 cross-evaluation, the model
fingerprints and every trajectory re-analysis had not been run.
