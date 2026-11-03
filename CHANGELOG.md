# Changelog

## 1.1.0 (2026-11-03)

- MatterSim evaluated against both DFT label sets with the Part B measures (amendment A4), after
  checking that the local model reproduces the forces stored in the July MatterSim runs. It enters
  Fig. 2 and the softening panel of Fig. 3.
- A new MD series, the stage-two 2x2x2 fine-tune at 600, 800 and 1000 K with the July settings
  (amendment A5, committed before the runs), removes the training-stage confound of Part C. Its
  force fingerprint, thermostat check and re-analysis follow the July runs; data/md now holds 45
  runs. At equal training stage the sigma(300 K) of the two fine-tunes differ by less than their
  1-sigma fit uncertainty (f2 = -0.20, ln(sigma_2x2x2/sigma_Gamma) = -0.51 +/- 0.54, read as
  "indistinguishable"); the training stage at equal labels changes D by factors of 0.81-1.11. The
  pre-registered reading with the stage-one run (f = -0.22, "not explained") stays reported.
- The main figures show the stage-two fine-tunes only; the stage-one models and the July 2x2x2 run
  appear in Extended Data, the run inventory of Fig. 1 and the stage comparison of Fig. 4.
- Figures restructured into four main figures (study design; forces against DFT; Li diffusion against
  AIMD; what fine-tuning changes) and nine Extended Data figures; measured conductivities appear only
  in Extended Data Figs. 8 and 9, as context.
- One encoding vocabulary for every figure (fine-tuned models blue, universal potentials red, DFT and
  AIMD references black and grey; fill marks the training stage), closed frames, and figures cited by
  registry key instead of hard-coded numbers.
- Automatic figure checks for panel alignment, closed frames, overlapping text and markers, clipping,
  font sizes and figure size (strict locally, warnings only in CI).
- The 1.0.0 entry below says eight Extended Data figures; 1.0.0 had nine (the AIMD reference was its
  ninth).

## 1.0.0 (2026-10-27)

First public release, rebuilt from the raw files of the July 2026 simulations.

- Reproduction rule and analysis plan committed before any re-analysis, with amendments A1 (MACE
  training-log errors) and A2 (where the trajectories are re-analysed).
- All 42 MD trajectories re-analysed in one environment (pymatgen, kinisi with 11 seeds, MSD,
  thermostat statistics, truncation windows, block estimates, trajectory divergence). All 33 published
  runs and 11 published fits are reproduced.
- New analyses of data made in July but never analysed: the 2x2x2 k-point DFT labels, the fine-tune
  on them and its MD, the 52-atom 100-ps runs, and the thermostat statistics of the runs without a
  surviving run log.
- Li diffusion of every model compared with published AIMD of the same ordered cell (Deng et al. 2017) at
  600-1000 K (amendment A3); measured conductivities kept as context only.
- Five models evaluated against both DFT label sets, per atom; the model behind each fine-tuned MD run
  identified from the forces stored in its trajectory.
- Four main and eight Extended Data figures with per-panel source data; generated results,
  reproduction and figure documents.
- The fine-tuned models are compared with DFT (labels and AIMD) and with the foundation model only, not
  with experiment.
