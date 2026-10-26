# Changelog

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
- Five models evaluated against both DFT label sets, per atom; the model behind each fine-tuned MD run
  identified from the forces stored in its trajectory.
- Four main and eight Extended Data figures with per-panel source data; generated results,
  reproduction and figure documents.
- The fine-tuned models are compared with DFT and with the foundation model only, not with
  experiment.
