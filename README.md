# MLIP-fine-tune

**How close do machine-learning potentials come to DFT for Li transport in Li<sub>6</sub>PS<sub>5</sub>Cl, and what does fine-tuning MACE-MP-0 on DFT change?**

[![CI](https://github.com/E1582271-dotcom/MLIP-fine-tune/actions/workflows/ci.yml/badge.svg)](https://github.com/E1582271-dotcom/MLIP-fine-tune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

The calibration project of a computational portfolio on sulfide solid electrolytes. Two universal potentials (MACE-MP-0 small and MatterSim) and two fine-tunes of MACE-MP-0 drive molecular dynamics of the argyrodite Li<sub>6</sub>PS<sub>5</sub>Cl, 42 trajectories in all. Every model is judged against DFT:

- **Li diffusion** against AIMD of the same ordered cell (Deng et al. 2017), at the simulated temperatures.
- **Forces and energies** against two sets of DFT labels (Γ point and 2×2×2 k-points).
- **Fine-tuned MD** also against the foundation model's MD under the same protocol.

Measured conductivities are of S/Cl-disordered polycrystals, not of the ordered cell simulated here, so they appear only as context.

![Study design](figures/main/fig1_design.png)

## Findings

- **Against AIMD, MACE-MP-0 diffuses Li 3 to 5 times too fast; MatterSim and both fine-tunes are within about 2 times.** These are kinisi D ratios at 800 and 1000 K:
  - MACE-MP-0: 5.1 and 3.2;
  - MatterSim: 1.9 and 1.9;
  - fine-tuned on Γ-point labels: 2.1 and 1.8;
  - fine-tuned on 2×2×2 labels: 2.6 and 1.9.

  At 600 K the AIMD Li ions barely leave their cages in 100 ps. MACE-MP-0 already diffuses there, while the other three models stay sub-diffusive, as AIMD does. The AIMD cell is smaller (a = 10.10 Å against 10.28 Å, 52 atoms), which may account for part of the remaining factor of 2.
- **Fine-tuning brings the forces close to DFT.** The test set is the three configurations held out by both fine-tunes, and the reference is the 2×2×2 labels:
  - the force RMSE falls from 216 meV Å<sup>−1</sup> (MACE-MP-0) to 59 and 70 meV Å<sup>−1</sup>;
  - the systematic softening of the foundation's forces (slope 0.81) disappears, consistent with its overestimated diffusion;
  - the Γ-point labels themselves differ from the 2×2×2 labels by 16 meV Å<sup>−1</sup>, about a quarter of the fine-tuned models' error.
- **The slowdown from fine-tuning is a move towards DFT.** Under the same MD protocol, fine-tuning lowers the extrapolated σ(300 K) from 5.5 to 0.45 mS cm<sup>−1</sup> (Γ labels) and 0.26 mS cm<sup>−1</sup> (2×2×2 labels), and raises E<sub>a</sub> from 0.27 to 0.33-0.36 eV.
  - Better k-point sampling of the labels does not change this: f = −0.22, which the pre-registered reading calls "not explained".
  - The two fine-tuned runs used models from different training stages, which confounds the comparison between them.
- **Extrapolated σ(300 K) is not a usable benchmark for this structure.**
  - The first 50 ps of the same 200-ps runs give 16 (MACE-MP-0) and 3.9 (MatterSim) mS cm<sup>−1</sup>.
  - Published AIMD extrapolations of ordered Li<sub>6</sub>PS<sub>5</sub>Cl range from 0.002 to 290 mS cm<sup>−1</sup>.
- **Cl excess, with MACE-MP-0.** σ(300 K) rises 17-fold [12, 24] from Li<sub>6</sub>PS<sub>5</sub>Cl to Li<sub>5.5</sub>PS<sub>4.5</sub>Cl<sub>1.5</sub>. For context, polycrystals measure about 4-fold.
- **The July 2026 analysis reproduces.** Rules fixed before the re-analysis ([protocol/](protocol/)):
  - all 33 published runs give pymatgen D identical to 1e-15 and kinisi D within the tolerance set by its Monte Carlo scatter;
  - all 11 published fits are reproduced, so the published values are quoted.

<!-- BEGIN GENERATED headline (python -m mlipft docs) -->
**Li diffusion against AIMD** (416 atoms, 200 ps; kinisi D divided by the AIMD D of the same ordered cell at the same temperature)

| Model | 800 K | 1000 K | 600 K<sup>a</sup> |
|---|---:|---:|---:|
| MACE-MP-0 (foundation) | 5.1 | 3.2 | 8.8 |
| MatterSim | 1.9 | 1.9 | 1.8 |
| Fine-tuned, Gamma labels (stage two) | 2.1 | 1.8 | 1.4 |
| Fine-tuned, 2x2x2 labels (stage one) | 2.6 | 1.9 | 1.4 |
| AIMD, D (cm<sup>2</sup> s<sup>−1</sup>) | 4.3e-6 | 1.7e-5 | 7.1e-7 |

<sup>a</sup> At 600 K the AIMD Li MSD is still almost flat in the fit window (log-log slope 0.07), so this column compares sub-diffusive motion.

**Forces against DFT** (RMSE on the 3 configurations held out by both fine-tunes, meV Å<sup>−1</sup>)

| Model | vs 2×2×2 labels | vs Γ-point labels | Softening slope |
|---|---:|---:|---:|
| MACE-MP-0 (foundation) | 216 | 220 | 0.81 |
| Fine-tuned, Gamma labels, stage two | 59.3 | 60.3 | 0.99 |
| Fine-tuned, 2x2x2 labels, stage one | 70.3 | 73.2 | 0.99 |
| Γ-point labels themselves (label floor) | 16.3 | | |

**Extrapolated to 300 K** (weighted Arrhenius fit of 600–1000 K, 1σ interval)

| Model | σ(300 K) (mS cm<sup>−1</sup>) | E<sub>a</sub> (eV) | σ / foundation |
|---|---:|---:|---:|
| MACE-MP-0 (foundation) | 5.54 [4.2, 7.3] | 0.266 ± 0.012 | 1.0 |
| Fine-tuned, Gamma labels (stage two) | 0.454 [0.31, 0.67] | 0.335 ± 0.016 | 0.082 |
| Fine-tuned, 2x2x2 labels (stage one) | 0.260 [0.19, 0.37] | 0.360 ± 0.014 | 0.047 |
| MatterSim | 0.349 [0.25, 0.49] | 0.346 ± 0.014 | |
<!-- END GENERATED headline -->

D comes from kinisi (fit from 30 % of each run); σ(300 K) is a weighted Arrhenius extrapolation from 600, 800 and 1000 K, with the 1σ interval of the fit in brackets. Every table: [docs/results.md](docs/results.md). Every figure with its caption: [docs/figures.md](docs/figures.md).

![Fine-tuned models against DFT](figures/main/fig3_dft.png)

![What fine-tuning changes in Li transport](figures/main/fig4_transport.png)

## Approach

1. **Simulations.** All of them were run in July 2026 on NVIDIA A40 GPUs:
   - Langevin NVT at 600, 800 and 1000 K, in the 52-atom cell and its 416-atom supercell, for 50-200 ps;
   - 27 configurations from 52-atom MACE-MP-0 MD, labelled with Quantum ESPRESSO (PBE) twice, at the Γ point and with a 2×2×2 k-point mesh;
   - two fine-tunes of MACE-MP-0 small, one on each label set ([docs/methods.md](docs/methods.md)).
2. **Rebuild.** In October 2026 every raw file was copied and checksummed: 42 trajectories, QE runs, model files and logs.
   - The rules for reproducing the published values were committed before any trajectory was re-analysed ([protocol/](protocol/)). Every later change has a dated amendment.
   - All 42 trajectories were re-analysed in one environment, with pymatgen and with kinisi over 11 random seeds.
3. **Model identity.** The forces stored in each fine-tuned trajectory identify the model that produced it:
   - the Γ-label run used the stage-two model;
   - the 2×2×2-label run used the stage-one model.
4. **AIMD reference** (amendment A3). The published Li<sub>6</sub>PS<sub>5</sub>Cl AIMD trajectories of Deng et al. (2017) have the same ordered S/Cl arrangement as the simulated cell. They are re-analysed with exactly the estimators used for the MLIP runs, and every model's D is compared with them at 600, 800 and 1000 K, without extrapolation.
5. **Comparisons.** The figures and tables keep forces against DFT, diffusion against AIMD, and fine-tuned against foundation apart. A test checks that no fine-tuned value is placed next to an experimental one.

## Reproduce

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
python -m mlipft all      # results/, figures/ and the generated docs from data/
python -m pytest          # reproduction of the published fits, data integrity, figures, the comparison rule
make check                # all of the above, repository checks, and an unchanged working tree
```

Everything is rebuilt from the committed `data/` in about a minute with numpy, pandas and matplotlib. Re-deriving `data/` from the raw files (`make ingest`) needs the raw backup and the packages of [requirements-ingest.lock.txt](requirements-ingest.lock.txt) ([docs/data.md](docs/data.md)).

## Data availability

- **In this repository.** Everything derived from the raw files is in `data/` as plain text:
  - per-run transport values, kinisi seeds, MSD curves and thermostat time series;
  - the DFT labels, the Quantum ESPRESSO inputs and outputs, and the model predictions per atom;
  - the July 2026 published analysis, byte for byte.
- **Not in this repository.**
  - The 3.3 GB of trajectories are listed with their checksums in [data/raw_manifest.csv](data/raw_manifest.csv).
  - The AIMD reference files (66 MB, BSD-3-Clause) are downloaded at a pinned commit and checked by [ingest/fetch_aimd.sh](ingest/fetch_aimd.sh); the values derived from them are in [data/references/aimd_deng2017/](data/references/aimd_deng2017/SOURCE.md).
  - The four fine-tuned models are attached to the [v1.0.0 release](https://github.com/E1582271-dotcom/MLIP-fine-tune/releases) under MIT, the licence of MACE-MP-0.
  - The GBRV pseudopotentials are downloaded and checked by [generation/labels/fetch_pseudos.sh](generation/labels/fetch_pseudos.sh).

## Limits

- **AIMD reference.**
  - One 100-ps trajectory per temperature, in a 52-atom cell at a = 10.10 Å; the MLIP runs use a = 10.28 Å.
  - The exchange-correlation functional is not recorded in the published files.
  - At 600 K neither AIMD nor three of the four models is in the diffusive regime, so that comparison is of caged motion.
- **Sampling.**
  - Each temperature has one trajectory, and σ(300 K) is an extrapolation from 600-1000 K with the Haven ratio taken as 1.
  - The fit interval does not include the spread between independent runs. That spread is large: independent 52-atom runs of MACE-MP-0 give 2.4-22 mS cm<sup>−1</sup>.
  - At 600 K the Li MSD of MatterSim and of both fine-tuned runs is still sub-diffusive in the fit window ([docs/results.md](docs/results.md)).
- **Structure.**
  - One ordered approximant of the S/Cl disorder.
  - A fixed cell at the PBE lattice parameter, 4.3 % larger than measured.
- **Part B statistics.**
  - Three configurations are held out by both fine-tunes, so no confidence intervals or significance statements are given.
  - No 1000 K configuration is held out.
  - The foundation and the Γ-label model had been evaluated against the Γ labels before the protocol was written (disclosed in [protocol/reproduction_rule.md](protocol/reproduction_rule.md)).
- **Part C.**
  - The two fine-tuned runs differ in training stage as well as in labels.
  - The pseudopotentials and the size of the training set, the other candidate causes of the slowdown, are not varied.
- **Model predictions.** All conductivities are model predictions, not measurements.

## Repository

| Path | Content |
|---|---|
| [protocol/](protocol/) | Reproduction rule and analysis plan, committed before the re-analysis, with dated amendments |
| [docs/results.md](docs/results.md), [docs/reproduction.md](docs/reproduction.md) | Every result table; the reproduction outcome, run by run (generated) |
| [docs/methods.md](docs/methods.md), [docs/figures.md](docs/figures.md) | Structure, MD, DFT, fine-tuning and analysis methods; all figures with captions |
| [docs/data.md](docs/data.md), [docs/provenance.md](docs/provenance.md) | What every file holds; where every number comes from, and facts established in the rebuild |
| [data/](data/) | Published July record, structures, MD, DFT labels and QE runs, model predictions, references |
| [results/](results/) | Tables built from `data/` by `python -m mlipft tables` |
| [figures/](figures/) | Four main and eight Extended Data figures (PDF, SVG, PNG) with per-panel source data |
| [src/mlipft/](src/mlipft/) | Transport estimators, DFT agreement, tables, figures, the ingest of the raw files |
| [generation/](generation/), [ingest/](ingest/) | Code that produced the raw data in July 2026; the cluster job of the re-analysis |

## Related projects

- [Solid-Electrolyte-Screening](https://github.com/E1582271-dotcom/Solid-Electrolyte-Screening) screens Materials Project entries down to 62 candidate Li–S compositions.
- [Solid-Electrolyte-MLIP-MD](https://github.com/E1582271-dotcom/Solid-Electrolyte-MLIP-MD) classifies those 62 compositions with the same un-fine-tuned MACE-MP-0 calibrated here.
- [Solid-Electrolyte-Inverse-Design](https://github.com/E1582271-dotcom/Solid-Electrolyte-Inverse-Design) generates new candidate structures.

## Citation and license

Cite with [CITATION.cff](CITATION.cff). Code and data are under the MIT license ([LICENSE](LICENSE)). Author: Jieao Jia.
