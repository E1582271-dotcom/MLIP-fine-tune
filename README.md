# MLIP-fine-tune

**How far from experiment do universal machine-learning potentials put the Li-ion conductivity of Li<sub>6</sub>PS<sub>5</sub>Cl, and what does fine-tuning MACE-MP-0 on DFT change?**

[![CI](https://github.com/E1582271-dotcom/MLIP-fine-tune/actions/workflows/ci.yml/badge.svg)](https://github.com/E1582271-dotcom/MLIP-fine-tune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

The calibration project of a computational portfolio on sulfide solid electrolytes. Two universal potentials (MACE-MP-0 small and MatterSim) and two fine-tunes of MACE-MP-0 drive molecular dynamics of the argyrodite Li<sub>6</sub>PS<sub>5</sub>Cl, 42 trajectories in all. The work makes three comparisons, and keeps them apart:

- **A.** Universal potentials against experiment.
- **B.** Fine-tuned models against their DFT reference.
- **C.** Fine-tuned MD against the foundation model's MD under the same protocol.

The fine-tuned models are never compared with experiment.

![Study design](figures/main/fig1_design.png)

## Findings

- **A. The two universal potentials land on opposite sides of experiment.** MACE-MP-0 gives σ(300 K) = 5.55 mS cm<sup>−1</sup>, 1.8 times the 3.15 mS cm<sup>−1</sup> of sintered Li<sub>6</sub>PS<sub>5</sub>Cl. MatterSim gives 0.35 mS cm<sup>−1</sup>, 0.11 times. These extrapolations are not converged in simulated time: the first 50 ps of the same 200-ps runs give 16 and 3.9 mS cm<sup>−1</sup>.
- **B. Fine-tuning brings the forces close to DFT, but not to the label floor.** The test set is the three configurations held out by both fine-tunes, and the reference is the 2×2×2 k-point labels:
  - the force RMSE falls from 216 meV Å<sup>−1</sup> (MACE-MP-0) to 59 and 70 meV Å<sup>−1</sup>;
  - the systematic softening of the foundation's forces (slope 0.81) disappears;
  - the Γ-point labels themselves differ from the 2×2×2 labels by 16 meV Å<sup>−1</sup>, about a quarter of the fine-tuned models' error.
- **C. Fine-tuning slows Li transport 12- to 21-fold.** Under the same MD protocol, σ(300 K) falls from 5.5 to 0.45 mS cm<sup>−1</sup> (Γ labels) and 0.26 mS cm<sup>−1</sup> (2×2×2 labels), and E<sub>a</sub> rises from 0.27 to 0.33-0.36 eV.
  - Better k-point sampling of the labels does not recover the slowdown: f = −0.22, which the pre-registered reading calls "not explained".
  - The two fine-tuned runs used models from different training stages, which confounds this comparison.
- **D. Cl excess: right direction, too large.** With MACE-MP-0, Li<sub>5.5</sub>PS<sub>4.5</sub>Cl<sub>1.5</sub> conducts 17 times better than Li<sub>6</sub>PS<sub>5</sub>Cl [12, 24]. The measured factor is about 4.
- **The July 2026 analysis reproduces.** Rules fixed before the re-analysis ([protocol/](protocol/)):
  - all 33 published runs give pymatgen D identical to 1e-15 and kinisi D within the tolerance set by its Monte Carlo scatter;
  - all 11 published fits are reproduced, so the published values are quoted.

<!-- BEGIN GENERATED headline (python -m mlipft docs) -->
**A. Universal potentials against experiment** (416 atoms, 200 ps)

| Potential | σ(300 K) (mS cm<sup>−1</sup>) | E<sub>a</sub> (eV) | σ / experiment |
|---|---:|---:|---:|
| MACE-MP-0 | 5.55 [4.2, 7.3] | 0.265 ± 0.012 | 1.8 |
| MatterSim | 0.349 [0.25, 0.49] | 0.346 ± 0.014 | 0.11 |
| Experiment, sintered (Wang et al. 2018) | 3.15 | | 1 |

**B. Fine-tuned models against DFT** (force RMSE on the 3 configurations held out by both fine-tunes, meV Å<sup>−1</sup>)

| Model | vs 2×2×2 labels | vs Γ-point labels | Softening slope |
|---|---:|---:|---:|
| MACE-MP-0 (foundation) | 216 | 220 | 0.81 |
| Fine-tuned, Gamma labels, stage two | 59.3 | 60.3 | 0.99 |
| Fine-tuned, 2x2x2 labels, stage one | 70.3 | 73.2 | 0.99 |
| Γ-point labels themselves (label floor) | 16.3 | | |

**C. Transport, fine-tuned against foundation** (same MD protocol; no experiment)

| Model | σ(300 K) (mS cm<sup>−1</sup>) | E<sub>a</sub> (eV) | σ / foundation |
|---|---:|---:|---:|
| MACE-MP-0 (foundation) | 5.54 [4.2, 7.3] | 0.266 ± 0.012 | 1.0 |
| Fine-tuned, Gamma labels (stage two) | 0.454 [0.31, 0.67] | 0.335 ± 0.016 | 0.082 |
| Fine-tuned, 2x2x2 labels (stage one) | 0.260 [0.19, 0.37] | 0.360 ± 0.014 | 0.047 |
<!-- END GENERATED headline -->

σ(300 K) is a weighted Arrhenius extrapolation from 600, 800 and 1000 K, with the 1σ interval of the fit in brackets. Every table: [docs/results.md](docs/results.md). Every figure with its caption: [docs/figures.md](docs/figures.md).

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
4. **Comparisons.** The figures and tables keep the three comparisons apart. A test checks that no fine-tuned value is placed next to an experimental one.

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
  - The four fine-tuned models are attached to the [v1.0.0 release](https://github.com/E1582271-dotcom/MLIP-fine-tune/releases) under MIT, the licence of MACE-MP-0.
  - The GBRV pseudopotentials are downloaded and checked by [generation/labels/fetch_pseudos.sh](generation/labels/fetch_pseudos.sh).

## Limits

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
