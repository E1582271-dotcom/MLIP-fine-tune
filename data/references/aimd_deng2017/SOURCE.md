# AIMD reference: Li6PS5Cl, Deng et al. (2017)

The transport reference of [amendment A3](../../../protocol/amendments.md). Derived here from the AIMD
records published with the notebooks of

> Z. Deng, Z. Zhu, I.-H. Chu and S. P. Ong, Data-driven first-principles methods for the study and design of alkali superionic conductors, Chem. Mater. 29, 281-288 (2017), doi:10.1021/acs.chemmater.6b02648.

## Source files

Repository `materialsvirtuallab/matgenb`, commit `2e77d515f23c507887856c265cdab84fa1a30082`, folder `notebooks/aimd_data/`. They are downloaded and
checked by [ingest/fetch_aimd.sh](../../../ingest/fetch_aimd.sh) and are not stored here.

| File | sha256 |
|---|---|
| `600.json` | `9157f2128ee88a508ca0560989d9b0c401f612d5c59d61a277ae89ca4a5deabe` |
| `800.json` | `fb31dce75edb856ae03a3fb0138c61dd7b8bb111501c2af2d661de0cf87c093a` |
| `1000.json` | `bf4ea1d1104279d422fa460e5d4c50dedeb163e846ea8cbbead285f2378a0701` |
| `1200.json` | `a5d283ca17110e17d1350bc1593ad43f45974ab54047959598d3964861f8ca49` |

What the files record: VASP AIMD of Li24P4S20Cl4, the ordered S/Cl arrangement of this repository's
cell (Cl on 4a, S on 4c), 52 atoms, a = 10.10 A, time step 2 fs, every 10th step stored, 100 ps at
600, 800, 1000 and 1200 K. The exchange-correlation functional and the thermostat are not recorded in
the files.

## Files here

- `transport.csv`: per temperature, pymatgen D (`smoothed="max"`) and its conductivity, kinisi D
  (seed 0, fit from 30 % of the run) with its standard deviation, the scatter over kinisi seeds 1-10,
  and the log-log slope of the Li MSD in the fit window.
- `kinisi_seeds.csv`: kinisi D for seeds 0-10.
- `msd.csv.gz`: MSD per species (all time origins, centre of mass removed) and the kinisi Li MSD.

Made by `mlipft ingest aimd` ([src/mlipft/ingest/aimd.py](../../../src/mlipft/ingest/aimd.py)).

## License of the source data

The source files are part of matgenb, distributed under the BSD 3-Clause License, Copyright (c) 2017,
Materials Virtual Lab. The values here are derived from them.
