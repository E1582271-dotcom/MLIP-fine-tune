"""Build the Li6PS5Cl cells: Ewald-ranked S/Cl orderings of mp-985592 and their supercells.

Thin entry point to mlipft.simulation.structure (port of the working repository's
01_build_structure.py without its figure). The base cell is read from ``--base-dir`` (the
committed cache data/structures/Li6PS5Cl_mp-985592.cif, so no network is needed); the cells are
written to ``--out-dir`` (default generation/output/structures/, ignored by git) so that the
committed files in data/structures/ are never overwritten. Compare with them, for example::

    python generation/structures/build_structure.py
    diff generation/output/structures/config0.cif data/structures/config0.cif

Commands that produced data/structures/ (pymatgen; CPU, seconds; run from the repository root):

    python generation/structures/build_structure.py                     # config0.cif, config1.cif (52 atoms)
    python generation/structures/build_structure.py --supercell 2,2,2   # config0_sc222.cif (416 atoms; production)

The second command also writes config1_sc222.cif, which no run used. The working repository
records that config0_sc222.cif was built locally with pymatgen as the 2x2x2 supercell of config0
but not the exact command. With pymatgen 2026.5 these commands rebuild all three committed CIFs
byte for byte.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from mlipft.simulation import structure as st

REPO = Path(__file__).resolve().parents[2]

# pymatgen warns when it serialises the disordered intermediate for hashing; written cells are ordered.
warnings.filterwarnings("ignore", message="Site labels are not unique")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(REPO / "data" / "structures"),
                    help=f"directory holding the cached {st.CACHE_CIF} (fetched from MP if absent)")
    ap.add_argument("--out-dir", default=str(REPO / "generation" / "output" / "structures"))
    ap.add_argument("--n-configs", type=int, default=2, help="number of distinct S/Cl orderings to keep")
    ap.add_argument("--supercell", default="1,1,1", help="diagonal supercell, e.g. 2,2,2 (52 -> 416 atoms)")
    ap.add_argument("--api-key", default=None, help="Materials Project API key (else $MP_API_KEY)")
    args = ap.parse_args()

    reps = tuple(int(x) for x in args.supercell.split(","))
    base = st.load_base(args.base_dir, api_key=args.api_key)
    base_summary = st.summary(base)
    print(f"base cell {st.MP_ID}: {base_summary['formula']}, {base_summary['n_atoms']} atoms, "
          f"a = {base_summary['abc'][0]} A ({base_summary['a_vs_expt_pct']:+} % vs experiment {st.EXPT_A} A)")

    configs, energies = st.enumerate_orderings(base, n_configs=args.n_configs)
    sc_tag = "" if reps == (1, 1, 1) else f"_sc{reps[0]}{reps[1]}{reps[2]}"
    records = []
    for j, cfg in enumerate(configs):
        cell = st.make_supercell(cfg, reps) if reps != (1, 1, 1) else cfg
        name = f"config{j}{sc_tag}.cif"
        st.write_structure(cell, os.path.join(args.out_dir, name))
        cell_summary = st.summary(cell)
        records.append({"config": j, "ewald_rank": j, "ewald_energy": round(float(energies[j]), 3), "cif": name,
                        **cell_summary})
        print(f"{name}: {cell_summary['n_atoms']} atoms, {cell_summary['n_Li']} Li, "
              f"Ewald energy {energies[j]:.3f} eV, density {cell_summary['density_g_cm3']} g/cm3")

    meta = {"source": f"Materials Project {st.MP_ID} (Li6PS5Cl, F-43m, ordered DFT approximant)",
            "supercell": list(reps), "expt_a_angstrom": st.EXPT_A, "configs": records}
    out = Path(args.out_dir) / f"structures{sc_tag}.json"
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} CIF(s) and {out.name} to {args.out_dir}")


if __name__ == "__main__":
    main()
