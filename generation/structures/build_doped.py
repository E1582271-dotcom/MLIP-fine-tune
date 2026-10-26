"""Build the Cl-excess series Li(6-x)PS(5-x)Cl(1+x), x = 0, 0.25, 0.5, 0.75, as 2x2x2 supercells.

Thin entry point to mlipft.simulation.structure (port of the working repository's
05_build_doped.py). On the free-anion sublattice of mp-985592, S2- is replaced by Cl- and the
charge is compensated by Li vacancies; the lowest-Ewald ordered approximant is kept and repeated
2x2x2. Per conventional cell (Z = 4, 8 free-anion sites, 24 Li sites):

    x = 0.00  Li6PS5Cl            4 free S, 4 Cl, 24 Li   -> 416 atoms in 2x2x2
    x = 0.25  Li5.75PS4.75Cl1.25  3 free S, 5 Cl, 23 Li   -> 408 atoms
    x = 0.50  Li5.5PS4.5Cl1.5     2 free S, 6 Cl, 22 Li   -> 400 atoms
    x = 0.75  Li5.25PS4.25Cl1.75  1 free S, 7 Cl, 21 Li   -> 392 atoms

Command that produced data/structures/doped_cl{100,125,150,175}_sc222.cif (pymatgen; CPU,
seconds; run from the repository root):

    python generation/structures/build_doped.py

The base cell is read from ``--base-dir`` (the committed cache, no network needed); the cells
go to ``--out-dir`` (default generation/output/structures/, ignored by git) as
doped_<tag>_sc222.cif plus doped.json. The working repository named the same files
data/doped/dope_<tag>.cif. With pymatgen 2026.5 this command rebuilds the four committed CIFs
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

warnings.filterwarnings("ignore", message="Site labels are not unique")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(REPO / "data" / "structures"),
                    help=f"directory holding the cached {st.CACHE_CIF} (fetched from MP if absent)")
    ap.add_argument("--out-dir", default=str(REPO / "generation" / "output" / "structures"))
    ap.add_argument("--x", type=float, default=None, help="a single composition (default: the full series)")
    ap.add_argument("--supercell", default="2,2,2", help="diagonal repetitions of the 52-atom cell")
    ap.add_argument("--api-key", default=None, help="Materials Project API key (else $MP_API_KEY)")
    args = ap.parse_args()

    xs = [args.x] if args.x is not None else list(st.CL_EXCESS_X)
    reps = tuple(int(v) for v in args.supercell.split(","))
    sc_tag = "" if reps == (1, 1, 1) else f"_sc{reps[0]}{reps[1]}{reps[2]}"
    base = st.load_base(args.base_dir, api_key=args.api_key)

    manifest = {
        "source": f"Materials Project {st.MP_ID} (Li6PS5Cl, F-43m, ordered DFT approximant)",
        "method": "Cl-excess Li6-x PS5-x Cl1+x: free-anion S2- -> Cl- substitution with Li-vacancy charge "
                  "compensation; lowest-Ewald ordered approximant; diagonal supercell.",
        "supercell": list(reps),
        "compositions": {},
    }
    for x in xs:
        s, energy, counts = st.enumerate_doped(base, x)
        cell = st.make_supercell(s, reps) if reps != (1, 1, 1) else s
        cell_summary = st.summary(cell)
        tag = st.cl_tag(counts["cl_content"])
        name = f"doped_{tag}{sc_tag}.cif"
        st.write_structure(cell, os.path.join(args.out_dir, name))
        manifest["compositions"][tag] = {
            "x": x, "formula": cell_summary["formula"], "cl_content": counts["cl_content"],
            "conv_counts": counts, "ewald_energy_eV": round(energy, 3), "unit_cell_atoms": len(s),
            "supercell": list(reps), "n_atoms": cell_summary["n_atoms"], "n_Li": cell_summary["n_Li"],
            "abc_A": cell_summary["abc"], "volume_A3": cell_summary["volume"],
            "density_g_cm3": cell_summary["density_g_cm3"], "cif": name,
        }
        print(f"x={x:<4} {cell_summary['formula']:<16} Cl={counts['cl_content']:<4} {cell_summary['n_atoms']} atoms "
              f"({cell_summary['n_Li']} Li), Ewald energy {energy:.3f} eV -> {name}")

    out = Path(args.out_dir) / "doped.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest['compositions'])} cells and {out.name} to {args.out_dir}")


if __name__ == "__main__":
    main()
