"""Step 3 -- sample decorrelated snapshots from MACE-MP-0 trajectories for DFT labelling.

Every ``--stride``-th frame per trajectory is kept after dropping the leading ``--skip-first``
fraction, pooled over the trajectories matched by ``--traj-glob`` (sorted by file name). The
output carries positions and cell only, plus ``source``, ``frame_in_traj`` and ``config_type``
in the info of each frame; energies and forces come from DFT in step 4.

Command that produced the 27 snapshots behind both label sets (data/dft/raw_inputs/
labelled_gamma.xyz and labelled_k222.xyz): the defaults, applied to the 50-ps 52-atom MACE-MP-0
runs at 600, 800 and 1000 K (1000 frames each, so frames 100, 200, ..., 900 of every run):

    python generation/labels/sample_snapshots.py      # -> generation/output/labels/snapshots.xyz

Ported from the working repository unchanged apart from paths: the trajectories are read from
generation/output/data/traj/ (generation/md/run_md.py) and the output goes to
generation/output/labels/ (ignored by git).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "generation" / "output"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-glob", default=str(OUTPUT / "data" / "traj" / "mace_*K.traj"),
                    help="trajectories to sample (default: the 50-ps 52-atom MACE-MP-0 runs)")
    ap.add_argument("--stride", type=int, default=100, help="keep every Nth production frame")
    ap.add_argument("--skip-first", type=float, default=0.1,
                    help="drop this leading fraction of each trajectory (residual equilibration)")
    ap.add_argument("--out", default=str(OUTPUT / "labels" / "snapshots.xyz"))
    args = ap.parse_args()

    from ase.io import read, write

    traj_files = sorted(glob.glob(args.traj_glob))
    if not traj_files:
        raise SystemExit(f"no trajectories matched {args.traj_glob} -- run generation/md/run_md.py first")

    pooled = []
    for tf in traj_files:
        frames = read(tf, index=":")
        n = len(frames)
        start = int(args.skip_first * n)
        picked = frames[start::args.stride]
        tag = os.path.basename(tf).replace(".traj", "")
        for i, at in enumerate(picked):
            at.info["source"] = tag
            at.info["frame_in_traj"] = start + i * args.stride
            at.info["config_type"] = "Li6PS5Cl_md_snapshot"
        pooled.extend(picked)
        print(f"[10] {tag}: {n} frames -> {len(picked)} sampled (stride {args.stride}, skip {args.skip_first})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write(args.out, pooled, format="extxyz")
    print(f"[10] wrote {len(pooled)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
