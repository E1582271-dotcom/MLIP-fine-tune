"""Step 5 -- split the DFT-labelled snapshots into the training and validation sets of mace_run_train.

    data/dft/raw_inputs/labelled_gamma.xyz  --(this)-->  generation/output/labels/train.xyz + valid.xyz

The configuration indices are shuffled with ``random.Random(seed)``; the first
``round(frac_valid * n)`` (at least 1) are the validation set, in shuffled order, and the rest
the training set, in file order. The DFT keys are carried through unchanged.

Commands that produced data/dft/training_sets/ (run from the repository root; compare the
outputs with ``cmp``):

    # ftG_train.xyz (22) and ftG_valid.xyz (5) from the 27 Gamma-point labels
    python generation/labels/split_train_valid.py --seed 0 --frac-valid 0.2
    # ftK_train.xyz (24) and ftK_valid.xyz (3) from the 27 2x2x2 k-point labels
    python generation/labels/split_train_valid.py --labelled data/dft/raw_inputs/labelled_k222.xyz \\
        --seed 0 --frac-valid 0.1 --out-dir generation/output/labels/k222

The split depends on the order of the labelled file; both label files keep the order in which
the 27 one-snapshot shards were concatenated (generation/README.md, step 4). Both have the same
order, so the 3 validation configurations of the 2x2x2 split are the first 3 of the 5 of the
Gamma split.

Changes from the working repository: the default ``--frac-valid`` is 0.2 (was 0.1, which gives
3 of 27: the 2x2x2 split was made with that default; the Gamma split has 5, which any value in
[0.167, 0.203] gives), the ``--tag`` option is not kept (use ``--out-dir``), the default input is
the published Gamma labels and the output goes to generation/output/labels/ (ignored by git).
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", default=str(REPO / "data" / "dft" / "raw_inputs" / "labelled_gamma.xyz"))
    ap.add_argument("--out-dir", default=str(REPO / "generation" / "output" / "labels"))
    ap.add_argument("--frac-valid", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ase.io import read, write

    frames = read(args.labelled, index=":")
    n = len(frames)
    if n < 4:
        raise SystemExit(f"[12] only {n} labelled configs -- label more before splitting")
    missing = [i for i, a in enumerate(frames) if "dft_energy" not in a.info]
    if missing:
        raise SystemExit(f"[12] {len(missing)} configs lack dft_energy (e.g. idx {missing[:5]})")

    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(round(args.frac_valid * n)))
    val_idx = set(idx[:n_val])
    train = [frames[i] for i in range(n) if i not in val_idx]
    valid = [frames[i] for i in idx[:n_val]]

    os.makedirs(args.out_dir, exist_ok=True)
    tp = os.path.join(args.out_dir, "train.xyz")
    vp = os.path.join(args.out_dir, "valid.xyz")
    write(tp, train, format="extxyz")
    write(vp, valid, format="extxyz")
    print(f"[12] {n} labelled -> {len(train)} train + {len(valid)} valid (seed {args.seed})")
    print(f"[12] wrote {tp}, {vp}")


if __name__ == "__main__":
    main()
