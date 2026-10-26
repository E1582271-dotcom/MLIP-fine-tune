"""Step 4 -- DFT-label the sampled snapshots with Quantum ESPRESSO (pw.x SCF): energy, forces, stress.

    generation/output/labels/snapshots.xyz  --(this)-->  generation/output/labels/labelled.xyz

Per snapshot: strip any placeholder labels, write a pw.x SCF input with ASE, run pw.x, parse the
output and append a frame carrying ``info['dft_energy']`` (eV), ``arrays['dft_forces']`` (eV/A),
``info['dft_stress']`` and ``info['labelled_by']`` -- the keys finetune_mace.sh reads.

SCF settings (the defaults are the values that produced data/dft/raw_inputs/labelled_gamma.xyz;
labelled_k222.xyz was produced with ``--kpts 2x2x2`` and otherwise the same settings):
  * PBE pseudopotentials found in ``--pseudo-dir`` by element prefix (fetch_pseudos.sh);
  * ecutwfc 60 Ry, ecutrho 480 Ry; Gaussian smearing, degauss 0.01 Ry;
  * conv_thr 1e-7 Ry, mixing_beta 0.4; tprnfor and tstress on; disk_io 'none';
  * Gamma point only with real wavefunctions (``--kpts gamma``; use ``2,2,2`` etc. for a mesh).

Resumable: appends to ``--out`` and skips the configurations already in it. A failed SCF is
logged to labelling.log next to ``--out`` and skipped. ``--start``/``--limit`` select a shard
(submit_shards.sh runs one snapshot per job). pw.x is invoked as ``--pw-cmd`` or $QE_PW_CMD
(default 'pw.x'; label_qe.pbs sets 'mpirun -np N pw.x -npool M'). Needs ase; CPU only.

Changes from the working repository: the defaults of ``--kpts`` (was 2,2,2) and ``--conv-thr``
(was 1e-8) are now the values submit_shards.sh passed for the Gamma labels (gamma, 1e-7);
the automatic cutoffs read from pseudo/sssp.json were removed (both label sets used the
fixed 60/480 Ry, consistent with the 135^3 FFT grid recorded for these runs); paths point to
generation/output/labels/ and generation/output/pseudo/ (both ignored by git).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "generation" / "output"
ELEMENTS = ["Li", "P", "S", "Cl"]
# fields that must not survive from the MD frames, so DFT labels never mix with placeholders
_STALE_INFO = ("energy", "free_energy", "stress", "dft_energy", "dft_stress")
_STALE_ARR = ("forces", "momenta", "energies")


def discover_pseudos(pseudo_dir: str) -> dict[str, str]:
    """Map each element to its UPF file in ``pseudo_dir`` (element-prefixed, case-insensitive)."""
    if not os.path.isdir(pseudo_dir):
        raise SystemExit(f"[11] no pseudo dir {pseudo_dir} -- run fetch_pseudos.sh first")
    upfs = [f for f in os.listdir(pseudo_dir) if f.lower().endswith(".upf")]
    pp = {}
    for el in ELEMENTS:
        cands = [f for f in upfs
                 if f.split(".")[0].split("_")[0].lower() == el.lower()] or \
                [f for f in upfs if f.lower().startswith(el.lower())]
        if not cands:
            raise SystemExit(f"[11] no UPF for {el} in {pseudo_dir} (have: {sorted(upfs)})")
        pp[el] = sorted(cands, key=len)[0]
    return pp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", default=str(OUTPUT / "labels" / "snapshots.xyz"))
    ap.add_argument("--out", default=str(OUTPUT / "labels" / "labelled.xyz"))
    ap.add_argument("--pseudo-dir", default=str(OUTPUT / "pseudo"))
    ap.add_argument("--pw-cmd", default=os.environ.get("QE_PW_CMD", "pw.x"),
                    help="pw.x invocation; label_qe.pbs sets $QE_PW_CMD='mpirun -np N pw.x -npool M'")
    ap.add_argument("--ecutwfc", type=float, default=60.0, help="Ry")
    ap.add_argument("--ecutrho", type=float, default=480.0, help="Ry")
    ap.add_argument("--kpts", default="gamma",
                    help="'gamma' for the Gamma point with real wavefunctions, or a Monkhorst-Pack mesh "
                         "such as 2,2,2 or 2x2x2 (the 'x' form survives qsub -v, which splits on commas)")
    ap.add_argument("--degauss", type=float, default=0.01, help="Ry")
    ap.add_argument("--conv-thr", type=float, default=1.0e-7, help="Ry")
    ap.add_argument("--workdir", default=str(OUTPUT / "labels" / "qe_work"))
    ap.add_argument("--limit", type=int, default=0, help="process only N configs from --start (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="first config index; shard the set over jobs with disjoint --start/--limit "
                         "(each with its own --workdir and --out)")
    args = ap.parse_args()

    from ase.io import read, write
    from ase.io.espresso import read_espresso_out, write_espresso_in

    frames = read(args.snapshots, index=":")
    start = max(0, args.start)
    end = min(start + args.limit, len(frames)) if args.limit else len(frames)
    pp = discover_pseudos(args.pseudo_dir)
    # kpts=None -> ASE writes 'K_POINTS gamma' (real wavefunctions); a single k-point cannot be pooled
    if args.kpts.strip().lower() in ("gamma", "g"):
        kpts, kpts_desc = None, "gamma (real WF)"
    else:
        kpts = tuple(int(x) for x in args.kpts.lower().replace("x", ",").split(","))
        kpts_desc = str(kpts)
    print(f"[11] {len(frames)} snapshots | shard [{start},{end}) | pseudos {pp} | "
          f"ecutwfc={args.ecutwfc} ecutrho={args.ecutrho} | kpts={kpts_desc} | pw='{args.pw_cmd}'")

    # resume: this shard's output holds configs start, start+1, ...; skip those already present
    done = 0
    if os.path.exists(args.out):
        done = len(read(args.out, index=":"))
        print(f"[11] resume: {done} already labelled in {args.out}")

    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": True, "disk_io": "none",
                    "prefix": "li6ps5cl", "pseudo_dir": os.path.abspath(args.pseudo_dir),
                    "outdir": os.path.abspath(args.workdir)},
        "system": {"ecutwfc": args.ecutwfc, "ecutrho": args.ecutrho, "occupations": "smearing",
                   "smearing": "gaussian", "degauss": args.degauss},
        "electrons": {"conv_thr": args.conv_thr, "mixing_beta": 0.4},
    }

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ok = fail = 0
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "labelling.log"), "a") as logf:
        for i, atoms in enumerate(frames):
            if i < start or i >= end:
                continue
            if i < start + done:          # already labelled in an earlier run of this shard
                continue
            atoms.calc = None
            for k in _STALE_INFO:
                atoms.info.pop(k, None)
            for k in _STALE_ARR:
                atoms.arrays.pop(k, None)
            inp = os.path.join(args.workdir, f"cfg{i:04d}.in")
            out = os.path.join(args.workdir, f"cfg{i:04d}.out")
            with open(inp, "w") as fd:
                write_espresso_in(fd, atoms, input_data=input_data, pseudopotentials=pp, kpts=kpts)
            try:
                with open(out, "w") as o:
                    subprocess.run(f"{args.pw_cmd} -in {inp}", shell=True, check=True,
                                   stdout=o, stderr=subprocess.STDOUT)
                with open(out) as fh:
                    res = list(read_espresso_out(fh))[-1]
                E, F = res.get_potential_energy(), res.get_forces()
                atoms.info["dft_energy"] = float(E)
                atoms.arrays["dft_forces"] = F
                try:
                    atoms.info["dft_stress"] = " ".join(f"{x:.8e}" for x in res.get_stress())
                except Exception:
                    pass
                # fixed string of the working repository, not a record of the UPF files used
                atoms.info["labelled_by"] = "QE-pw.x-PBE-SSSP"
                write(args.out, atoms, format="extxyz", append=True)
                ok += 1
                print(f"[11] cfg{i:04d} OK  E={E:.4f} eV  |F|max={abs(F).max():.3f} eV/A")
            except Exception as e:
                fail += 1
                logf.write(f"cfg{i:04d} FAILED: {e}\n")
                logf.flush()
                print(f"[11] cfg{i:04d} FAILED ({e}) -- logged, continuing")
    print(f"[11] done: +{ok} labelled, {fail} failed -> {args.out}")


if __name__ == "__main__":
    main()
