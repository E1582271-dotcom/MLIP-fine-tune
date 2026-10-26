"""NVT Langevin MD of a Li6PS5Cl cell at several temperatures (one trajectory per potential and T).

Thin entry point to mlipft.simulation.md (port of the working repository's 02_baseline_md.py
without its figure). For each temperature it equilibrates for ``--equilib`` steps, then records
``--steps`` production steps with one frame every ``--log-every`` steps (1 fs steps, friction
0.01 1/fs, NVT at the fixed input cell, velocity seed 0). Output, ignored by git:

    generation/output/data/traj{tag}/<mlip>_<T>K.traj   production trajectory (input of mlipft ingest trajectories)
    generation/output/data/traj{tag}/<mlip>_<T>K.json   run record: settings, thermostat statistics, energy
                                                        drift and the per-frame temperature/energy series

Below generation/output/ the layout is that of the July raw backup (data/traj{tag}/<mlip>_<T>K.traj).

Commands that produced the July trajectories (on a GPU, run from the repository root;
generation/hpc/run_in_container.sh runs exactly these through generation/hpc/run_md.pbs, adding
``--device cuda``; generation/README.md gives the qsub command of every series). $T is one of
600, 800, 1000 where the runs were submitted as one job per temperature:

    # 52 atoms (config0.cif), 50 ps after 5 ps -> traj/
    python generation/md/run_md.py --mlip mace --temps 600,800,1000 --steps 50000 --equilib 5000
    python generation/md/run_md.py --mlip mattersim --temps 600,800,1000 --steps 50000 --equilib 5000

    # 52 atoms, 100 ps after 5 ps -> traj_100ps/ (MatterSim likewise)
    python generation/md/run_md.py --mlip mace --temps 600,800,1000 --steps 100000 --equilib 5000 --traj-tag _100ps

    # 52 atoms, 150 ps after 5 ps -> traj_long/
    python generation/md/run_md.py --mlip mace --temps 600,800,1000 --steps 150000 --equilib 5000 --traj-tag _long
    python generation/md/run_md.py --mlip mattersim --temps $T --steps 150000 --equilib 5000 --traj-tag _long

    # 416 atoms (config0_sc222.cif), 200 ps after 10 ps -> traj_prod/, traj_prod_ms/
    python generation/md/run_md.py --mlip mace --temps $T --steps 200000 --equilib 10000 \\
        --supercell-tag _sc222 --traj-tag _prod
    python generation/md/run_md.py --mlip mattersim --temps $T --steps 200000 --equilib 10000 \\
        --supercell-tag _sc222 --traj-tag _prod_ms

    # the fine-tuned models (generation/README.md, step 6), same protocol -> traj_ft/, traj_ft27k/
    python generation/md/run_md.py --mlip mace \\
        --mace-model generation/output/finetune/models/li6ps5cl_ft_stagetwo.model \\
        --temps $T --steps 200000 --equilib 10000 --supercell-tag _sc222 --traj-tag _ft
    python generation/md/run_md.py --mlip mace \\
        --mace-model generation/output/finetune/models/li6ps5cl_ft27k.model \\
        --temps $T --steps 200000 --equilib 10000 --supercell-tag _sc222 --traj-tag _ft27k

    # Cl-excess cells (build_doped.py), MACE-MP-0 small, 150 ps after 10 ps -> traj_dope_cl100/ ... traj_dope_cl175/
    python generation/md/run_md.py --mlip mace --cif data/structures/doped_cl125_sc222.cif \\
        --temps $T --steps 150000 --equilib 10000 --traj-tag _dope_cl125

All runs used the defaults --timestep 1.0 --friction 0.01 --log-every 50 and MACE-MP-0 "small"
(float32) or the default MatterSim model (v1.0.0-1M). The equilibration length is not in the run
records of the working repository; the values above are the ones printed in the July job log of
every run. The Gamma-label fine-tuned model file is the stage-two (SWA) model named in the
working repository's report (a comment written there before the training names li6ps5cl_ft.model,
the stage-one model); the July command of the 2x2x2-label runs named li6ps5cl_ft27k.model, the
stage-one model of that fine-tune. Which model drove each fine-tuned trajectory is established
from the trajectories themselves (protocol/analysis_plan.md, "Fingerprints").

Changes from the working repository: ``--friction`` is exposed (the value is the one run_nvt used
by default); structures are read from data/structures/; one run record per trajectory is written
next to it instead of a shared data/md_runs{tag}.json (which per-temperature jobs overwrote).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mlipft.simulation import md

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlip", default="both", choices=["mace", "mattersim", "both", "lj"],
                    help="'lj' = Lennard-Jones test calculator (non-physical; CPU smoke tests only)")
    ap.add_argument("--temps", default="600,800,1000", help="comma-separated temperatures (K)")
    ap.add_argument("--steps", type=int, default=50000, help="production MD steps")
    ap.add_argument("--equilib", type=int, default=5000, help="equilibration steps before recording")
    ap.add_argument("--timestep", type=float, default=1.0, help="fs")
    ap.add_argument("--friction", type=float, default=0.01, help="Langevin friction (1/fs)")
    ap.add_argument("--log-every", type=int, default=50, help="steps between recorded frames")
    ap.add_argument("--config", type=int, default=0, help="which config{N}.cif of --structure-dir")
    ap.add_argument("--supercell-tag", default="", help="e.g. _sc222 for config0_sc222.cif")
    ap.add_argument("--cif", default=None,
                    help="explicit structure file (relative paths: repository root); overrides --config")
    ap.add_argument("--structure-dir", default=str(REPO / "data" / "structures"))
    ap.add_argument("--mace-model", default="small", help="'small'|'medium'|'large' or a fine-tuned .model path")
    ap.add_argument("--mattersim-model", default=None, help="MatterSim checkpoint path (default model if unset)")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    ap.add_argument("--data-dir", default=str(REPO / "generation" / "output" / "data"),
                    help="trajectories go to <data-dir>/traj<tag>/")
    ap.add_argument("--traj-tag", default="", help="suffix of the trajectory directory, e.g. _prod")
    args = ap.parse_args()

    from ase.io import read

    temps = [float(t) for t in args.temps.split(",")]
    mlips = ["mace", "mattersim"] if args.mlip == "both" else [args.mlip]
    if args.cif:
        cif = Path(args.cif) if Path(args.cif).is_absolute() else REPO / args.cif
    else:
        cif = Path(args.structure_dir) / f"config{args.config}{args.supercell_tag}.cif"
    if not cif.is_file():
        sys.exit(f"missing structure {cif} -- see generation/structures/build_structure.py / build_doped.py")
    traj_dir = Path(args.data_dir) / f"traj{args.traj_tag}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    atoms0 = read(cif)
    print(f"cell {cif.name}: {atoms0.get_chemical_formula()}, {len(atoms0)} atoms, a = {atoms0.cell.lengths()[0]:.3f} A")
    print(f"{args.steps * args.timestep / 1000:g} ps production after {args.equilib * args.timestep / 1000:g} ps "
          f"equilibration; MLIPs {mlips}; T = {temps}")

    for mlip in mlips:
        calc = md.load_calculator(mlip, device=args.device, mace_model=args.mace_model,
                                  mattersim_model=args.mattersim_model)
        for T in temps:
            traj_path = traj_dir / f"{mlip}_{int(T)}K.traj"
            summary = md.run_nvt(atoms0, calc, temperature_K=T, n_steps=args.steps, timestep_fs=args.timestep,
                                 friction=args.friction, equilib_steps=args.equilib, traj_path=str(traj_path),
                                 log_every=args.log_every)
            summary["traj_path"] = os.path.relpath(traj_path, REPO) if traj_path.is_relative_to(REPO) else str(traj_path)
            record = {"mlip": mlip, "cif": cif.name, "mace_model": args.mace_model if mlip == "mace" else None,
                      "mattersim_model": args.mattersim_model if mlip == "mattersim" else None,
                      "equilib_steps": args.equilib, "friction_per_fs": args.friction, **summary}
            traj_path.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
            print(f"{mlip} {int(T)} K: mean T {summary['mean_T']} K, E/atom {summary['mean_E_per_atom']} eV, "
                  f"drift {summary['drift_meV_atom_ps']} meV/atom/ps, {summary['n_frames']} log samples, "
                  f"{summary['wall_seconds']} s -> {traj_path.name}")


if __name__ == "__main__":
    main()
