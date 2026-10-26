"""Re-analyse the three runs of one MD series (the cluster job of amendment A2).

    python3 ingest/run_series.py <repository> <raw backup> <series>

Writes <repository>/work/md/<run>.json and .npz, which ``mlipft ingest collect`` (with MLIPFT_WORK
pointing at <repository>/work) turns into data/md/. The main guard matters: the worker pool uses the
"spawn" start method, which re-imports this file in every worker.
"""
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # also for the spawned workers


def main() -> None:
    from mlipft.ingest import trajectories

    project, raw, series = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    packages = ("numpy", "ase", "pymatgen", "pymatgen-analysis-diffusion", "kinisi", "scipp", "scipy", "emcee")
    print(platform.platform(), [(name, metadata.version(name)) for name in packages], flush=True)
    done = trajectories.analyse_all(raw, project / "work", workers=3, only=[series])
    print("runs", len(done), flush=True)


if __name__ == "__main__":
    main()
