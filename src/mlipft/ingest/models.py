"""MACE models: identity, reproduction of the training logs, predictions on the DFT labels, and the
fingerprints that show which model drove each fine-tuned MD run.

Reads the raw backup (``foundation/``, ``finetune/{models,checkpoints,logs,results}``,
``data/traj_ft*``, ``data/traj_prod``) and writes ``data/models/``:

* ``registry.csv``: every model file with its sha256 and the model it belongs to; files that give
  identical predictions are one model. Models: ``foundation`` (MACE-MP-0 small) and, for the
  fine-tunes on Gamma (``ftG``) and on 2x2x2 (``ftK``) labels, stage one (``_s1``, best epoch
  before stochastic weight averaging) and stage two (``_s2``). Stages are assigned by reproducing
  the error tables of the training logs (``protocol/reproduction_rule.md`` item 8).
* ``log_reproduction.csv``: logged against re-evaluated errors on each model's own sets.
* ``predictions_energy.csv`` (energy and pressure per model and configuration) and
  ``predictions_forces.csv.gz`` (DFT forces of both label sets and the forces of every model).
* ``fingerprints.csv``: RMS difference between the forces stored in the MD trajectories and the
  forces of each candidate model on the same frames, with the assignment rule of the analysis plan.
* ``training/``: the MACE training records (JSON lines) and a summary of each run's settings.

Predictions on the labels use float64 on a CPU; fingerprints use float32, the precision of the MD.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np

from .. import io, paths
from ..constants import EV_A3_TO_GPA

FOUNDATION_FILE = "foundation/20231210mace128L0_energy_epoch249model"
FINETUNE_FILES = {
    "ftG": ("finetune/models/li6ps5cl_ft.model", "finetune/models/li6ps5cl_ft_stagetwo.model",
            "finetune/checkpoints/li6ps5cl_ft_run-0.model", "finetune/checkpoints/li6ps5cl_ft_run-0_stagetwo.model"),
    "ftK": ("finetune/models/li6ps5cl_ft27k.model", "finetune/models/li6ps5cl_ft27k_stagetwo.model",
            "finetune/checkpoints/li6ps5cl_ft27k_run-0.model",
            "finetune/checkpoints/li6ps5cl_ft27k_run-0_stagetwo.model"),
}
COMPILED_FILES = ("finetune/models/li6ps5cl_ft_compiled.model", "finetune/models/li6ps5cl_ft_stagetwo_compiled.model",
                  "finetune/models/li6ps5cl_ft27k_compiled.model",
                  "finetune/models/li6ps5cl_ft27k_stagetwo_compiled.model")
LOGS = {"ftG": ("finetune/logs/li6ps5cl_ft_run-0.log", "finetune/results/li6ps5cl_ft_run-0_train.txt"),
        "ftK": ("finetune/logs/li6ps5cl_ft27k_run-0.log", "finetune/results/li6ps5cl_ft27k_run-0_train.txt")}
OWN_LABELS = {"ftG": "gamma", "ftK": "k222"}
FINGERPRINT_RUNS = (("ftG_416_200", "data/traj_ft"), ("ftK_416_200", "data/traj_ft27k"),
                    ("mace_416_200", "data/traj_prod"))
FINGERPRINT_MATCH_meV_A = 1.0
FINGERPRINT_RATIO = 10.0
IDENTICAL_eV = 1e-8
EV_A3_TO_KBAR = EV_A3_TO_GPA * 10.0


def _calculator(path: Path, dtype: str):
    from mace.calculators import MACECalculator

    kwargs = {"head": "Default"} if "finetune" in str(path) else {}
    return MACECalculator(model_paths=str(path), device="cpu", default_dtype=dtype, **kwargs)


def _predict(calc, frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ase import Atoms

    energies, forces, pressures = [], [], []
    for frame in frames:
        atoms = Atoms(symbols=frame["species"], positions=frame["pos"], cell=frame["lattice"], pbc=True)
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        pressures.append(-atoms.get_stress(voigt=True)[:3].mean() * EV_A3_TO_KBAR)
    return np.array(energies), np.array(forces), np.array(pressures)


def _rmse_meV(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - reference) ** 2)) * 1000.0)


def _train_check(squared: np.ndarray, logged: float, batch_size: int, n_atoms: int) -> dict:
    """Compare a logged training error with the re-evaluation (amendment A1).

    MACE evaluates the training set with its training loader, which drops the last incomplete
    batch. If the set is not a multiple of the batch size, the logged value is checked against
    every subset of the evaluated size."""
    from itertools import combinations

    n = len(squared)
    evaluated = n // batch_size * batch_size
    if evaluated == n:
        value = float(np.sqrt(squared.sum() / (n * n_atoms * 3)) * 1000.0)
        return {"train_evaluated_by_log": n, "train_subsets_matching": "", "train_check": round(value, 1) == logged}
    hits = total = 0
    for subset in combinations(range(n), evaluated):
        total += 1
        hits += round(float(np.sqrt(squared[list(subset)].sum() / (evaluated * n_atoms * 3)) * 1000.0), 1) == logged
    return {"train_evaluated_by_log": evaluated, "train_subsets_matching": f"{hits}/{total}", "train_check": hits > 0}


def _error_tables(log: str) -> dict:
    """Stage one and stage two train/valid tables of a MACE log."""
    tables = {}
    for stage, marker in (("s1", "Loaded Stage one model"), ("s2", "Loaded Stage two model")):
        block = log[log.index(marker):]
        epoch = int(re.search(r"from epoch (\d+)", block)[1])
        row = {"epoch": epoch}
        for split in ("train", "valid"):
            match = re.search(rf"\|\s*{split}_Default\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", block)
            row[f"{split}_rmse_e_meV_atom"], row[f"{split}_rmse_f_meV_A"] = float(match[1]), float(match[2])
        tables[stage] = row
    initial = re.search(r"Initial: head: Default, loss=[\d.]+, RMSE_E_per_atom=\s*([\d.]+) meV, RMSE_F=\s*([\d.]+)",
                        log)
    tables["foundation_valid_rmse_f_meV_A"] = float(initial[2])
    return tables


def training_summary(log: str) -> dict:
    e0 = re.search(r"Atomic Energies used \(z: eV\) for head Default: (\{.*?\})", log)[1]
    stage_two = re.search(r"Stage Two \(after (\d+) epochs\) with loss function: .*?energy_weight=([\d.]+), "
                          r"forces_weight=([\d.]+)\).*?learning rate : ([\d.]+)", log)
    tables = _error_tables(log)
    return {
        "mace_version": re.search(r"MACE version: ([\d.]+)", log)[1],
        "foundation_file": Path(re.search(r"Using foundation model (\S+) as", log)[1]).name,
        "n_train": int(re.search(r"Total number of configurations: train=(\d+)", log)[1]),
        "n_valid": int(re.search(r"Total number of configurations: train=\d+, valid=(\d+)", log)[1]),
        "atomic_energies_eV": {str(k): v for k, v in ast.literal_eval(e0).items()},
        "hidden_irreps": re.search(r"hidden irreps (\S+?)\)", log)[1],
        "interaction_layers": int(re.search(r"(\d+) layers, each with correlation order", log)[1]),
        "correlation_order": int(re.search(r"correlation order: (\d+)", log)[1]),
        "r_cut_A": float(re.search(r"Radial cutoff: ([\d.]+) A", log)[1]),
        "optimizer": re.search(r"Using (\w+) as parameter optimizer", log)[1],
        "batch_size": int(re.search(r"Batch size: (\d+)", log)[1]),
        "learning_rate": float(re.search(r"Learning rate: ([\d.]+), weight decay", log)[1]),
        "weight_decay": float(re.search(r"weight decay: ([\d.e-]+)", log)[1]),
        "stage_one_loss_weights": [float(v) for v in
                                   re.search(r"WeightedEnergyForcesLoss\(energy_weight=([\d.]+), forces_weight=([\d.]+)\)",
                                             log).groups()],
        "stage_two_start_epoch": int(stage_two[1]),
        "stage_two_loss_weights": [float(stage_two[2]), float(stage_two[3])],
        "stage_two_learning_rate": float(stage_two[4]),
        "error_tables": tables,
    }


def ingest(raw: Path | None = None, out: Path | None = None) -> dict:
    import ase
    import mace
    import torch

    raw = Path(raw or paths.raw())
    out = Path(out or paths.MODELS)
    out.mkdir(parents=True, exist_ok=True)
    configs = {row["config_id"]: row for row in _read_csv(paths.DFT / "configs.csv")}
    order = sorted(configs, key=lambda c: int(configs[c]["snapshot_index"]))
    labels = {}
    for name in ("gamma", "k222"):
        frames = io.read_extxyz(paths.DFT / "raw_inputs" / f"labelled_{name}.xyz")
        labels[name] = [frames[int(configs[c][f"index_{name}"])] for c in order]
    frames = labels["gamma"]

    # predictions of every candidate file (float64)
    files = {"foundation": [FOUNDATION_FILE], **{family: list(names) for family, names in FINETUNE_FILES.items()}}
    predictions, registry = {}, []
    for family, names in files.items():
        groups: list[tuple[str, tuple]] = []
        for name in names:
            result = _predict(_calculator(raw / name, "float64"), frames)
            match = next((key for key, value in groups
                          if all(np.abs(a - b).max() < IDENTICAL_eV for a, b in zip(value, result))), None)
            if match is None:
                match = f"{family}_candidate{len(groups)}"
                groups.append((match, result))
            registry.append({"file": name, "sha256": io.sha256_file(raw / name), "group": match})
        for key, value in groups:
            predictions[key] = value

    # stage assignment by the training logs
    summaries, reproduction, rename = {}, [], {"foundation_candidate0": "foundation"}
    for family, (log_name, record_name) in LOGS.items():
        log = (raw / log_name).read_text(encoding="utf-8")
        summaries[family] = training_summary(log)
        own = OWN_LABELS[family]
        split_column = f"{family}_split"
        for key, (_, forces, _) in predictions.items():
            if not key.startswith(family):
                continue
            errors, squared = {}, {}
            for split in ("train", "valid"):
                index = [i for i, c in enumerate(order) if configs[c][split_column] == split]
                reference = np.array([labels[own][i]["dft_forces"] for i in index])
                errors[split] = _rmse_meV(forces[index], reference)
                squared[split] = ((forces[index] - reference) ** 2).sum(axis=(1, 2))
            for stage in ("s1", "s2"):
                logged = summaries[family]["error_tables"][stage]
                if round(errors["valid"], 1) != logged["valid_rmse_f_meV_A"]:
                    continue
                rename[key] = f"{family}_{stage}"
                train_check = _train_check(squared["train"], logged["train_rmse_f_meV_A"],
                                           summaries[family]["batch_size"], forces.shape[1])
                reproduction.append({"model": f"{family}_{stage}", "labels": own, "epoch": logged["epoch"],
                                     "logged_train_rmse_f_meV_A": logged["train_rmse_f_meV_A"],
                                     "reevaluated_train_rmse_f_meV_A": errors["train"],
                                     "logged_valid_rmse_f_meV_A": logged["valid_rmse_f_meV_A"],
                                     "reevaluated_valid_rmse_f_meV_A": errors["valid"], **train_check})
        (out / "training").mkdir(exist_ok=True)
        (out / "training" / f"{family}_record.jsonl").write_bytes((raw / record_name).read_bytes())
    # the foundation on the Gamma validation set, as logged before training
    index = [i for i, c in enumerate(order) if configs[c]["ftG_split"] == "valid"]
    found_valid = _rmse_meV(predictions["foundation_candidate0"][1][index],
                            np.array([labels["gamma"][i]["dft_forces"] for i in index]))
    reproduction.append({"model": "foundation", "labels": "gamma",
                         "logged_valid_rmse_f_meV_A": summaries["ftG"]["error_tables"]["foundation_valid_rmse_f_meV_A"],
                         "reevaluated_valid_rmse_f_meV_A": found_valid})
    unassigned = [key for key in predictions if key not in rename]
    assert not unassigned, f"files not matched to a logged model: {unassigned}"
    predictions = {rename[key]: value for key, value in predictions.items()}
    for row in registry:
        row["model"] = rename[row.pop("group")]
    for name in COMPILED_FILES:
        registry.append({"file": name, "sha256": io.sha256_file(raw / name), "model": "",
                         "note": "TorchScript export of a model above; not loaded"})
    model_ids = ["foundation", "ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2"]

    # fingerprints (float32, the precision of the MD)
    from ase.io.trajectory import Trajectory

    fingerprints = []
    calculators = {model: _calculator(raw / next(r["file"] for r in registry if r["model"] == model), "float32")
                   for model in model_ids}
    for series, folder in FINGERPRINT_RUNS:
        for temperature in (600, 800, 1000):
            trajectory = Trajectory(str(raw / folder / f"mace_{temperature}K.traj"))
            n = len(trajectory)
            chosen = [trajectory[i] for i in (0, n // 3, 2 * n // 3, n - 1)]
            residual = {}
            for model, calc in calculators.items():
                squared = []
                for atoms in chosen:
                    stored = atoms.get_forces()
                    copy = atoms.copy()
                    copy.calc = calc
                    squared.append(np.mean((copy.get_forces() - stored) ** 2))
                residual[model] = float(np.sqrt(np.mean(squared)) * 1000.0)
            ranked = sorted(residual, key=residual.get)
            best, second = ranked[0], ranked[1]
            assigned = (best if residual[best] < FINGERPRINT_MATCH_meV_A
                        and residual[second] >= FINGERPRINT_RATIO * residual[best] else "unassigned")
            fingerprints.append({"series": series, "temperature_K": temperature, "frames": "0;n/3;2n/3;n-1",
                                 **{f"rms_{m}_meV_A": residual[m] for m in model_ids}, "assigned_model": assigned})

    # tables
    io.write_csv(out / "registry.csv", ("model", "file", "sha256", "note"), registry)
    io.write_csv(out / "log_reproduction.csv", ("model", "labels", "epoch", "logged_valid_rmse_f_meV_A",
                                                "reevaluated_valid_rmse_f_meV_A", "logged_train_rmse_f_meV_A",
                                                "reevaluated_train_rmse_f_meV_A", "train_evaluated_by_log",
                                                "train_subsets_matching", "train_check"), reproduction)
    io.write_csv(out / "fingerprints.csv", ("series", "temperature_K", "frames",
                                            *(f"rms_{m}_meV_A" for m in model_ids), "assigned_model"), fingerprints)
    energy_rows = []
    for i, cid in enumerate(order):
        row = {"config_id": cid, "n_atoms": len(frames[i]["species"])}
        for name in ("gamma", "k222"):
            row[f"dft_{name}_energy_eV"] = float(labels[name][i]["info"]["dft_energy"])
            row[f"dft_{name}_pressure_kbar"] = float(configs[cid][f"pressure_{name}_kbar"])
        for model in model_ids:
            energies, _, pressures = predictions[model]
            row[f"{model}_energy_eV"], row[f"{model}_pressure_kbar"] = energies[i], pressures[i]
        energy_rows.append(row)
    columns = ["config_id", "n_atoms"] + [f"dft_{n}_{q}" for n in ("gamma", "k222") for q in ("energy_eV", "pressure_kbar")]
    columns += [f"{m}_{q}" for m in model_ids for q in ("energy_eV", "pressure_kbar")]
    io.write_csv(out / "predictions_energy.csv", columns, energy_rows, digits=17)
    force_rows = []
    for i, cid in enumerate(order):
        for atom, species in enumerate(frames[i]["species"]):
            row = {"config_id": cid, "atom": atom, "species": species}
            for name in ("gamma", "k222"):
                for axis, value in zip("xyz", labels[name][i]["dft_forces"][atom]):
                    row[f"dft_{name}_f{axis}"] = value
            for model in model_ids:
                for axis, value in zip("xyz", predictions[model][1][i][atom]):
                    row[f"{model}_f{axis}"] = value
            force_rows.append(row)
    columns = ["config_id", "atom", "species"] + [f"dft_{n}_f{a}" for n in ("gamma", "k222") for a in "xyz"]
    columns += [f"{m}_f{a}" for m in model_ids for a in "xyz"]
    io.write_csv(out / "predictions_forces.csv.gz", columns, force_rows, digits=10)
    meta = {"evaluated": "2026-10-26", "device": "cpu", "labels_dtype": "float64", "fingerprint_dtype": "float32",
            "versions": {"mace-torch": mace.__version__, "torch": torch.__version__, "ase": ase.__version__},
            "training": summaries}
    (out / "evaluation.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return {"models": model_ids, "registry": len(registry), "fingerprints": fingerprints,
            "reproduction": reproduction}


def _read_csv(path: Path) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
