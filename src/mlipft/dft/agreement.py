"""Part B: the fine-tuned models against DFT, and the two DFT label sets against each other.

All quantities follow ``protocol/analysis_plan.md``:

* force errors per Cartesian component (meV/A), overall and per element;
* softening slope: least-squares slope through the origin of model force against DFT force;
* energy errors per atom after removing the constant offset of each (model, label set) pair,
  taken as the mean error over T22 (every configuration has the composition Li24P4S20Cl4, so
  reference energies can only add a constant); the foundation's absolute energy error is never used;
* pressure errors (kbar), a diagnostic only;
* the label-error floor: the same measures for Gamma labels against 2x2x2 labels;
* the decomposition RMSE(m vs k222)^2 = RMSE(m vs gamma)^2 + floor^2 + cross term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths

MODELS = ("foundation", "ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2")
LABELS = ("gamma", "k222")
ELEMENTS = ("Li", "P", "S", "Cl")
SUBSETS = ("H3", "H2", "T22", "All27")
OWN = {"ftG_s1": ("gamma", "ftG"), "ftG_s2": ("gamma", "ftG"), "ftK_s1": ("k222", "ftK"), "ftK_s2": ("k222", "ftK")}
DROVE_MD = {"foundation": "mace_416_200", "ftG_s2": "ftG_416_200", "ftK_s1": "ftK_416_200"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = pd.read_csv(paths.DFT / "configs.csv")
    energy = pd.read_csv(paths.MODELS / "predictions_energy.csv").merge(configs[["config_id", "subset", "ftG_split",
                                                                                 "ftK_split"]], on="config_id")
    forces = pd.read_csv(paths.MODELS / "predictions_forces.csv.gz").merge(configs[["config_id", "subset"]],
                                                                           on="config_id")
    return configs, energy, forces


def _components(forces: pd.DataFrame, prefix: str) -> np.ndarray:
    return forces[[f"{prefix}_f{axis}" for axis in "xyz"]].to_numpy()


def _subset_mask(frame: pd.DataFrame, subset: str, configs: pd.DataFrame | None = None, own=None) -> np.ndarray:
    if subset == "All27":
        return np.ones(len(frame), bool)
    if subset == "own_valid":
        ids = set(configs.loc[configs[f"{own}_split"] == "valid", "config_id"])
        return frame["config_id"].isin(ids).to_numpy()
    return (frame["subset"] == subset).to_numpy()


def _force_stats(predicted: np.ndarray, reference: np.ndarray) -> dict:
    diff = predicted - reference
    flat_ref, flat_pred = reference.ravel(), predicted.ravel()
    return {"force_rmse_meV_A": float(np.sqrt(np.mean(diff ** 2)) * 1000.0),
            "force_mae_meV_A": float(np.mean(np.abs(diff)) * 1000.0),
            "softening_slope": float(flat_pred @ flat_ref / (flat_ref @ flat_ref)),
            "n_components": int(diff.size)}


def _centred_energy(energy: pd.DataFrame, predicted: str, reference: str, offset_from: str = "T22") -> np.ndarray:
    error = (energy[predicted] - energy[reference]) / energy["n_atoms"] * 1000.0
    base = error if offset_from == "All27" else error[energy["subset"] == "T22"]
    return (error - base.mean()).to_numpy()


def label_floor() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gamma labels against 2x2x2 labels, per subset and element, and per configuration."""
    configs, energy, forces = load()
    gamma, k222 = _components(forces, "dft_gamma"), _components(forces, "dft_k222")
    centred = _centred_energy(energy, "dft_gamma_energy_eV", "dft_k222_energy_eV")
    raw = ((energy["dft_gamma_energy_eV"] - energy["dft_k222_energy_eV"]) / energy["n_atoms"] * 1000.0).to_numpy()
    pressure = (energy["dft_gamma_pressure_kbar"] - energy["dft_k222_pressure_kbar"]).to_numpy()
    rows = []
    for subset in ("All27", "H3"):
        mask = _subset_mask(forces, subset)
        emask = _subset_mask(energy, subset)
        for element in ("all", *ELEMENTS):
            keep = mask if element == "all" else mask & (forces["species"] == element).to_numpy()
            row = {"subset": subset, "element": element, **_force_stats(gamma[keep], k222[keep])}
            if element == "all":
                row.update({"energy_rmse_centred_meV_atom": float(np.sqrt(np.mean(centred[emask] ** 2))),
                            "energy_mean_raw_meV_atom": float(raw[emask].mean()),
                            "pressure_rmse_kbar": float(np.sqrt(np.mean(pressure[emask] ** 2)))})
            rows.append(row)
    per_config = []
    for index, cid in enumerate(energy["config_id"]):
        keep = (forces["config_id"] == cid).to_numpy()
        per_config.append({"config_id": cid, "subset": energy["subset"].iloc[index],
                           "force_rmse_meV_A": float(np.sqrt(np.mean((gamma[keep] - k222[keep]) ** 2)) * 1000.0),
                           "energy_raw_meV_atom": raw[index], "energy_centred_meV_atom": centred[index],
                           "pressure_kbar": pressure[index]})
    return pd.DataFrame(rows), pd.DataFrame(per_config)


def cross_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every model against both label sets, per subset and element, and per configuration."""
    configs, energy, forces = load()
    rows, per_config = [], []
    for model in MODELS:
        predicted = _components(forces, model)
        subsets = list(SUBSETS) + (["own_valid"] if model in OWN else [])
        for labels in LABELS:
            reference = _components(forces, f"dft_{labels}")
            centred = _centred_energy(energy, f"{model}_energy_eV", f"dft_{labels}_energy_eV")
            centred_all = _centred_energy(energy, f"{model}_energy_eV", f"dft_{labels}_energy_eV", "All27")
            pressure = (energy[f"{model}_pressure_kbar"] - energy[f"dft_{labels}_pressure_kbar"]).to_numpy()
            for subset in subsets:
                own = OWN.get(model, (None, None))[1]
                mask = _subset_mask(forces, subset, configs, own)
                emask = _subset_mask(energy, subset, configs, own)
                for element in ("all", *ELEMENTS):
                    keep = mask if element == "all" else mask & (forces["species"] == element).to_numpy()
                    row = {"model": model, "drove_md": DROVE_MD.get(model, ""), "labels": labels, "subset": subset,
                           "n_configs": int(emask.sum()), "element": element,
                           **_force_stats(predicted[keep], reference[keep])}
                    if element == "all":
                        row.update({"energy_rmse_centred_meV_atom": float(np.sqrt(np.mean(centred[emask] ** 2))),
                                    "energy_rmse_centred_all27_offset_meV_atom":
                                        float(np.sqrt(np.mean(centred_all[emask] ** 2))),
                                    "pressure_rmse_kbar": float(np.sqrt(np.mean(pressure[emask] ** 2)))})
                    rows.append(row)
            for index, cid in enumerate(energy["config_id"]):
                keep = (forces["config_id"] == cid).to_numpy()
                per_config.append({"model": model, "labels": labels, "config_id": cid,
                                   "subset": energy["subset"].iloc[index],
                                   "force_rmse_meV_A": float(np.sqrt(np.mean((predicted[keep] - reference[keep]) ** 2))
                                                             * 1000.0),
                                   "energy_centred_meV_atom": centred[index], "pressure_error_kbar": pressure[index]})
    return pd.DataFrame(rows), pd.DataFrame(per_config)


def floor_decomposition() -> pd.DataFrame:
    """RMSE(m vs k222)^2 = RMSE(m vs gamma)^2 + floor^2 + cross, per model and subset (all atoms).

    ``cross_fraction`` = cross / floor^2: about 0 if the model's error is unrelated to the Gamma-2x2x2
    difference, about -2 if the model reproduces the 2x2x2 forces as well as the Gamma ones are off,
    and the model has "learned" the Gamma artefact if its error against 2x2x2 approaches the floor."""
    configs, _, forces = load()
    gamma, k222 = _components(forces, "dft_gamma"), _components(forces, "dft_k222")
    rows = []
    for model in MODELS:
        predicted = _components(forces, model)
        for subset in ("H3", "All27"):
            mask = _subset_mask(forces, subset)
            e_gamma, d = (predicted - gamma)[mask], (gamma - k222)[mask]
            rmse = lambda x: float(np.sqrt(np.mean(x ** 2)) * 1000.0)  # noqa: E731
            cross = float(2.0 * np.mean(e_gamma * d) * 1e6)
            floor = rmse(d)
            rows.append({"model": model, "subset": subset, "rmse_vs_gamma_meV_A": rmse(e_gamma),
                         "rmse_vs_k222_meV_A": rmse(predicted[mask] - k222[mask]), "floor_meV_A": floor,
                         "cross_term_meV2_A2": cross, "cross_fraction": cross / floor ** 2})
    return pd.DataFrame(rows)
