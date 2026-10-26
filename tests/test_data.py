"""Contracts of the committed data: counts, identities and checksums."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from mlipft import io, paths


def _csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip")


def test_published_files_match_their_checksums():
    for line in (paths.PUBLISHED / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((paths.PUBLISHED / name).read_bytes()).hexdigest() == digest


def test_configurations_and_subsets():
    configs = _csv(paths.DFT / "configs.csv")
    assert len(configs) == 27 and configs["config_id"].is_unique
    assert configs["subset"].value_counts().to_dict() == {"T22": 22, "H3": 3, "H2": 2}
    held_g = set(configs.loc[configs["ftG_split"] == "valid", "config_id"])
    held_k = set(configs.loc[configs["ftK_split"] == "valid", "config_id"])
    assert held_k < held_g and len(held_g) == 5 and len(held_k) == 3
    assert not any(cid.startswith("T1000") for cid in held_g)


def test_label_files_share_positions_and_match_the_configs():
    configs = _csv(paths.DFT / "configs.csv").set_index("config_id")
    gamma = io.read_extxyz(paths.DFT / "raw_inputs" / "labelled_gamma.xyz")
    k222 = io.read_extxyz(paths.DFT / "raw_inputs" / "labelled_k222.xyz")
    for _, row in configs.iterrows():
        a, b = gamma[int(row["index_gamma"])], k222[int(row["index_k222"])]
        assert np.abs(a["pos"] - b["pos"]).max() < 1e-6
        assert float(a["info"]["dft_energy"]) == row["dft_energy_gamma_eV"]
        assert float(b["info"]["dft_energy"]) == row["dft_energy_k222_eV"]


def test_qe_runs_and_pseudopotentials():
    runs = _csv(paths.DFT / "qe_runs.csv")
    assert len(runs) == 54
    assert set(runs.groupby("label_set")["k_points"].unique().apply(tuple)) == {(1,), (8,)}
    assert (runs[["ecutwfc_Ry", "ecutrho_Ry", "conv_thr_Ry", "smearing_Ry"]].drop_duplicates().values.tolist()
            == [[60.0, 480.0, 1e-07, 0.01]])
    pseudo = _csv(paths.DFT / "pseudopotentials.csv")
    assert sorted(pseudo["file"]) == ["cl_pbe_v1.4.uspp.F.UPF", "li_pbe_v1.4.uspp.F.UPF", "p_pbe_v1.5.uspp.F.UPF",
                                      "s_pbe_v1.4.uspp.F.UPF"]


def test_models_registry_and_log_reproduction():
    registry = _csv(paths.MODELS / "registry.csv")
    assert sorted(set(registry["model"].dropna())) == ["foundation", "ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2"]
    logs = _csv(paths.MODELS / "log_reproduction.csv")
    fine = logs[logs["model"] != "foundation"]
    assert (fine["reevaluated_valid_rmse_f_meV_A"].round(1) == fine["logged_valid_rmse_f_meV_A"]).all()
    assert fine["train_check"].astype(str).str.lower().eq("true").all()
    foundation = logs[logs["model"] == "foundation"].iloc[0]
    assert round(foundation["reevaluated_valid_rmse_f_meV_A"], 2) == foundation["logged_valid_rmse_f_meV_A"]


def test_fingerprints_assign_every_md_run():
    prints = _csv(paths.MODELS / "fingerprints.csv")
    assigned = prints.groupby("series")["assigned_model"].unique().apply(list).to_dict()
    assert assigned == {"ftG_416_200": ["ftG_s2"], "ftK_416_200": ["ftK_s1"], "mace_416_200": ["foundation"]}


def test_predictions_carry_the_dft_labels():
    forces = _csv(paths.MODELS / "predictions_forces.csv.gz")
    configs = _csv(paths.DFT / "configs.csv").set_index("config_id")
    gamma = io.read_extxyz(paths.DFT / "raw_inputs" / "labelled_gamma.xyz")
    assert len(forces) == 27 * 52
    for cid, part in forces.groupby("config_id", sort=False):
        frame = gamma[int(configs.loc[cid, "index_gamma"])]
        assert list(part["species"]) == frame["species"]
        assert np.allclose(part[["dft_gamma_fx", "dft_gamma_fy", "dft_gamma_fz"]].to_numpy(), frame["dft_forces"],
                           rtol=1e-9, atol=1e-12)
