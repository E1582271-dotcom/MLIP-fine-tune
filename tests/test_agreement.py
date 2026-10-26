"""Part B arithmetic: offset removal and the floor decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlipft.dft import agreement


def test_constant_energy_shift_does_not_change_centred_errors():
    energy = pd.DataFrame({"n_atoms": [52] * 4, "subset": ["T22", "T22", "H3", "H3"],
                           "model": [1.0, 2.0, 3.0, 4.5], "dft": [0.9, 2.1, 3.0, 4.0]})
    base = agreement._centred_energy(energy, "model", "dft")
    energy["model"] += 1234.5
    assert np.allclose(agreement._centred_energy(energy, "model", "dft"), base)


def test_floor_decomposition_identity():
    table = agreement.floor_decomposition()
    lhs = table["rmse_vs_k222_meV_A"] ** 2
    rhs = table["rmse_vs_gamma_meV_A"] ** 2 + table["floor_meV_A"] ** 2 + table["cross_term_meV2_A2"]
    assert np.allclose(lhs, rhs, rtol=1e-9)


def test_primary_estimand_is_present():
    cross, _ = agreement.cross_evaluation()
    primary = cross[(cross["subset"] == "H3") & (cross["labels"] == "k222") & (cross["element"] == "all")]
    assert sorted(primary["model"]) == sorted(agreement.MODELS)
    assert (primary["n_configs"] == 3).all()
