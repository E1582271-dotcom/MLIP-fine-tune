"""The AIMD reference of amendment A3: provenance of the derived values and coverage of the comparison."""
from __future__ import annotations

import pandas as pd

from mlipft import constants as C
from mlipft import paths
from mlipft.ingest import aimd

FOLDER = paths.REFERENCES / "aimd_deng2017"


def test_derived_values_come_from_the_pinned_files():
    transport = pd.read_csv(FOLDER / "transport.csv")
    assert list(transport["temperature_K"]) == list(aimd.TEMPERATURES)
    assert dict(zip(transport["temperature_K"], transport["sha256"])) == aimd.SHA256
    source = (FOLDER / "SOURCE.md").read_text(encoding="utf-8")
    assert aimd.COMMIT in source and "BSD 3-Clause" in source
    assert all(value in source for value in aimd.SHA256.values())


def test_every_production_model_is_compared_at_every_simulated_temperature():
    comparison = pd.read_csv(paths.RESULTS / "aimd_comparison.csv")
    for series in C.PRODUCTION_SERIES:
        assert sorted(comparison.loc[comparison["series"] == series, "temperature_K"]) == list(C.TEMPERATURES)
    assert (comparison["kinisi_D_ratio_to_aimd"] > 0).all()
    assert (comparison["kinisi_D_ratio_rel_err"] > 0).all()
