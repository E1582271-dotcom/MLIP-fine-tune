"""results/ is rebuilt from data/ exactly, reproduces the published fits, and follows the comparison rule."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from mlipft import constants as C
from mlipft import paths, tables
from mlipft.figures import registry

# Phase-2 switch: while False, measured values left in the source data of figures not yet migrated are an
# expected failure (xfail) listing them; phase 2 sets it to True.
STRICT_SOURCE_DATA = True

FIELDS = ("Ea_eV", "Ea_err_eV", "sigma300_mS_cm", "sigma300_min_mS_cm", "sigma300_max_mS_cm", "R2", "slope",
          "intercept", "slope_err", "intercept_err")


def _results(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.RESULTS / name, keep_default_na=False, na_values=[""], float_precision="round_trip")


@pytest.mark.parametrize("series", [s for s in C.SERIES if s["published"]], ids=lambda s: s["id"])
def test_published_per_run_values_refit_to_the_published_fit(series):
    published = tables.published_runs()
    fit = tables._fit(published[published["series"] == series["id"]])
    reference = tables.published_fit(series)
    for field in FIELDS:
        assert math.isclose(fit[field], reference[field], rel_tol=1e-9), field


def test_results_are_current(tmp_path):
    written = tables.build(tmp_path)
    for path in written:
        assert path.read_text(encoding="utf-8") == (paths.RESULTS / path.name).read_text(encoding="utf-8"), path.name


def test_fine_tuned_models_are_never_compared_with_experiment():
    fits = _results("fits.csv")
    fine_tuned = fits[fits["model"].isin(["ftG", "ftK"])]
    assert fine_tuned["ratio_to_experiment"].isna().all()
    universal = fits[fits["potential"].isin(C.UNIVERSAL_POTENTIALS) & (fits["composition"] == "Li6PS5Cl")]
    assert universal["ratio_to_experiment"].notna().all()


def _experiment_tokens() -> list[str]:
    """'experiment' and the ids of the measured values in the literature table."""
    literature = pd.read_csv(paths.REFERENCES / "literature.csv", keep_default_na=False)
    return ["experiment"] + sorted(literature.loc[literature["kind"] == "experiment", "ref_id"])


def test_source_data_keep_measured_values_away_from_the_comparisons():
    """No measured value in main-figure source data or next to a fine-tuned model; only the AIMD reference and
    the Cl-excess Extended Data figures may show measured values (as grey context)."""
    tokens = _experiment_tokens()
    assert {"wang2018", "li2025_compiled", "adeli2019"} <= set(tokens)
    allowed = tuple(f"{registry.number(key)}_" for key in ("aimd", "clexcess"))
    problems = []
    for path in sorted((paths.FIGURES / "source_data").glob("*.csv")):
        text = path.read_text(encoding="utf-8")
        found = [token for token in tokens if token in text.lower()]
        if not found:
            continue
        if path.name.startswith("fig"):
            problems.append(f"{path.name}: main figure with {found}")
        if "ftG" in text or "ftK" in text:
            problems.append(f"{path.name}: fine-tuned model next to {found}")
        if not path.name.startswith(allowed):
            problems.append(f"{path.name}: {found} outside {', '.join(allowed)}")
    if problems and not STRICT_SOURCE_DATA:
        pytest.xfail("measured values in comparison source data (switched on in phase 2): " + "; ".join(problems))
    assert not problems, problems


def test_every_published_run_reproduces_pymatgen():
    runs = _results("reproduction_runs.csv")
    assert len(runs) == 33
    assert set(runs["pymatgen_class"]) <= {"R1", "R2"}


def test_quoted_values_follow_the_rule():
    fits = _results("fits.csv")
    classes = _results("reproduction_fits.csv").set_index("series")
    quoted = fits[fits["quoted"].astype(str).str.lower() == "true"].set_index("series")
    assert len(quoted) == len(C.SERIES)
    for series in C.SERIES:
        source = quoted.loc[series["id"], "source"]
        if not series["published"]:
            assert source == "reanalysis"
        else:
            expected = "published" if classes.loc[series["id"], "class"] in ("R1", "R2") else "reanalysis"
            assert source == expected, series["id"]
