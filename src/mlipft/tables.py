"""``mlipft tables``: every table of ``results/``, from the committed data only.

Part A (universal potentials against experiment), B (fine-tuned models against DFT), C (transport,
model against model), D (Cl-excess series), and the reproduction of the July values, following
``protocol/``. Only rows of universal potentials carry a ratio to experiment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as C
from . import io, paths
from .dft import agreement
from .transport import reproduction as R
from .transport.arrhenius import arrhenius_fit

FIT_FIELDS = ("n_points", "weighted", "Ea_eV", "Ea_err_eV", "sigma300_mS_cm", "sigma300_min_mS_cm",
              "sigma300_max_mS_cm", "R2", "slope", "slope_err", "intercept", "intercept_err")
PART_C = {"foundation": "mace_416_200", "ftG": "ftG_416_200", "ftK": "ftK_416_200"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip", **kwargs)


def published_runs() -> pd.DataFrame:
    """Per-run values published in July, one row per (series, temperature)."""
    rows = []
    for series in C.SERIES:
        if not series["published"]:
            continue
        name, key = series["published"]
        record = json.loads((paths.PUBLISHED / name).read_text(encoding="utf-8"))
        for run in record["per_run"]:
            if run["mlip"] == key:
                rows.append({"series": series["id"], "temperature_K": int(run["T"]),
                             "run_id": C.run_id(series["id"], int(run["T"])),
                             **{k: run[k] for k in ("D_cm2_s", "sigma_mS_cm", "kinisi_D_cm2_s", "kinisi_D_std_cm2_s",
                                                    "kinisi_sigma_mS_cm", "kinisi_sigma_std_mS_cm")}})
    return pd.DataFrame(rows)


def published_fit(series: dict) -> dict:
    name, key = series["published"]
    return json.loads((paths.PUBLISHED / name).read_text(encoding="utf-8"))["arrhenius"][key]


def _fit(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("temperature_K")
    return arrhenius_fit(frame["temperature_K"], frame["sigma_mS_cm"], frame["kinisi_sigma_std_mS_cm"])


def fits(transport: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Published (refitted from the published per-run values) and re-analysis fits of every series."""
    rows = []
    for series in C.SERIES:
        base = {"series": series["id"], "label": series["label"], "potential": series["potential"],
                "model": series["model"], "composition": series["composition"], "cl_content": series["cl_content"],
                "n_atoms": series["n_atoms"], "production_ps": series["production_ps"], "part": series["part"]}
        if series["published"]:
            rows.append({**base, "source": "published", **_fit(published[published["series"] == series["id"]])})
        rows.append({**base, "source": "reanalysis", **_fit(transport[transport["series"] == series["id"]])})
    return pd.DataFrame(rows)


def reproduction(transport: pd.DataFrame, published: pd.DataFrame, fit_table: pd.DataFrame):
    """Run-level and fit-level classes, and the source quoted for each series (rule items 2-5)."""
    merged = published.merge(transport, on=["run_id", "series", "temperature_K"], suffixes=("_published", ""))
    run_rows = []
    for row in merged.to_dict("records"):
        run_rows.append({
            "run_id": row["run_id"], "series": row["series"], "temperature_K": row["temperature_K"],
            "pymatgen_D_rel_diff": R.relative(row["D_cm2_s"], row["D_cm2_s_published"]),
            "pymatgen_class": R.classify_pymatgen(row["D_cm2_s"], row["D_cm2_s_published"]),
            "pymatgen_sigma_class": R.classify_pymatgen(row["sigma_mS_cm"], row["sigma_mS_cm_published"]),
            "kinisi_D_rel_diff": R.relative(row["kinisi_D_cm2_s"], row["kinisi_D_cm2_s_published"]),
            "kinisi_D_seed_scatter": row["seed_scatter_D_rel"],
            "kinisi_D_class": R.classify_kinisi(row["kinisi_D_cm2_s"], row["kinisi_D_cm2_s_published"],
                                                row["seed_scatter_D_rel"], R.KINISI_D_FLOOR),
            "kinisi_std_rel_diff": R.relative(row["kinisi_D_std_cm2_s"], row["kinisi_D_std_cm2_s_published"]),
            "kinisi_std_seed_scatter": row["seed_scatter_D_std_rel"],
            "kinisi_std_class": R.classify_kinisi(row["kinisi_D_std_cm2_s"], row["kinisi_D_std_cm2_s_published"],
                                                  row["seed_scatter_D_std_rel"], R.KINISI_STD_FLOOR)})
    runs = pd.DataFrame(run_rows)
    if (runs["pymatgen_class"] == "R3").any() or (runs["pymatgen_sigma_class"] == "R3").any():
        bad = runs.loc[runs["pymatgen_class"] == "R3", "run_id"].tolist()
        raise SystemExit(f"reproduction rule item 4: pymatgen D is R3 for {bad}; stop and investigate")
    fit_rows, quoted = [], {}
    for series in C.SERIES:
        if not series["published"]:
            quoted[series["id"]] = "reanalysis"
            continue
        new = fit_table[(fit_table["series"] == series["id"]) & (fit_table["source"] == "reanalysis")].iloc[0]
        published_values = published_fit(series)
        klass, measures = R.classify_fit(new.to_dict(), published_values)
        pymatgen_r1 = bool((runs.loc[runs["series"] == series["id"], "pymatgen_class"] == "R1").all())
        fit_rows.append({"series": series["id"], "class": klass, "pymatgen_runs_all_R1": pymatgen_r1,
                         "published_sigma300_mS_cm": published_values["sigma300_mS_cm"],
                         "reanalysis_sigma300_mS_cm": new["sigma300_mS_cm"],
                         "published_Ea_eV": published_values["Ea_eV"], "reanalysis_Ea_eV": new["Ea_eV"], **measures})
    fits_frame = pd.DataFrame(fit_rows)
    for row in fit_rows:
        if row["class"] in ("R1", "R2"):
            quoted[row["series"]] = "published"
        else:  # item 3: pymatgen reproduces, so the difference comes from the kinisi weights
            quoted[row["series"]] = "reanalysis" if row["pymatgen_runs_all_R1"] else "stop"
    if "stop" in quoted.values():
        raise SystemExit(f"reproduction rule: fit discrepancy without pymatgen R1: {quoted}")
    return runs, fits_frame, quoted


def thermostat(transport: pd.DataFrame) -> pd.DataFrame:
    stats = _read(paths.MD / "thermostat.csv")
    logs = _read(paths.MD / "run_logs.csv", dtype=str)
    rows = []
    for row in stats.merge(transport[["run_id", "series", "temperature_K", "n_atoms", "trajectory"]],
                           on="run_id").to_dict("records"):
        out = {**row, "canonical_rel_std_T": math.sqrt(2.0 / (3.0 * row["n_atoms"])),
               "rel_std_T": row["std_T_K"] / row["temperature_K"]}
        match = logs[logs["trajectory"] == row["trajectory"].removeprefix("data/")]
        out["run_log"] = "" if match.empty else match["log_file"].iloc[0]
        if not match.empty:
            out.update(R.thermostat_check(row, match.iloc[0].to_dict(), row["n_atoms"]))
        rows.append(out)
    return pd.DataFrame(rows)


def convergence(transport: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    """Part A: sigma(300 K) and Ea against simulated time (re-analysis values)."""
    windows = _read(paths.MD / "windows.csv")
    rows = []
    reanalysis = fit_table[fit_table["source"] == "reanalysis"].set_index("series")
    for series in C.SERIES:
        if series["part"] not in ("A",) and series["id"] != "clx000_416_150":
            continue
        kind = "independent_run" if series["id"] == "clx000_416_150" else (
            "cell_52" if series["n_atoms"] == 52 else "production_full")
        fit = reanalysis.loc[series["id"]]
        rows.append({"series": series["id"], "potential": series["potential"], "kind": kind,
                     "n_atoms": series["n_atoms"], "simulated_ps": series["production_ps"],
                     **{k: fit[k] for k in FIT_FIELDS}})
        if series["id"] in C.PRODUCTION_SERIES and series["part"] == "A":
            for window in C.WINDOW_PS[:-1]:
                part = windows[(windows["kind"] == "window") & (windows["value"] == window)
                               & windows["run_id"].str.startswith(series["id"] + "_")].copy()
                part["temperature_K"] = part["run_id"].str.extract(r"_(\d+)K$").astype(int)[0]
                part["kinisi_sigma_std_mS_cm"] = part["kinisi_sigma_std_mS_cm"]
                fit = _fit(part)
                rows.append({"series": series["id"], "potential": series["potential"], "kind": "production_window",
                             "n_atoms": series["n_atoms"], "simulated_ps": window, **{k: fit[k] for k in FIT_FIELDS}})
    return pd.DataFrame(rows)


def transport_comparison(transport: pd.DataFrame, fit_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part C: per-temperature D ratios and the fitted sigma(300 K) of the three models (re-analysis)."""
    per_run = transport.set_index("run_id")
    ratio_rows = []
    for name in ("ftG", "ftK"):
        for t in C.TEMPERATURES:
            model, base = per_run.loc[C.run_id(PART_C[name], t)], per_run.loc[C.run_id(PART_C["foundation"], t)]
            rel = math.hypot(model["kinisi_D_std_cm2_s"] / model["kinisi_D_cm2_s"],
                             base["kinisi_D_std_cm2_s"] / base["kinisi_D_cm2_s"])
            ratio_rows.append({"model": name, "temperature_K": t,
                               "pymatgen_D_ratio": model["D_cm2_s"] / base["D_cm2_s"],
                               "kinisi_D_ratio": model["kinisi_D_cm2_s"] / base["kinisi_D_cm2_s"],
                               "kinisi_D_ratio_rel_err": rel})
    reanalysis = fit_table[fit_table["source"] == "reanalysis"].set_index("series")
    summary = []
    half = {}
    for name, series in PART_C.items():
        fit = reanalysis.loc[series]
        half[name] = (math.log(fit["sigma300_max_mS_cm"]) - math.log(fit["sigma300_min_mS_cm"])) / 2.0
        summary.append({"model": name, "series": series, **{k: fit[k] for k in FIT_FIELDS},
                        "sigma300_ratio_to_foundation": fit["sigma300_mS_cm"]
                        / reanalysis.loc[PART_C["foundation"], "sigma300_mS_cm"]})
    s = {name: reanalysis.loc[series, "sigma300_mS_cm"] for name, series in PART_C.items()}
    gap = math.log(s["ftK"] / s["ftG"])
    gap_err = math.hypot(half["ftK"], half["ftG"])
    f = gap / math.log(s["foundation"] / s["ftG"])
    if abs(gap) < gap_err:
        reading = "indistinguishable"
    else:
        reading = "not explained" if f < 0.25 else ("explains most" if f > 0.75 else "partial")
    summary.append({"model": "reading", "series": "", "f_recovered_fraction": f, "ln_sigma_ftK_over_ftG": gap,
                    "ln_sigma_ftK_over_ftG_err": gap_err, "reading": reading})
    return pd.DataFrame(ratio_rows), pd.DataFrame(summary)


def clexcess(fit_table: pd.DataFrame, quoted: dict) -> pd.DataFrame:
    rows = []
    selected = {s: fit_table[(fit_table["series"] == s) & (fit_table["source"] == quoted[s])].iloc[0]
                for s in ("clx000_416_150", "clx025_408_150", "clx050_400_150", "clx075_392_150")}
    base = selected["clx000_416_150"]
    base_half = (math.log(base["sigma300_max_mS_cm"]) - math.log(base["sigma300_min_mS_cm"])) / 2.0
    for series, fit in selected.items():
        half = (math.log(fit["sigma300_max_mS_cm"]) - math.log(fit["sigma300_min_mS_cm"])) / 2.0
        enhancement = fit["sigma300_mS_cm"] / base["sigma300_mS_cm"]
        err = math.hypot(half, base_half) if series != "clx000_416_150" else 0.0
        rows.append({"series": series, "source": quoted[series], "cl_content": fit["cl_content"],
                     "composition": fit["composition"], "n_atoms": fit["n_atoms"],
                     **{k: fit[k] for k in FIT_FIELDS}, "enhancement_over_x0": enhancement,
                     "enhancement_min": enhancement * math.exp(-err), "enhancement_max": enhancement * math.exp(err)})
    return pd.DataFrame(rows)


def build(out: Path | None = None) -> list[Path]:
    out = Path(out or paths.RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    transport = _read(paths.MD / "transport.csv")
    published = published_runs()
    fit_table = fits(transport, published)
    runs, fit_classes, quoted = reproduction(transport, published, fit_table)
    fit_table["quoted"] = [quoted[s] == src for s, src in zip(fit_table["series"], fit_table["source"])]
    experiment = (fit_table["potential"].isin(C.UNIVERSAL_POTENTIALS) & (fit_table["composition"] == "Li6PS5Cl"))
    fit_table["ratio_to_experiment"] = np.where(experiment, fit_table["sigma300_mS_cm"] / C.EXPERIMENT_mS_cm, np.nan)
    tables = {
        "fits.csv": fit_table, "reproduction_runs.csv": runs, "reproduction_fits.csv": fit_classes,
        "thermostat.csv": thermostat(transport), "convergence.csv": convergence(transport, fit_table),
        "clexcess.csv": clexcess(fit_table, quoted),
    }
    ratios, summary = transport_comparison(transport, fit_table)
    tables["transport_ratios.csv"], tables["transport_comparison.csv"] = ratios, summary
    floor, floor_configs = agreement.label_floor()
    cross, cross_configs = agreement.cross_evaluation()
    tables.update({"label_floor.csv": floor, "label_floor_configs.csv": floor_configs,
                   "cross_evaluation.csv": cross, "cross_evaluation_configs.csv": cross_configs,
                   "floor_decomposition.csv": agreement.floor_decomposition()})
    written = []
    for name, frame in tables.items():
        records = frame.to_dict("records")
        written.append(io.write_csv(out / name, list(frame.columns), records))
    return written
