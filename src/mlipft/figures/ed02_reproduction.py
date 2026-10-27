"""Re-analysis against the values published in July 2026 (Extended Data, key "reproduction").

a (hero), pymatgen D of each of the 33 published runs, relative difference to the published value
(exact zeros drawn on an "identical" row), with the R1 and R2 limits; b, kinisi D and c, its kinisi
standard deviation, relative difference against each run's threshold max(3 s_seed, floor); d, the 11
Arrhenius fits: differences in ln sigma(300 K) and Ea in units of the published uncertainty, with the
R2 (0.1) and R2- (1) limits; e, training logs: logged minus re-evaluated force RMSE of the fine-tuned
models and of the foundation model, on the validation and the training set, against the printed
precision; f, thermostat statistics recomputed from the trajectories against the surviving July run
logs, in units of the tolerance of the rule; g, the Monte-Carlo scatter s_seed of kinisi D over seeds
1-10 against the width of its posterior, for all 42 runs.

Encoding: in a-d and f, colour is the temperature (a-c, f) or the quantity (d); marker shape the run
type (``style.RUN_MARKERS``); fill the training stage. In e and g, colour is the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

from .. import paths
from ..constants import SERIES, SERIES_BY_ID, TEMPERATURES
from ..transport import reproduction as rule
from . import common, registry, style

KEY = "reproduction"
NAME = registry.number(KEY)
PANELS = "abcdefg"
WIDTH, HEIGHT = 183.0, 186.0
RUNS = paths.RESULTS / "reproduction_runs.csv"
FITS = paths.RESULTS / "reproduction_fits.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
LOGS = paths.MODELS / "log_reproduction.csv"
CONFIGS = paths.DFT / "configs.csv"
TRANSPORT = paths.MD / "transport.csv"
SEEDS = paths.MD / "kinisi_seeds.csv"
RULE = paths.PROTOCOL / "reproduction_rule.md"
AMENDMENTS = paths.PROTOCOL / "amendments.md"

ZERO_FLOOR = 1e-17  # exact zeros of the pymatgen difference are drawn here
FLOAT64_EPS = float(np.finfo(np.float64).eps)
GROUP = 4  # x units per series: three runs and a gap
LOG_MODELS = ("ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2", "foundation")
THERMO_CHECKS = (("mean_T", "d_mean_T_K", "mean_T_ok", "Mean T"),
                 ("mean_E_per_atom", "d_mean_E_per_atom", "mean_E_per_atom_ok", "Mean E"),
                 ("drift_meV_atom_ps", "d_drift_meV_atom_ps", "drift_meV_atom_ps_ok", "Drift"))
PART_TEXT = {"A": "Universal potentials", "C": "Fine-tuned", "D": "Cl excess, MACE-MP-0"}
KIND_LABELS = {"cell_52": "52 atoms", "production": "416 atoms, 200 ps", "clexcess": "Cl excess, 392-416 atoms"}
QUANTITY_COLORS = {"sigma": style.BLACK, "Ea": style.GRAY}  # the two fit quantities of d
MARKER = 3.0  # marker diameter (pt) of a run in a-d and f
FAMILY_MODELS = {"foundation": "foundation", "mattersim": "mattersim", "ftG": "ftG", "ftK": "ftK"}

# Layout (mm from the top-left corner)
LEFT, STRIP_WIDTH = 20.0, 161.0
STRIPS = ((13.0, 24.0), (44.0, 18.0), (69.0, 18.0), (94.0, 18.0))  # (top, height) of a (hero), b, c, d
BOTTOM_TOP, BOTTOM_HEIGHT = 139.0, 34.0
E_LEFT, E_VALID, E_GAP, E_TRAIN = 34.0, 19.0, 3.0, 22.0
F_LEFT, F_WIDTH = 92.0, 32.0
G_LEFT, G_WIDTH = 141.0, 40.0


def _load():
    runs = common.read(RUNS)
    fits = common.read(FITS)
    thermostat = common.read(THERMOSTAT)
    logs = common.read(LOGS)
    configs = common.read(CONFIGS)
    return runs, fits, thermostat, logs, configs


def _published_series(runs: pd.DataFrame) -> list[dict]:
    present = set(runs["series"])
    series = [s for s in SERIES if s["id"] in present]
    if len(series) != len(present):
        raise ValueError("a published series is missing from the catalogue")
    return series


def _positions(runs: pd.DataFrame, series: list[dict]) -> pd.DataFrame:
    """Runs ordered by series (catalogue order) and temperature, with their x position."""
    order = {s["id"]: i for i, s in enumerate(series)}
    out = runs.assign(group=runs["series"].map(order),
                      t_index=runs["temperature_K"].map({t: i for i, t in enumerate(TEMPERATURES)}))
    out = out.sort_values(["group", "t_index"]).reset_index(drop=True)
    out["x"] = out["group"] * GROUP + out["t_index"]
    return out


# ------------------------------------------------------------------------------------- run encoding


def _kind(series_id: str) -> str:
    """Run type of a series, a key of ``style.RUN_MARKERS``."""
    entry = SERIES_BY_ID[series_id]
    if entry["part"] == "D":
        return "clexcess"
    return "cell_52" if entry["n_atoms"] < 100 else "production"


def _series_style(series_id: str, color: str, size: float = MARKER, **overrides) -> dict:
    """``ax.plot`` keywords of a run or series in ``color``: run-type shape, fill by training stage."""
    kwargs = {"marker": style.RUN_MARKERS[_kind(series_id)], "markersize": size, "markeredgewidth": 0.7,
              "markeredgecolor": color,
              "markerfacecolor": color if common.filled(SERIES_BY_ID[series_id]["model"]) else style.WHITE,
              "linestyle": "none"}
    kwargs.update(overrides)
    return kwargs


def _run_style(series_id: str, temperature: int, size: float = MARKER, **overrides) -> dict:
    return _series_style(series_id, style.TEMPERATURE_COLORS[int(temperature)], size, **overrides)


def _key_legend(ax) -> None:
    """Temperature and run-type key above a (it applies to a-d and f), with the machine-epsilon line of a."""
    neutral = {"linestyle": "none", "markersize": MARKER, "markeredgewidth": 0.7, "markeredgecolor": style.GRAY,
               "markerfacecolor": style.GRAY}
    temperature = [Line2D([0], [0], **_run_style("mace_416_200", t)) for t in TEMPERATURES]
    kinds = [Line2D([0], [0], marker=style.RUN_MARKERS[kind], **neutral) for kind in KIND_LABELS]
    epsilon = Line2D([0], [0], color=style.GRAY, lw=0.5, ls=(0, (1, 1.5)))
    blank = Line2D([0], [0], linestyle="none", marker="none")
    handles = [temperature[0], kinds[0], temperature[1], kinds[1], temperature[2], kinds[2], epsilon, blank]
    labels = [f"{TEMPERATURES[0]} K", KIND_LABELS["cell_52"], f"{TEMPERATURES[1]} K", KIND_LABELS["production"],
              f"{TEMPERATURES[2]} K", KIND_LABELS["clexcess"], "float64 machine epsilon (a)", ""]
    ax.legend(handles=handles, labels=labels, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4,
              fontsize=style.SMALL, handletextpad=0.4, columnspacing=1.5, labelspacing=0.35, borderaxespad=0.0,
              handlelength=1.6)


def _frame_runs(ax, series: list[dict], labels: bool) -> None:
    n = len(series)
    ax.set_xlim(-0.9, (n - 1) * GROUP + 2.9)
    for g in range(1, n):
        ax.axvline(g * GROUP - 1.0, color=style.LIGHT_GRAY, lw=0.4, zorder=0)
    ax.set_xticks([g * GROUP + 1.0 for g in range(n)], [_short(s) for s in series] if labels else [])
    ax.tick_params(axis="x", length=0, pad=2.0, labelsize=style.SMALL)
    if not labels:
        return
    blend = ax.get_xaxis_transform()
    line = transforms.offset_copy(blend, fig=ax.figure, y=-21.0, units="points")
    text = transforms.offset_copy(blend, fig=ax.figure, y=-22.5, units="points")
    parts = [s["part"] for s in series]
    for part in dict.fromkeys(parts):
        members = [g for g, p in enumerate(parts) if p == part]
        x0, x1 = members[0] * GROUP - 0.6, members[-1] * GROUP + 2.6
        ax.plot([x0, x1], [0, 0], color=style.BLACK, lw=0.5, transform=line, clip_on=False)
        ax.text((x0 + x1) / 2.0, 0, PART_TEXT[part], transform=text, ha="center", va="top", fontsize=style.SMALL)


def _model_name(family: str) -> str:
    """Name of the model that drove a run family: MACE-MP-0, MatterSim, 'Gamma, stage 2', '2x2x2, stage 1'."""
    if family in style.UNIVERSAL_MODELS:
        return style.MODEL_LABELS[family]
    return common.SHORT_LABELS[f"{family}_{style.MD_STAGE[family]}"]


def _short(series: dict) -> str:
    if series["part"] == "D":
        x = series["cl_content"] - 1.0
        name = f"x = {x:.2f}" if x else "x = 0"
    else:
        name = _model_name(series["model"])
    return f"{name}\n{series['n_atoms']} atoms\n{series['production_ps']} ps"


def _title(ax, text: str) -> None:
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=style.FONT_SIZE)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, runs: pd.DataFrame, series: list[dict]) -> Path:
    _frame_runs(ax, series, labels=False)
    zero = runs["pymatgen_D_rel_diff"] == 0.0
    plotted = runs["pymatgen_D_rel_diff"].where(~zero, ZERO_FLOOR)
    for row, y in zip(runs.itertuples(), plotted):
        ax.plot(row.x, y, **_run_style(row.series, row.temperature_K), zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-4)
    ax.axhline(rule.PYMATGEN_EXACT, color=style.BLACK, lw=0.7, zorder=2)
    ax.axhline(rule.PYMATGEN_REPRODUCED, color=style.BLACK, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.axhline(FLOAT64_EPS, color=style.GRAY, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    right = ax.get_xlim()[1]
    for value, name in ((rule.PYMATGEN_EXACT, "R1"), (rule.PYMATGEN_REPRODUCED, "R2")):
        ax.text(right - 0.3, value * 2.0, f"{name} limit, $10^{{{int(round(np.log10(value)))}}}$", ha="right",
                va="bottom", fontsize=style.SMALL)
    style.log_ticks(ax, "y")
    ax.set_yticks([ZERO_FLOOR, 1e-15, 1e-12, 1e-9, 1e-6],
                  ["identical", "$10^{-15}$", "$10^{-12}$", "$10^{-9}$", "$10^{-6}$"])
    ax.yaxis.set_minor_locator(_minor_decades(1e-16, 1e-5, ax.get_yticks()))
    ax.set_ylabel("Relative difference")
    _title(ax, "pymatgen D of each run")
    _key_legend(ax)
    source = runs[["run_id", "series", "temperature_K", "x", "pymatgen_D_rel_diff", "pymatgen_class"]].assign(
        run_type=runs["series"].map(_kind), plotted_value=plotted, identical=zero, r1_limit=rule.PYMATGEN_EXACT,
        r2_limit=rule.PYMATGEN_REPRODUCED)
    return style.save_source(source, NAME, "a")


def _minor_decades(start: float, stop: float, majors) -> object:
    """Minor ticks at the unlabelled decades from ``start`` to ``stop`` (none next to the zero floor)."""
    exponents = range(int(round(np.log10(start))), int(round(np.log10(stop))) + 1)
    labelled = {int(round(np.log10(m))) for m in majors}
    return FixedLocator([10.0 ** e for e in exponents if e not in labelled])


def _panel_kinisi(ax, runs: pd.DataFrame, series: list[dict], quantity: str, floor: float, title: str,
                  top: float, letter: str) -> Path:
    _frame_runs(ax, series, labels=False)
    diff = runs[f"{quantity}_rel_diff"] * 100.0
    three_s = rule.KINISI_SEED_FACTOR * runs[f"{quantity}_seed_scatter"] * 100.0
    threshold = np.maximum(three_s, floor * 100.0)
    ratio = diff / threshold
    ax.axhline(floor * 100.0, color=style.GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for x, t, s in zip(runs["x"], threshold, three_s):
        ax.plot([x - 0.38, x + 0.38], [t, t], color=style.BLACK, lw=1.0, solid_capstyle="butt", zorder=2)
        if s < t:
            ax.plot([x - 0.25, x + 0.25], [s, s], color=style.GRAY, lw=0.8, solid_capstyle="butt", zorder=2)
    for row, y in zip(runs.itertuples(), diff):
        ax.plot(row.x, y, **_run_style(row.series, row.temperature_K), zorder=3)
    ax.set_ylim(-0.07 * top, top)
    ax.set_ylabel("Relative\ndifference (%)")
    _title(ax, f"{title} (closest: {ratio.max():.3f} of its threshold)")
    handles = [Line2D([0], [0], color=style.BLACK, lw=1.0),
               Line2D([0], [0], color=style.GRAY, lw=0.8),
               Line2D([0], [0], color=style.GRAY, lw=0.6, ls=(0, (3, 2)))]
    labels = [f"threshold max(3 s$_{{seed}}$, {floor * 100:g} %)", "3 s$_{seed}$ (when below the floor)",
              f"floor {floor * 100:g} %"]
    ax.legend(handles=handles, labels=labels, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3,
              fontsize=style.SMALL, handlelength=1.4, handletextpad=0.4, columnspacing=1.5, borderaxespad=0.0)
    source = runs[["run_id", "series", "temperature_K", "x", f"{quantity}_rel_diff", f"{quantity}_seed_scatter",
                   f"{quantity}_class"]].assign(run_type=runs["series"].map(_kind), rel_diff_percent=diff,
                                                three_s_seed_percent=three_s, threshold_percent=threshold,
                                                difference_over_threshold=ratio)
    return style.save_source(source, NAME, letter)


def _panel_d(ax, fits: pd.DataFrame, series: list[dict]) -> Path:
    order = {s["id"]: i for i, s in enumerate(series)}
    fits = fits.assign(group=fits["series"].map(order)).sort_values("group").reset_index(drop=True)
    x = fits["group"] * GROUP + 1.0
    _frame_runs(ax, series, labels=True)
    lo, hi = 1e-4, 4.0
    ax.axhspan(rule.FIT_REPRODUCED, rule.FIT_CONSISTENT, **style.BAND_FILL)
    ax.axhspan(rule.FIT_CONSISTENT, hi, **{**style.BAND_FILL, "color": style.LIGHT_GRAY})
    ax.axhline(rule.FIT_REPRODUCED, color=style.BLACK, lw=0.7, zorder=1)
    ax.axhline(rule.FIT_CONSISTENT, color=style.BLACK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for row, xi in zip(fits.itertuples(), x):
        ax.plot(xi - 0.45, row.d_ln_sigma300_in_uncertainties,
                **_series_style(row.series, QUANTITY_COLORS["sigma"], 3.2), zorder=3)
        ax.plot(xi + 0.45, row.d_Ea_in_uncertainties, **_series_style(row.series, QUANTITY_COLORS["Ea"], 3.2),
                zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    style.log_ticks(ax, "y")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["$10^{-4}$", "$10^{-3}$", "0.01", "0.1", "1"])
    right = ax.get_xlim()[1]
    for y, text in ((np.sqrt(lo * rule.FIT_REPRODUCED) * 3.0, "R2"),
                    (np.sqrt(rule.FIT_REPRODUCED * rule.FIT_CONSISTENT), "R2-"),
                    (np.sqrt(rule.FIT_CONSISTENT * hi), "R3")):
        ax.text(right - 0.3, y, text, ha="right", va="center", fontsize=style.SMALL, color=style.DARK_GRAY)
    ax.set_ylabel("|Difference| /\nuncertainty")
    _title(ax, "Arrhenius fit of each series")
    handles = [Line2D([0], [0], color=QUANTITY_COLORS[q], lw=2.0, solid_capstyle="butt") for q in ("sigma", "Ea")]
    labels = ["|d ln $\\sigma$(300 K)| / published log half-width (left)",
              "|d E$_a$| / published E$_a$ error (right)"]
    ax.legend(handles=handles, labels=labels, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
              fontsize=style.SMALL, handlelength=1.0, handletextpad=0.4, columnspacing=1.5,
              borderaxespad=0.0)
    source = fits[["series", "class", "d_ln_sigma300", "published_log_half_width",
                   "d_ln_sigma300_in_uncertainties", "d_Ea_eV", "published_Ea_err_eV",
                   "d_Ea_in_uncertainties"]].assign(run_type=fits["series"].map(_kind), x=x,
                                                    r2_limit=rule.FIT_REPRODUCED,
                                                    r2_minus_limit=rule.FIT_CONSISTENT)
    return style.save_source(source, NAME, "d")


def _decimals(value: float) -> int:
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def _panel_e(axes, logs: pd.DataFrame, configs: pd.DataFrame) -> Path:
    """Logged minus re-evaluated force RMSE: validation set (left axes) and training set (right axes)."""
    logs = logs.set_index("model").loc[list(LOG_MODELS)].reset_index()
    n_train = {run: int((configs[f"{run}_split"] == "train").sum()) for run in ("ftG", "ftK")}
    rows = []
    subset_text = ""
    for ax, kind in zip(axes, ("valid", "train")):
        for y, row in logs.iterrows():
            model = row["model"]
            logged = row[f"logged_{kind}_rmse_f_meV_A"]
            if pd.isna(logged):
                continue
            half = 0.5 * 10.0 ** -_decimals(logged)
            ax.add_patch(Rectangle((-half, y - 0.27), 2 * half, 0.54, facecolor=style.BAND, edgecolor="none",
                                   zorder=0))
            difference = logged - row[f"reevaluated_{kind}_rmse_f_meV_A"]
            subset = kind == "train" and model != "foundation" and \
                int(row["train_evaluated_by_log"]) < n_train[model.split("_")[0]]
            ax.plot(difference, y, **common.model_style(model, size=3.4), zorder=3)
            if subset:
                subset_text = (f"logged over {int(row['train_evaluated_by_log'])} of "
                               f"{n_train[model.split('_')[0]]} training configurations")
                ax.text(difference + 0.016, y - 0.08, "*", ha="left", va="center", fontsize=style.LABEL)
            rows.append({"model": model, "labels": row["labels"], "epoch": row["epoch"], "set": kind,
                         "logged_meV_A": logged, "reevaluated_meV_A": row[f"reevaluated_{kind}_rmse_f_meV_A"],
                         "difference_meV_A": difference, "half_printed_digit_meV_A": half,
                         "configurations_in_log": int(row["train_evaluated_by_log"])
                         if kind == "train" and not pd.isna(row["train_evaluated_by_log"]) else "",
                         "subset_in_log": subset})
        ax.axvline(0, **style.UNITY_LINE)
        ax.set_ylim(len(logs) - 0.5, -0.5)
        ax.set_yticks(range(len(logs)))
        ax.tick_params(axis="y", length=0, pad=2.0)
    valid, train = axes
    valid.set_xlim(-0.08, 0.08)
    valid.set_xticks([-0.05, 0.0, 0.05], ["-0.05", "0", "0.05"])
    train.set_xlim(-0.08, 0.28)
    train.set_xticks([0.0, 0.1, 0.2], ["0", "0.1", "0.2"])
    names = []
    for row in logs.itertuples():
        name = common.SHORT_LABELS[row.model]
        names.append(f"{name}, epoch {int(row.epoch)}" if row.model != "foundation"
                     else f"{name}, before fine-tuning")
    valid.set_yticks(range(len(logs)), names, fontsize=style.SMALL)
    for tick, model in zip(valid.get_yticklabels(), LOG_MODELS):
        tick.set_color(style.MODEL_COLORS[model])
    train.set_yticks(range(len(logs)), [])
    valid.set_title("validation set", fontsize=style.SMALL, pad=2.0)
    train.set_title("training set", fontsize=style.SMALL, pad=2.0)
    handles = [Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none"),
               Line2D([0], [0], marker="$*$", linestyle="none", markersize=4.0, markerfacecolor=style.BLACK,
                      markeredgecolor=style.BLACK, markeredgewidth=0.0)]
    valid.legend(handles=handles, labels=["± half the last printed digit", subset_text], loc="lower left",
                 bbox_to_anchor=(-0.02, 1.13), ncol=1, fontsize=style.SMALL, handlelength=1.2, handletextpad=0.4,
                 labelspacing=0.3, borderaxespad=0.0)
    return style.save_source(pd.DataFrame(rows), NAME, "e")


def _panel_f(ax, thermostat: pd.DataFrame) -> Path:
    checked = thermostat[thermostat["run_log"].fillna("").astype(str) != ""].sort_values(
        ["series", "temperature_K"]).reset_index(drop=True)
    rows = []
    for index, (key, column, ok_column, _) in enumerate(THERMO_CHECKS):
        values = checked[["run_id", "series", "temperature_K", "n_atoms", column, ok_column]].dropna(subset=[column])
        if key == "mean_T":
            tolerance = np.where(values["n_atoms"] > 100, rule.THERMOSTAT_T_TOLERANCE_K[416],
                                 rule.THERMOSTAT_T_TOLERANCE_K[52])
        else:
            tolerance = np.full(len(values), 10.0 ** -rule.LOGGED_DECIMALS[key])
        scaled = values[column].abs().to_numpy() / tolerance
        offsets = common.jitter(len(values), 0.6, seed=index)
        for (_, run), y, dx, tol in zip(values.iterrows(), scaled, offsets, tolerance):
            ax.plot(index + dx, y, **_run_style(run["series"], run["temperature_K"], 2.8), zorder=3)
            rows.append({"run_id": run["run_id"], "run_type": _kind(run["series"]), "quantity": key,
                         "difference": run[column], "tolerance": tol, "difference_over_tolerance": y,
                         "x": index + dx, "ok": str(run[ok_column]).lower() == "true"})
    ax.axhline(1.0, color=style.BLACK, lw=0.7, zorder=1)
    ax.text(0.03, 1.04, "tolerance", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=style.SMALL)
    ax.set_xticks(range(len(THERMO_CHECKS)), [c[3] for c in THERMO_CHECKS])
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_xlim(-0.6, len(THERMO_CHECKS) - 0.4)
    top = max(1.6, max(r["difference_over_tolerance"] for r in rows) * 1.12)
    ax.set_ylim(-0.06 * top, top)
    ticks = np.arange(0.0, top, 0.5)
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_ylabel("|Difference| /\ntolerance")
    return style.save_source(pd.DataFrame(rows), NAME, "f")


def _seed_scatter(seeds: pd.DataFrame) -> pd.DataFrame:
    """Relative standard deviation of kinisi D over seeds 1-10 (seed 0 is the primary value), the s_seed of
    the reproduction rule."""
    scatter = seeds[seeds["seed"] != 0].groupby("run_id", sort=False)["D_cm2_s"]
    out = (scatter.std(ddof=1) / scatter.mean()).rename("s_seed").reset_index()
    out["n_seeds"] = scatter.size().to_numpy()
    return out


def _seed_table() -> pd.DataFrame:
    transport = common.read(TRANSPORT)
    seeds = common.read(SEEDS)
    scatter = _seed_scatter(seeds).merge(transport[["run_id", "series", "temperature_K", "kinisi_D_cm2_s",
                                                    "kinisi_D_std_cm2_s", "seed_scatter_D_rel"]], on="run_id")
    if not np.allclose(scatter["s_seed"], scatter["seed_scatter_D_rel"], rtol=1e-9, atol=0):
        raise ValueError("seed scatter differs from transport.csv")
    scatter["posterior_rel"] = scatter["kinisi_D_std_cm2_s"] / scatter["kinisi_D_cm2_s"]
    return scatter


def _panel_g(ax) -> Path:
    scatter = _seed_table()
    crossover = rule.KINISI_D_FLOOR / rule.KINISI_SEED_FACTOR * 100.0
    ax.axhline(crossover, color=style.BLACK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, crossover * 1.1, f"{rule.KINISI_SEED_FACTOR:g} s$_{{seed}}$ = {100 * rule.KINISI_D_FLOOR:g} %",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=style.SMALL)
    rows = []
    for row in scatter.itertuples():
        model = FAMILY_MODELS[SERIES_BY_ID[row.series]["model"]]
        ax.plot(row.posterior_rel * 100.0, row.s_seed * 100.0,
                **_series_style(row.series, style.MODEL_COLORS[model]), zorder=3)
        rows.append({"run_id": row.run_id, "series": row.series, "model": model, "run_type": _kind(row.series),
                     "temperature_K": row.temperature_K,
                     "kinisi_posterior_rel_std_percent": row.posterior_rel * 100.0,
                     "s_seed_percent": row.s_seed * 100.0, "n_seeds": row.n_seeds,
                     "rule_threshold_percent": max(rule.KINISI_SEED_FACTOR * row.s_seed,
                                                   rule.KINISI_D_FLOOR) * 100.0})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.0, 50.0)
    ax.set_ylim(0.09, 2.2)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
    ax.set_xlabel("kinisi posterior std / D (%)")
    ax.set_ylabel("s$_{seed}$ of kinisi D (%)")
    handles = [common.handle(common.model_style(model, size=MARKER), _model_name(model))
               for model in ("foundation", "mattersim", "ftG", "ftK")]
    ax.legend(handles=handles, loc="lower right", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.3,
              labelspacing=0.3, borderaxespad=0.4)
    return style.save_source(pd.DataFrame(rows), NAME, "g")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    runs, fits, thermostat, logs, configs = _load()
    series = _published_series(runs)
    runs = _positions(runs, series)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    ax_a, ax_b, ax_c, ax_d = (L.axes(LEFT, top, STRIP_WIDTH, height, row=letter, col="strip")
                              for letter, (top, height) in zip("abcd", STRIPS))
    sources = [_panel_a(ax_a, runs, series)]
    sources.append(_panel_kinisi(ax_b, runs, series, "kinisi_D", rule.KINISI_D_FLOOR, "kinisi D of each run",
                                 3.6, "b"))
    sources.append(_panel_kinisi(ax_c, runs, series, "kinisi_std", rule.KINISI_STD_FLOOR,
                                 "kinisi standard deviation of D", 12.0, "c"))
    sources.append(_panel_d(ax_d, fits, series))

    ax_e = [L.axes(E_LEFT, BOTTOM_TOP, E_VALID, BOTTOM_HEIGHT, row="bottom"),
            L.axes(E_LEFT + E_VALID + E_GAP, BOTTOM_TOP, E_TRAIN, BOTTOM_HEIGHT, row="bottom")]
    sources.append(_panel_e(ax_e, logs, configs))
    L.text(E_LEFT + (E_VALID + E_GAP + E_TRAIN) / 2, BOTTOM_TOP + BOTTOM_HEIGHT + 4.4,
           "Logged - re-evaluated force RMSE (meV/Å)", ha="center", va="top")
    ax_f = L.axes(F_LEFT, BOTTOM_TOP, F_WIDTH, BOTTOM_HEIGHT, row="bottom")
    sources.append(_panel_f(ax_f, thermostat))
    ax_g = L.axes(G_LEFT, BOTTOM_TOP, G_WIDTH, BOTTOM_HEIGHT, row="bottom")
    sources.append(_panel_g(ax_g))

    for letter, ax, (top, _) in zip("abcd", (ax_a, ax_b, ax_c, ax_d), STRIPS):
        L.letter(ax, letter, 4.0, top - 4.0 if letter != "a" else 7.5)
    title_top = BOTTOM_TOP - 11.0
    L.letter(ax_e[0], "e", 4.0, title_top)
    L.letter(ax_f, "f", F_LEFT - 13.0, title_top)
    L.letter(ax_g, "g", G_LEFT - 13.0, title_top)
    L.text(7.0, title_top - 0.4, "Force RMSE in the MACE training logs", ha="left", va="bottom")
    L.text(F_LEFT - 10.0, title_top - 0.4, "Thermostat statistics\nagainst the July run logs", ha="left",
           va="bottom", linespacing=1.1)
    L.text(G_LEFT - 10.0, title_top - 0.4, "Seed scatter of kinisi D,\nall runs", ha="left", va="bottom",
           linespacing=1.1)
    return L.save(*registry.target(KEY)) + sources


def _caption() -> str:
    runs, fits, thermostat, logs, configs = _load()
    series = _published_series(runs)
    nonzero = runs.loc[runs["pymatgen_D_rel_diff"] > 0, "pymatgen_D_rel_diff"]
    n_zero = int((runs["pymatgen_D_rel_diff"] == 0).sum())
    kd = runs["kinisi_D_rel_diff"] / np.maximum(rule.KINISI_SEED_FACTOR * runs["kinisi_D_seed_scatter"],
                                                rule.KINISI_D_FLOOR)
    ks = runs["kinisi_std_rel_diff"] / np.maximum(rule.KINISI_SEED_FACTOR * runs["kinisi_std_seed_scatter"],
                                                  rule.KINISI_STD_FLOOR)
    worst_d = runs.loc[kd.idxmax()]
    kinisi_classes = sorted(set(runs["kinisi_D_class"]) | set(runs["kinisi_std_class"]))
    kinisi_text = (f"Within the threshold: {int((kd <= 1.0).sum())} of {len(runs)} runs in b and "
                   f"{int((ks <= 1.0).sum())} of {len(runs)} in c (class {'/'.join(kinisi_classes)}).")
    threshold_d = max(rule.KINISI_SEED_FACTOR * worst_d["kinisi_D_seed_scatter"], rule.KINISI_D_FLOOR)
    worst_s = runs.loc[ks.idxmax()]
    max_3s_std = float((rule.KINISI_SEED_FACTOR * runs["kinisi_std_seed_scatter"]).max())
    fit_worst = float(fits[["d_ln_sigma300_in_uncertainties", "d_Ea_in_uncertainties"]].to_numpy().max())
    fit_worst_series = fits.loc[fits[["d_ln_sigma300_in_uncertainties", "d_Ea_in_uncertainties"]].max(axis=1).idxmax(),
                                "series"]
    classes = ", ".join(sorted(set(fits["class"])))
    quoted = (" The published values are therefore quoted for these series (rule item 2)."
              if set(fits["class"]) <= {"R1", "R2"} else "")
    pymatgen_classes = "/".join(sorted(set(runs["pymatgen_class"]) | set(runs["pymatgen_sigma_class"])))
    logs_valid = (logs["logged_valid_rmse_f_meV_A"] - logs["reevaluated_valid_rmse_f_meV_A"]).abs().max()
    ftg_s1 = logs.set_index("model").loc["ftG_s1"]
    ftg_s2 = logs.set_index("model").loc["ftG_s2"]
    n_train_g = int((configs["ftG_split"] == "train").sum())
    first, total = (int(v) for v in str(ftg_s1["train_subsets_matching"]).split("/"))
    second, total_2 = (int(v) for v in str(ftg_s2["train_subsets_matching"]).split("/"))
    if total != total_2:
        raise ValueError("different numbers of training subsets")
    matching = (first, second, total)
    train_diff = [row["logged_train_rmse_f_meV_A"] - row["reevaluated_train_rmse_f_meV_A"] for row in (ftg_s1, ftg_s2)]
    checked = thermostat[thermostat["run_log"].fillna("").astype(str) != ""]
    failed = checked[checked["drift_meV_atom_ps_ok"].astype(str).str.lower() == "false"]
    t_ok = int((checked["mean_T_ok"].astype(str).str.lower() == "true").sum())
    e_ok = int((checked["mean_E_per_atom_ok"].astype(str).str.lower() == "true").sum())
    d_ok = len(checked) - len(failed)
    short = set(SERIES_BY_ID[s]["production_ps"] for s in failed["series"])
    reason = (f" are all from 52-atom, {short.pop()}-ps runs, where the logged fit also includes the starting frame "
              "that the trajectory does not store" if (failed["n_atoms"] == 52).all() and len(short) == 1 else "")
    labels = {s["id"]: s["label"] for s in SERIES}
    n_new_logs = int((~checked["series"].isin(set(runs["series"]))).sum())
    scatter = _seed_table()
    crossover = rule.KINISI_D_FLOOR / rule.KINISI_SEED_FACTOR
    above = scatter[scatter["s_seed"] > crossover]
    above_52 = int((above["series"].map(_kind) == "cell_52").sum())
    above_other = [f"{labels[r.series]} at {r.temperature_K} K" for r in above.itertuples()
                   if _kind(r.series) != "cell_52"]
    above_text = f"{above_52} of the {int((scatter['series'].map(_kind) == 'cell_52').sum())} 52-atom runs" + (
        f" and {', '.join(above_other)}" if above_other else "")
    share = scatter["s_seed"] / scatter["posterior_rel"]
    n_valid = int(logs["logged_valid_rmse_f_meV_A"].notna().sum())
    n_train_logs = int(logs["logged_train_rmse_f_meV_A"].notna().sum())
    return (
        registry.heading(KEY) + "The comparison follows a "
        "rule committed before any trajectory was re-analysed (protocol/reproduction_rule.md), applied to the "
        f"{len(runs)} published runs of {len(series)} series (x axis of a-d: series in the order of "
        + registry.label("design", "c") + ", three runs each at 600, 800 and 1000 K). In a-d, f and g, marker "
        "shape is the run type (triangles, 52-atom cells; circles, the 416-atom, 200-ps production runs; inverted "
        "triangles, the 392-416-atom Cl-excess runs of 150 ps) and open markers (f, g) are the runs of the 2x2x2-label "
        "fine-tune (stage 1; every other model is filled); in a-c and f colour is the temperature. "
        f"**a**, Relative difference between the re-analysed and the published pymatgen D of each run (n = "
        f"{len(runs)}; one value per run, no error bars). In {n_zero} runs the two are bit-identical (drawn on the "
        f"'identical' row, at {common.sci(ZERO_FLOOR, 1)}); the others differ by "
        f"{nonzero.min():.2g}-{nonzero.max():.2g}, a few float64 rounding steps (dotted line, machine epsilon "
        f"{FLOAT64_EPS:.3g}). Solid line, the R1 limit ({common.sci(rule.PYMATGEN_EXACT, 1)}); dashed line, the R2 "
        f"limit ({common.sci(rule.PYMATGEN_REPRODUCED, 1)}). Classes: {pymatgen_classes} for every run in D and in "
        f"sigma. "
        f"**b**, Relative difference of the kinisi D (seed 0 here, an unseeded sampler in July) and **c**, of its "
        f"kinisi standard deviation (n = {len(runs)} runs each), with each run's threshold max(3 s_seed, "
        f"{rule.KINISI_D_FLOOR * 100:g} %) and max(3 s_seed, {rule.KINISI_STD_FLOOR * 100:g} %) (black bars), "
        f"where s_seed is the relative standard deviation over seeds 1-10 (n = 10; grey bars, 3 s_seed when below "
        f"the floor; dashed line, the floor). {kinisi_text} "
        f"The closest is {labels[worst_d['series']]} at {worst_d['temperature_K']} K in b "
        f"({worst_d['kinisi_D_rel_diff'] * 100:.3f} % against {threshold_d * 100:.3f} %, "
        f"{kd.max():.4f} of the threshold) and {labels[worst_s['series']]} at {worst_s['temperature_K']} K in c "
        f"({ks.max():.3f} of the threshold); in c, 3 s_seed never exceeds {max_3s_std * 100:.3g} %, so the "
        f"{rule.KINISI_STD_FLOOR * 100:g} % floor sets every threshold. "
        f"**d**, The n = {len(fits)} Arrhenius fits (one per series, three temperatures each) re-done with the "
        "re-analysed values: |d ln sigma(300 K)| in units of the published log half-width of the 1-sigma interval "
        "(black, left of each series) and |d Ea| in units of the published Ea error (grey, right). Light band, "
        f"R2-; darker band, R3. Solid line, the R2 limit ({rule.FIT_REPRODUCED:g}); dashed line, the R2- limit "
        f"({rule.FIT_CONSISTENT:g}). Classes: {classes}; the largest difference is {fit_worst:.3f} "
        f"({labels[fit_worst_series]}).{quoted} "
        f"**e**, Force RMSE written in the MACE training logs minus the same error re-evaluated from the model "
        f"files (meV/A), on the validation set (n = {n_valid} models) and on the training set (n = {n_train_logs}; "
        "MACE-MP-0 has no training log), for the four fine-tuned models (colour, model; stage 1 open, stage 2 "
        "filled) and the foundation model before fine-tuning (validation set of the Gamma fine-tune). Grey "
        f"bands, half the last printed digit. Validation errors agree to within {logs_valid:.3f} meV/A. The logged "
        f"training errors of the Gamma fine-tune (asterisks) cover {int(ftg_s1['train_evaluated_by_log'])} of its "
        f"{n_train_g} training configurations because MACE drops the last incomplete batch (protocol amendment A1); "
        f"against all {n_train_g} they differ by {train_diff[0]:+.2f} "
        f"and {train_diff[1]:+.2f} meV/A, and "
        f"{matching[0]} (stage 1) and {matching[1]} (stage 2) of the {matching[2]} subsets of "
        f"{int(ftg_s1['train_evaluated_by_log'])} configurations reproduce the logged values exactly. "
        f"**f**, Thermostat statistics recomputed from the trajectory against the n = {len(checked)} surviving "
        f"July run logs ({n_new_logs} of them belong to runs never analysed in July; per run in "
        + registry.label("stability", "e") + "): absolute difference in units of the "
        f"tolerance of the rule ({rule.THERMOSTAT_T_TOLERANCE_K[416]:g} K for 392-416 atoms and "
        f"{rule.THERMOSTAT_T_TOLERANCE_K[52]:g} K for 52 atoms for the mean temperature; one unit of the logged "
        f"rounding, {10.0 ** -rule.LOGGED_DECIMALS['mean_E_per_atom']:g} eV/atom and "
        f"{10.0 ** -rule.LOGGED_DECIMALS['drift_meV_atom_ps']:g} meV/atom/ps, for the mean energy and the drift; "
        "runs spread horizontally at random within each column). "
        f"The mean temperature agrees in {t_ok} of {len(checked)} runs and the mean energy in {e_ok} of "
        f"{len(checked)}; the drift agrees in {d_ok} of {len(checked)}. The {len(failed)} drift values outside the "
        f"tolerance ({', '.join(failed['run_id'])}; at most {failed['d_drift_meV_atom_ps'].abs().max():.4f} "
        f"meV/atom/ps){reason}; the check is reported as failed, as the rule states. "
        f"**g**, Monte-Carlo scatter of kinisi D, s_seed (relative standard deviation of D over seeds 1-10, n = 10 "
        f"per run; seed 0 is the primary value), against the relative width of the kinisi posterior (posterior "
        f"standard deviation over D, seed 0), for all n = {len(scatter)} runs (colour, the model that drove the "
        f"run). The rule accepts a kinisi D within max({rule.KINISI_SEED_FACTOR:g} s_seed, "
        f"{100 * rule.KINISI_D_FLOOR:g} %); above the dashed line (s_seed = {100 * crossover:.3f} %, where "
        f"{rule.KINISI_SEED_FACTOR:g} s_seed = {100 * rule.KINISI_D_FLOOR:g} %) the seed term sets the threshold, "
        f"which is the case in {len(above)} of {len(scatter)} runs ({above_text}). s_seed is "
        f"{100 * scatter['s_seed'].min():.2f}-{100 * scatter['s_seed'].max():.2f} %, "
        f"{share.min():.3f}-{share.max():.3f} times the posterior width. "
        + common.sources([RUNS, FITS, THERMOSTAT, LOGS, CONFIGS, TRANSPORT, SEEDS, RULE, AMENDMENTS])
    )


CAPTION = _caption()
