"""Extended Data Fig. 2: re-analysis against the values published in July 2026.

a, pymatgen D of each of the 33 published runs, relative difference to the published value (exact
zeros drawn at a floor), with the R1 and R2 limits; b, kinisi D and c, its kinisi standard deviation,
relative difference against each run's threshold max(3 s_seed, floor); d, the 11 Arrhenius fits:
differences in ln sigma(300 K) and Ea in units of the published uncertainty, with the R2 (0.1) and
R2- (1) limits; e, training logs: logged minus re-evaluated force RMSE of the fine-tuned models and of
the foundation model, against the printed precision; f, thermostat statistics recomputed from the
trajectories against the surviving July run logs, in units of the tolerance of the rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from ..constants import SERIES, SERIES_BY_ID, TEMPERATURES
from ..transport import reproduction as rule
from . import common, style

NAME = "ed02"
WIDTH, HEIGHT = 183.0, 156.0
RUNS = paths.RESULTS / "reproduction_runs.csv"
FITS = paths.RESULTS / "reproduction_fits.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
LOGS = paths.MODELS / "log_reproduction.csv"
CONFIGS = paths.DFT / "configs.csv"
RULE = paths.PROTOCOL / "reproduction_rule.md"
AMENDMENTS = paths.PROTOCOL / "amendments.md"

ZERO_FLOOR = 1e-17  # exact zeros of the pymatgen difference are drawn here
FLOAT64_EPS = float(np.finfo(np.float64).eps)
GROUP = 4  # x units per series: three runs and a gap
LOG_MODELS = ("ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2", "foundation")
THERMO_CHECKS = (("mean_T", "d_mean_T_K", "mean_T_ok", "Mean T"),
                 ("mean_E_per_atom", "d_mean_E_per_atom", "mean_E_per_atom_ok", "Mean E"),
                 ("drift_meV_atom_ps", "d_drift_meV_atom_ps", "drift_meV_atom_ps_ok", "Drift"))
PART_TEXT = {"A": "A: universal potentials", "C": "C", "D": "D: Cl excess, MACE-MP-0"}


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


def _short(series: dict) -> str:
    if series["part"] == "D":
        x = series["cl_content"] - 1.0
        name = f"x = {x:.2f}" if x else "x = 0"
    elif series["model"] in ("foundation", "mattersim"):
        name = style.MODEL_LABELS[series["model"]]
    else:
        name = common.SHORT_LABELS[f"{series['model']}_s2"]
    return f"{name}\n{series['n_atoms']} atoms\n{series['production_ps']} ps"


def _frame_runs(ax, series: list[dict], labels: bool) -> None:
    n = len(series)
    ax.set_xlim(-0.9, (n - 1) * GROUP + 2.9)
    for g in range(1, n):
        ax.axvline(g * GROUP - 1.0, color=style.LIGHT_GREY, lw=0.4, zorder=0)
    ax.set_xticks([g * GROUP + 1.0 for g in range(n)], [_short(s) for s in series] if labels else [])
    ax.tick_params(axis="x", length=0, pad=2.0, labelsize=style.SMALL)
    if not labels:
        return
    blend = ax.get_xaxis_transform()
    parts = [s["part"] for s in series]
    for part in dict.fromkeys(parts):
        members = [g for g, p in enumerate(parts) if p == part]
        x0, x1 = members[0] * GROUP - 0.6, members[-1] * GROUP + 2.6
        ax.annotate("", xy=(x0, -0.36), xytext=(x1, -0.36), xycoords=blend, textcoords=blend,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "black", "shrinkA": 0, "shrinkB": 0})
        ax.text((x0 + x1) / 2.0, -0.40, PART_TEXT[part], transform=blend, ha="center", va="top",
                fontsize=style.SMALL)


def _sci(value: float) -> str:
    """1e-09 -> '1e-9' (caption text)."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _title(ax, text: str) -> None:
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=style.FONT_SIZE)


def _temperature_handles(marker: str = "o") -> list[Line2D]:
    return [Line2D([0], [0], marker=marker, linestyle="none", markersize=3.2,
                   markerfacecolor=style.TEMPERATURE_COLORS[t],
                   markeredgecolor=style.TEMPERATURE_COLORS[t], label=f"{t} K") for t in TEMPERATURES]


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, runs: pd.DataFrame, series: list[dict]) -> Path:
    _frame_runs(ax, series, labels=False)
    zero = runs["pymatgen_D_rel_diff"] == 0.0
    plotted = runs["pymatgen_D_rel_diff"].where(~zero, ZERO_FLOOR)
    for temperature in TEMPERATURES:
        keep = runs["temperature_K"] == temperature
        color = style.TEMPERATURE_COLORS[temperature]
        ax.plot(runs.loc[keep & ~zero, "x"], plotted[keep & ~zero], marker="o", markersize=3.0, linestyle="none",
                markerfacecolor=color, markeredgecolor=color, zorder=3)
        ax.plot(runs.loc[keep & zero, "x"], plotted[keep & zero], marker="o", markersize=3.0, linestyle="none",
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(4e-18, 2e-5)
    ax.axhline(rule.PYMATGEN_EXACT, color="black", lw=0.7, zorder=2)
    ax.axhline(rule.PYMATGEN_REPRODUCED, color="black", lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.axhline(FLOAT64_EPS, color=style.GREY, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    right = ax.get_xlim()[1]
    for value, name in ((rule.PYMATGEN_EXACT, "R1"), (rule.PYMATGEN_REPRODUCED, "R2")):
        ax.text(right - 0.2, value * 1.6, f"{name} limit, $10^{{{int(round(np.log10(value)))}}}$", ha="right",
                va="bottom", fontsize=style.SMALL)
    style.log_ticks(ax, "y")
    ax.set_yticks([ZERO_FLOOR, 1e-15, 1e-12, 1e-9, 1e-6],
                  ["identical", "$10^{-15}$", "$10^{-12}$", "$10^{-9}$", "$10^{-6}$"])
    ax.yaxis.set_minor_locator(_minor_decades(1e-16, 1e-5, ax.get_yticks()))
    ax.set_ylabel("Relative\ndifference")
    _title(ax, "pymatgen D of each run")
    handles = _temperature_handles()
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markerfacecolor="white",
                          markeredgecolor="black", markeredgewidth=0.8, label="identical to the published value"))
    handles.append(Line2D([0], [0], color=style.GREY, lw=0.5, ls=(0, (1, 1.5)), label="float64 machine epsilon"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, fontsize=style.SMALL,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2)
    source = runs[["run_id", "series", "temperature_K", "x", "pymatgen_D_rel_diff", "pymatgen_class"]].assign(
        plotted_value=plotted, identical=zero, r1_limit=rule.PYMATGEN_EXACT, r2_limit=rule.PYMATGEN_REPRODUCED)
    return style.save_source(source, NAME, "a")


def _minor_decades(start: float, stop: float, majors) -> object:
    """Minor ticks at the unlabelled decades from ``start`` to ``stop`` (none next to the zero floor)."""
    from matplotlib.ticker import FixedLocator

    exponents = range(int(round(np.log10(start))), int(round(np.log10(stop))) + 1)
    labelled = {int(round(np.log10(m))) for m in majors}
    return FixedLocator([10.0 ** e for e in exponents if e not in labelled])


def _panel_kinisi(ax, runs: pd.DataFrame, series: list[dict], quantity: str, floor: float, title: str,
                  top: float, letter: str, labels: bool) -> Path:
    _frame_runs(ax, series, labels=labels)
    diff = runs[f"{quantity}_rel_diff"] * 100.0
    three_s = rule.KINISI_SEED_FACTOR * runs[f"{quantity}_seed_scatter"] * 100.0
    threshold = np.maximum(three_s, floor * 100.0)
    ratio = diff / threshold
    ax.axhline(floor * 100.0, color=style.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for x, t, s in zip(runs["x"], threshold, three_s):
        ax.plot([x - 0.38, x + 0.38], [t, t], color="black", lw=1.0, solid_capstyle="butt", zorder=2)
        if s < t:
            ax.plot([x - 0.25, x + 0.25], [s, s], color=style.GREY, lw=0.8, solid_capstyle="butt", zorder=2)
    for temperature in TEMPERATURES:
        keep = runs["temperature_K"] == temperature
        color = style.TEMPERATURE_COLORS[temperature]
        ax.plot(runs.loc[keep, "x"], diff[keep], marker="o", markersize=3.0, linestyle="none",
                markerfacecolor=color, markeredgecolor=color, zorder=3)
    worst = int(np.argmax(ratio.to_numpy()))
    wx = runs["x"].iloc[worst]
    left_side = wx > np.mean(ax.get_xlim())
    ax.annotate(f"{ratio.iloc[worst]:.3f} of its threshold", xy=(wx, diff.iloc[worst]),
                xytext=(wx + (-1.6 if left_side else 1.6), top * 0.93), ha="right" if left_side else "left",
                va="top", fontsize=style.SMALL,
                arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "black", "shrinkA": 0.5, "shrinkB": 2.0})
    ax.set_ylim(0, top)
    ax.set_ylabel("Relative\ndifference (%)")
    _title(ax, title)
    handles = [Line2D([0], [0], color="black", lw=1.0, label=f"threshold max(3 s$_{{seed}}$, {floor * 100:g} %)"),
               Line2D([0], [0], color=style.GREY, lw=0.8, label="3 s$_{seed}$ (when below the floor)"),
               Line2D([0], [0], color=style.GREY, lw=0.6, ls=(0, (3, 2)), label=f"floor {floor * 100:g} %")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=style.SMALL,
              handlelength=1.4, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    source = runs[["run_id", "series", "temperature_K", "x", f"{quantity}_rel_diff", f"{quantity}_seed_scatter",
                   f"{quantity}_class"]].assign(rel_diff_percent=diff, three_s_seed_percent=three_s,
                                                threshold_percent=threshold, difference_over_threshold=ratio)
    return style.save_source(source, NAME, letter)


def _panel_d(ax, fits: pd.DataFrame, series: list[dict]) -> Path:
    order = {s["id"]: i for i, s in enumerate(series)}
    fits = fits.assign(group=fits["series"].map(order)).sort_values("group").reset_index(drop=True)
    x = fits["group"] * GROUP + 1.0
    _frame_runs(ax, series, labels=True)
    lo, hi = 1e-4, 4.0
    ax.axhspan(rule.FIT_REPRODUCED, rule.FIT_CONSISTENT, color=style.LIGHT_GREY, alpha=0.45, lw=0, zorder=0)
    ax.axhspan(rule.FIT_CONSISTENT, hi, color=style.LIGHT_GREY, alpha=0.9, lw=0, zorder=0)
    ax.axhline(rule.FIT_REPRODUCED, color="black", lw=0.7, zorder=1)
    ax.axhline(rule.FIT_CONSISTENT, color="black", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(x - 0.45, fits["d_ln_sigma300_in_uncertainties"], marker="o", markersize=3.2, linestyle="none",
            markerfacecolor="black", markeredgecolor="black", zorder=3)
    ax.plot(x + 0.45, fits["d_Ea_in_uncertainties"], marker="s", markersize=3.0, linestyle="none",
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    style.log_ticks(ax, "y")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["$10^{-4}$", "$10^{-3}$", "0.01", "0.1", "1"])
    right = ax.get_xlim()[1]
    for y, text in ((np.sqrt(lo * rule.FIT_REPRODUCED) * 3.0, "R2"),
                    (np.sqrt(rule.FIT_REPRODUCED * rule.FIT_CONSISTENT), "R2-"),
                    (np.sqrt(rule.FIT_CONSISTENT * hi), "R3")):
        ax.text(right - 0.3, y, text, ha="right", va="center", fontsize=style.SMALL, color="#333333")
    ax.set_ylabel("|Difference| /\nuncertainty")
    _title(ax, "Arrhenius fit of each series")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markerfacecolor="black",
                      markeredgecolor="black", label="|d ln $\\sigma$(300 K)| / published log half-width"),
               Line2D([0], [0], marker="s", linestyle="none", markersize=3.0, markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=0.8, label="|d E$_a$| / published E$_a$ error")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=style.SMALL,
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0.0)
    source = fits[["series", "class", "d_ln_sigma300", "published_log_half_width",
                   "d_ln_sigma300_in_uncertainties", "d_Ea_eV", "published_Ea_err_eV",
                   "d_Ea_in_uncertainties"]].assign(x=x, r2_limit=rule.FIT_REPRODUCED,
                                                    r2_minus_limit=rule.FIT_CONSISTENT)
    return style.save_source(source, NAME, "d")


def _decimals(value: float) -> int:
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def _panel_e(ax, logs: pd.DataFrame, configs: pd.DataFrame) -> Path:
    logs = logs.set_index("model").loc[list(LOG_MODELS)].reset_index()
    n_train = {run: int((configs[f"{run}_split"] == "train").sum()) for run in ("ftG", "ftK")}
    rows = []
    span = 0.3
    for index, row in logs.iterrows():
        y = index
        model = row["model"]
        color = style.MODEL_COLORS[model]
        filled = common.filled(model)
        for kind, dy, marker in (("valid", -0.17, "o"), ("train", 0.17, "s")):
            logged = row[f"logged_{kind}_rmse_f_meV_A"]
            if pd.isna(logged):
                continue
            half = 0.5 * 10.0 ** -_decimals(logged)
            ax.add_patch(Rectangle((-half, y + dy - 0.15), 2 * half, 0.3, facecolor=style.LIGHT_GREY,
                                   edgecolor="none", zorder=0))
            difference = logged - row[f"reevaluated_{kind}_rmse_f_meV_A"]
            subset = kind == "train" and model != "foundation" and \
                int(row["train_evaluated_by_log"]) < n_train[model.split("_")[0]]
            ax.plot(difference, y + dy, marker=marker, markersize=3.2, linestyle="none",
                    markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=0.8, zorder=3)
            if subset:
                subset_text = (f"logged over {int(row['train_evaluated_by_log'])} of "
                               f"{n_train[model.split('_')[0]]} training configurations")
                ax.text(difference + 0.01, y + dy - 0.05, "*", ha="left", va="center", fontsize=style.LABEL)
            rows.append({"model": model, "labels": row["labels"], "epoch": row["epoch"], "set": kind,
                         "logged_meV_A": logged, "reevaluated_meV_A": row[f"reevaluated_{kind}_rmse_f_meV_A"],
                         "difference_meV_A": difference, "half_printed_digit_meV_A": half,
                         "configurations_in_log": int(row["train_evaluated_by_log"])
                         if kind == "train" and not pd.isna(row["train_evaluated_by_log"]) else "",
                         "subset_in_log": subset})
    ax.axvline(0, color=style.GREY, lw=0.5, zorder=1)
    ax.set_xlim(-span / 2.0, span)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3], ["-0.1", "0", "0.1", "0.2", "0.3"])
    ax.set_ylim(len(logs) - 0.45, -0.55)
    names = []
    for row in logs.itertuples():
        name = common.SHORT_LABELS[row.model]
        names.append(f"{name}, epoch {int(row.epoch)}" if row.model != "foundation"
                     else f"{name}, before fine-tuning")
    ax.set_yticks(range(len(logs)), names, fontsize=style.SMALL)
    for tick, model in zip(ax.get_yticklabels(), LOG_MODELS):
        tick.set_color(style.MODEL_COLORS[model])
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.set_xlabel("Logged - re-evaluated force RMSE (meV/Å)")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markerfacecolor="black",
                      markeredgecolor="black", label="validation"),
               Line2D([0], [0], marker="s", linestyle="none", markersize=3.2, markerfacecolor="black",
                      markeredgecolor="black", label="training"),
               Rectangle((0, 0), 1, 1, facecolor=style.LIGHT_GREY, edgecolor="none",
                         label="± half the last printed digit"),
               Line2D([0], [0], marker="$*$", linestyle="none", markersize=4.0, markerfacecolor="black",
                      markeredgecolor="black", markeredgewidth=0.0, label=subset_text)]
    ax.legend(handles=[handles[i] for i in (0, 2, 1, 3)], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
              fontsize=style.SMALL, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2, labelspacing=0.3)
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
        offsets = common.jitter(len(values), 0.55, seed=index)
        for (_, run), y, dx, tol in zip(values.iterrows(), scaled, offsets, tolerance):
            color = style.TEMPERATURE_COLORS[int(run["temperature_K"])]
            small = run["n_atoms"] < 100
            ax.plot(index + dx, y, marker="o" if small else "s", markersize=2.8, linestyle="none",
                    markerfacecolor="white" if small else color, markeredgecolor=color, markeredgewidth=0.7,
                    zorder=3)
            rows.append({"run_id": run["run_id"], "quantity": key, "difference": run[column], "tolerance": tol,
                         "difference_over_tolerance": y, "x": index + dx,
                         "ok": str(run[ok_column]).lower() == "true"})
    ax.axhline(1.0, color="black", lw=0.7, zorder=1)
    ax.text(-0.55, 1.04, "tolerance", ha="left", va="bottom", fontsize=style.SMALL)
    ax.set_xticks(range(len(THERMO_CHECKS)), [c[3] for c in THERMO_CHECKS])
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_xlim(-0.6, len(THERMO_CHECKS) - 0.4)
    top = max(1.6, max(r["difference_over_tolerance"] for r in rows) * 1.12)
    ax.set_ylim(0, top)
    ax.set_ylabel("|Difference| /\ntolerance")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=2.8, markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=0.7, label="52 atoms"),
               Line2D([0], [0], marker="s", linestyle="none", markersize=2.8, markerfacecolor="black",
                      markeredgecolor="black", label="392-416 atoms")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=style.SMALL,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    return style.save_source(pd.DataFrame(rows), NAME, "f")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    runs, fits, thermostat, logs, configs = _load()
    series = _published_series(runs)
    runs = _positions(runs, series)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    left, width, height = 20.0, 161.0, 18.0
    ax_a = L.axes(left, 8.0, width, height)
    sources = [_panel_a(ax_a, runs, series)]
    ax_b = L.axes(left, 33.0, width, height)
    sources.append(_panel_kinisi(ax_b, runs, series, "kinisi_D", rule.KINISI_D_FLOOR,
                                 "kinisi D of each run", 3.6, "b", labels=False))
    ax_c = L.axes(left, 58.0, width, height)
    sources.append(_panel_kinisi(ax_c, runs, series, "kinisi_std", rule.KINISI_STD_FLOOR,
                                 "kinisi standard deviation of D of each run", 12.0, "c", labels=False))
    ax_d = L.axes(left, 83.0, width, height)
    sources.append(_panel_d(ax_d, fits, series))

    top2, height2 = 125.0, 23.0
    ax_e = L.axes(36.0, top2, 58.0, height2)
    sources.append(_panel_e(ax_e, logs, configs))
    ax_f = L.axes(127.0, top2, 54.0, height2)
    sources.append(_panel_f(ax_f, thermostat))

    L.letter(ax_a, "a", 3.0, 5.0)
    L.letter(ax_b, "b", 3.0, 30.0)
    L.letter(ax_c, "c", 3.0, 55.0)
    L.letter(ax_d, "d", 3.0, 80.0)
    L.letter(ax_e, "e", 3.0, 117.0)
    L.letter(ax_f, "f", 108.0, 117.0)
    L.text(6.0, 116.6, "Force RMSE in the MACE training logs", ha="left", va="bottom")
    L.text(111.0, 116.6, "Thermostat statistics against the July run logs", ha="left", va="bottom")
    return style.save(fig, "ed02_reproduction", "extended") + sources


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
    return (
        "Extended Data Fig. 2 | Re-analysis against the values published in July 2026. The comparison follows a "
        "rule committed before any trajectory was re-analysed (protocol/reproduction_rule.md), applied to the "
        f"{len(runs)} published runs of {len(series)} series (x axis of a-d: series in the order of Fig. 1c, three "
        "runs each at 600, 800 and 1000 K, coloured by temperature). "
        f"**a**, Relative difference between the re-analysed and the published pymatgen D of each run. In "
        f"{n_zero} runs the two are bit-identical (open circles, drawn at {_sci(ZERO_FLOOR)}); the others differ by "
        f"{nonzero.min():.2g}-{nonzero.max():.2g}, a few float64 rounding steps (dotted line, machine epsilon "
        f"{FLOAT64_EPS:.3g}). Solid line, the R1 limit ({_sci(rule.PYMATGEN_EXACT)}); dashed line, the R2 limit "
        f"({_sci(rule.PYMATGEN_REPRODUCED)}). Classes: {pymatgen_classes} for every run in D and in sigma. "
        f"**b**, Relative difference of the kinisi D (seed 0 here, an unseeded sampler in July) and **c**, of its "
        f"kinisi standard deviation, with each run's threshold max(3 s_seed, {rule.KINISI_D_FLOOR * 100:g} %) and "
        f"max(3 s_seed, {rule.KINISI_STD_FLOOR * 100:g} %) (black bars), where s_seed is the relative scatter of "
        f"seeds 1-10 (grey bars, 3 s_seed when below the floor; dashed line, the floor). {kinisi_text} "
        f"The closest is {labels[worst_d['series']]} at {worst_d['temperature_K']} K in b "
        f"({worst_d['kinisi_D_rel_diff'] * 100:.3f} % against {threshold_d * 100:.3f} %, "
        f"{kd.max():.4f} of the threshold) and {labels[worst_s['series']]} at {worst_s['temperature_K']} K in c "
        f"({ks.max():.3f} of the threshold); in c, 3 s_seed never exceeds {max_3s_std * 100:.3g} %, so the "
        f"{rule.KINISI_STD_FLOOR * 100:g} % floor sets every threshold. "
        f"**d**, The {len(fits)} Arrhenius fits re-done with the re-analysed values: |d ln sigma(300 K)| in units of "
        f"the published log half-width of the 1-sigma interval (filled circles) and |d Ea| in units of the "
        f"published Ea error (open squares). Solid line, the R2 limit ({rule.FIT_REPRODUCED:g}); dashed line, the "
        f"R2- limit ({rule.FIT_CONSISTENT:g}). Classes: {classes}; the largest difference is {fit_worst:.3f} "
        f"({labels[fit_worst_series]}).{quoted} "
        f"**e**, Force RMSE written in the MACE training logs minus the same error re-evaluated from the model "
        f"files (meV/A), for the four fine-tuned models (stage 1 open, stage 2 filled; colours as in Fig. 3) and "
        f"the foundation model before fine-tuning (validation set of the Gamma fine-tune). Grey bars, half the "
        f"last printed digit. Validation errors agree to within {logs_valid:.3f} meV/A. The logged training errors "
        f"of the Gamma fine-tune (asterisks) cover {int(ftg_s1['train_evaluated_by_log'])} of its "
        f"{n_train_g} training configurations because MACE drops the last incomplete batch (protocol amendment A1); "
        f"against all {n_train_g} they differ by {train_diff[0]:+.2f} "
        f"and {train_diff[1]:+.2f} meV/A, and "
        f"{matching[0]} (stage 1) and {matching[1]} (stage 2) of the {matching[2]} subsets of "
        f"{int(ftg_s1['train_evaluated_by_log'])} configurations reproduce the logged values exactly. "
        f"**f**, Thermostat statistics recomputed from the trajectory against the {len(checked)} surviving July run "
        f"logs ({n_new_logs} of them belong to runs never analysed in July): absolute difference in units of the "
        f"tolerance of the rule ({rule.THERMOSTAT_T_TOLERANCE_K[416]:g} K for 392-416 atoms and "
        f"{rule.THERMOSTAT_T_TOLERANCE_K[52]:g} K for 52 atoms for the mean temperature; one unit of the logged "
        f"rounding, {10.0 ** -rule.LOGGED_DECIMALS['mean_E_per_atom']:g} eV/atom and "
        f"{10.0 ** -rule.LOGGED_DECIMALS['drift_meV_atom_ps']:g} meV/atom/ps, for the mean energy and the drift); "
        f"colour, temperature; circles, 52 atoms; squares, 392-416 atoms. "
        f"The mean temperature agrees in {t_ok} of {len(checked)} runs and the mean energy in {e_ok} of "
        f"{len(checked)}; the drift agrees in {d_ok} of {len(checked)}. The {len(failed)} drift values outside the "
        f"tolerance ({', '.join(failed['run_id'])}; at most {failed['d_drift_meV_atom_ps'].abs().max():.4f} "
        f"meV/atom/ps){reason}; the check is reported as failed, as the rule states. "
        + common.sources([RUNS, FITS, THERMOSTAT, LOGS, CONFIGS, RULE, AMENDMENTS])
    )


CAPTION = _caption()
