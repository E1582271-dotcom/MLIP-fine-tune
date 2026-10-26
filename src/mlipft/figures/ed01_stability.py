"""Extended Data Fig. 1: temperature, energy and pressure of every MD run.

a, mean temperature minus the target for all 42 runs, with the standard error of the mean from
5-ps blocks; b, relative temperature fluctuation std(T)/T against the canonical sqrt(2/(3N));
c, drift of the potential energy per atom; d, 0.5-ps block series of temperature, potential energy
and pressure of the four 416-atom, 200-ps production runs at 1000 K; e, the statistics recomputed
from the trajectories against the July run logs, for the 19 runs whose log survives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import constants as C
from .. import paths
from ..transport import reproduction as R
from . import common, style

NAME = "ed01"
WIDTH, HEIGHT = 183.0, 193.0
THERMOSTAT = paths.RESULTS / "thermostat.csv"
TIMESERIES = paths.MD / "timeseries.csv.gz"
BLOCK_PS = 5.0  # blocks for the standard errors (ten 0.5-ps blocks)
SERIES_TEMPERATURE = 1000  # temperature of the production time series in d
SERIES_ORDER = ("mace_52_050", "mace_52_100", "mace_52_150", "mattersim_52_050", "mattersim_52_100",
                "mattersim_52_150", "mace_416_200", "mattersim_416_200", "ftG_416_200", "ftK_416_200",
                "clx000_416_150", "clx025_408_150", "clx050_400_150", "clx075_392_150")
TICKS = {"mace_52_050": "50 ps", "mace_52_100": "100 ps", "mace_52_150": "150 ps",
         "mattersim_52_050": "50 ps", "mattersim_52_100": "100 ps", "mattersim_52_150": "150 ps",
         "mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim", "ftG_416_200": "Gamma labels",
         "ftK_416_200": "2x2x2 labels", "clx000_416_150": "x = 0", "clx025_408_150": "x = 0.25",
         "clx050_400_150": "x = 0.50", "clx075_392_150": "x = 0.75"}
GROUPS = (("MACE-MP-0\n52 atoms", SERIES_ORDER[0:3]), ("MatterSim\n52 atoms", SERIES_ORDER[3:6]),
          ("416 atoms\n200 ps", SERIES_ORDER[6:10]), ("Cl-excess\n150 ps", SERIES_ORDER[10:14]))
GROUP_GAP = 0.8
GROUP_GAP_E, WIDE_SLOT = 0.6, 1.6  # panel e: gap between groups, slot width of the ~400-atom runs
T_OFFSETS = dict(zip(C.TEMPERATURES, (-0.24, 0.0, 0.24)))
PRODUCTION_MODELS = {"mace_416_200": "foundation", "mattersim_416_200": "mattersim", "ftG_416_200": "ftG",
                     "ftK_416_200": "ftK"}
LOG_TICKS = {"mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim", "ftG_416_200": "Gamma",
             "ftK_416_200": "2x2x2", "clx000_416_150": "x = 0", "clx025_408_150": "x = 0.25",
             "clx050_400_150": "x = 0.50", "clx075_392_150": "x = 0.75"}


def _load():
    thermo = common.read(THERMOSTAT)
    series = common.read(TIMESERIES)
    return thermo, series


def _is_true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower() == "true"


def _logged(thermo: pd.DataFrame) -> pd.Series:
    return thermo["run_log"].fillna("").astype(str) != ""


def _block_errors(series: pd.DataFrame) -> pd.DataFrame:
    """Per run: standard error of the mean temperature and of the energy drift from ``BLOCK_PS`` blocks.

    The 0.5-ps block means are averaged into 5-ps blocks, which are close to uncorrelated; the
    error of the mean is std(blocks)/sqrt(n), the error of the drift that of a straight-line fit.
    """
    rows = []
    for run, frame in series.groupby("run_id", sort=False):
        frame = frame.sort_values("time_ps")
        step = float(np.median(np.diff(frame["time_ps"])))
        size = int(round(BLOCK_PS / step))
        n = len(frame) // size * size
        time = frame["time_ps"].to_numpy()[:n].reshape(-1, size).mean(axis=1)
        temperature = frame["T_K"].to_numpy()[:n].reshape(-1, size).mean(axis=1)
        energy = frame["E_eV_atom"].to_numpy()[:n].reshape(-1, size).mean(axis=1) * 1000.0
        slope, intercept = np.polyfit(time, energy, 1)
        residual = energy - (slope * time + intercept)
        drift_se = np.sqrt((residual ** 2).sum() / (len(time) - 2) / ((time - time.mean()) ** 2).sum())
        rows.append({"run_id": run, "n_blocks": len(time),
                     "sem_T_K": float(temperature.std(ddof=1) / np.sqrt(len(temperature))),
                     "drift_se_meV_atom_ps": float(drift_se)})
    return pd.DataFrame(rows)


def _positions() -> dict[str, float]:
    positions, x = {}, 0.0
    for index, (_, members) in enumerate(GROUPS):
        if index:
            x += GROUP_GAP
        for series in members:
            positions[series] = x
            x += 1.0
    return positions


def _series_axis(ax, positions: dict[str, float], labels: bool) -> None:
    """Series as x ticks (rotated) with a bracket and name per group below them."""
    xs = [positions[s] for s in SERIES_ORDER]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_xticks(xs, [TICKS[s] for s in SERIES_ORDER] if labels else [])
    ax.tick_params(axis="x", length=1.5, pad=1.5, labelsize=style.SMALL)
    for index, (_, members) in enumerate(GROUPS):
        if index:
            ax.axvline(positions[members[0]] - (1.0 + GROUP_GAP) / 2, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    if not labels:
        return
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    blend = ax.get_xaxis_transform()
    line = transforms.offset_copy(blend, fig=ax.figure, y=-38.0, units="points")
    text = transforms.offset_copy(blend, fig=ax.figure, y=-39.5, units="points")
    for name, members in GROUPS:
        first, last = positions[members[0]], positions[members[-1]]
        ax.plot([first - 0.35, last + 0.35], [0, 0], color="black", lw=0.5, transform=line, clip_on=False)
        ax.text((first + last) / 2, 0, name, transform=text, ha="center", va="top", fontsize=style.SMALL,
                linespacing=1.1)


def _run_markers(ax, x: float, y: float, temperature: int, logged: bool, yerr: float | None = None,
                 size: float = 3.2) -> None:
    color = style.TEMPERATURE_COLORS[temperature]
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.0, capthick=0.6,
                    zorder=2)
    ax.plot(x, y, marker="o", linestyle="none", markersize=size, markeredgewidth=0.7, markeredgecolor=color,
            markerfacecolor=color if logged else "white", zorder=3)


def _run_legend(ax, error_label: str | None, **kwargs) -> None:
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markeredgewidth=0.7,
                      markeredgecolor=style.TEMPERATURE_COLORS[t], markerfacecolor=style.TEMPERATURE_COLORS[t],
                      label=f"{t} K") for t in C.TEMPERATURES]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markeredgewidth=0.7,
                          markeredgecolor="black", markerfacecolor="black", label="July run log survives"))
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markeredgewidth=0.7,
                          markeredgecolor="black", markerfacecolor="white", label="from the trajectory only"))
    if error_label:
        handles.append(Line2D([0], [0], color="black", lw=0.6, marker="_", markersize=2.5, label=error_label))
    ax.legend(handles=handles, fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.0, labelspacing=0.3,
              borderaxespad=0.0, **kwargs)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, thermo: pd.DataFrame, errors: pd.DataFrame, positions: dict[str, float]) -> None:
    frame = thermo.merge(errors, on="run_id")
    logged = _logged(frame)
    rows = []
    ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    for (_, row), has_log in zip(frame.iterrows(), logged):
        x = positions[row["series"]] + T_OFFSETS[row["temperature_K"]]
        dt = row["mean_T_K"] - row["temperature_K"]
        _run_markers(ax, x, dt, row["temperature_K"], has_log, row["sem_T_K"])
        rows.append({"run_id": row["run_id"], "series": row["series"], "temperature_K": row["temperature_K"],
                     "n_atoms": row["n_atoms"], "x": x, "mean_T_minus_target_K": dt,
                     "sem_T_K_5ps_blocks": row["sem_T_K"], "n_blocks": row["n_blocks"], "run_log": has_log})
    _series_axis(ax, positions, labels=False)
    ax.set_ylim(-22, 42)
    ax.set_yticks([-20, 0, 20, 40])
    ax.set_ylabel("Mean T - target (K)")
    _run_legend(ax, "±1 s.e. (5-ps blocks)", loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3)
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, thermo: pd.DataFrame, positions: dict[str, float]) -> None:
    logged = _logged(thermo)
    rows = []
    for series in SERIES_ORDER:
        block = thermo[thermo["series"] == series]
        canonical = float(block["canonical_rel_std_T"].iloc[0]) * 100.0
        x = positions[series]
        ax.plot([x - 0.45, x + 0.45], [canonical, canonical], color="black", lw=0.8, solid_capstyle="butt",
                zorder=2)
    for (_, row), has_log in zip(thermo.iterrows(), logged):
        x = positions[row["series"]] + T_OFFSETS[row["temperature_K"]]
        _run_markers(ax, x, row["rel_std_T"] * 100.0, row["temperature_K"], has_log)
        rows.append({"run_id": row["run_id"], "series": row["series"], "temperature_K": row["temperature_K"],
                     "n_atoms": row["n_atoms"], "x": x, "rel_std_T_percent": row["rel_std_T"] * 100.0,
                     "canonical_sqrt_2_over_3N_percent": row["canonical_rel_std_T"] * 100.0, "run_log": has_log})
    _series_axis(ax, positions, labels=False)
    ax.set_yscale("log")
    ax.set_ylim(3.3, 16.0)
    style.log_ticks(ax, "y")
    ax.set_yticks([4, 6, 8, 10, 15], ["4", "6", "8", "10", "15"])
    ax.set_ylabel("std(T)/T (%)")
    n52 = 52
    big = thermo.loc[thermo["n_atoms"] > 100, "n_atoms"]
    first_big = positions[GROUPS[2][1][0]]
    ax.text(positions[GROUPS[0][1][0]] - 0.45, 100.0 * np.sqrt(2.0 / (3.0 * n52)) * 0.86,
            f"canonical $\\sqrt{{2/(3N)}}$, N = {n52}", ha="left", va="top", fontsize=style.SMALL)
    ax.text(first_big - 0.45, 100.0 * np.sqrt(2.0 / (3.0 * big.min())) * 1.14,
            f"canonical $\\sqrt{{2/(3N)}}$, N = {big.min()}-{big.max()}", ha="left", va="bottom",
            fontsize=style.SMALL)
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(ax, thermo: pd.DataFrame, errors: pd.DataFrame, positions: dict[str, float]) -> None:
    frame = thermo.merge(errors, on="run_id")
    logged = _logged(frame)
    rows = []
    ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    for (_, row), has_log in zip(frame.iterrows(), logged):
        x = positions[row["series"]] + T_OFFSETS[row["temperature_K"]]
        _run_markers(ax, x, row["drift_meV_atom_ps"], row["temperature_K"], has_log, row["drift_se_meV_atom_ps"])
        rows.append({"run_id": row["run_id"], "series": row["series"], "temperature_K": row["temperature_K"],
                     "x": x, "drift_meV_atom_ps": row["drift_meV_atom_ps"],
                     "drift_se_meV_atom_ps_5ps_blocks": row["drift_se_meV_atom_ps"], "run_log": has_log})
    _series_axis(ax, positions, labels=True)
    ax.set_ylim(-0.32, 0.32)
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"])
    ax.set_ylabel("Energy drift\n(meV atom$^{-1}$ ps$^{-1}$)")
    ax.text(0.995, 0.97, "±1 s.e. of the slope (5-ps blocks)", transform=ax.transAxes, ha="right", va="top",
            fontsize=style.SMALL)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(axes, series: pd.DataFrame, thermo: pd.DataFrame) -> None:
    runs = {s: C.run_id(s, SERIES_TEMPERATURE) for s in C.PRODUCTION_SERIES}
    rows = []
    for series_id, run in runs.items():
        frame = series[series["run_id"] == run].sort_values("time_ps")
        model = PRODUCTION_MODELS[series_id]
        energy = (frame["E_eV_atom"] - frame["E_eV_atom"].mean()) * 1000.0
        values = (frame["T_K"], energy, frame["P_GPa"])
        size = int(round(BLOCK_PS / float(np.median(np.diff(frame["time_ps"])))))
        n = len(frame) // size * size
        coarse_time = frame["time_ps"].to_numpy()[:n].reshape(-1, size).mean(axis=1)
        coarse = []
        for ax, value in zip(axes, values):
            value = np.asarray(value)
            ax.plot(frame["time_ps"], value, color=common.lighten(style.MODEL_COLORS[model], 0.6), lw=0.3,
                    zorder=1)
            mean = value[:n].reshape(-1, size).mean(axis=1)
            coarse.append(np.repeat(mean, size))
            ax.plot(coarse_time, mean, color=style.MODEL_COLORS[model], lw=0.9, zorder=3)
        rows.extend({"run_id": run, "model": model, "time_ps": t, "T_K": a, "E_minus_run_mean_meV_atom": b,
                     "P_GPa": c, "T_K_5ps": ca, "E_minus_run_mean_meV_atom_5ps": cb, "P_GPa_5ps": cc}
                    for t, a, b, c, ca, cb, cc in zip(frame["time_ps"], frame["T_K"], energy, frame["P_GPa"],
                                                      *coarse))
    axes[0].axhline(SERIES_TEMPERATURE, color="black", lw=0.5, ls=(0, (3, 2)), zorder=3)
    axes[1].axhline(0, color="black", lw=0.5, ls=(0, (3, 2)), zorder=3)
    labels = ("T (K)", "E - mean\n(meV atom$^{-1}$)", "P (GPa)")
    for index, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_ylabel(label)
        if index < len(axes) - 1:
            ax.tick_params(labelbottom=False)
    axes[0].set_ylim(895, 1110)
    axes[0].set_yticks([900, 950, 1000, 1050, 1100])
    axes[1].set_ylim(-13, 13)
    axes[1].set_yticks([-10, -5, 0, 5, 10])
    axes[2].set_ylim(-0.75, 1.35)
    axes[2].set_yticks([-0.5, 0, 0.5, 1.0])
    axes[-1].set_xlabel("Production time (ps)")
    handles = [Line2D([0], [0], color=style.MODEL_COLORS[PRODUCTION_MODELS[s]], lw=1.0,
                      label=style.MODEL_LABELS[PRODUCTION_MODELS[s]]) for s in C.PRODUCTION_SERIES]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=style.SMALL,
                   handlelength=1.2, handletextpad=0.4, columnspacing=1.0, labelspacing=0.3, borderaxespad=0.2)
    axes[2].text(1.0, 0.97, f"{SERIES_TEMPERATURE} K; thin, 0.5-ps blocks; thick, {BLOCK_PS:g}-ps blocks",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=style.SMALL)
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _log_order(thermo: pd.DataFrame) -> pd.DataFrame:
    checked = thermo[_logged(thermo)].copy()
    checked["order"] = checked["series"].map(SERIES_ORDER.index)
    return checked.sort_values(["order", "temperature_K"]).reset_index(drop=True)


def _panel_e(axes, thermo: pd.DataFrame) -> None:
    checked = _log_order(thermo)
    groups, x, positions = [], 0.0, []
    previous = None
    for row in checked.itertuples():
        family = row.series if row.n_atoms < 100 else ("production" if row.series in C.PRODUCTION_SERIES
                                                        else "clexcess")
        step = 1.0 if row.n_atoms < 100 else WIDE_SLOT
        if previous is not None and family != previous:
            x += GROUP_GAP_E + step / 2
        elif previous is not None:
            x += step / 2
        if family != previous:
            groups.append([family, []])
        groups[-1][1].append(x)
        positions.append(x)
        previous = family
        x += step / 2
    positions = np.array(positions)
    t_tol = np.array([R.THERMOSTAT_T_TOLERANCE_K[416 if n > 100 else 52] for n in checked["n_atoms"]])
    e_tol = 10.0 ** -R.LOGGED_DECIMALS["mean_E_per_atom"] * 1000.0  # meV/atom
    d_tol = 10.0 ** -R.LOGGED_DECIMALS["drift_meV_atom_ps"]
    quantities = (
        ("d_mean_T_K", 1.0, t_tol, "mean_T_ok", "Mean T\n(K)", (-1.35, 1.35), ([-1, 0, 1], ["-1", "0", "1"])),
        ("d_mean_E_per_atom", 1000.0, np.full(len(checked), e_tol), "mean_E_per_atom_ok",
         "Mean E\n(meV/atom)", (-0.135, 0.135), ([-0.1, 0, 0.1], ["-0.1", "0", "0.1"])),
        ("d_drift_meV_atom_ps", 1000.0, np.full(len(checked), d_tol * 1000.0), "drift_meV_atom_ps_ok",
         "Drift ($10^{-3}$\nmeV/atom/ps)", (-1.9, 1.9), ([-1, 0, 1], ["-1", "0", "1"])),
    )
    source = checked[["run_id", "series", "temperature_K", "n_atoms", "run_log"]].copy()
    source["x"] = positions
    for ax, (column, scale, tolerance, ok_column, label, limits, ticks) in zip(axes, quantities):
        values = checked[column].to_numpy() * scale
        ok = _is_true(checked[ok_column]).to_numpy()
        for xi, tol in zip(positions, tolerance):
            ax.add_patch(Rectangle((xi - 0.45, -tol), 0.9, 2 * tol, facecolor=style.LIGHT_GREY, edgecolor="none",
                                   zorder=0))
        ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
        for xi, value, temperature, passed in zip(positions, values, checked["temperature_K"], ok):
            color = style.TEMPERATURE_COLORS[temperature]
            ax.plot(xi, value, marker="o" if passed else "D", linestyle="none", markersize=3.2 if passed else 3.4,
                    markerfacecolor=color if passed else "white", markeredgecolor=color, markeredgewidth=0.8,
                    zorder=3)
        ax.set_ylim(*limits)
        ax.set_yticks(*ticks)
        ax.set_ylabel(label, linespacing=1.1)
        ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
        for previous_group, group in zip(groups[:-1], groups[1:]):
            ax.axvline((previous_group[1][-1] + group[1][0]) / 2, color=style.LIGHT_GREY, lw=0.5, zorder=0)
        name = column.replace("_meV_atom_ps", "").replace("_per_atom", "_E").replace("d_mean_E_E", "d_mean_E")
        source[f"{name}_value"] = values
        source[f"{name}_tolerance"] = tolerance
        source[f"{name}_within"] = ok
    source = source.rename(columns={"d_mean_T_K_value": "d_mean_T_K", "d_mean_T_K_tolerance": "tolerance_T_K",
                                    "d_mean_E_value": "d_mean_E_meV_atom",
                                    "d_mean_E_tolerance": "tolerance_E_meV_atom",
                                    "d_drift_value": "d_drift_1e-3_meV_atom_ps",
                                    "d_drift_tolerance": "tolerance_drift_1e-3_meV_atom_ps"})
    for ax in axes[:-1]:
        ax.set_xticks(positions, [])
        ax.tick_params(axis="x", length=1.5)
    bottom = axes[-1]
    ticks = []
    for row in checked.itertuples():
        if row.n_atoms < 100:
            ticks.append(f"{row.temperature_K} K")
        else:
            ticks.append(f"{LOG_TICKS[row.series]}\n{row.temperature_K} K")
    bottom.set_xticks(positions, ticks)
    bottom.tick_params(axis="x", length=1.5, pad=1.5, labelsize=style.SMALL)
    names = {"mace_52_050": "MACE-MP-0, 52 atoms\n50 ps", "mattersim_52_050": "MatterSim, 52 atoms\n50 ps",
             "mattersim_52_100": "MatterSim, 52 atoms\n100 ps", "mace_52_150": "MACE-MP-0, 52 atoms\n150 ps",
             "production": "416 atoms, 200 ps", "clexcess": "Cl-excess, 150 ps"}
    blend = bottom.get_xaxis_transform()
    line = transforms.offset_copy(blend, fig=bottom.figure, y=-15.0, units="points")
    text = transforms.offset_copy(blend, fig=bottom.figure, y=-16.5, units="points")
    for family, members in groups:
        bottom.plot([members[0] - 0.35, members[-1] + 0.35], [0, 0], color="black", lw=0.5, transform=line,
                    clip_on=False)
        bottom.text((members[0] + members[-1]) / 2, 0, names[family], transform=text, ha="center", va="top",
                    fontsize=style.SMALL, linespacing=1.1)
    failed = checked[~_is_true(checked["drift_meV_atom_ps_ok"])]
    if len(failed):
        values = failed["d_drift_meV_atom_ps"].abs() * 1000.0
        axes[2].text(1.0, 0.03, f"{len(failed)} runs outside the rounding: |d| = {values.min():.2f}-"
                                f"{values.max():.2f} x $10^{{-3}}$", transform=axes[2].transAxes, ha="right",
                     va="bottom", fontsize=style.SMALL)
    handles = [Rectangle((0, 0), 1, 1, facecolor=style.LIGHT_GREY, edgecolor="none",
                         label="tolerance (protocol item 7)"),
               Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markerfacecolor="black",
                      markeredgecolor="black", label="within"),
               Line2D([0], [0], marker="D", linestyle="none", markersize=3.2, markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=0.8, label="outside")]
    handles += [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, markeredgecolor="none",
                       markerfacecolor=style.TEMPERATURE_COLORS[t], label=f"{t} K") for t in C.TEMPERATURES]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                   fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    axes[0].text(1.0, 1.04, "recomputed from the trajectory minus the July run log", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=style.SMALL)
    style.save_source(source, NAME, "e")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    thermo, series = _load()
    errors = _block_errors(series)
    positions = _positions()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    left, width = 17.0, 94.0
    ax_a = L.axes(left, 15.0, width, 25.0)
    ax_b = L.axes(left, 45.0, width, 25.0)
    ax_c = L.axes(left, 75.0, width, 25.0)
    _panel_a(ax_a, thermo, errors, positions)
    _panel_b(ax_b, thermo, positions)
    _panel_c(ax_c, thermo, errors, positions)

    d_left, d_width = 130.0, 51.0
    ax_d = [L.axes(d_left, 15.0 + i * 29.5, d_width, 26.0) for i in range(3)]
    _panel_d(ax_d, series, thermo)

    e_left, e_width = 21.0, 160.0
    ax_e = [L.axes(e_left, 132.0 + i * 17.0, e_width, 14.5) for i in range(3)]
    _panel_e(ax_e, thermo)

    L.letter(ax_a, "a", 3.0, 9.0)
    L.letter(ax_b, "b", 3.0, 44.0)
    L.letter(ax_c, "c", 3.0, 74.0)
    L.letter(ax_d[0], "d", 120.0, 9.0)
    L.letter(ax_e[0], "e", 3.0, 129.0)
    return style.save(fig, "ed01_stability", "extended")


def _caption() -> str:
    thermo, series = _load()
    errors = _block_errors(series)
    frame = thermo.merge(errors, on="run_id")
    small = frame[frame["n_atoms"] == 52]
    big = frame[frame["n_atoms"] > 100]
    d_small = small["mean_T_K"] - small["temperature_K"]
    d_big = big["mean_T_K"] - big["temperature_K"]
    z_small = d_small / small["sem_T_K"]
    z_big = d_big / big["sem_T_K"]
    rel_small = (d_small / small["temperature_K"]).mean()
    rel_big = (d_big / big["temperature_K"]).mean()
    ratio = frame["rel_std_T"] / frame["canonical_rel_std_T"]
    drift_z = (frame["drift_meV_atom_ps"] / frame["drift_se_meV_atom_ps"]).abs()
    checked = _log_order(thermo)
    n_logged = len(checked)
    t_ok = int(_is_true(checked["mean_T_ok"]).sum())
    e_ok = int(_is_true(checked["mean_E_per_atom_ok"]).sum())
    failed = checked[~_is_true(checked["drift_meV_atom_ps_ok"])]
    failed_text = ", ".join(f"{r.series.split('_')[0].replace('mace', 'MACE-MP-0').replace('mattersim', 'MatterSim')}"
                            f" {r.temperature_K} K" for r in failed.itertuples())
    failed_sizes = sorted(set(failed["n_atoms"]))
    failed_lengths = sorted(set(int(C.SERIES_BY_ID[s]["production_ps"]) for s in failed["series"]))
    p = {s: thermo.loc[thermo["run_id"] == C.run_id(s, SERIES_TEMPERATURE), "mean_P_GPa"].iloc[0]
         for s in C.PRODUCTION_SERIES}
    p_text = ", ".join(f"{style.MODEL_LABELS[PRODUCTION_MODELS[s]]} {p[s]:+.2f}" for s in C.PRODUCTION_SERIES)
    ts = series[series["run_id"].isin([C.run_id(s, SERIES_TEMPERATURE) for s in C.PRODUCTION_SERIES])]
    t_range = (ts["T_K"].min(), ts["T_K"].max())
    big_n = sorted(set(big["n_atoms"]))
    return (
        "Extended Data Fig. 1 | Temperature, energy and pressure of every MD run. All 42 runs (14 series at 600, "
        "800 and 1000 K; Langevin thermostat, 1 fs step). Filled markers, runs whose July run log survives "
        f"({n_logged}); open markers, statistics recovered from the trajectory only "
        f"({len(thermo) - n_logged}). Instantaneous temperature from the kinetic energy with 3N degrees of freedom; "
        "pressure -tr(stress)/3 + N k_B T / V. 'Gamma labels' and '2x2x2 labels' are the MACE-MP-0 fine-tunes on "
        "the two DFT label sets; x is the Cl excess of Li(6-x)PS(5-x)Cl(1+x). "
        f"**a**, Mean temperature minus the target. Error bars, standard error of the mean from {BLOCK_PS:g}-ps block "
        "averages (the standard deviation of the block means divided by the square root of their number). The "
        f"52-atom runs lie on average {d_small.mean():+.1f} K ({100 * rel_small:+.1f} %) from the target (range "
        f"{d_small.min():+.1f} to {d_small.max():+.1f} K; {int((z_small > 2).sum())} of {len(small)} more than two "
        f"standard errors above it and {int((z_small < -2).sum())} below), the {big_n[0]}-{big_n[-1]}-atom runs "
        f"{d_big.mean():+.1f} K ({100 * rel_big:+.1f} %; range {d_big.min():+.1f} to {d_big.max():+.1f} K; "
        f"{int((z_big.abs() > 2).sum())} of {len(big)} beyond two standard errors). "
        f"**b**, Relative temperature fluctuation std(T)/T against the canonical value sqrt(2/(3N)) (black bars, "
        f"for the atom count of each series; logarithmic axis). The ratio of the two is {ratio.min():.2f}-"
        f"{ratio.max():.2f} over all runs. "
        "**c**, Drift of the potential energy per atom, the slope of a straight-line fit against time over the "
        f"production run. Error bars, standard error of that slope from {BLOCK_PS:g}-ps block averages; "
        f"{int((drift_z > 2).sum())} of 42 drifts exceed two standard errors (largest "
        f"{frame['drift_meV_atom_ps'].abs().max():.3f} meV/atom/ps). "
        f"**d**, Temperature, potential energy per atom (minus its run mean) and pressure of the four 416-atom, "
        f"200-ps production runs at {SERIES_TEMPERATURE} K in 0.5-ps blocks (temperature {t_range[0]:.0f}-"
        f"{t_range[1]:.0f} K). Mean pressure (GPa): {p_text}. "
        f"**e**, Statistics recomputed from the trajectories minus the values in the July run logs, for the "
        f"{n_logged} runs whose log survives. Grey bands, the tolerance of protocol item 7: |d mean T| <= "
        f"{R.THERMOSTAT_T_TOLERANCE_K[52]:g} K for 52 atoms and {R.THERMOSTAT_T_TOLERANCE_K[416]:g} K for 416 "
        f"atoms (the logs include one extra sample at step 0 and round to 0.1 K); mean energy and drift within one "
        f"unit of the logged rounding (10^-{R.LOGGED_DECIMALS['mean_E_per_atom']} eV/atom and "
        f"10^-{R.LOGGED_DECIMALS['drift_meV_atom_ps']} meV/atom/ps). The mean temperature agrees in {t_ok} of "
        f"{n_logged} runs and the mean energy in {e_ok} of {n_logged}. The drift agrees in "
        f"{n_logged - len(failed)} of {n_logged}; it does not in {len(failed)} "
        f"{'/'.join(str(n) for n in failed_sizes)}-atom, "
        f"{'/'.join(str(n) for n in failed_lengths)}-ps runs ({failed_text}; diamonds), which differ by "
        f"{failed['d_drift_meV_atom_ps'].abs().min() * 1000:.2f}-{failed['d_drift_meV_atom_ps'].abs().max() * 1000:.2f}"
        " x 10^-3 meV/atom/ps. The logged fit also includes the production starting frame, which the trajectory "
        "does not store; in a 52-atom run this one end point moves the 50-ps slope by about 10^-3 meV/atom/ps. "
        "The check is reported as failed, as the rule states. "
        + common.sources([THERMOSTAT, TIMESERIES])
    )


CAPTION = _caption()
