"""Fig. 2: the universal potentials against AIMD of the same ordered cell (amendment A3).

a, Li diffusion coefficient against 1000/T: AIMD (Deng et al. 2017, re-analysed with the estimators
of the MLIP runs), the 416-atom production runs of MACE-MP-0 and MatterSim, their 52-atom runs and a
second 416-atom MACE-MP-0 run; b, kinisi D of every run divided by the AIMD value at the same
temperature; c, Li mean-squared displacement at 1000 K and 600 K of AIMD and the 52-atom MLIP runs;
d, sigma(300 K) and e, activation energy against simulated time. Measured conductivities (S/Cl-
disordered polycrystals) and published AIMD extrapolations to 300 K appear in d as grey context only.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import paths
from ..constants import KINISI_START_FRACTION, SERIES_BY_ID, TEMPERATURES
from . import common, style

NAME = "fig2"
WIDTH, HEIGHT = 183.0, 162.0
COMPARISON = paths.RESULTS / "aimd_comparison.csv"
AIMD = paths.REFERENCES / "aimd_deng2017" / "transport.csv"
AIMD_MSD = paths.REFERENCES / "aimd_deng2017" / "msd.csv.gz"
CONVERGENCE = paths.RESULTS / "convergence.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
LITERATURE = paths.REFERENCES / "literature.csv"

MODELS = ("foundation", "mattersim")
PRODUCTION = {"foundation": "mace_416_200", "mattersim": "mattersim_416_200"}
CELL_52 = {"foundation": ("mace_52_050", "mace_52_100", "mace_52_150"),
           "mattersim": ("mattersim_52_050", "mattersim_52_100", "mattersim_52_150")}
SECOND_RUN = "clx000_416_150"
AIMD_COLOR = style.OKABE_ITO["black"]
CONTEXT_COLOR = style.GREY
CONTEXT_TEXT = "#6E6E6E"
MSD_TEMPERATURES = (1000, 600)
# Run kinds: marker, filled; the same encoding in every panel.
KIND_MARKERS = {"cell_52": "o", "production_window": "s", "production_full": "s", "independent_run": "^"}
KIND_FILLED = {"cell_52": False, "production_window": True, "production_full": True, "independent_run": True}
# Horizontal offsets so that markers at the same position do not overlap: d and e in ps, a in 1000/T.
KIND_OFFSET = {"cell_52": -6.0, "production_window": 2.0, "production_full": 2.0, "independent_run": 10.0}
MODEL_OFFSET = {"foundation": -2.0, "mattersim": 2.0}
A_OFFSET_52 = {"foundation": (-0.058, -0.047, -0.036), "mattersim": (0.034, 0.045, 0.056)}
A_OFFSET_SECOND = 0.014
AIMD_EXTRAPOLATIONS = ("deng2017_sigma300", "deklerk2016_sigma300", "baktash2020")


def _num(value: float, digits: int = 3) -> str:
    """``digits`` significant figures with trailing zeros; scientific notation below 1e-3."""
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits - 1}e}".replace("e-0", "e-").replace("e+0", "e")
    rounded = float(f"{value:.{digits}g}")
    decimals = max(digits - 1 - math.floor(math.log10(abs(rounded))), 0)
    return f"{rounded:.{decimals}f}"


def _load():
    comparison = common.read(COMPARISON)
    aimd = common.read(AIMD).sort_values("temperature_K").reset_index(drop=True)
    aimd_msd = common.read(AIMD_MSD)
    convergence = common.read(CONVERGENCE)
    transport = common.read(TRANSPORT)
    msd = common.read(MSD)
    literature = common.read(LITERATURE).set_index("ref_id")
    return comparison, aimd, aimd_msd, convergence, transport, msd, literature


def _measured(literature: pd.DataFrame) -> tuple[float, float, float]:
    """Measured sigma_RT of sintered Li6PS5Cl (Wang et al. 2018) and the compiled range (Li et al. 2025), mS/cm."""
    return (float(literature.loc["wang2018", "value"]), float(literature.loc["li2025_compiled", "value_low"]),
            float(literature.loc["li2025_compiled", "value_high"]))


def _extrapolations(literature: pd.DataFrame) -> pd.DataFrame:
    """Published AIMD extrapolations of the ordered cell to 300 K: low and high value per study (mS/cm)."""
    rows = literature.loc[list(AIMD_EXTRAPOLATIONS)]
    low = rows["value_low"].fillna(rows["value"])
    high = rows["value_high"].fillna(rows["value"])
    return pd.DataFrame({"ref_id": rows.index, "low_mS_cm": low.to_numpy(float), "high_mS_cm": high.to_numpy(float)})


def _label(model: str) -> str:
    return style.MODEL_LABELS[model]


def _kind(series: str) -> str:
    if series == SECOND_RUN:
        return "independent_run"
    return "cell_52" if SERIES_BY_ID[series]["n_atoms"] == 52 else "production_full"


def _marker(kind: str, color: str, small: bool = False) -> dict:
    size = {"cell_52": 2.8 if small else 3.4, "production_full": 3.6, "production_window": 3.6,
            "independent_run": 3.8}[kind]
    return {"marker": KIND_MARKERS[kind], "markersize": size, "markeredgewidth": 0.6 if not KIND_FILLED[kind] else 0.0,
            "markeredgecolor": color, "markerfacecolor": color if KIND_FILLED[kind] else "white", "linestyle": "none"}


def _inverse(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 1000.0 / np.where(x > 0, x, 1.0), np.inf)


def _log_axis(ax, ticks: list[float], labels: list[str], axis: str = "y") -> None:
    style.log_ticks(ax, axis)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks, labels)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, comparison: pd.DataFrame, aimd: pd.DataFrame) -> Path:
    rows = []
    x = 1000.0 / aimd["temperature_K"].to_numpy(float)
    d, err = aimd["kinisi_D_cm2_s"].to_numpy(), aimd["kinisi_D_std_cm2_s"].to_numpy()
    lowest = int(aimd["temperature_K"].idxmin())
    ax.plot(x[1:], d[1:], color=AIMD_COLOR, lw=0.8, zorder=5)
    ax.plot(x[:2], d[:2], color=AIMD_COLOR, lw=0.8, ls=(0, (1, 1.5)), zorder=5)
    ax.errorbar(x, d, yerr=err, fmt="o", color=AIMD_COLOR, markersize=3.6, markeredgewidth=0.0, elinewidth=0.7,
                capsize=1.5, zorder=6)
    rows.extend({"run": "AIMD (Deng et al. 2017)", "kind": "aimd", "temperature_K": t, "x_plot_1000_per_K": xx,
                 "kinisi_D_cm2_s": v, "kinisi_D_std_cm2_s": e}
                for t, xx, v, e in zip(aimd["temperature_K"], x, d, err))
    slope = aimd.loc[lowest, "msd_loglog_slope_fit_window"]
    ax.annotate(f"AIMD {aimd.loc[lowest, 'temperature_K']:.0f} K: not diffusive\n(MSD slope {_num(slope)})",
                xy=(x[lowest] - 0.01, d[lowest]), xytext=(x[lowest] - 0.12, d[lowest] * 0.45), ha="right",
                va="center", fontsize=style.SMALL, arrowprops={"arrowstyle": "-", "lw": 0.5, "color": AIMD_COLOR,
                                                               "shrinkA": 1.0, "shrinkB": 2.5})
    for model in MODELS:
        color = style.MODEL_COLORS[model]
        runs = [PRODUCTION[model], *CELL_52[model]] + ([SECOND_RUN] if model == "foundation" else [])
        for series in runs:
            part = comparison[comparison["series"] == series].sort_values("temperature_K")
            kind = _kind(series)
            base = 1000.0 / part["temperature_K"].to_numpy(float)
            if kind == "cell_52":
                offset = A_OFFSET_52[model][CELL_52[model].index(series)]
            else:
                offset = A_OFFSET_SECOND if kind == "independent_run" else 0.0
            xp = base + offset
            values, stds = part["kinisi_D_cm2_s"].to_numpy(), part["kinisi_D_std_cm2_s"].to_numpy()
            if kind == "production_full":
                ax.plot(xp, values, color=color, lw=0.8, zorder=3)
            small = kind == "cell_52"
            ax.errorbar(xp, values, yerr=stds, fmt="none", ecolor=common.lighten(color, 0.4) if small else color,
                        elinewidth=0.5 if small else 0.7, capsize=0.0 if small else 1.3, zorder=2 if small else 4)
            ax.plot(xp, values, **_marker(kind, color, small=True), zorder=3 if small else 4)
            rows.extend({"run": series, "kind": kind, "temperature_K": t, "x_plot_1000_per_K": xx,
                         "kinisi_D_cm2_s": v, "kinisi_D_std_cm2_s": e}
                        for t, xx, v, e in zip(part["temperature_K"], xp, values, stds))
    ax.set_yscale("log")
    ax.set_ylim(1.5e-7, 2.0e-4)
    _log_axis(ax, [1e-6, 1e-5, 1e-4], ["10$^{-6}$", "10$^{-5}$", "10$^{-4}$"])
    ax.set_xlim(0.78, 1.73)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("Li D, kinisi (cm$^2$ s$^{-1}$)")
    top = ax.secondary_xaxis("top", functions=(_inverse, _inverse))
    ticks = sorted(aimd["temperature_K"].astype(int), reverse=True)
    top.set_xticks(ticks, [str(t) for t in ticks])
    top.tick_params(axis="x", labelsize=style.SMALL, pad=1.0)
    top.set_xlabel("T (K)", labelpad=2.0)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _b_positions() -> dict[str, float]:
    """Offset of each run inside a temperature group: MACE-MP-0 runs, a gap, MatterSim runs."""
    order = [PRODUCTION["foundation"], SECOND_RUN, *CELL_52["foundation"], None, PRODUCTION["mattersim"],
             *CELL_52["mattersim"]]
    offsets = np.linspace(-0.4, 0.4, len(order))
    return {series: float(offset) for series, offset in zip(order, offsets) if series is not None}


def _panel_b(ax, comparison: pd.DataFrame, aimd: pd.DataFrame) -> Path:
    positions = _b_positions()
    temperatures = list(TEMPERATURES)
    lowest = min(temperatures)
    ax.axvspan(-0.5, 0.5, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
    slope = aimd.set_index("temperature_K").loc[lowest, "msd_loglog_slope_fit_window"]
    ax.text(0.0, 0.985, f"AIMD sub-diffusive\n(MSD slope {_num(slope)})", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=style.SMALL, color="#555555", linespacing=1.1)
    ax.axhline(1.0, color=AIMD_COLOR, lw=0.7, zorder=1)
    ax.text(-0.46, 1.05, "AIMD", ha="left", va="bottom", fontsize=style.SMALL)
    rows = []
    for series, offset in positions.items():
        part = comparison[comparison["series"] == series].sort_values("temperature_K")
        model, kind = part["model"].iloc[0], _kind(series)
        color = style.MODEL_COLORS[model]
        xp = np.array([temperatures.index(t) for t in part["temperature_K"]], float) + offset
        ratio = part["kinisi_D_ratio_to_aimd"].to_numpy()
        err = ratio * part["kinisi_D_ratio_rel_err"].to_numpy()
        ax.errorbar(xp, ratio, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.0, zorder=2)
        ax.plot(xp, ratio, **_marker(kind, color), zorder=3)
        rows.extend({"run": series, "potential": part["potential"].iloc[0], "kind": kind, "temperature_K": t,
                     "x_plot": xx, "kinisi_D_ratio_to_aimd": r, "kinisi_D_ratio_rel_err": q, "err_abs": e,
                     "pymatgen_D_ratio_to_aimd": p, "msd_loglog_slope": s, "aimd_msd_loglog_slope": a}
                    for t, xx, r, q, e, p, s, a in zip(part["temperature_K"], xp, ratio, part["kinisi_D_ratio_rel_err"],
                                                       err, part["pymatgen_D_ratio_to_aimd"],
                                                       part["msd_loglog_slope"], part["aimd_msd_loglog_slope"]))
    for index in range(1, len(temperatures)):
        ax.axvline(index - 0.5, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 30)
    _log_axis(ax, [0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(-0.5, len(temperatures) - 0.5)
    ax.set_xticks(range(len(temperatures)), [f"{t} K" for t in temperatures])
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_ylabel("Li D, run / AIMD")
    return style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(axes, msd: pd.DataFrame, aimd_msd: pd.DataFrame, aimd: pd.DataFrame) -> Path:
    rows = []
    length = float(aimd["length_ps"].iloc[0])
    start = KINISI_START_FRACTION * length
    slopes = aimd.set_index("temperature_K")["msd_loglog_slope_fit_window"]
    for index, (ax, temperature) in enumerate(zip(axes, MSD_TEMPERATURES)):
        ax.axvspan(start, length, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
        for model in MODELS:
            for series in CELL_52[model]:
                matched = SERIES_BY_ID[series]["production_ps"] == round(length)
                color = style.MODEL_COLORS[model] if matched else common.lighten(style.MODEL_COLORS[model], 0.5)
                run = f"{series}_{temperature}K"
                curve = msd[msd["run_id"] == run].sort_values("lag_ps")
                ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=color, lw=0.9 if matched else 0.55,
                        zorder=3 if matched else 2)
                rows.extend({"run": run, "temperature_K": temperature, "lag_ps": lag, "msd_Li_A2": value}
                            for lag, value in zip(curve["lag_ps"], curve["msd_Li_A2"]))
        curve = aimd_msd[aimd_msd["temperature_K"] == temperature].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=AIMD_COLOR, lw=1.0, zorder=4)
        rows.extend({"run": f"AIMD_{temperature}K", "temperature_K": temperature, "lag_ps": lag, "msd_Li_A2": value}
                    for lag, value in zip(curve["lag_ps"], curve["msd_Li_A2"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.05, 200)
        ax.set_ylim(0.2, 1000)
        _log_axis(ax, [0.1, 1, 10, 100], ["0.1", "1", "10", "100"], "x")
        _log_axis(ax, [1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_xlabel("Lag time (ps)")
        ax.text(0.05, 0.96, f"{temperature} K", transform=ax.transAxes, ha="left", va="top",
                fontsize=style.FONT_SIZE, fontweight="bold")
        ax.text(0.05, 0.85, f"AIMD slope {_num(slopes[temperature])}", transform=ax.transAxes, ha="left", va="top",
                fontsize=style.SMALL)
        if index == 0:
            ax.set_ylabel("Li MSD (Å$^2$)")
            ax.text(np.sqrt(start * length), 0.04, "fit\nwindow", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=style.SMALL, color="#555555", linespacing=1.0)
        else:
            ax.tick_params(labelleft=False)
    rows.append({"run": "kinisi fit window (AIMD and 100-ps runs)", "temperature_K": np.nan, "lag_ps": start,
                 "msd_Li_A2": np.nan})
    return style.save_source(pd.DataFrame(rows), NAME, "c")


def _convergence_rows(convergence: pd.DataFrame) -> pd.DataFrame:
    frame = convergence.copy()
    model = {"MACE-MP-0": "foundation", "MatterSim": "mattersim"}
    frame["model"] = frame["potential"].map(model)
    frame["x_plot_ps"] = (frame["simulated_ps"] + frame["kind"].map(KIND_OFFSET)
                          + frame["model"].map(MODEL_OFFSET))
    return frame.sort_values(["kind", "model", "simulated_ps"]).reset_index(drop=True)


def _draw_runs(ax, frame: pd.DataFrame, value: str, low: np.ndarray, high: np.ndarray) -> None:
    for model in MODELS:
        color = style.MODEL_COLORS[model]
        windows = frame[(frame["model"] == model) & frame["kind"].isin(["production_window", "production_full"])]
        windows = windows.sort_values("simulated_ps")
        ax.plot(windows["x_plot_ps"], windows[value], color=color, lw=0.6, zorder=2)
    for row, lo, hi in zip(frame.itertuples(), low, high):
        color = style.MODEL_COLORS[row.model]
        y = getattr(row, value)
        ax.errorbar(row.x_plot_ps, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=1.3, zorder=3)
        ax.plot(row.x_plot_ps, y, **_marker(row.kind, color), zorder=4)


def _time_axis(ax) -> None:
    ax.set_xlim(35, 220)
    ax.set_xticks([50, 100, 150, 200])
    ax.set_xlabel("Simulated time per run (ps)")


def _panel_d(ax, frame: pd.DataFrame, measured: tuple[float, float, float], extrapolations: pd.DataFrame) -> Path:
    value, low, high = measured
    ax.axhspan(low, high, color=style.LIGHT_GREY, alpha=0.7, lw=0, zorder=0)
    ax.axhline(value, color=CONTEXT_COLOR, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(38, low * 1.2, "measured, S/Cl-disordered polycrystals (context)", ha="left", va="bottom",
            fontsize=style.SMALL, color=CONTEXT_TEXT)
    ax.text(217, value / 1.18, f"sintered, {value:g} (context)", ha="right", va="top", fontsize=style.SMALL,
            color=CONTEXT_TEXT)
    _draw_runs(ax, frame, "sigma300_mS_cm", frame["sigma300_min_mS_cm"].to_numpy(),
               frame["sigma300_max_mS_cm"].to_numpy())
    # Published AIMD extrapolations of the ordered cell: a bracket just right of the axes.
    bracket = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    span_low, span_high = extrapolations["low_mS_cm"].min(), extrapolations["high_mS_cm"].max()
    xb = 1.03
    ax.plot([xb, xb], [span_low, span_high], color=CONTEXT_COLOR, lw=0.7, transform=bracket, clip_on=False)
    for y in (span_low, span_high):
        ax.plot([xb - 0.012, xb], [y, y], color=CONTEXT_COLOR, lw=0.7, transform=bracket, clip_on=False)
    for row in extrapolations.itertuples():
        if row.high_mS_cm > row.low_mS_cm:
            ax.plot([xb, xb], [row.low_mS_cm, row.high_mS_cm], color=CONTEXT_COLOR, lw=2.4, transform=bracket,
                    clip_on=False, solid_capstyle="butt")
    ax.text(xb + 0.02, math.sqrt(span_low * span_high),
            f"published AIMD extrapolations,\nordered cell, {span_low:g}-{span_high:g} mS cm$^{{-1}}$",
            transform=bracket, rotation=90, ha="left", va="center", fontsize=style.SMALL, color=CONTEXT_TEXT,
            linespacing=1.1)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e3)
    _log_axis(ax, [1e-3, 1e-2, 0.1, 1, 10, 100, 1000], ["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_ylabel(r"$\sigma$(300 K), extrapolated (mS cm$^{-1}$)")
    _time_axis(ax)
    source = frame[["series", "potential", "kind", "n_atoms", "simulated_ps", "x_plot_ps", "sigma300_mS_cm",
                    "sigma300_min_mS_cm", "sigma300_max_mS_cm"]].copy()
    context = [{"series": "measured_wang2018", "kind": "context_measured", "sigma300_mS_cm": value},
               {"series": "measured_li2025_compiled", "kind": "context_measured", "sigma300_min_mS_cm": low,
                "sigma300_max_mS_cm": high}]
    context += [{"series": row.ref_id, "kind": "context_aimd_extrapolation", "sigma300_min_mS_cm": row.low_mS_cm,
                 "sigma300_max_mS_cm": row.high_mS_cm} for row in extrapolations.itertuples()]
    return style.save_source(pd.concat([source, pd.DataFrame(context)], ignore_index=True), NAME, "d")


def _panel_e(ax, frame: pd.DataFrame) -> Path:
    ea, err = frame["Ea_eV"].to_numpy(), frame["Ea_err_eV"].to_numpy()
    _draw_runs(ax, frame, "Ea_eV", ea - err, ea + err)
    ax.set_ylim(0.15, 0.40)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_ylabel("$E_a$ (eV)")
    _time_axis(ax)
    source = frame[["series", "potential", "kind", "n_atoms", "simulated_ps", "x_plot_ps", "Ea_eV", "Ea_err_eV"]]
    return style.save_source(source, NAME, "e")


def _legend(ax) -> None:
    """One key for every panel: colour = potential, marker = run type."""
    ax.axis("off")
    grey = "#4D4D4D"
    handles = [
        Line2D([0], [0], color=AIMD_COLOR, lw=0.8, marker="o", markersize=3.4, markeredgewidth=0.0,
               label="AIMD, same ordered cell (Deng et al. 2017), 52 atoms, 100 ps"),
        Patch(facecolor=style.MODEL_COLORS["foundation"], edgecolor="none", label=_label("foundation")),
        Patch(facecolor=style.MODEL_COLORS["mattersim"], edgecolor="none", label=_label("mattersim")),
        Line2D([0], [0], color=grey, lw=0.8, **{k: v for k, v in _marker("production_full", grey).items()
                                                 if k != "linestyle"},
               label="416-atom production run, 200 ps (d, e: also its first 50, 100, 150 ps)"),
        Line2D([0], [0], **_marker("cell_52", grey), label="52-atom runs of 50, 100 and 150 ps (AIMD cell size)"),
        Line2D([0], [0], **_marker("independent_run", grey),
               label="second 416-atom MACE-MP-0 run (Cl-excess x = 0), 150 ps"),
        Line2D([0], [0], color=grey, lw=0.9, label="c: 52-atom run of 100 ps (AIMD length)"),
        Line2D([0], [0], color=common.lighten(grey, 0.5), lw=0.55, label="c: 52-atom runs of 50 and 150 ps"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.8, handletextpad=0.6,
              labelspacing=0.55, borderaxespad=0.0, frameon=False)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    comparison, aimd, aimd_msd, convergence, transport, msd, literature = _load()
    frame = _convergence_rows(convergence)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    # row 1: a (D against 1000/T), b (ratio to AIMD)
    ax_a = L.axes(14.0, 13.0, 66.0, 44.0)
    sources = [_panel_a(ax_a, comparison, aimd)]
    ax_b = L.axes(98.0, 13.0, 83.0, 44.0)
    sources.append(_panel_b(ax_b, comparison, aimd))

    # row 2: c (MSD at 1000 K and 600 K), key
    top2, height2 = 72.0, 33.0
    ax_c = [L.axes(14.0, top2, 40.0, height2), L.axes(57.0, top2, 40.0, height2)]
    sources.append(_panel_c(ax_c, msd, aimd_msd, aimd))
    _legend(L.axes(106.0, top2 - 1.0, 75.0, height2))

    # row 3: d (sigma(300 K)), e (Ea)
    top3, height3 = 119.0, 33.0
    ax_d = L.axes(14.0, top3, 70.0, height3)
    sources.append(_panel_d(ax_d, frame, _measured(literature), _extrapolations(literature)))
    ax_e = L.axes(111.0, top3, 70.0, height3)
    sources.append(_panel_e(ax_e, frame))

    L.letter(ax_a, "a", 3.0, 6.0)
    L.letter(ax_b, "b", 89.0, 6.0)
    L.letter(ax_c[0], "c", 3.0, top2 - 3.0)
    L.letter(ax_d, "d", 3.0, top3 - 3.0)
    L.letter(ax_e, "e", 100.0, top3 - 3.0)
    return style.save(fig, "fig2_universal", "main") + sources


def _caption() -> str:
    comparison, aimd, aimd_msd, convergence, transport, msd, literature = _load()
    frame = _convergence_rows(convergence)
    value, low, high = _measured(literature)
    extrapolations = _extrapolations(literature).set_index("ref_id")
    reference = aimd.set_index("temperature_K")
    by_run = comparison.set_index(["series", "temperature_K"])
    a_aimd = float(aimd["a_A"].iloc[0])
    volume_52 = float(transport.loc[transport["n_atoms"] == 52, "volume_A3"].iloc[0])
    a_mlip = volume_52 ** (1.0 / 3.0)
    lowest = min(TEMPERATURES)

    def ratios(series: str) -> str:
        return ", ".join(_num(by_run.loc[(series, t), "kinisi_D_ratio_to_aimd"]) for t in TEMPERATURES)

    def ratio_range(model: str) -> str:
        values = [by_run.loc[(s, t), "kinisi_D_ratio_to_aimd"] for s in CELL_52[model] for t in TEMPERATURES
                  if t != lowest]
        return f"{_num(min(values))}-{_num(max(values))}"

    def slope(series: str, temperature: int) -> str:
        return _num(by_run.loc[(series, temperature), "msd_loglog_slope"])

    def by_kind(potential: str, kind: str) -> pd.DataFrame:
        return frame[(frame["potential"] == potential) & (frame["kind"] == kind)].sort_values("simulated_ps")

    def values(rows: pd.DataFrame, column: str) -> str:
        return ", ".join(_num(v) for v in rows[column])

    mace_w = pd.concat([by_kind("MACE-MP-0", "production_window"), by_kind("MACE-MP-0", "production_full")])
    ms_w = pd.concat([by_kind("MatterSim", "production_window"), by_kind("MatterSim", "production_full")])
    mace_52, ms_52 = by_kind("MACE-MP-0", "cell_52"), by_kind("MatterSim", "cell_52")
    independent = by_kind("MACE-MP-0", "independent_run").iloc[0]
    length = float(reference["length_ps"].iloc[0])
    aimd_d = ", ".join(_num(reference.loc[t, "kinisi_D_cm2_s"]) for t in reference.index)
    aimd_t = ", ".join(str(int(t)) for t in reference.index[:-1]) + f" and {int(reference.index[-1])}"
    matched = {m: next(s for s in CELL_52[m] if SERIES_BY_ID[s]["production_ps"] == round(length)) for m in MODELS}
    baktash = extrapolations.loc["baktash2020"]
    return (
        "Fig. 2 | Universal potentials against AIMD. "
        "The reference is AIMD of the same ordered S/Cl arrangement (Cl on 4a, S on 4c) by Deng et al. (2017): "
        f"52 atoms, a = {a_aimd:.2f} A (here {a_mlip:.2f} A, a {100 * ((a_mlip / a_aimd) ** 3 - 1):.0f} % larger volume), "
        f"{length:.0f} ps per temperature, "
        "re-analysed with the estimators and settings of the MLIP runs (protocol amendment A3). "
        "Every D is the kinisi value (seed 0, fitted from 30 % of the run) with its 1-sigma uncertainty. "
        f"**a**, Li D against 1000/T. Black circles, AIMD ({aimd_d} cm2/s at {aimd_t} K); squares joined by lines, "
        "the 416-atom, 200-ps production runs of MACE-MP-0 and MatterSim; small open circles, their 52-atom runs of "
        "50, 100 and 150 ps (independent trajectories, offset horizontally); triangle, a second 416-atom MACE-MP-0 "
        f"run (the x = 0 run of the Cl-excess series, 150 ps). At {lowest} K the AIMD Li MSD is flat in the fit "
        f"window (log-log slope {_num(reference.loc[lowest, 'msd_loglog_slope_fit_window'])}): that point measures "
        "caged motion, not diffusion (dotted line). "
        "**b**, D of each run divided by the AIMD D at the same temperature; error bars, the two relative standard "
        "deviations added in quadrature. Within each temperature, from left: MACE-MP-0 production run, second run and "
        "52-atom runs of 50, 100 and 150 ps, then MatterSim production run and 52-atom runs of 50, 100 and 150 ps. "
        "Production runs at 600, 800 and 1000 K: MACE-MP-0 "
        f"{ratios(PRODUCTION['foundation'])}; MatterSim {ratios(PRODUCTION['mattersim'])}; second MACE-MP-0 run "
        f"{ratios(SECOND_RUN)}. The 52-atom runs, which match the AIMD cell size, give {ratio_range('foundation')} "
        f"(MACE-MP-0) and {ratio_range('mattersim')} (MatterSim) at 800 and 1000 K. Grey column, the {lowest} K "
        "ratios compare caged, sub-diffusive motion, not diffusion coefficients. "
        f"**c**, Li mean-squared displacement against lag time at 1000 K and {lowest} K of AIMD (black) and the "
        f"52-atom MLIP runs: full colour, the {length:.0f}-ps runs that match the AIMD length (log-log slope in the "
        f"fit window {slope(matched['foundation'], 1000)} and {slope(matched['foundation'], lowest)} for MACE-MP-0, "
        f"{slope(matched['mattersim'], 1000)} and {slope(matched['mattersim'], lowest)} for MatterSim, at 1000 and "
        f"{lowest} K); faded, the 50- and 150-ps runs. Grey, the fit window of AIMD and the {length:.0f}-ps runs (lag "
        f"of at least {KINISI_START_FRACTION * length:.0f} ps). "
        "**d**, sigma(300 K) from weighted Arrhenius fits (re-analysis) against simulated time per run, with 1-sigma "
        "intervals: 52-atom runs (open circles; MACE-MP-0 "
        f"{values(mace_52, 'sigma300_mS_cm')} and MatterSim {values(ms_52, 'sigma300_mS_cm')} mS/cm at 50, 100 and "
        "150 ps), the 416-atom production run truncated to its first 50, 100 and 150 ps and in full (squares; "
        f"MACE-MP-0 {values(mace_w, 'sigma300_mS_cm')}, MatterSim {values(ms_w, 'sigma300_mS_cm')} mS/cm) and the "
        f"second 416-atom run (triangle, {_num(independent['sigma300_mS_cm'])} mS/cm). In grey, context only: the "
        f"{value:g} mS/cm measured on sintered, S/Cl-disordered Li6PS5Cl (dashed; Wang et al. 2018) and the range "
        f"{low:g}-{high:g} mS/cm compiled by Li et al. (2025); bracket, published AIMD extrapolations of "
        f"the ordered cell to 300 K, from {extrapolations.loc['deng2017_sigma300', 'low_mS_cm']:g} (Deng et al. "
        f"2017) to {extrapolations.loc['deklerk2016_sigma300', 'high_mS_cm']:g} mS/cm (de Klerk et al. 2016), "
        f"with {baktash['low_mS_cm']:g}-{baktash['high_mS_cm']:g} mS/cm from Baktash et al. (2020) "
        "(thick segment). These span five orders of magnitude, so an extrapolated sigma(300 K) is not a usable "
        "benchmark, and the measured values refer to a different (disordered) structure. "
        "**e**, Activation energy of the same fits with 1-sigma errors (MACE-MP-0 production windows "
        f"{values(mace_w, 'Ea_eV')} eV, MatterSim {values(ms_w, 'Ea_eV')} eV). "
        "In d and e, markers at the same simulated time are offset horizontally, and the truncated windows are "
        "nested parts of one trajectory. AIMD and MLIP runs have one trajectory per temperature each; the "
        "production runs have 416 atoms against 52 in AIMD. Logarithmic axes in a-d. "
        + common.sources([AIMD, AIMD_MSD, COMPARISON, CONVERGENCE, TRANSPORT, MSD, LITERATURE])
    )


CAPTION = _caption()
