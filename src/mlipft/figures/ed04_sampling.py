"""Sampling and estimator checks (registry key "sampling", an Extended Data figure).

a, RMS position difference between runs that share their settings (the three 52-atom tiers of each
universal potential, and the 416-atom MACE-MP-0 production run against the Cl-excess x = 0 run)
against production time; b, pymatgen D of the four consecutive 50-ps blocks of the four 416-atom,
200-ps production runs relative to the full-run value; c, kinisi D against the start of its fit
window, relative to the primary start at 30 %; d, pymatgen D against kinisi D for all 42 runs;
e, sigma(300 K) and f, activation energy of the Arrhenius fits against simulated time per run;
g, kinisi D of the 52-atom runs (the AIMD cell size) and of the second 416-atom MACE-MP-0 run
relative to AIMD; h, Li mean-squared displacement of the 100-ps, 52-atom runs against AIMD.

Temperatures are colour-coded (red tiers) only in a-c, where the models are separated by facets or
positions; d-h colour the potentials and shape the run types (style.MODEL_COLORS, style.RUN_MARKERS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from .. import constants as C
from .. import paths
from . import common, registry, style

KEY = "sampling"
NAME = registry.number(KEY)
PANELS = "abcdefgh"
WIDTH, HEIGHT = 183.0, 217.0
TRANSPORT = paths.MD / "transport.csv"
WINDOWS = paths.MD / "windows.csv"
DIVERGENCE = paths.MD / "divergence.csv"
MSD = paths.MD / "msd.csv.gz"
CONVERGENCE = paths.RESULTS / "convergence.csv"
COMPARISON = paths.RESULTS / "aimd_comparison.csv"
AIMD = paths.REFERENCES / "aimd_deng2017" / "transport.csv"
AIMD_MSD = paths.REFERENCES / "aimd_deng2017" / "msd.csv.gz"

PRODUCTION_TITLES = {"mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim",
                     "ftG_416_200": "Fine-tuned, Gamma", "ftK_416_200": "Fine-tuned, 2x2x2"}
PRODUCTION_TICKS = {"mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim",
                    "ftG_416_200": "Fine-tuned,\nGamma", "ftK_416_200": "Fine-tuned,\n2x2x2"}
PRODUCTION = tuple(s for s in C.PRODUCTION_SERIES if s not in common.PENDING_SERIES)
UNIVERSAL = ("foundation", "mattersim")
CELL_52 = {"foundation": ("mace_52_050", "mace_52_100", "mace_52_150"),
           "mattersim": ("mattersim_52_050", "mattersim_52_100", "mattersim_52_150")}
SECOND_RUN = "clx000_416_150"  # the x = 0 run of the Cl-excess series: a second 416-atom MACE-MP-0 run
MSD_SERIES = {"foundation": "mace_52_100", "mattersim": "mattersim_52_100"}  # the 52-atom runs of AIMD length
MSD_TEMPERATURES = (600, 1000)
DIVERGENCE_STYLES = {"mace_52": ("MACE-MP-0, 52-atom tiers", "-", 0.7),
                     "mattersim_52": ("MatterSim, 52-atom tiers", (0, (3, 1.5)), 0.7),
                     "mace_416": ("MACE-MP-0 416 atoms vs Cl x = 0", "-", 1.6)}
T_OFFSETS = dict(zip(C.TEMPERATURES, (-0.27, 0.0, 0.27)))
BLOCK_OFFSETS = (-0.09, -0.03, 0.03, 0.09)
C_OFFSETS = dict(zip(C.TEMPERATURES, (-0.028, 0.0, 0.028)))  # c: start-fraction offsets of the temperatures
# e, f: run kind of results/convergence.csv -> marker of style.RUN_MARKERS, and horizontal offsets (ps) so
# that markers at the same simulated time stay apart.
CONVERGENCE_KINDS = {"cell_52": "cell_52", "production_window": "production", "production_full": "production",
                     "independent_run": "clexcess"}
TIME_OFFSETS = {("cell_52", "foundation"): -12.0, ("cell_52", "mattersim"): -6.0,
                ("production", "foundation"): 0.0, ("production", "mattersim"): 6.0,
                ("clexcess", "foundation"): 12.0}
# g: slot of each run inside a temperature group (MACE-MP-0 runs, a gap, MatterSim runs).
G_ORDER = (*CELL_52["foundation"], SECOND_RUN, None, *CELL_52["mattersim"])
MARKER = 3.2
D_MARKER, D_EDGE = 2.6, 0.6
# d: slot order inside each temperature group (MACE-MP-0 runs, MatterSim runs, the two fine-tunes).
D_ORDER = (*CELL_52["foundation"], "mace_416_200", SECOND_RUN, "clx025_408_150", "clx050_400_150", "clx075_392_150",
           *CELL_52["mattersim"], "mattersim_416_200", "ftG_416_200", "ftK_416_200")
D_KEY = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG": "Gamma fine-tune", "ftK": "2x2x2 fine-tune"}


def _load():
    transport = common.read(TRANSPORT)
    transport = transport[~transport["series"].isin(common.PENDING_SERIES)].reset_index(drop=True)
    windows = common.read(WINDOWS)
    windows = windows[windows["run_id"].isin(transport["run_id"])].reset_index(drop=True)
    return {"transport": transport, "windows": windows,
            "divergence": common.read(DIVERGENCE), "msd": common.read(MSD), "convergence": common.read(CONVERGENCE),
            "comparison": common.read(COMPARISON),
            "aimd": common.read(AIMD).sort_values("temperature_K").reset_index(drop=True),
            "aimd_msd": common.read(AIMD_MSD)}


def _and(items) -> str:
    """'600, 800 and 1000' from a sequence."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _run_label(run: str) -> str:
    """Readable name of a run id, e.g. 'Fine-tuned on Gamma labels, 416 atoms, 200 ps, 600 K'."""
    series, temperature = run.rsplit("_", 1)
    label = C.SERIES_BY_ID[series]["label"].replace("Fine-tuned", "fine-tuned")
    return f"{label}, {temperature.replace('K', ' K')}"


def _series_style(series: str) -> tuple[str, str]:
    """(model, run kind) of a series: colour by potential, marker shape by run type."""
    info = C.SERIES_BY_ID[series]
    if series.startswith("clx"):
        return "foundation", "clexcess"
    return info["model"], "cell_52" if info["n_atoms"] == 52 else "production"


def _temperature_marker(series: str, temperature: int, size: float) -> dict:
    """``ax.plot`` marker keywords of a production run in a temperature-coloured panel (b, c): colour by
    temperature, fill by the training stage of the model (the 2x2x2 fine-tune ran MD with its stage-one model,
    so its markers are open)."""
    color = style.TEMPERATURE_COLORS[temperature]
    if common.filled(C.SERIES_BY_ID[series]["model"]):
        return {"marker": "o", "markersize": size, "markeredgewidth": 0, "color": color, "markerfacecolor": color}
    return {"marker": "o", "markersize": size, "markeredgewidth": 0.6, "color": color, "markeredgecolor": color,
            "markerfacecolor": style.WHITE}


def _divergence_key(run: str) -> str:
    parts = run.split("_")
    return f"{parts[0]}_{parts[1]}"


def _symlog_axis(ax, linthresh: float) -> None:
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.35)
    ax.set_xlim(-0.1 * linthresh, 32)  # a little room left of 0, so the markers at time 0 clear the frame
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10]))
    ax.set_xticks([0, 0.1, 1, 10], ["0", "0.1", "1", "10"])
    minor = [m * 10.0 ** e for e in (-2, -1, 0, 1) for m in range(2, 10) if linthresh <= m * 10.0 ** e <= 32]
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _log_axis(ax, axis: str, ticks: list[float]) -> None:
    style.log_ticks(ax, axis)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks, [f"{t:g}" for t in ticks])


def _below(L: common.Layout, axes: list, offset_mm: float) -> tuple[float, float]:
    """(x, top) in mm of a shared x label centred under ``axes``, ``offset_mm`` below their bottom frame."""
    left = L.records[[r.ax for r in L.records].index(axes[0])].box_mm
    right = L.records[[r.ax for r in L.records].index(axes[-1])].box_mm
    return (left[0] + right[0] + right[2]) / 2, left[1] + left[3] + offset_mm


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, divergence: pd.DataFrame) -> None:
    frame = divergence.copy()
    frame["kind"] = frame["run_a"].map(_divergence_key)
    frame["temperature_K"] = frame["run_a"].str.extract(r"_(\d+)K$")[0].astype(int)
    first = frame.loc[frame["time_ps"] > 0, "time_ps"].min()
    for _, pair in frame.groupby(["run_a", "run_b"], sort=False):
        pair = pair.sort_values("time_ps")
        kind = pair["kind"].iloc[0]
        _, dash, width = DIVERGENCE_STYLES[kind]
        color = style.TEMPERATURE_COLORS[int(pair["temperature_K"].iloc[0])]
        ax.plot(pair["time_ps"], pair["rms_position_difference_A"], color=color, lw=width, ls=dash,
                marker="o" if kind == "mace_416" else None, markersize=2.0, markeredgewidth=0,
                zorder=3 if kind == "mace_416" else 2)
    _symlog_axis(ax, first)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Production time (ps)")
    ax.set_ylabel("RMS position difference (Å)")
    at_zero = frame[frame["time_ps"] == 0]["rms_position_difference_A"]
    ax.axhspan(0, at_zero.min(), **style.BAND_FILL)
    ax.text(0.02, at_zero.min() / 2, f"all pairs differ by ≥ {at_zero.min():.2f} Å at the first frame",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=style.SMALL)
    handles = [Line2D([0], [0], color=style.BLACK, lw=width, ls=dash, marker="o" if key == "mace_416" else None,
                      markersize=2.0, markeredgewidth=0, label=label)
               for key, (label, dash, width) in DIVERGENCE_STYLES.items()]
    handles += [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=1.0, label=f"{t} K") for t in C.TEMPERATURES]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=style.SMALL, handlelength=1.8, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.3, borderaxespad=0.3)
    source = frame[["run_a", "run_b", "kind", "temperature_K", "frame", "time_ps", "rms_position_difference_A"]]
    style.save_source(source, NAME, "a")


def _panel_b(ax, transport: pd.DataFrame, windows: pd.DataFrame) -> None:
    full = transport.set_index("run_id")["D_cm2_s"]
    blocks = windows[windows["kind"] == "block_start_ps"].copy()
    rows = []
    ax.axhline(1.0, **style.UNITY_LINE)
    for index, series in enumerate(PRODUCTION):
        for temperature in C.TEMPERATURES:
            run = C.run_id(series, temperature)
            part = blocks[blocks["run_id"] == run].sort_values("value")
            ratio = (part["D_cm2_s"] / full[run]).to_numpy()
            x = index + T_OFFSETS[temperature] + np.array(BLOCK_OFFSETS[:len(part)])
            color = style.TEMPERATURE_COLORS[temperature]
            ax.plot(x, ratio, color=color, lw=0.5, zorder=2)
            ax.plot(x, ratio, linestyle="none", zorder=3, **_temperature_marker(series, temperature, 2.6))
            rows.extend({"run_id": run, "series": series, "temperature_K": temperature, "block_start_ps": start,
                         "x": xi, "D_block_cm2_s": d, "D_full_run_cm2_s": full[run], "ratio": r}
                        for start, xi, d, r in zip(part["value"], x, part["D_cm2_s"], ratio))
    for index in range(1, len(PRODUCTION)):
        ax.axvline(index - 0.5, color=style.LIGHT_GRAY, lw=0.5, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 5.0)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.7, 1, 1.5, 2, 3], ["0.7", "1", "1.5", "2", "3"])
    ax.set_xlim(-0.5, len(PRODUCTION) - 0.5)
    ax.set_xticks(range(len(PRODUCTION)), [PRODUCTION_TICKS[s] for s in PRODUCTION])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("D of a 50-ps block / D of the\nfull 200-ps run (pymatgen)")
    handles = [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=0.5, marker="o", markersize=2.6,
                      markeredgewidth=0, label=f"{t} K") for t in C.TEMPERATURES]
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.6, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.3)
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(axes, transport: pd.DataFrame, windows: pd.DataFrame) -> None:
    primary = transport.set_index("run_id")
    starts = windows[windows["kind"] == "kinisi_start_fraction"]
    rows = []
    for index, (ax, series) in enumerate(zip(axes, PRODUCTION)):
        ax.axhline(1.0, **style.UNITY_LINE)
        ax.axvline(C.KINISI_START_FRACTION, color=style.LIGHT_GRAY, lw=0.6, zorder=0)
        for temperature in C.TEMPERATURES:
            run = C.run_id(series, temperature)
            reference = primary.loc[run, "kinisi_D_cm2_s"]
            part = pd.concat([
                starts.loc[starts["run_id"] == run, ["value", "kinisi_D_cm2_s", "kinisi_D_std_cm2_s"]],
                pd.DataFrame({"value": [C.KINISI_START_FRACTION], "kinisi_D_cm2_s": [reference],
                              "kinisi_D_std_cm2_s": [primary.loc[run, "kinisi_D_std_cm2_s"]]}),
            ]).sort_values("value")
            ratio = part["kinisi_D_cm2_s"] / reference
            error = part["kinisi_D_std_cm2_s"] / reference
            dx = C_OFFSETS[temperature]
            ax.errorbar(part["value"] + dx, ratio, yerr=error, lw=0.7, elinewidth=0.5, capsize=1.0, capthick=0.5,
                        zorder=3, **_temperature_marker(series, temperature, 2.4))
            rows.extend({"run_id": run, "series": series, "temperature_K": temperature, "start_fraction": f,
                         "x": f + dx, "kinisi_D_cm2_s": d, "kinisi_D_std_cm2_s": s, "D_at_0.3_cm2_s": reference,
                         "ratio": r, "ratio_error": e}
                        for f, d, s, r, e in zip(part["value"], part["kinisi_D_cm2_s"], part["kinisi_D_std_cm2_s"],
                                                 ratio, error))
        ax.set_xlim(0.05, 0.55)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"])
        ax.set_ylim(0.55, 2.1)
        ax.set_yticks([0.6, 1.0, 1.4, 1.8])
        ax.set_title(PRODUCTION_TITLES[series], fontsize=style.SMALL, pad=2.0, color=style.BLACK)
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0.6, 1.0, 1.4, 1.8], ["0.6", "1.0", "1.4", "1.8"])
            ax.set_ylabel("kinisi D / D at 0.3")
    handles = [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=0.7, marker="o", markersize=2.4,
                      markeredgewidth=0, label=f"{t} K") for t in C.TEMPERATURES]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.3,
                    labelspacing=0.25, borderaxespad=0.3)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, transport: pd.DataFrame) -> None:
    """pymatgen D / kinisi D of every run, one slot per series inside each temperature group."""
    if set(D_ORDER) != set(transport["series"]):
        raise ValueError("panel d must show every series of transport.csv")
    temperatures = list(C.TEMPERATURES)
    offsets = dict(zip(D_ORDER, np.linspace(-0.45, 0.45, len(D_ORDER))))
    ax.axhline(1.0, **style.UNITY_LINE)
    for index in range(1, len(temperatures)):
        ax.axvline(index - 0.5, color=style.LIGHT_GRAY, lw=0.5, zorder=0)
    rows = []
    for series in D_ORDER:
        model, kind = _series_style(series)
        part = transport[transport["series"] == series].set_index("temperature_K").loc[temperatures]
        ratio = (part["D_cm2_s"] / part["kinisi_D_cm2_s"]).to_numpy()
        x = np.arange(len(temperatures)) + offsets[series]
        ax.plot(x, ratio, **common.model_style(model, kind, size=D_MARKER, edge=D_EDGE, zorder=3))
        rows.extend({"run_id": C.run_id(series, t), "series": series, "model": model, "run_kind": kind,
                     "temperature_K": t, "x": xi, "kinisi_D_cm2_s": k, "kinisi_D_std_cm2_s": e,
                     "pymatgen_D_cm2_s": p, "ratio_pymatgen_over_kinisi": r}
                    for t, xi, k, e, p, r in zip(temperatures, x, part["kinisi_D_cm2_s"], part["kinisi_D_std_cm2_s"],
                                                 part["D_cm2_s"], ratio))
    ax.set_yscale("log")
    ax.set_ylim(0.85, 3.2)
    _log_axis(ax, "y", [1, 1.5, 2, 3])
    ax.set_xlim(-0.5, len(temperatures) - 0.5)
    ax.set_xticks(range(len(temperatures)), [f"{t} K" for t in temperatures])
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_ylabel("pymatgen D / kinisi D")
    handles = [common.handle(common.model_style(model, size=D_MARKER, edge=D_EDGE), label)
               for model, label in D_KEY.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=style.SMALL, handlelength=0.8, handletextpad=0.4,
              labelspacing=0.2, borderaxespad=0.4, borderpad=0.2)
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _convergence_rows(convergence: pd.DataFrame) -> pd.DataFrame:
    frame = convergence.copy()
    frame["model"] = frame["potential"].map({"MACE-MP-0": "foundation", "MatterSim": "mattersim"})
    frame["run_kind"] = frame["kind"].map(CONVERGENCE_KINDS)
    frame["x_plot_ps"] = frame["simulated_ps"] + [TIME_OFFSETS[(k, m)] for k, m in zip(frame["run_kind"],
                                                                                         frame["model"])]
    return frame.sort_values(["run_kind", "model", "simulated_ps"]).reset_index(drop=True)


def _draw_runs(ax, frame: pd.DataFrame, value: str, low: np.ndarray, high: np.ndarray) -> None:
    """Every fit of ``frame`` with its interval; the nested windows of a production run are joined by a line."""
    for model in UNIVERSAL:
        windows = frame[(frame["model"] == model) & (frame["run_kind"] == "production")].sort_values("simulated_ps")
        ax.plot(windows["x_plot_ps"], windows[value], color=style.MODEL_COLORS[model], lw=0.6, zorder=2)
    for row, lo, hi in zip(frame.itertuples(), low, high):
        color = style.MODEL_COLORS[row.model]
        y = getattr(row, value)
        ax.errorbar(row.x_plot_ps, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=0.6,
                    capsize=1.0, capthick=0.6, zorder=3)
        ax.plot(row.x_plot_ps, y, **common.model_style(row.model, row.run_kind, size=MARKER, zorder=4))


def _time_axis(ax) -> None:
    ax.set_xlim(28, 216)
    ax.set_xticks([50, 100, 150, 200])
    ax.set_xlabel("Simulated time per run (ps)")


def _panel_e(ax, frame: pd.DataFrame) -> None:
    _draw_runs(ax, frame, "sigma300_mS_cm", frame["sigma300_min_mS_cm"].to_numpy(),
               frame["sigma300_max_mS_cm"].to_numpy())
    ax.set_yscale("log")
    ax.set_ylim(0.1, 200)
    _log_axis(ax, "y", [0.1, 1, 10, 100])
    ax.set_ylabel(r"$\sigma$(300 K), extrapolated (mS cm$^{-1}$)")
    _time_axis(ax)
    source = frame[["series", "potential", "model", "kind", "run_kind", "n_atoms", "simulated_ps", "x_plot_ps",
                    "sigma300_mS_cm", "sigma300_min_mS_cm", "sigma300_max_mS_cm"]]
    style.save_source(source, NAME, "e")


def _panel_f(ax, frame: pd.DataFrame) -> None:
    ea, err = frame["Ea_eV"].to_numpy(), frame["Ea_err_eV"].to_numpy()
    _draw_runs(ax, frame, "Ea_eV", ea - err, ea + err)
    ax.set_ylim(0.15, 0.42)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40], ["0.15", "0.20", "0.25", "0.30", "0.35", "0.40"])
    ax.set_ylabel("$E_a$ (eV)")
    _time_axis(ax)
    source = frame[["series", "potential", "model", "kind", "run_kind", "n_atoms", "simulated_ps", "x_plot_ps",
                    "Ea_eV", "Ea_err_eV"]]
    style.save_source(source, NAME, "f")


def _panel_g(ax, comparison: pd.DataFrame, aimd: pd.DataFrame) -> None:
    temperatures = list(C.TEMPERATURES)
    offsets = np.linspace(-0.39, 0.39, len(G_ORDER))
    caged = common.caged_temperatures(aimd)
    for temperature in caged:
        index = temperatures.index(temperature)
        ax.axvspan(index - 0.5, index + 0.5, **style.BAND_FILL)
        ax.text(index, 0.965, "AIMD caged", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    ax.axhline(1.0, **style.UNITY_LINE)
    rows = []
    for series, offset in zip(G_ORDER, offsets):
        if series is None:
            continue
        model, kind = _series_style(series)
        part = comparison[comparison["series"] == series].set_index("temperature_K").loc[temperatures]
        x = np.arange(len(temperatures)) + offset
        ratio = part["kinisi_D_ratio_to_aimd"].to_numpy()
        rel = part["kinisi_D_ratio_rel_err"].to_numpy()
        color = style.MODEL_COLORS[model]
        ax.errorbar(x, ratio, yerr=ratio * rel, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.0,
                    capthick=0.6, zorder=2)
        ax.plot(x, ratio, **common.model_style(model, kind, size=MARKER, zorder=3))
        rows.extend({"series": series, "model": model, "run_kind": kind,
                     "production_ps": int(C.SERIES_BY_ID[series]["production_ps"]),
                     "n_atoms": int(C.SERIES_BY_ID[series]["n_atoms"]), "temperature_K": t, "x": xi,
                     "kinisi_D_ratio_to_aimd": r, "kinisi_D_ratio_rel_err": e, "error_bar_abs": r * e,
                     "pymatgen_D_ratio_to_aimd": p, "mlip_msd_loglog_slope": m, "aimd_msd_loglog_slope": a,
                     "aimd_caged": t in caged}
                    for t, xi, r, e, p, m, a in zip(temperatures, x, ratio, rel, part["pymatgen_D_ratio_to_aimd"],
                                                    part["msd_loglog_slope"], part["aimd_msd_loglog_slope"]))
    for index in range(1, len(temperatures)):
        ax.axvline(index - 0.5, color=style.LIGHT_GRAY, lw=0.5, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 40)
    _log_axis(ax, "y", [0.5, 1, 2, 5, 10, 20])
    ax.set_xlim(-0.5, len(temperatures) - 0.5)
    ax.set_xticks(range(len(temperatures)), [f"{t} K" for t in temperatures])
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_ylabel("kinisi D, run / AIMD")
    style.save_source(pd.DataFrame(rows), NAME, "g")


def _panel_h(axes, msd: pd.DataFrame, aimd_msd: pd.DataFrame, aimd: pd.DataFrame) -> None:
    rows = []
    info = aimd.set_index("temperature_K")
    length = float(info["length_ps"].iloc[0])
    start = C.KINISI_START_FRACTION * length
    for index, (ax, temperature) in enumerate(zip(axes, MSD_TEMPERATURES)):
        ax.axvspan(start, length, **style.BAND_FILL)
        for model in UNIVERSAL:
            run = C.run_id(MSD_SERIES[model], temperature)
            curve = msd[msd["run_id"] == run].sort_values("lag_ps")
            ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.MODEL_COLORS[model], lw=0.9, zorder=3)
            rows.extend({"run": run, "model": model, "temperature_K": temperature, "lag_ps": lag, "msd_Li_A2": v}
                        for lag, v in zip(curve["lag_ps"], curve["msd_Li_A2"]))
        curve = aimd_msd[aimd_msd["temperature_K"] == temperature].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.AIMD_COLOR, lw=0.9, zorder=4)
        rows.extend({"run": f"AIMD_{temperature}K", "model": "aimd", "temperature_K": temperature, "lag_ps": lag,
                     "msd_Li_A2": v} for lag, v in zip(curve["lag_ps"], curve["msd_Li_A2"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.012, 200)
        ax.set_ylim(0.04, 3000)
        _log_axis(ax, "x", [0.1, 1, 10, 100])
        _log_axis(ax, "y", [0.1, 1, 10, 100, 1000])
        ax.text(0.05, 0.95, f"{temperature} K", transform=ax.transAxes, ha="left", va="top",
                fontsize=style.FONT_SIZE, fontweight="bold")
        ax.text(0.05, 0.83, f"AIMD slope {common.sig(info.loc[temperature, 'msd_loglog_slope_fit_window'])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=style.SMALL)
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Li MSD (Å$^2$)")
            ax.text(np.sqrt(start * length), 0.04, "fit", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    rows.append({"run": "kinisi fit window (AIMD and the 100-ps runs)", "model": "", "temperature_K": np.nan,
                 "lag_ps": start, "msd_Li_A2": np.nan})
    style.save_source(pd.DataFrame(rows), NAME, "h")


def _key(ax) -> None:
    """Shared key of e-h (colour = potential) and of d-g (marker shape = run type)."""
    ax.set_xticks([])
    ax.set_yticks([])
    neutral = style.DARK_GRAY
    colours = [common.handle(common.model_style(m, size=MARKER), style.MODEL_LABELS[m]) for m in UNIVERSAL]
    colours.append(Line2D([0], [0], color=style.AIMD_COLOR, lw=0.9, label="AIMD, 52 atoms, 100 ps (h)"))
    shapes = [
        common.handle(common.model_style("foundation", "production", size=MARKER, color=neutral,
                                         markeredgecolor=neutral, markerfacecolor=neutral),
                      "416-atom production run, 200 ps\n(e, f: also its first 50, 100, 150 ps)"),
        common.handle(common.model_style("foundation", "cell_52", size=MARKER, color=neutral,
                                         markeredgecolor=neutral, markerfacecolor=neutral),
                      "52-atom runs (AIMD cell size),\n50, 100 and 150 ps"),
        common.handle(common.model_style("foundation", "clexcess", size=MARKER, color=neutral,
                                         markeredgecolor=neutral, markerfacecolor=neutral),
                      "Cl-excess x = 0: second 416-atom\nMACE-MP-0 run, 150 ps (d: all x)"),
    ]
    first = ax.legend(handles=colours, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=style.SMALL,
                      handlelength=1.4, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.0,
                      title="Colour: potential (e-h)", title_fontsize=style.SMALL, alignment="left")
    ax.add_artist(first)
    ax.legend(handles=shapes, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=style.SMALL, handlelength=1.4,
              handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0, title="Shape: run type (d-g)",
              title_fontsize=style.SMALL, alignment="left")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    data = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    # row 1: a (hero), b
    ax_a = L.axes(15.0, 8.0, 88.0, 46.0, row="1", col="left")
    _panel_a(ax_a, data["divergence"])
    ax_b = L.axes(119.0, 8.0, 62.0, 46.0, row="1")
    _panel_b(ax_b, data["transport"], data["windows"])

    # row 2: c (four facets), d
    width, gap = 21.5, 2.5
    ax_c = [L.axes(15.0 + i * (width + gap), 72.0, width, 34.0, row="2", col="left" if i == 0 else None)
            for i in range(4)]
    _panel_c(ax_c, data["transport"], data["windows"])
    L.text(*_below(L, ax_c, 5.4), "Start of the kinisi fit (fraction of the run)", ha="center", va="top")
    ax_d = L.axes(126.0, 72.0, 55.0, 34.0, row="2")
    _panel_d(ax_d, data["transport"])

    # row 3: e, f, key
    frame = _convergence_rows(data["convergence"])
    ax_e = L.axes(15.0, 124.0, 50.0, 36.0, row="3", col="left")
    _panel_e(ax_e, frame)
    ax_f = L.axes(81.0, 124.0, 50.0, 36.0, row="3")
    _panel_f(ax_f, frame)
    _key(L.axes(137.0, 124.0, 45.0, 36.0, frame="none"))

    # row 4: g, h (two temperatures)
    ax_g = L.axes(15.0, 174.0, 58.0, 34.0, row="4", col="left")
    _panel_g(ax_g, data["comparison"], data["aimd"])
    ax_h = [L.axes(91.0, 174.0, 43.5, 34.0, row="4"), L.axes(137.5, 174.0, 43.5, 34.0, row="4")]
    _panel_h(ax_h, data["msd"], data["aimd_msd"], data["aimd"])
    L.text(*_below(L, ax_h, 5.4), "Lag time (ps)", ha="center", va="top")

    L.letter(ax_a, "a", 4.0, 6.0)
    L.letter(ax_b, "b", 106.0, 6.0)
    L.letter(ax_c[0], "c", 4.0, 68.0)
    L.letter(ax_d, "d", 114.0, 68.0)
    L.letter(ax_e, "e", 4.0, 121.0)
    L.letter(ax_f, "f", 69.0, 121.0)
    L.letter(ax_g, "g", 4.0, 171.0)
    L.letter(ax_h[0], "h", 79.0, 171.0)
    return L.save(*registry.target(KEY))


# ---------------------------------------------------------------------------------------------- caption


def _caption() -> str:
    data = _load()
    transport, windows, divergence = data["transport"], data["windows"], data["divergence"]
    comparison, aimd = data["comparison"], data["aimd"]
    frame = _convergence_rows(data["convergence"])
    sig = common.sig

    at_zero = divergence[divergence["time_ps"] == 0]
    small_zero = at_zero[at_zero["run_a"].str.contains("_52_")]["rms_position_difference_A"]
    big_zero = at_zero[~at_zero["run_a"].str.contains("_52_")]["rms_position_difference_A"]
    first_frame = int(divergence["frame"][divergence["frame"] > 0].min())
    first_time = divergence.loc[divergence["frame"] == first_frame, "time_ps"].iloc[0]
    last = divergence[divergence["time_ps"] == divergence["time_ps"].max()]["rms_position_difference_A"]
    pairs = divergence.groupby(["run_a", "run_b"]).ngroups
    pairs_52 = divergence[divergence["run_a"].str.contains("_52_")].groupby(["run_a", "run_b"]).ngroups

    full = transport.set_index("run_id")["D_cm2_s"]
    blocks = windows[windows["kind"] == "block_start_ps"].copy()
    blocks["ratio"] = blocks["D_cm2_s"] / blocks["run_id"].map(full)
    blocks["temperature_K"] = blocks["run_id"].str.extract(r"_(\d+)K$")[0].astype(int)
    ranges = blocks.groupby("temperature_K")["ratio"].agg(["min", "max"])

    starts = windows[windows["kind"] == "kinisi_start_fraction"].copy()
    starts["ratio"] = starts["kinisi_D_cm2_s"] / starts["run_id"].map(transport.set_index("run_id")["kinisi_D_cm2_s"])
    extreme = starts.loc[(starts["ratio"] - 1).abs().idxmax()]
    hot = starts[~starts["run_id"].str.endswith(f"_{min(C.TEMPERATURES)}K")]
    fractions = sorted(set(starts["value"]) | {C.KINISI_START_FRACTION})

    ratio = transport["D_cm2_s"] / transport["kinisi_D_cm2_s"]
    by_t = transport.assign(ratio=ratio).groupby("temperature_K")["ratio"].agg(["min", "max"])
    worst = transport.loc[ratio.idxmax()]
    posterior = transport["kinisi_D_std_cm2_s"] / transport["kinisi_D_cm2_s"]

    def fits(model: str, kind: str) -> pd.DataFrame:
        return frame[(frame["model"] == model) & (frame["run_kind"] == kind)].sort_values("simulated_ps")

    def values(rows: pd.DataFrame, column: str) -> str:
        return ", ".join(sig(v) for v in rows[column])

    second = fits("foundation", "clexcess").iloc[0]
    caged = common.caged_temperatures(aimd)
    info = aimd.set_index("temperature_K")
    by_run = comparison.set_index(["series", "temperature_K"])
    warm = [t for t in C.TEMPERATURES if t not in caged]

    def ratio_range(series_list, temperatures) -> str:
        found = [by_run.loc[(s, t), "kinisi_D_ratio_to_aimd"] for s in series_list for t in temperatures]
        return f"{sig(min(found))}-{sig(max(found))}"

    def slopes(series: str) -> str:
        return " and ".join(sig(by_run.loc[(series, t), "msd_loglog_slope"]) for t in MSD_TEMPERATURES)

    length = float(info["length_ps"].iloc[0])
    a_mlip = float(transport.loc[transport["n_atoms"] == 52, "volume_A3"].iloc[0]) ** (1.0 / 3.0)
    a_aimd = float(info["a_A"].iloc[0])
    smaller = 100.0 * (1.0 - (a_aimd / a_mlip) ** 3)
    caged_text = " and ".join(f"{t} K" for t in caged)
    caged_slope = ", ".join(sig(info.loc[t, "msd_loglog_slope_fit_window"]) for t in caged)
    n_fits = len(frame)
    n_clx = int(transport["series"].str.startswith("clx").sum())
    n_g = sum(series is not None for series in G_ORDER) * len(C.TEMPERATURES)
    return (
        registry.heading(KEY)
        + "Every run is a single trajectory per temperature; D is the Li tracer diffusion coefficient. "
        f"**a**, Root-mean-square difference of the atomic positions between two runs with the same potential, "
        f"cell and temperature, against production time (n = {pairs} pairs: {pairs_52} among the 52-atom runs of "
        f"50, 100 and 150 ps, three per universal potential and temperature, and the 416-atom, 200-ps MACE-MP-0 "
        f"production run against the 416-atom Cl-excess x = 0 run at each temperature, thick lines). Positions are "
        f"stored at frames 0, 1, 2, 5, 10, 20, 50, 100, 200 and 500 (one frame every 50 fs); the axis is linear "
        f"below {first_time:g} ps and logarithmic above. At the first production frame the pairs already differ "
        f"by {small_zero.min():.2f}-{small_zero.max():.2f} A (52 atoms) and {big_zero.min():.2f}-"
        f"{big_zero.max():.2f} A (416 atoms), and by {last.min():.2f}-{last.max():.2f} A after "
        f"{divergence['time_ps'].max():g} ps, so the runs are independent trajectories and not extensions of one "
        f"another. Single values, no error bars. "
        f"**b**, pymatgen D of the four consecutive 50-ps blocks of each 416-atom, 200-ps production run divided by "
        f"the value from the full run (n = {len(blocks)} blocks of {len(PRODUCTION)} runs at "
        f"{len(C.TEMPERATURES)} temperatures; blocks in time order from left to right; logarithmic axis; pymatgen "
        "gives no uncertainty, so no error bars). Open markers, the 2x2x2 fine-tune, whose MD used its stage-one "
        "model (fill marks the training stage, as in every figure). The ratio is "
        + "; ".join(f"{ranges.loc[t, 'min']:.2f}-{ranges.loc[t, 'max']:.2f} at {t} K" for t in C.TEMPERATURES)
        + f". At {min(C.TEMPERATURES)} K the slower models give block values well above the full-run value, i.e. "
        "pymatgen D still depends on the length of the analysed window there. "
        f"**c**, kinisi D of the same runs with the fit window starting at "
        f"{_and(f'{100 * f:.0f}' for f in fractions)} % of the run, divided by the primary value (start at "
        f"{100 * C.KINISI_START_FRACTION:.0f} %, grey line; n = {len(fractions)} starts for each of "
        f"{len(PRODUCTION) * len(C.TEMPERATURES)} runs); error bars, kinisi posterior standard deviation "
        f"divided by the same primary value; markers as in b, the three temperatures offset horizontally. Above {min(C.TEMPERATURES)} K the ratio stays within "
        f"{hot['ratio'].min():.2f}-{hot['ratio'].max():.2f}; the largest change is {extreme['ratio']:.2f} "
        f"({_run_label(extreme['run_id'])}, start {extreme['value']:g}). "
        f"**d**, pymatgen D divided by kinisi D for all {len(transport)} runs (grey line, equality; logarithmic "
        f"axis; single values, no error bars: the kinisi posterior standard deviation is {100 * posterior.min():.0f}-"
        f"{100 * posterior.max():.0f} % of D). Colour, potential (key in the panel); circles, the 416-atom "
        f"production runs; triangles, the 52-atom runs; inverted triangles, the {n_clx} runs of the four 150-ps "
        f"Cl-excess series of MACE-MP-0 (x = 0-0.75, {registry.label('clexcess')}). Within each temperature, from "
        "left: MACE-MP-0 52-atom runs of 50, 100 and 150 ps, production run and Cl-excess runs x = 0, 0.25, 0.5 and "
        "0.75, then MatterSim 52-atom runs and production run, then the Gamma and 2x2x2 fine-tunes. The ratio is "
        + "; ".join(f"{by_t.loc[t, 'min']:.2f}-{by_t.loc[t, 'max']:.2f} at {t} K" for t in C.TEMPERATURES)
        + f" (largest {ratio.max():.2f}, {_run_label(worst['run_id'])}). "
        f"**e**, sigma(300 K) extrapolated from weighted Arrhenius fits of the pymatgen conductivity at "
        f"{_and(C.TEMPERATURES)} K (re-analysis fits, weights from the kinisi relative standard deviation; "
        f"{registry.label('arrhenius')} shows the quoted fit of each series, which equals the published fit for the "
        f"published series) against simulated time per run (n = {n_fits} fits, each on "
        f"{len(C.TEMPERATURES)} runs); error bars, the 1-sigma interval propagated from the fit covariance. "
        f"Triangles, the 52-atom runs (MACE-MP-0 {values(fits('foundation', 'cell_52'), 'sigma300_mS_cm')} and "
        f"MatterSim {values(fits('mattersim', 'cell_52'), 'sigma300_mS_cm')} mS/cm at 50, 100 and 150 ps); circles "
        "joined by lines, the 416-atom production run truncated to its first 50, 100 and 150 ps and in full "
        f"(MACE-MP-0 {values(fits('foundation', 'production'), 'sigma300_mS_cm')}, MatterSim "
        f"{values(fits('mattersim', 'production'), 'sigma300_mS_cm')} mS/cm); inverted triangle, the second "
        f"416-atom MACE-MP-0 run (Cl-excess x = 0, 150 ps, {sig(second['sigma300_mS_cm'])} mS/cm). The truncated "
        "windows are nested parts of one trajectory, and markers at the same simulated time are offset "
        "horizontally. "
        f"**f**, Activation energy of the same fits; error bars, 1 standard deviation of the fitted slope "
        f"(production windows: MACE-MP-0 {values(fits('foundation', 'production'), 'Ea_eV')} eV, MatterSim "
        f"{values(fits('mattersim', 'production'), 'Ea_eV')} eV; 52-atom runs "
        f"{sig(frame.loc[frame['run_kind'] == 'cell_52', 'Ea_eV'].min())}-"
        f"{sig(frame.loc[frame['run_kind'] == 'cell_52', 'Ea_eV'].max())} eV). "
        f"**g**, kinisi D (seed 0, fitted from {100 * C.KINISI_START_FRACTION:.0f} % of the run) of the 52-atom "
        f"runs, which have the 52-atom cell of the AIMD reference ({registry.label('aimd')}; lattice parameter "
        f"{a_mlip:.2f} A here against {a_aimd:.2f} A, a {sig(smaller)} % smaller AIMD volume), and of the second "
        "416-atom MACE-MP-0 run, divided by the AIMD kinisi D at the same temperature (n = "
        f"{n_g} ratios); error bars, the ratio times the two relative standard deviations added in quadrature. "
        "Within each temperature, from left: MACE-MP-0 52-atom runs of 50, 100 and 150 ps and the second 416-atom "
        "run, then MatterSim 52-atom runs of 50, 100 and 150 ps. At "
        f"{_and(warm)} K the 52-atom runs give {ratio_range(CELL_52['foundation'], warm)} "
        f"(MACE-MP-0) and {ratio_range(CELL_52['mattersim'], warm)} (MatterSim), the second run "
        f"{ratio_range([SECOND_RUN], warm)}; the production runs are compared in {registry.label('diffusion')}. "
        f"Grey column, at {caged_text} the AIMD Li MSD is almost flat in the fit window (log-log slope "
        f"{caged_slope}): those ratios compare caged motion, not diffusion coefficients "
        f"({registry.label('aimd', short=False)} shows all AIMD temperatures). "
        f"**h**, Li mean-squared displacement against lag time at {MSD_TEMPERATURES[0]} and {MSD_TEMPERATURES[1]} K "
        f"(all time origins, centre-of-mass drift removed) of AIMD (black) and of the {length:.0f}-ps, 52-atom runs "
        f"that match the AIMD length and cell (one trajectory each; log-log slope in the fit window "
        f"{slopes(MSD_SERIES['foundation'])} for MACE-MP-0 and {slopes(MSD_SERIES['mattersim'])} for MatterSim at "
        f"{MSD_TEMPERATURES[0]} and {MSD_TEMPERATURES[1]} K; AIMD slope printed in each panel). Grey, the kinisi fit "
        f"window of all three (lags from {C.KINISI_START_FRACTION * length:.0f} ps). Single curves, no error bars. "
        + common.sources([DIVERGENCE, WINDOWS, TRANSPORT, CONVERGENCE, COMPARISON, AIMD, AIMD_MSD, MSD])
    )


CAPTION = _caption()
