"""Extended Data Fig. 4: sampling and estimator checks.

a, RMS position difference between runs that share their settings (the three 52-atom tiers of each
universal potential, and the 416-atom MACE-MP-0 production run against the Cl-excess x = 0 run)
against production time; b, pymatgen D of the four consecutive 50-ps blocks of the four 416-atom,
200-ps production runs relative to the full-run value; c, kinisi D against the start of its fit
window, relative to the primary start at 30 %; d, pymatgen D against kinisi D for all 42 runs;
e, mean-squared displacement of P, S and Cl next to Li in the production runs at 1000 K; f, the
Monte-Carlo scatter of kinisi D over seeds 1-10 against the width of its posterior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from .. import constants as C
from .. import paths
from ..transport import reproduction as R
from . import common, style

NAME = "ed04"
WIDTH, HEIGHT = 183.0, 172.0
TRANSPORT = paths.MD / "transport.csv"
SEEDS = paths.MD / "kinisi_seeds.csv"
WINDOWS = paths.MD / "windows.csv"
DIVERGENCE = paths.MD / "divergence.csv"
MSD = paths.MD / "msd.csv.gz"
MSD_TEMPERATURE = 1000
PRODUCTION_MODELS = {"mace_416_200": "foundation", "mattersim_416_200": "mattersim", "ftG_416_200": "ftG",
                     "ftK_416_200": "ftK"}
PRODUCTION_TITLES = {"mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim",
                     "ftG_416_200": "Fine-tuned, Gamma", "ftK_416_200": "Fine-tuned, 2x2x2"}
# Series families of d and f: (label, colour key, marker, filled, series ids)
FAMILIES = (
    ("MACE-MP-0, 52 atoms", "foundation", "o", False, ("mace_52_050", "mace_52_100", "mace_52_150")),
    ("MatterSim, 52 atoms", "mattersim", "o", False, ("mattersim_52_050", "mattersim_52_100", "mattersim_52_150")),
    ("MACE-MP-0, 416 atoms", "foundation", "o", True, ("mace_416_200",)),
    ("MatterSim, 416 atoms", "mattersim", "o", True, ("mattersim_416_200",)),
    ("Fine-tuned, Gamma labels", "ftG", "o", True, ("ftG_416_200",)),
    ("Fine-tuned, 2x2x2 labels", "ftK", "o", True, ("ftK_416_200",)),
    ("Cl-excess (MACE-MP-0), 392-416 atoms", "clexcess", "s", True,
     ("clx000_416_150", "clx025_408_150", "clx050_400_150", "clx075_392_150")),
)
FAMILY_COLORS = {**style.MODEL_COLORS, "clexcess": style.OKABE_ITO["purple"]}
DIVERGENCE_STYLES = {"mace_52": ("MACE-MP-0, 52-atom tiers", "-", 0.7),
                     "mattersim_52": ("MatterSim, 52-atom tiers", (0, (3, 1.5)), 0.7),
                     "mace_416": ("MACE-MP-0 416 atoms vs Cl x = 0", "-", 1.6)}
T_OFFSETS = dict(zip(C.TEMPERATURES, (-0.27, 0.0, 0.27)))
BLOCK_OFFSETS = (-0.09, -0.03, 0.03, 0.09)


def _load():
    transport = common.read(TRANSPORT)
    seeds = common.read(SEEDS)
    windows = common.read(WINDOWS)
    divergence = common.read(DIVERGENCE)
    msd = common.read(MSD)
    return transport, seeds, windows, divergence, msd


def _family_of(series: str) -> tuple:
    for family in FAMILIES:
        if series in family[4]:
            return family
    raise ValueError(f"series {series!r} in no family")


def _family_marker(family: tuple, size: float = 3.4) -> dict:
    color = FAMILY_COLORS[family[1]]
    return {"marker": family[2], "markersize": size, "markeredgewidth": 0.7, "markeredgecolor": color,
            "markerfacecolor": color if family[3] else "white", "linestyle": "none"}


def _seed_scatter(seeds: pd.DataFrame) -> pd.DataFrame:
    """Relative standard deviation of kinisi D over seeds 1-10 (seed 0 is the primary value), the s_seed of
    the reproduction rule."""
    scatter = seeds[seeds["seed"] != 0].groupby("run_id", sort=False)["D_cm2_s"]
    out = (scatter.std(ddof=1) / scatter.mean()).rename("s_seed").reset_index()
    out["n_seeds"] = scatter.size().to_numpy()
    return out


def _run_label(run: str) -> str:
    """Readable name of a run id, e.g. 'Fine-tuned on Gamma labels, 416 atoms, 200 ps, 600 K'."""
    series, temperature = run.rsplit("_", 1)
    return f"{C.SERIES_BY_ID[series]['label']}, {temperature.replace('K', ' K')}"


def _divergence_key(run: str) -> str:
    parts = run.split("_")
    return f"{parts[0]}_{parts[1]}"


def _symlog_axis(ax, linthresh: float) -> None:
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.35)
    ax.set_xlim(0, 32)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10]))
    ax.set_xticks([0, 0.1, 1, 10], ["0", "0.1", "1", "10"])
    minor = [m * 10.0 ** e for e in (-2, -1, 0, 1) for m in range(2, 10) if linthresh <= m * 10.0 ** e <= 32]
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    ax.xaxis.set_minor_formatter(NullFormatter())


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
    ax.set_ylim(0, 10.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Production time (ps)")
    ax.set_ylabel("RMS position difference (Å)")
    at_zero = frame[frame["time_ps"] == 0]["rms_position_difference_A"]
    ax.axhspan(0, at_zero.min(), color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.text(0.02, at_zero.min() / 2, f"all pairs differ by ≥ {at_zero.min():.2f} Å at the first frame",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=style.SMALL)
    handles = [Line2D([0], [0], color="black", lw=width, ls=dash, marker="o" if key == "mace_416" else None,
                      markersize=2.0, markeredgewidth=0, label=label)
               for key, (label, dash, width) in DIVERGENCE_STYLES.items()]
    handles += [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=1.0, label=f"{t} K") for t in C.TEMPERATURES]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=style.SMALL, handlelength=1.8, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.3, borderaxespad=0.2)
    source = frame[["run_a", "run_b", "kind", "temperature_K", "frame", "time_ps", "rms_position_difference_A"]]
    style.save_source(source, NAME, "a")


def _panel_b(ax, transport: pd.DataFrame, windows: pd.DataFrame) -> None:
    full = transport.set_index("run_id")["D_cm2_s"]
    blocks = windows[windows["kind"] == "block_start_ps"].copy()
    rows = []
    ax.axhline(1.0, color="black", lw=0.6, zorder=1)
    for index, series in enumerate(C.PRODUCTION_SERIES):
        for temperature in C.TEMPERATURES:
            run = C.run_id(series, temperature)
            part = blocks[blocks["run_id"] == run].sort_values("value")
            ratio = (part["D_cm2_s"] / full[run]).to_numpy()
            x = index + T_OFFSETS[temperature] + np.array(BLOCK_OFFSETS[:len(part)])
            color = style.TEMPERATURE_COLORS[temperature]
            ax.plot(x, ratio, color=color, lw=0.5, zorder=2)
            ax.plot(x, ratio, marker="o", linestyle="none", markersize=2.6, markeredgewidth=0, color=color,
                    zorder=3)
            rows.extend({"run_id": run, "series": series, "temperature_K": temperature, "block_start_ps": start,
                         "x": xi, "D_block_cm2_s": d, "D_full_run_cm2_s": full[run], "ratio": r}
                        for start, xi, d, r in zip(part["value"], x, part["D_cm2_s"], ratio))
    for index in range(1, len(C.PRODUCTION_SERIES)):
        ax.axvline(index - 0.5, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 4.6)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.7, 1, 1.5, 2, 3], ["0.7", "1", "1.5", "2", "3"])
    ax.set_xlim(-0.5, len(C.PRODUCTION_SERIES) - 0.5)
    ax.set_xticks(range(len(C.PRODUCTION_SERIES)), [PRODUCTION_TITLES[s] for s in C.PRODUCTION_SERIES])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("D of a 50-ps block / D of the\nfull 200-ps run (pymatgen)")
    handles = [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=0.5, marker="o", markersize=2.6,
                      markeredgewidth=0, label=f"{t} K") for t in C.TEMPERATURES]
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=style.SMALL, handlelength=1.6, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.2)
    ax.text(0.99, 0.97, "four consecutive blocks, left to right", transform=ax.transAxes, ha="right", va="top",
            fontsize=style.SMALL)
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(axes, transport: pd.DataFrame, windows: pd.DataFrame) -> None:
    primary = transport.set_index("run_id")
    starts = windows[windows["kind"] == "kinisi_start_fraction"]
    rows = []
    for index, (ax, series) in enumerate(zip(axes, C.PRODUCTION_SERIES)):
        ax.axhline(1.0, color=style.GREY, lw=0.5, zorder=1)
        ax.axvline(C.KINISI_START_FRACTION, color=style.LIGHT_GREY, lw=0.6, zorder=0)
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
            dx = (temperature - 800) / 200 * 0.012
            color = style.TEMPERATURE_COLORS[temperature]
            ax.errorbar(part["value"] + dx, ratio, yerr=error, color=color, lw=0.7, elinewidth=0.5, capsize=1.0,
                        capthick=0.5, marker="o", markersize=2.4, markeredgewidth=0, zorder=3)
            rows.extend({"run_id": run, "series": series, "temperature_K": temperature, "start_fraction": f,
                         "kinisi_D_cm2_s": d, "kinisi_D_std_cm2_s": s, "D_at_0.3_cm2_s": reference,
                         "ratio": r, "ratio_error": e}
                        for f, d, s, r, e in zip(part["value"], part["kinisi_D_cm2_s"], part["kinisi_D_std_cm2_s"],
                                                 ratio, error))
        ax.set_xlim(0.05, 0.55)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"])
        ax.set_ylim(0.55, 2.05)
        ax.set_yticks([0.6, 1.0, 1.4, 1.8])
        ax.set_title(PRODUCTION_TITLES[series], fontsize=style.SMALL, pad=2.0,
                     color=style.MODEL_COLORS[PRODUCTION_MODELS[series]])
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0.6, 1.0, 1.4, 1.8], ["0.6", "1.0", "1.4", "1.8"])
            ax.set_ylabel("kinisi D / D at 0.3")
    axes[0].figure.text(*_between(axes[1], axes[2]), "Start of the kinisi fit (fraction of the run)",
                        ha="center", va="top", fontsize=style.FONT_SIZE)
    handles = [Line2D([0], [0], color=style.TEMPERATURE_COLORS[t], lw=0.7, marker="o", markersize=2.4,
                      markeredgewidth=0, label=f"{t} K") for t in C.TEMPERATURES]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=style.SMALL, handlelength=1.4, handletextpad=0.4,
                    labelspacing=0.25, borderaxespad=0.2)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _between(left_ax, right_ax) -> tuple[float, float]:
    """Figure position centred between two axes, just below their tick labels."""
    a, b = left_ax.get_position(), right_ax.get_position()
    return (a.x1 + b.x0) / 2, a.y0 - 0.032


def _panel_d(ax, transport: pd.DataFrame) -> None:
    rows = []
    lo, hi = 5e-7, 2e-4
    ax.plot([lo, hi], [lo, hi], color=style.GREY, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for row in transport.itertuples():
        family = _family_of(row.series)
        ax.plot(row.kinisi_D_cm2_s, row.D_cm2_s, **_family_marker(family, 3.0), zorder=3)
        rows.append({"run_id": row.run_id, "series": row.series, "family": family[0],
                     "temperature_K": row.temperature_K, "kinisi_D_cm2_s": row.kinisi_D_cm2_s,
                     "pymatgen_D_cm2_s": row.D_cm2_s, "ratio_pymatgen_over_kinisi": row.D_cm2_s / row.kinisi_D_cm2_s})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_aspect("equal")
    ax.set_xlabel("kinisi D (cm$^2$ s$^{-1}$)")
    ax.set_ylabel("pymatgen D (cm$^2$ s$^{-1}$)")
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _panel_e(axes, msd: pd.DataFrame) -> None:
    rows = []
    for index, (ax, series) in enumerate(zip(axes, C.PRODUCTION_SERIES)):
        run = C.run_id(series, MSD_TEMPERATURE)
        frame = msd[msd["run_id"] == run].sort_values("lag_ps")
        for element in common.ELEMENTS:
            ax.plot(frame["lag_ps"], frame[f"msd_{element}_A2"], color=style.ELEMENT_COLORS[element], lw=0.9,
                    zorder=3 if element == "Li" else 2)
        rows.extend({"run_id": run, "series": series, "lag_ps": lag, "msd_Li_A2": li, "msd_P_A2": p, "msd_S_A2": s,
                     "msd_Cl_A2": cl}
                    for lag, li, p, s, cl in zip(frame["lag_ps"], frame["msd_Li_A2"], frame["msd_P_A2"],
                                                 frame["msd_S_A2"], frame["msd_Cl_A2"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.04, 250)
        ax.set_ylim(0.03, 2000)
        style.log_ticks(ax, "x")
        style.log_ticks(ax, "y")
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_title(PRODUCTION_TITLES[series], fontsize=style.SMALL, pad=2.0,
                     color=style.MODEL_COLORS[PRODUCTION_MODELS[series]])
        last = frame.iloc[-1]
        ax.text(0.97, 0.04, f"P {last['msd_P_A2']:.2f}\nS {last['msd_S_A2']:.2f}\nCl {last['msd_Cl_A2']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=style.SMALL, linespacing=1.15)
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
            ax.set_ylabel(f"MSD at {MSD_TEMPERATURE} K (Å$^2$)")
    axes[0].figure.text(*_between(axes[1], axes[2]), "Lag time (ps)", ha="center", va="top",
                        fontsize=style.FONT_SIZE)
    handles = [Line2D([0], [0], color=style.ELEMENT_COLORS[e], lw=1.0, label=e) for e in common.ELEMENTS]
    axes[0].legend(handles=handles, loc="upper left", ncol=2, fontsize=style.SMALL, handlelength=1.2,
                   handletextpad=0.3, columnspacing=0.6, labelspacing=0.25, borderaxespad=0.2)
    style.save_source(pd.DataFrame(rows), NAME, "e")


def _panel_f(ax, transport: pd.DataFrame, seeds: pd.DataFrame) -> None:
    scatter = _seed_scatter(seeds).merge(transport[["run_id", "series", "temperature_K", "kinisi_D_cm2_s",
                                                    "kinisi_D_std_cm2_s", "seed_scatter_D_rel"]], on="run_id")
    if not np.allclose(scatter["s_seed"], scatter["seed_scatter_D_rel"], rtol=1e-9, atol=0):
        raise ValueError("seed scatter differs from transport.csv")
    scatter["posterior_rel"] = scatter["kinisi_D_std_cm2_s"] / scatter["kinisi_D_cm2_s"]
    crossover = R.KINISI_D_FLOOR / R.KINISI_SEED_FACTOR * 100.0
    ax.axhline(crossover, color="black", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, crossover * 1.08, f"{R.KINISI_SEED_FACTOR:g} s$_{{seed}}$ = {100 * R.KINISI_D_FLOOR:g} % floor",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=style.SMALL)
    rows = []
    for row in scatter.itertuples():
        family = _family_of(row.series)
        ax.plot(row.posterior_rel * 100.0, row.s_seed * 100.0, **_family_marker(family, 3.0), zorder=3)
        rows.append({"run_id": row.run_id, "series": row.series, "family": family[0],
                     "temperature_K": row.temperature_K, "kinisi_posterior_rel_std_percent": row.posterior_rel * 100,
                     "s_seed_percent": row.s_seed * 100.0, "n_seeds": row.n_seeds,
                     "rule_tolerance_percent": max(R.KINISI_SEED_FACTOR * row.s_seed, R.KINISI_D_FLOOR) * 100.0})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.5, 50)
    ax.set_ylim(0.08, 2.5)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
    ax.set_xlabel("kinisi posterior std / D (%)")
    ax.set_ylabel("Seed scatter s$_{seed}$ of kinisi D (%)")
    style.save_source(pd.DataFrame(rows), NAME, "f")


def _family_legend(fig, L: common.Layout, top: float) -> None:
    handles = [Line2D([0], [0], label=family[0], **_family_marker(family, 3.2)) for family in FAMILIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, L.y(top)), ncol=len(handles),
               fontsize=style.SMALL, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0,
               title="Series in d and f:", title_fontsize=style.SMALL)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    transport, seeds, windows, divergence, msd = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    ax_a = L.axes(15.0, 8.0, 68.0, 38.0)
    _panel_a(ax_a, divergence)
    ax_b = L.axes(103.0, 8.0, 78.0, 38.0)
    _panel_b(ax_b, transport, windows)

    width, gap = 23.5, 3.0
    ax_c = [L.axes(15.0 + i * (width + gap), 63.0, width, 31.0) for i in range(4)]
    _panel_c(ax_c, transport, windows)
    ax_d = L.axes(140.0, 60.0, 36.0, 36.0)
    _panel_d(ax_d, transport)

    ax_e = [L.axes(15.0 + i * (width + gap), 115.0, width, 31.0) for i in range(4)]
    _panel_e(ax_e, msd)
    ax_f = L.axes(140.0, 113.0, 36.0, 33.0)
    _panel_f(ax_f, transport, seeds)

    _family_legend(fig, L, 160.0)

    L.letter(ax_a, "a", 3.0, 7.0)
    L.letter(ax_b, "b", 90.0, 7.0)
    L.letter(ax_c[0], "c", 3.0, 58.0)
    L.letter(ax_d, "d", 124.0, 58.0)
    L.letter(ax_e[0], "e", 3.0, 110.0)
    L.letter(ax_f, "f", 124.0, 110.0)
    return style.save(fig, "ed04_sampling", "extended")


def _caption() -> str:
    transport, seeds, windows, divergence, msd = _load()
    at_zero = divergence[divergence["time_ps"] == 0]
    small_zero = at_zero[at_zero["run_a"].str.contains("_52_")]["rms_position_difference_A"]
    big_zero = at_zero[~at_zero["run_a"].str.contains("_52_")]["rms_position_difference_A"]
    first_frame = int(divergence["frame"][divergence["frame"] > 0].min())
    first_time = divergence.loc[divergence["frame"] == first_frame, "time_ps"].iloc[0]
    last = divergence[divergence["time_ps"] == divergence["time_ps"].max()]["rms_position_difference_A"]
    full = transport.set_index("run_id")["D_cm2_s"]
    blocks = windows[windows["kind"] == "block_start_ps"].copy()
    blocks["ratio"] = blocks["D_cm2_s"] / blocks["run_id"].map(full)
    blocks["temperature_K"] = blocks["run_id"].str.extract(r"_(\d+)K$")[0].astype(int)
    ranges = blocks.groupby("temperature_K")["ratio"].agg(["min", "max"])
    starts = windows[windows["kind"] == "kinisi_start_fraction"].copy()
    starts["ratio"] = starts["kinisi_D_cm2_s"] / starts["run_id"].map(transport.set_index("run_id")["kinisi_D_cm2_s"])
    extreme = starts.loc[(starts["ratio"] - 1).abs().idxmax()]
    hot = starts[~starts["run_id"].str.endswith("_600K")]
    ratio = transport["D_cm2_s"] / transport["kinisi_D_cm2_s"]
    by_t = transport.assign(ratio=ratio).groupby("temperature_K")["ratio"].agg(["min", "max"])
    worst = transport.loc[ratio.idxmax()]
    msd_runs = msd[msd["run_id"].isin([C.run_id(s, MSD_TEMPERATURE) for s in C.PRODUCTION_SERIES])]
    tail = msd_runs.sort_values("lag_ps").groupby("run_id").tail(1).set_index("run_id")
    tails = {s: tail.loc[C.run_id(s, MSD_TEMPERATURE)] for s in C.PRODUCTION_SERIES}
    max_lag = float(tail["lag_ps"].max())
    s_text = ", ".join(f"{PRODUCTION_TITLES[s]} {tails[s]['msd_S_A2']:.2f}" for s in C.PRODUCTION_SERIES)
    p_max = max(tails[s]["msd_P_A2"] for s in C.PRODUCTION_SERIES)
    cl_max = max(tails[s]["msd_Cl_A2"] for s in C.PRODUCTION_SERIES)
    li_min = min(tails[s]["msd_Li_A2"] for s in C.PRODUCTION_SERIES)
    scatter = _seed_scatter(seeds).merge(transport[["run_id", "kinisi_D_cm2_s", "kinisi_D_std_cm2_s"]], on="run_id")
    posterior = scatter["kinisi_D_std_cm2_s"] / scatter["kinisi_D_cm2_s"]
    crossover = R.KINISI_D_FLOOR / R.KINISI_SEED_FACTOR
    above = int((scatter["s_seed"] > crossover).sum())
    share = scatter["s_seed"] / posterior
    return (
        "Extended Data Fig. 4 | Sampling and estimator checks. "
        f"**a**, Root-mean-square difference of the atomic positions between two runs with the same potential, "
        f"cell and temperature, against production time: the 52-atom runs of 50, 100 and 150 ps of each universal "
        f"potential (three pairs per potential and temperature) and the 416-atom, 200-ps MACE-MP-0 production run "
        f"against the 416-atom Cl-excess x = 0 run (thick lines). Positions are stored at frames 0, 1, 2, 5, 10, "
        f"20, 50, 100, 200 and 500 (one frame every 50 fs); the axis is linear below {first_time:g} ps and "
        f"logarithmic above. At the first production frame the pairs already differ by "
        f"{small_zero.min():.2f}-{small_zero.max():.2f} A (52 atoms) and {big_zero.min():.2f}-{big_zero.max():.2f} "
        f"A (416 atoms), and by {last.min():.2f}-{last.max():.2f} A after {divergence['time_ps'].max():g} ps, so "
        f"the runs are independent trajectories and not extensions of one another. "
        f"**b**, pymatgen D of the four consecutive 50-ps blocks of each 416-atom, 200-ps production run divided by "
        f"the value from the full run (blocks in time order from left to right; logarithmic axis). The ratio is "
        + "; ".join(f"{ranges.loc[t, 'min']:.2f}-{ranges.loc[t, 'max']:.2f} at {t} K" for t in C.TEMPERATURES)
        + ". At 600 K the slower models give block values well above the full-run value, i.e. pymatgen D still "
        "depends on the length of the analysed window there. "
        f"**c**, kinisi D with the fit window starting at 10-50 % of the run, divided by the primary value (start "
        f"at {100 * C.KINISI_START_FRACTION:.0f} %, grey line); error bars, kinisi posterior standard deviation "
        f"divided by the same primary value. Above 600 K the ratio stays within {hot['ratio'].min():.2f}-"
        f"{hot['ratio'].max():.2f}; the largest change is {extreme['ratio']:.2f} ({_run_label(extreme['run_id'])}, start "
        f"{extreme['value']:g}). "
        f"**d**, pymatgen D against kinisi D for all 42 runs (dashed line, equality). pymatgen D / kinisi D is "
        + "; ".join(f"{by_t.loc[t, 'min']:.2f}-{by_t.loc[t, 'max']:.2f} at {t} K" for t in C.TEMPERATURES)
        + f" (largest {ratio.max():.2f}, {_run_label(worst['run_id'])}). "
        f"**e**, Mean-squared displacement of Li, P, S and Cl (all time origins, centre-of-mass drift removed) in "
        f"the four production runs at {MSD_TEMPERATURE} K. At the longest lag ({max_lag:g} ps, a single time "
        f"origin) Li reaches at least {li_min:.0f} A^2, P at most {p_max:.2f} A^2 and Cl at most {cl_max:.2f} A^2: "
        f"the P framework and the Cl sublattice stay in place. S reaches {s_text} A^2 (values printed in each "
        "panel); S moving further than P is consistent with reorientation of PS4 tetrahedra about fixed P. "
        f"**f**, Monte-Carlo scatter of kinisi D, s_seed = relative standard deviation of D over seeds 1-10 (seed 0 "
        f"is the primary value), against the relative width of the kinisi posterior (seed 0). The reproduction rule "
        f"accepts a kinisi D within max({R.KINISI_SEED_FACTOR:g} s_seed, {100 * R.KINISI_D_FLOOR:g} %); above the "
        f"dashed line ({100 * crossover:.3f} %) the seed term sets the tolerance, which is the case in {above} of "
        f"42 runs. s_seed is {100 * scatter['s_seed'].min():.2f}-{100 * scatter['s_seed'].max():.2f} %, "
        f"{share.min():.3f}-{share.max():.3f} times the posterior width. "
        + common.sources([DIVERGENCE, WINDOWS, TRANSPORT, MSD, SEEDS])
    )


CAPTION = _caption()
