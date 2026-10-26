"""Fig. 2: the universal potentials against experiment.

a, Arrhenius plot of the two 416-atom, 200-ps production series (MACE-MP-0 and MatterSim): per-run
conductivity with its kinisi uncertainty, the quoted weighted fit extrapolated to 300 K with its
1-sigma band, the measured value and the compiled experimental range; b, Li mean-squared
displacement of the same six runs against lag time with the kinisi fit window; c, sigma(300 K)
and d, activation energy against simulated time for the 52-atom tiers, truncations of the
production runs and a second independent 416-atom MACE-MP-0 run. Only the universal potentials
are compared with experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import paths
from ..constants import KINISI_START_FRACTION, SERIES_BY_ID, T_REF_K, TEMPERATURES
from . import common, style

NAME = "fig2"
WIDTH, HEIGHT = 183.0, 134.0
FITS = paths.RESULTS / "fits.csv"
CONVERGENCE = paths.RESULTS / "convergence.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
LITERATURE = paths.REFERENCES / "literature.csv"
PRODUCTION = ("mace_416_200", "mattersim_416_200")
MODELS = ("foundation", "mattersim")
KIND_MARKERS = {"cell_52": "o", "production_window": "s", "production_full": "s", "independent_run": "^"}
KIND_FILLED = {"cell_52": False, "production_window": True, "production_full": True, "independent_run": True}
# Horizontal offsets (ps) so that markers at the same simulated time do not overlap.
KIND_OFFSET = {"cell_52": -6.0, "production_window": 2.0, "production_full": 2.0, "independent_run": 10.0}
MODEL_OFFSET = {"foundation": -2.0, "mattersim": 2.0}
T_TOP_TICKS = (1000, 800, 600, 500, 400, 300)


def _load():
    fits = common.read(FITS)
    fits = fits[fits["quoted"]].set_index("series")
    convergence = common.read(CONVERGENCE)
    transport = common.read(TRANSPORT)
    msd = common.read(MSD)
    literature = common.read(LITERATURE).set_index("ref_id")
    return fits, convergence, transport, msd, literature


def _experiment(literature: pd.DataFrame) -> tuple[float, float, float]:
    """Measured sigma_RT (Wang et al. 2018) and the compiled range (Li et al. 2025), mS/cm."""
    return (float(literature.loc["wang2018", "value"]), float(literature.loc["li2025_compiled", "value_low"]),
            float(literature.loc["li2025_compiled", "value_high"]))


def _fit_band(row: pd.Series, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sigma(T) of the fit ln(sigma T) = slope/T + intercept and its 1-sigma band (mS/cm).

    The slope-intercept covariance is recovered from the stored 300 K interval, so the band equals
    the quoted interval at 300 K exactly.
    """
    x0 = 1.0 / T_REF_K
    var_s, var_i = row["slope_err"] ** 2, row["intercept_err"] ** 2
    half = 0.5 * (np.log(row["sigma300_max_mS_cm"]) - np.log(row["sigma300_min_mS_cm"]))
    covariance = (half ** 2 - x0 ** 2 * var_s - var_i) / (2.0 * x0)
    x = 1.0 / temperature
    y = row["slope"] * x + row["intercept"]
    sd = np.sqrt(np.maximum(x ** 2 * var_s + var_i + 2.0 * x * covariance, 0.0))
    return np.exp(y) / temperature, np.exp(y - sd) / temperature, np.exp(y + sd) / temperature


def _label(model: str) -> str:
    return style.MODEL_LABELS[model]


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, fits: pd.DataFrame, transport: pd.DataFrame, experiment: tuple[float, float, float]) -> Path:
    measured, low, high = experiment
    x300 = 1000.0 / T_REF_K
    rows = []
    x_lo, x_split = 1000.0 / max(TEMPERATURES), 1000.0 / min(TEMPERATURES)
    x_solid = np.linspace(x_lo, x_split, 40)
    x_dashed = np.linspace(x_split, x300, 80)
    handles = []
    for series in PRODUCTION:
        fit = fits.loc[series]
        model = fit["model"]
        color = style.MODEL_COLORS[model]
        x = np.concatenate([x_solid, x_dashed[1:]])
        t_fit = 1000.0 / x
        sigma, lo, hi = _fit_band(fit, t_fit)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=1)
        solid = x <= x_split
        ax.plot(x[solid], sigma[solid], color=color, lw=0.9, zorder=3)
        ax.plot(x[x >= x_split], sigma[x >= x_split], color=color, lw=0.9, ls=(0, (3, 1.5)), zorder=3)
        rows.extend({"kind": "fit", "series": series, "potential": fit["potential"], "temperature_K": t,
                     "inv_temperature_1000_per_K": xx, "sigma_mS_cm": s, "err_mS_cm": np.nan,
                     "band_low_mS_cm": a, "band_high_mS_cm": b, "source": fit["source"]}
                    for t, xx, s, a, b in zip(t_fit, x, sigma, lo, hi))
        points = transport[transport["series"] == series].sort_values("temperature_K")
        ax.errorbar(1000.0 / points["temperature_K"], points["sigma_mS_cm"], yerr=points["kinisi_sigma_std_mS_cm"],
                    fmt="o", color=color, markersize=3.4, markeredgewidth=0.0, elinewidth=0.7, capsize=1.5,
                    zorder=4)
        rows.extend({"kind": "run", "series": series, "potential": fit["potential"], "temperature_K": t,
                     "inv_temperature_1000_per_K": 1000.0 / t, "sigma_mS_cm": s, "err_mS_cm": e,
                     "band_low_mS_cm": np.nan, "band_high_mS_cm": np.nan, "source": "pymatgen; kinisi 1-sigma"}
                    for t, s, e in zip(points["temperature_K"], points["sigma_mS_cm"],
                                       points["kinisi_sigma_std_mS_cm"]))
        handles.append(Line2D([0], [0], color=color, marker="o", markersize=3.0, markeredgewidth=0.0, lw=0.9,
                              label=f"{_label(model)}: {fit['sigma300_mS_cm']:.3g} "
                                    f"[{fit['sigma300_min_mS_cm']:.3g}, {fit['sigma300_max_mS_cm']:.3g}] "
                                    f"mS cm$^{{-1}}$, $E_a$ {fit['Ea_eV']:.3f} eV"))
    ax.plot([x300, x300], [low, high], color=style.LIGHT_GREY, lw=4.0, solid_capstyle="butt", zorder=0.5)
    ax.plot(x300, measured, marker="D", color=style.EXPERIMENT_COLOR, markersize=3.6, markeredgewidth=0.0,
            linestyle="none", zorder=6)
    rows.append({"kind": "experiment_wang2018", "series": "", "potential": "", "temperature_K": T_REF_K,
                 "inv_temperature_1000_per_K": x300, "sigma_mS_cm": measured, "err_mS_cm": np.nan,
                 "band_low_mS_cm": np.nan, "band_high_mS_cm": np.nan, "source": "literature"})
    rows.append({"kind": "experiment_compiled_range", "series": "", "potential": "", "temperature_K": T_REF_K,
                 "inv_temperature_1000_per_K": x300, "sigma_mS_cm": np.nan, "err_mS_cm": np.nan,
                 "band_low_mS_cm": low, "band_high_mS_cm": high, "source": "literature"})
    handles.append(Line2D([0], [0], marker="D", color=style.EXPERIMENT_COLOR, markersize=3.2, markeredgewidth=0.0,
                          linestyle="none", label=f"Experiment (Wang et al. 2018), {measured:g} mS cm$^{{-1}}$"))
    handles.append(Line2D([0], [0], color=style.LIGHT_GREY, lw=4.0, solid_capstyle="butt",
                          label=f"Compiled experimental range (Li et al. 2025), {low:g}-{high:g}"))
    ax.set_yscale("log")
    ax.set_ylim(0.04, 2.0e4)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1000", "10000"])
    ax.set_xlim(0.9, 3.5)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel(r"$\sigma$ (mS cm$^{-1}$)")
    top = ax.secondary_xaxis("top", functions=(_inverse, _inverse))
    top.set_xticks(list(T_TOP_TICKS))
    top.set_xticklabels([str(t) for t in T_TOP_TICKS])
    top.tick_params(axis="x", labelsize=style.SMALL, pad=1.0)
    top.set_xlabel("T (K)", labelpad=2.0)
    ax.legend(handles=handles, loc="upper right", fontsize=style.SMALL, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.2)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _inverse(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 1000.0 / np.where(x > 0, x, 1.0), np.inf)


def _panel_b(axes, msd: pd.DataFrame, fits: pd.DataFrame) -> Path:
    rows = []
    for index, (ax, series) in enumerate(zip(axes, PRODUCTION)):
        model = fits.loc[series, "model"]
        run_lags = []
        for temperature in TEMPERATURES:
            run = f"{series}_{temperature}K"
            curve = msd[msd["run_id"] == run].sort_values("lag_ps")
            ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.TEMPERATURE_COLORS[temperature], lw=0.9,
                    label=f"{temperature} K", zorder=3)
            run_lags.append(curve["lag_ps"].max())
            rows.extend({"series": series, "run_id": run, "temperature_K": temperature, "lag_ps": lag,
                         "msd_Li_A2": value} for lag, value in zip(curve["lag_ps"], curve["msd_Li_A2"]))
        start = KINISI_START_FRACTION * max(run_lags)
        end = max(run_lags)
        ax.axvspan(start, end, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
        ax.text(np.sqrt(start * end), 0.97, "fit\nwindow", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=style.SMALL, color="#555555", linespacing=1.0)
        guide_t = np.array([0.4, 12.0])
        guide = 40.0 * guide_t
        ax.plot(guide_t, guide, color="black", lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.text(1.5, 40.0 * 1.5 * 1.6, "slope 1", fontsize=style.SMALL, ha="right", va="bottom")
        rows.extend({"series": series, "run_id": "slope-1 guide", "temperature_K": np.nan, "lag_ps": t,
                     "msd_Li_A2": v} for t, v in zip(guide_t, guide))
        rows.append({"series": series, "run_id": "kinisi fit window start", "temperature_K": np.nan, "lag_ps": start,
                     "msd_Li_A2": np.nan})
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.05, 250)
        ax.set_ylim(0.2, 2000)
        style.log_ticks(ax, "x")
        style.log_ticks(ax, "y")
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.text(0.04, 0.97, _label(model), transform=ax.transAxes, ha="left", va="top", fontsize=style.FONT_SIZE,
                color=style.MODEL_COLORS[model], fontweight="bold")
        ax.set_xlabel("Lag time (ps)")
        if index == 0:
            ax.set_ylabel("Li MSD (Å$^2$)")
            ax.legend(loc="lower right", fontsize=style.SMALL, handlelength=1.2, handletextpad=0.4,
                      labelspacing=0.3, borderaxespad=0.3)
        else:
            ax.tick_params(labelleft=False)
    return style.save_source(pd.DataFrame(rows), NAME, "b")


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
        ax.plot(row.x_plot_ps, y, marker=KIND_MARKERS[row.kind], markersize=3.6 if row.kind != "cell_52" else 3.4,
                markeredgewidth=0.7, markeredgecolor=color,
                markerfacecolor=color if KIND_FILLED[row.kind] else "white", linestyle="none", zorder=4)


def _time_axis(ax) -> None:
    ax.set_xlim(35, 220)
    ax.set_xticks([50, 100, 150, 200])
    ax.set_xlabel("Simulated time per run (ps)")


def _panel_c(ax, frame: pd.DataFrame, experiment: tuple[float, float, float]) -> Path:
    measured, low, high = experiment
    ax.axhspan(low, high, color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.axhline(measured, color=style.EXPERIMENT_COLOR, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(125, measured * 1.12, f"experiment, {measured:g}", ha="center", va="bottom", fontsize=style.SMALL)
    ax.text(38, low * 1.25, "compiled experimental range", ha="left", va="bottom", fontsize=style.SMALL,
            color="#555555")
    _draw_runs(ax, frame, "sigma300_mS_cm", frame["sigma300_min_mS_cm"].to_numpy(),
               frame["sigma300_max_mS_cm"].to_numpy())
    ax.set_yscale("log")
    ax.set_ylim(0.04, 100)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_ylabel(r"$\sigma$(300 K) (mS cm$^{-1}$)")
    _time_axis(ax)
    source = frame[["series", "potential", "kind", "n_atoms", "simulated_ps", "x_plot_ps", "sigma300_mS_cm",
                    "sigma300_min_mS_cm", "sigma300_max_mS_cm"]].copy()
    extra = pd.DataFrame([{"series": "experiment_wang2018", "kind": "reference", "sigma300_mS_cm": measured},
                          {"series": "experiment_compiled_range", "kind": "reference", "sigma300_min_mS_cm": low,
                           "sigma300_max_mS_cm": high}])
    return style.save_source(pd.concat([source, extra], ignore_index=True), NAME, "c")


def _panel_d(ax, frame: pd.DataFrame) -> Path:
    ea, err = frame["Ea_eV"].to_numpy(), frame["Ea_err_eV"].to_numpy()
    _draw_runs(ax, frame, "Ea_eV", ea - err, ea + err)
    ax.set_ylim(0.15, 0.40)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_ylabel("$E_a$ (eV)")
    _time_axis(ax)
    source = frame[["series", "potential", "kind", "n_atoms", "simulated_ps", "x_plot_ps", "Ea_eV",
                    "Ea_err_eV"]]
    return style.save_source(source, NAME, "d")


def _legend_cd(ax) -> None:
    """Marker shape = run type (first row), colour = potential (second row)."""
    kinds = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.4, markeredgewidth=0.7,
                    markeredgecolor="black", markerfacecolor="white",
                    label="52 atoms, independent runs of 50, 100 and 150 ps"),
             Line2D([0], [0], marker="s", color="black", lw=0.6, markersize=3.6, markeredgewidth=0.0,
                    label="416-atom production run, first 50, 100, 150 ps and full 200 ps"),
             Line2D([0], [0], marker="^", linestyle="none", markersize=3.6, markeredgewidth=0.0, color="black",
                    label="second independent 416-atom run (x = 0), 150 ps")]
    colours = [Patch(facecolor=style.MODEL_COLORS[m], edgecolor="none", label=_label(m)) for m in MODELS]
    handles = [kinds[0], colours[0], kinds[1], colours[1], kinds[2]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=3, fontsize=style.SMALL,
              handlelength=1.4, handletextpad=0.4, columnspacing=1.6, labelspacing=0.4, borderaxespad=0.0)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    fits, convergence, transport, msd, literature = _load()
    experiment = _experiment(literature)
    frame = _convergence_rows(convergence)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    # row 1: a (Arrhenius), b (two MSD panels)
    ax_a = L.axes(13.0, 12.0, 74.0, 50.0)
    sources = [_panel_a(ax_a, fits, transport, experiment)]
    ax_b = [L.axes(104.0, 12.0, 37.0, 50.0), L.axes(144.0, 12.0, 37.0, 50.0)]
    sources.append(_panel_b(ax_b, msd, fits))

    # row 2: c, d with a shared legend above
    top2, height2 = 86.0, 37.0
    ax_c = L.axes(13.0, top2, 74.0, height2)
    sources.append(_panel_c(ax_c, frame, experiment))
    ax_d = L.axes(107.0, top2, 74.0, height2)
    sources.append(_panel_d(ax_d, frame))
    _legend_cd(ax_c)

    L.letter(ax_a, "a", 3.0, 6.0)
    L.letter(ax_b[0], "b", 95.0, 6.0)
    L.letter(ax_c, "c", 3.0, 76.0)
    L.letter(ax_d, "d", 97.0, 76.0)
    return style.save(fig, "fig2_universal", "main") + sources


def _caption() -> str:
    fits, convergence, transport, msd, literature = _load()
    measured, low, high = _experiment(literature)
    frame = _convergence_rows(convergence)

    def fit_text(series: str) -> str:
        fit = fits.loc[series]
        return (f"{_label(fit['model'])} {fit['sigma300_mS_cm']:.3g} [{fit['sigma300_min_mS_cm']:.3g}, "
                f"{fit['sigma300_max_mS_cm']:.3g}] mS/cm with Ea = {fit['Ea_eV']:.3f} +/- {fit['Ea_err_eV']:.3f} eV "
                f"({fit['source']} value; {fit['ratio_to_experiment']:.2f} times the measured value)")

    slopes = transport.set_index("run_id")["msd_loglog_slope_fit_window"]
    slope_text = "; ".join(
        f"{_label(fits.loc[s, 'model'])} " + ", ".join(f"{slopes[f'{s}_{t}K']:.2f}" for t in TEMPERATURES)
        for s in PRODUCTION)

    def by_kind(potential: str, kind: str) -> pd.DataFrame:
        return frame[(frame["potential"] == potential) & (frame["kind"] == kind)].sort_values("simulated_ps")

    def values(rows: pd.DataFrame, column: str, digits: str) -> str:
        return ", ".join(f"{v:{digits}}" for v in rows[column])

    mace_w = pd.concat([by_kind("MACE-MP-0", "production_window"), by_kind("MACE-MP-0", "production_full")])
    ms_w = pd.concat([by_kind("MatterSim", "production_window"), by_kind("MatterSim", "production_full")])
    mace_52, ms_52 = by_kind("MACE-MP-0", "cell_52"), by_kind("MatterSim", "cell_52")
    independent = by_kind("MACE-MP-0", "independent_run").iloc[0]
    start = KINISI_START_FRACTION * SERIES_BY_ID["mace_416_200"]["production_ps"]
    return (
        "Fig. 2 | Universal potentials against experiment. "
        "**a**, Arrhenius plot of the 416-atom, 200-ps production runs at 600, 800 and 1000 K: conductivity of each run "
        "from the pymatgen diffusion coefficient (markers; error bars, kinisi 1-sigma uncertainty, which also weights "
        "the fit), the quoted weighted fit of ln(sigma T) against 1/T (solid over the simulated range, dashed where "
        "extrapolated to 300 K) and its 1-sigma band from the fit covariance; sigma(300 K) = "
        f"{fit_text('mace_416_200')} and {fit_text('mattersim_416_200')}. Black diamond, the measured "
        f"{measured:g} mS/cm of sintered Li6PS5Cl (Wang et al. 2018); grey bar, the range {low:g}-{high:g} mS/cm "
        "compiled by Li et al. (2025). "
        "**b**, Li mean-squared displacement (all time origins, cell centre of mass removed) of the same six runs against "
        f"lag time; grey, the kinisi fit window (lag of at least {start:g} ps, {KINISI_START_FRACTION:.0%} of the "
        f"run); dashed line, slope 1. Log-log slopes in the fit window at 600, 800 and 1000 K: {slope_text}. "
        "**c**, sigma(300 K) of re-analysis fits against simulated time per run, with 1-sigma intervals: 52-atom cells "
        "(open circles; three independent trajectories per potential, "
        f"MACE-MP-0 {values(mace_52, 'sigma300_mS_cm', '.3g')} and MatterSim {values(ms_52, 'sigma300_mS_cm', '.3g')} "
        "mS/cm at 50, 100 and 150 ps), the 416-atom production run truncated to its first 50, 100 and 150 ps and in "
        f"full (squares joined by a line; MACE-MP-0 {values(mace_w, 'sigma300_mS_cm', '.3g')}, MatterSim "
        f"{values(ms_w, 'sigma300_mS_cm', '.3g')} mS/cm), and a second independent 416-atom MACE-MP-0 run "
        f"(triangle; the x = 0 run of the Cl-excess series, {independent['simulated_ps']:g} ps, "
        f"{independent['sigma300_mS_cm']:.3g} mS/cm). Dashed line, the measured value; grey band, the compiled "
        "range. Markers at the same simulated time are offset horizontally for legibility. "
        "**d**, Activation energy of the same fits with 1-sigma errors (MACE-MP-0 production windows "
        f"{values(mace_w, 'Ea_eV', '.3f')} eV, MatterSim {values(ms_w, 'Ea_eV', '.3f')} eV). "
        "The truncated windows are nested parts of one trajectory and not independent of the full run. "
        "Logarithmic axes in a, b and c. "
        + common.sources([FITS, CONVERGENCE, TRANSPORT, MSD, LITERATURE])
    )


CAPTION = _caption()
