"""Fig. 4: what fine-tuning changes in Li transport, against AIMD of the same ordered cell.

a, Li diffusion coefficient (kinisi) of the MACE-MP-0 production run, the two fine-tuned MD runs and
AIMD of the same ordered cell (Deng et al. 2017) against 1000/T; b, kinisi D of each run divided by
that of AIMD at 600, 800 and 1000 K, and below it the pre-registered ratio of each fine-tuned run to
the foundation; c, sigma(300 K) and Ea of the three runs and the reading fixed in the analysis plan;
d, Li mean-squared displacement at 600 K and at 1000 K, with AIMD; e, Li force softening slope against
the 2x2x2 labels against the D ratio to AIMD at 1000 K (n = 3, descriptive); f, mean pressure of each
run in the fixed cell. The reference is AIMD at the simulated temperatures (amendment A3); the
fine-tuned models are compared with DFT-level dynamics and with the foundation model only.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import constants as C
from .. import paths
from . import common, style

NAME = "fig4"
WIDTH, HEIGHT = 183.0, 130.0
COMPARISON = paths.RESULTS / "transport_comparison.csv"
RATIOS = paths.RESULTS / "transport_ratios.csv"
AIMD_COMPARISON = paths.RESULTS / "aimd_comparison.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
TIMESERIES = paths.MD / "timeseries.csv.gz"
AIMD_TRANSPORT = paths.REFERENCES / "aimd_deng2017" / "transport.csv"
AIMD_MSD = paths.REFERENCES / "aimd_deng2017" / "msd.csv.gz"
MODELS = ("foundation", "ftG", "ftK")
FINE_TUNED = ("ftG", "ftK")
ROW_LABELS = {"foundation": "MACE-MP-0", "ftG": "Gamma", "ftK": "2x2x2"}
SUBSCRIPTS = {"foundation": "MP0", "ftG": "Gamma", "ftK": "2x2x2"}
CAPTION_NAMES = {"foundation": "MACE-MP-0", "ftG": "Gamma-label run", "ftK": "2x2x2-label run"}
AIMD_COLOR = style.OKABE_ITO["black"]
AIMD_MARKER = "s"
AIMD_LABEL = "AIMD, same ordered cell (Deng et al. 2017)"
CAGED_SLOPE = 0.25  # AIMD runs whose Li MSD log-log slope in the fit window is below this are caged
SOFTENING_SUBSET = "All27"
MSD_TEMPERATURES = (600, 1000)
RATIO_TEMPERATURE = 1000  # temperature of the D ratio in panel e
OFFSETS = {"foundation": -0.22, "ftG": 0.0, "ftK": 0.22}  # horizontal offsets in panel b (category units)
A_OFFSETS = {"foundation": 0.0, "ftG": -0.025, "ftK": 0.025, "aimd": 0.0}  # in 1000/T units, panel a


def _load() -> dict:
    return {"comparison": common.read(COMPARISON), "ratios": common.read(RATIOS),
            "aimd_ratios": common.read(AIMD_COMPARISON), "cross": common.read(CROSS),
            "thermostat": common.read(THERMOSTAT), "transport": common.read(TRANSPORT), "msd": common.read(MSD),
            "blocks": common.read(TIMESERIES).groupby("run_id").size().rename("n_blocks"),
            "aimd": common.read(AIMD_TRANSPORT).sort_values("temperature_K").reset_index(drop=True),
            "aimd_msd": common.read(AIMD_MSD)}


def _fits(comparison: pd.DataFrame) -> pd.DataFrame:
    """One fit row per model (foundation, ftG, ftK), indexed by model."""
    return comparison[comparison["model"].isin(MODELS)].set_index("model").loc[list(MODELS)]


def _reading(comparison: pd.DataFrame) -> pd.Series:
    rows = comparison[comparison["model"] == "reading"]
    if len(rows) != 1:
        raise ValueError("expected one reading row in transport_comparison.csv")
    return rows.iloc[0]


def _md_models(cross: pd.DataFrame, fits: pd.DataFrame) -> dict[str, str]:
    """model (foundation, ftG, ftK) -> id of the model that drove its MD run, from ``drove_md``."""
    drove = cross[cross["drove_md"].fillna("").astype(str) != ""].groupby("drove_md")["model"].unique()
    out = {}
    for model in MODELS:
        ids = drove[fits.loc[model, "series"]]
        if len(ids) != 1:
            raise ValueError(f"expected one model for the run of {model}")
        out[model] = str(ids[0])
    return out


def _stage_word(model_id: str) -> str:
    return {"s1": "stage-one", "s2": "stage-two"}[model_id.split("_")[1]]


def _sig(value: float, digits: int = 3) -> str:
    """``value`` to ``digits`` significant figures, keeping trailing zeros (0.260, not 0.26)."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    decimals = max(digits - 1 - math.floor(math.log10(abs(value))), 0)
    return f"{value:.{decimals}f}"


def _sci(value: float) -> str:
    """Three significant figures in a x 10^n form, ASCII (for the caption)."""
    exponent = math.floor(math.log10(abs(value)))
    return f"{_sig(value / 10 ** exponent)}e{exponent}"


def _softening(cross: pd.DataFrame, model_id: str, subset: str) -> float:
    rows = cross[(cross["model"] == model_id) & (cross["labels"] == "k222") & (cross["subset"] == subset)
                 & (cross["element"] == "Li")]
    if len(rows) != 1:
        raise ValueError(f"expected one Li row for {model_id} on {subset}")
    return float(rows["softening_slope"].iloc[0])


def _aimd_ratio(aimd_ratios: pd.DataFrame, series: str, temperature: int) -> pd.Series:
    rows = aimd_ratios[(aimd_ratios["series"] == series) & (aimd_ratios["temperature_K"] == temperature)]
    if len(rows) != 1:
        raise ValueError(f"expected one AIMD comparison row for {series} at {temperature} K")
    return rows.iloc[0]


def _caged(aimd: pd.DataFrame) -> set[int]:
    """AIMD temperatures at which the Li MSD is almost flat in the fit window."""
    return set(aimd.loc[aimd["msd_loglog_slope_fit_window"] < CAGED_SLOPE, "temperature_K"].astype(int))


def _inverse(values):
    values = np.asarray(values, float)
    return np.divide(1000.0, values, out=np.full_like(values, np.inf), where=values != 0)


def _handle(model: str, md_model: str) -> Line2D:
    return Line2D([0], [0], label=style.MODEL_LABELS[model], **common.marker_kwargs(md_model, "o", 3.6))


def _aimd_kwargs(caged: bool, size: float = 3.4) -> dict:
    return {"marker": AIMD_MARKER, "markersize": size, "markeredgewidth": 0.7, "markeredgecolor": AIMD_COLOR,
            "markerfacecolor": "white" if caged else AIMD_COLOR, "linestyle": "none"}


def _window_start(production_ps: float) -> float:
    return C.KINISI_START_FRACTION * production_ps


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    rows = []
    for model in MODELS:
        color = style.MODEL_COLORS[model]
        runs = d["transport"][d["transport"]["series"] == fits.loc[model, "series"]].sort_values("temperature_K")
        temps = runs["temperature_K"].to_numpy(float)
        x = 1000.0 / temps + A_OFFSETS[model]
        y, err = runs["kinisi_D_cm2_s"].to_numpy(), runs["kinisi_D_std_cm2_s"].to_numpy()
        ax.plot(x, y, color=color, lw=0.7, zorder=2)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, zorder=3)
        ax.plot(x, y, **common.marker_kwargs(md[model], "o", 3.4), zorder=4)
        rows.extend({"source": "MLIP MD", "model": model, "series": fits.loc[model, "series"], "temperature_K": t,
                     "x_inv_T_1000_per_K": xi, "kinisi_D_cm2_s": yi, "kinisi_D_std_cm2_s": ei,
                     "pymatgen_D_cm2_s": p, "msd_loglog_slope": s, "caged": False}
                    for t, xi, yi, ei, p, s in zip(temps, x, y, err, runs["D_cm2_s"],
                                                   runs["msd_loglog_slope_fit_window"]))
    aimd, caged = d["aimd"], _caged(d["aimd"])
    temps = aimd["temperature_K"].to_numpy(float)
    x = 1000.0 / temps + A_OFFSETS["aimd"]
    y, err = aimd["kinisi_D_cm2_s"].to_numpy(), aimd["kinisi_D_std_cm2_s"].to_numpy()
    ax.plot(x, y, color=AIMD_COLOR, lw=0.7, zorder=2)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=AIMD_COLOR, elinewidth=0.6, capsize=1.3, zorder=3)
    for xi, yi, t in zip(x, y, temps):
        ax.plot(xi, yi, **_aimd_kwargs(int(t) in caged), zorder=5)
        if int(t) in caged:
            ax.annotate("AIMD not\ndiffusive", (xi, yi), xytext=(0, -9), textcoords="offset points", ha="center",
                        va="top", fontsize=style.SMALL, linespacing=1.05)
    rows.extend({"source": "AIMD", "model": "AIMD (Deng et al. 2017)", "series": "aimd_deng2017", "temperature_K": t,
                 "x_inv_T_1000_per_K": xi, "kinisi_D_cm2_s": yi, "kinisi_D_std_cm2_s": ei, "pymatgen_D_cm2_s": p,
                 "msd_loglog_slope": s, "caged": int(t) in caged}
                for t, xi, yi, ei, p, s in zip(temps, x, y, err, aimd["D_cm2_s"], aimd["msd_loglog_slope_fit_window"]))
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 2e-4)
    style.log_ticks(ax, "y")
    ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4], ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$", "$10^{-4}$"])
    ax.set_xlim(0.76, 1.76)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("Li diffusion coefficient (cm$^2$ s$^{-1}$)")
    top = ax.secondary_xaxis("top", functions=(_inverse, _inverse))
    top.set_xticks([1200, 1000, 800, 600], ["1200", "1000", "800", "600"])
    top.tick_params(axis="x", labelsize=style.SMALL, pad=1.0, length=2.0)
    top.set_xlabel("T (K)", fontsize=style.SMALL, labelpad=1.5)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _ratio_marks(ax, x: float, kinisi: float, err: float, pymatgen: float, model: str, md_model: str) -> None:
    color = style.MODEL_COLORS[model]
    ax.errorbar(x, kinisi, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, zorder=2)
    ax.plot(x, kinisi, **common.marker_kwargs(md_model, "o", 3.4), zorder=3)
    ax.plot(x, pymatgen, marker="_", markersize=6.5, markeredgewidth=1.1, color=color, linestyle="none", zorder=4)


def _panel_b(ax, ax_low, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    temps = list(C.TEMPERATURES)
    caged = _caged(d["aimd"])
    rows = []
    for index, temperature in enumerate(temps):
        if temperature in caged:
            ax.axvspan(index - 0.5, index + 0.5, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
            ax.text(index, 1.02, "AIMD sub-\ndiffusive", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=style.SMALL, color="#555555", linespacing=1.05)
        for model in MODELS:
            row = _aimd_ratio(d["aimd_ratios"], fits.loc[model, "series"], temperature)
            kinisi = row["kinisi_D_ratio_to_aimd"]
            err = kinisi * row["kinisi_D_ratio_rel_err"]
            _ratio_marks(ax, index + OFFSETS[model], kinisi, err, row["pymatgen_D_ratio_to_aimd"], model, md[model])
            rows.append({"reference": "AIMD", "model": model, "temperature_K": temperature,
                         "x": index + OFFSETS[model], "kinisi_D_ratio": kinisi, "kinisi_D_ratio_err": err,
                         "pymatgen_D_ratio": row["pymatgen_D_ratio_to_aimd"], "msd_loglog_slope": row["msd_loglog_slope"],
                         "reference_msd_loglog_slope": row["aimd_msd_loglog_slope"]})
        for model in FINE_TUNED:
            row = d["ratios"][(d["ratios"]["model"] == model) & (d["ratios"]["temperature_K"] == temperature)].iloc[0]
            kinisi = row["kinisi_D_ratio"]
            err = kinisi * row["kinisi_D_ratio_rel_err"]
            _ratio_marks(ax_low, index + OFFSETS[model], kinisi, err, row["pymatgen_D_ratio"], model, md[model])
            rows.append({"reference": "MACE-MP-0", "model": model, "temperature_K": temperature,
                         "x": index + OFFSETS[model], "kinisi_D_ratio": kinisi, "kinisi_D_ratio_err": err,
                         "pymatgen_D_ratio": row["pymatgen_D_ratio"]})
    ax.axhline(1.0, color=AIMD_COLOR, lw=0.6, zorder=1)
    ax.text(2.47, 1.06, "AIMD", ha="right", va="bottom", fontsize=style.SMALL, color=AIMD_COLOR)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 20)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_ylabel("Li D / D(AIMD)")
    ax_low.axhline(1.0, color=style.MODEL_COLORS["foundation"], lw=0.6, zorder=1)
    ax_low.text(2.47, 1.0, "MACE-MP-0", ha="right", va="bottom", fontsize=style.SMALL,
                color=style.MODEL_COLORS["foundation"])
    ax_low.set_ylim(0, 1.3)
    ax_low.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax_low.set_ylabel("D / D(MP-0)", labelpad=6.5)
    for axis in (ax, ax_low):
        axis.set_xlim(-0.5, len(temps) - 0.5)
        axis.set_xticks(range(len(temps)), [str(t) for t in temps])
    ax.tick_params(axis="x", labelbottom=False)
    ax_low.set_xlabel("Temperature (K)")
    handles = [Line2D([0], [0], marker="o", linestyle="-", color="black", markersize=3.0, lw=0.6,
                      label="kinisi, $\\pm$1 s.d."),
               Line2D([0], [0], marker="_", linestyle="none", color="black", markersize=6.5, markeredgewidth=1.1,
                      label="pymatgen")]
    ax.legend(handles=handles, loc="upper right", fontsize=style.SMALL, borderaxespad=0.2, labelspacing=0.3,
              handlelength=1.2)
    return style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(ax_s, ax_e, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    """Forest plot: one row per model, sigma(300 K) on the left axes and Ea on the right."""
    rows = []
    for index, model in enumerate(MODELS):
        fit, color = fits.loc[model], style.MODEL_COLORS[model]
        s300, lo, hi = fit["sigma300_mS_cm"], fit["sigma300_min_mS_cm"], fit["sigma300_max_mS_cm"]
        ax_s.errorbar(s300, index, xerr=[[s300 - lo], [hi - s300]], fmt="none", ecolor=color, elinewidth=0.7,
                      capsize=1.5, zorder=2)
        ax_s.plot(s300, index, **common.marker_kwargs(md[model], "o", 3.6), zorder=3)
        ax_e.errorbar(fit["Ea_eV"], index, xerr=fit["Ea_err_eV"], fmt="none", ecolor=color, elinewidth=0.7,
                      capsize=1.5, zorder=2)
        ax_e.plot(fit["Ea_eV"], index, **common.marker_kwargs(md[model], "o", 3.6), zorder=3)
        rows.append({"model": model, "series": fit["series"], "md_model": md[model], "sigma300_mS_cm": s300,
                     "sigma300_min_mS_cm": lo, "sigma300_max_mS_cm": hi, "Ea_eV": fit["Ea_eV"],
                     "Ea_err_eV": fit["Ea_err_eV"]})
    for axis in (ax_s, ax_e):
        axis.set_ylim(len(MODELS) - 0.4, -0.6)
        axis.tick_params(axis="y", length=0, pad=2.0)
    ax_s.set_yticks(range(len(MODELS)), [ROW_LABELS[m] for m in MODELS])
    for label, model in zip(ax_s.get_yticklabels(), MODELS):
        label.set_color(style.MODEL_COLORS[model])
    ax_e.set_yticks(range(len(MODELS)), [])
    ax_s.set_xscale("log")
    ax_s.set_xlim(0.1, 20)
    style.log_ticks(ax_s, "x")
    ax_s.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax_s.set_xlabel("$\\sigma$(300 K) (mS cm$^{-1}$)")
    ax_e.set_xlim(0.2, 0.4)
    ax_e.set_xticks([0.2, 0.3, 0.4])
    ax_e.set_xticks([0.25, 0.35], minor=True)
    ax_e.set_xlabel("$E_a$ (eV)")
    return style.save_source(pd.DataFrame(rows), NAME, "c")


def _reading_text(reading: pd.Series, md: dict[str, str]) -> str:
    g, k, f0 = SUBSCRIPTS["ftG"], SUBSCRIPTS["ftK"], SUBSCRIPTS["foundation"]
    return (f"$f$ = ln($\\sigma_{{{k}}}$/$\\sigma_{{{g}}}$) / ln($\\sigma_{{{f0}}}$/$\\sigma_{{{g}}}$)"
            f" = {_sig(reading['f_recovered_fraction'])}\n"
            f"ln($\\sigma_{{{k}}}$/$\\sigma_{{{g}}}$) = {_sig(reading['ln_sigma_ftK_over_ftG'])}"
            f" $\\pm$ {_sig(reading['ln_sigma_ftK_over_ftG_err'])}\n"
            f"Pre-registered reading: {reading['reading']}\n"
            f"MD models: Gamma {_stage_word(md['ftG'])}, 2x2x2 {_stage_word(md['ftK'])}")


def _msd_axis(ax, d: dict, fits: pd.DataFrame, temperature: int, slopes_text: bool) -> list[pd.DataFrame]:
    production = float(C.SERIES_BY_ID[fits.loc["foundation", "series"]]["production_ps"])
    aimd_length = float(d["aimd"]["length_ps"].iloc[0])
    start, aimd_start = _window_start(production), _window_start(aimd_length)
    ax.axvspan(start, 250.0, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
    ax.plot([aimd_start, aimd_length], [0.99, 0.99], color=AIMD_COLOR, lw=1.4, solid_capstyle="butt",
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
    frames, slopes = [], []
    for model in MODELS:
        run_id = C.run_id(fits.loc[model, "series"], temperature)
        curve = d["msd"][d["msd"]["run_id"] == run_id].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.MODEL_COLORS[model], lw=0.9, zorder=2)
        frames.append(curve[["run_id", "lag_ps", "msd_Li_A2"]].assign(
            model=model, temperature_K=temperature, fit_window_start_ps=start))
        slope = d["transport"].loc[d["transport"]["run_id"] == run_id, "msd_loglog_slope_fit_window"].iloc[0]
        slopes.append((ROW_LABELS[model], style.MODEL_COLORS[model], slope))
    curve = d["aimd_msd"][d["aimd_msd"]["temperature_K"] == temperature].sort_values("lag_ps")
    ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=AIMD_COLOR, lw=0.9, zorder=3)
    frames.append(curve[["lag_ps", "msd_Li_A2"]].assign(run_id=f"aimd_deng2017_{temperature}K", model="AIMD",
                                                        temperature_K=temperature, fit_window_start_ps=aimd_start))
    aimd_slope = d["aimd"].loc[d["aimd"]["temperature_K"] == temperature, "msd_loglog_slope_fit_window"].iloc[0]
    slopes.append(("AIMD", AIMD_COLOR, aimd_slope))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 200.0)
    ax.set_ylim(0.05, 2000.0)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax.text(0.95, 0.04, f"{temperature} K", transform=ax.transAxes, ha="right", va="bottom", fontsize=style.FONT_SIZE,
            fontweight="bold")
    if slopes_text:
        ax.text(0.04, 0.95, "Log-log slope in\nthe fit window", transform=ax.transAxes, ha="left", va="top",
                fontsize=style.SMALL, linespacing=1.1)
        for index, (label, color, slope) in enumerate(slopes):
            ax.text(0.04, 0.80 - index * 0.075, f"{label}  {_sig(slope)}", transform=ax.transAxes, ha="left",
                    va="top", fontsize=style.SMALL, color=color)
    out = pd.concat(frames, ignore_index=True)
    out["loglog_slope_fit_window"] = out["model"].map({m: s for m, (_, _, s) in zip([*MODELS, "AIMD"], slopes)})
    return [out]


def _panel_d(ax_low, ax_high, d: dict, fits: pd.DataFrame) -> Path:
    frames = _msd_axis(ax_low, d, fits, MSD_TEMPERATURES[0], slopes_text=True)
    frames += _msd_axis(ax_high, d, fits, MSD_TEMPERATURES[1], slopes_text=False)
    ax_low.set_ylabel("Li MSD (Å$^2$)")
    ax_high.tick_params(axis="y", labelleft=False)
    handles = [Patch(facecolor=style.LIGHT_GREY, edgecolor="none", label="MLIP fit\nwindow"),
               Line2D([0], [0], color=AIMD_COLOR, lw=1.4, solid_capstyle="butt", label="AIMD fit\nwindow")]
    ax_high.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.4,
                   labelspacing=0.5, borderaxespad=0.3)
    source = pd.concat(frames, ignore_index=True)[["model", "run_id", "temperature_K", "lag_ps", "msd_Li_A2",
                                                   "fit_window_start_ps", "loglog_slope_fit_window"]]
    return style.save_source(source, NAME, "d")


def _panel_e(ax, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    rows = []
    ax.axhline(1.0, color=AIMD_COLOR, lw=0.6, zorder=1)
    ax.text(0.995, 1.04, "AIMD", ha="right", va="bottom", fontsize=style.SMALL, color=AIMD_COLOR)
    for model in MODELS:
        slope = _softening(d["cross"], md[model], SOFTENING_SUBSET)
        row = _aimd_ratio(d["aimd_ratios"], fits.loc[model, "series"], RATIO_TEMPERATURE)
        ratio = row["kinisi_D_ratio_to_aimd"]
        err = ratio * row["kinisi_D_ratio_rel_err"]
        color = style.MODEL_COLORS[model]
        ax.errorbar(slope, ratio, yerr=err, fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=2)
        ax.plot(slope, ratio, **common.marker_kwargs(md[model], "o", 3.8), zorder=3)
        rows.append({"model": model, "md_model": md[model], "series": fits.loc[model, "series"], "labels": "k222",
                     "element": "Li", "subset": SOFTENING_SUBSET, "softening_slope": slope,
                     "temperature_K": RATIO_TEMPERATURE, "kinisi_D_ratio_to_aimd": ratio,
                     "kinisi_D_ratio_err": err})
    ax.set_xlim(0.83, 1.01)
    ax.set_xticks([0.85, 0.9, 0.95, 1.0])
    ax.set_ylim(0, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel(f"Li softening slope vs 2x2x2 ({SOFTENING_SUBSET})")
    ax.set_ylabel(f"Li D / D(AIMD) at {RATIO_TEMPERATURE} K")
    ax.text(0.96, 0.97, "n = 3, descriptive", transform=ax.transAxes, ha="right", va="top",
            fontsize=style.SMALL, color="#555555")
    return style.save_source(pd.DataFrame(rows), NAME, "e")


def _panel_f(ax, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    ax.axhline(0.0, color=style.GREY, lw=0.5, zorder=1)
    thermostat, blocks = d["thermostat"], d["blocks"]
    rows = []
    for model in MODELS:
        runs = thermostat[thermostat["series"] == fits.loc[model, "series"]].sort_values("temperature_K")
        ax.plot(runs["temperature_K"], runs["mean_P_GPa"], color=style.MODEL_COLORS[model], lw=0.8, zorder=2)
        ax.plot(runs["temperature_K"], runs["mean_P_GPa"], **common.marker_kwargs(md[model], "o", 3.6), zorder=3)
        for run in runs.itertuples():
            n = int(blocks[run.run_id])
            rows.append({"model": model, "run_id": run.run_id, "temperature_K": run.temperature_K,
                         "mean_P_GPa": run.mean_P_GPa, "std_P_GPa": run.std_P_GPa, "n_blocks_0p5ps": n,
                         "std_over_sqrt_blocks_GPa": run.std_P_GPa / math.sqrt(n)})
    ax.set_xticks(list(C.TEMPERATURES))
    ax.set_xlim(540, 1060)
    ax.set_ylim(-1.0, 0.6)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean pressure, fixed cell (GPa)")
    return style.save_source(pd.DataFrame(rows), NAME, "f")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    d = _load()
    fits = _fits(d["comparison"])
    md = _md_models(d["cross"], fits)
    reading = _reading(d["comparison"])
    fig, L = common.new_figure(WIDTH, HEIGHT)

    handles = [_handle(m, md[m]) for m in MODELS]
    handles.append(Line2D([0], [0], label=AIMD_LABEL, **_aimd_kwargs(False, 3.6)))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L.x(10.0), L.y(0.8)), ncol=len(handles),
               fontsize=style.FONT_SIZE, handletextpad=0.3, columnspacing=1.8, borderaxespad=0.0)

    # row 1: a, b (two stacked axes), c
    top1, h1 = 16.0, 38.0
    ax_a = L.axes(13.0, top1, 44.0, h1)
    ax_b = L.axes(74.0, top1, 36.0, 25.0)
    ax_b_low = L.axes(74.0, top1 + 27.0, 36.0, h1 - 27.0)
    ax_cs = L.axes(139.0, top1, 17.0, 19.0)
    ax_ce = L.axes(162.0, top1, 17.0, 19.0)
    sources = [_panel_a(ax_a, d, fits, md), _panel_b(ax_b, ax_b_low, d, fits, md), _panel_c(ax_cs, ax_ce, fits, md)]
    L.text(126.0, top1 + 28.0, _reading_text(reading, md), ha="left", va="top", fontsize=style.SMALL,
           linespacing=1.5)

    # row 2: d (600 K and 1000 K), e, f
    top2, h2 = 77.0, 38.0
    ax_d_low = L.axes(13.0, top2, 31.0, h2)
    ax_d_high = L.axes(47.0, top2, 22.0, h2)
    ax_e = L.axes(86.0, top2, 34.0, h2)
    ax_f = L.axes(140.0, top2, 41.0, h2)
    sources += [_panel_d(ax_d_low, ax_d_high, d, fits), _panel_e(ax_e, d, fits, md), _panel_f(ax_f, d, fits, md)]
    L.text(41.0, top2 + h2 + 7.5, "Lag time (ps)", ha="center", va="top", fontsize=style.FONT_SIZE)

    L.letter(ax_a, "a", 3.0, 9.5)
    L.letter(ax_b, "b", 64.0, 9.5)
    L.letter(ax_cs, "c", 124.0, 9.5)
    L.letter(ax_d_low, "d", 3.0, 71.0)
    L.letter(ax_e, "e", 76.0, 71.0)
    L.letter(ax_f, "f", 128.0, 71.0)
    return style.save(fig, "fig4_transport", "main") + sources


def _caption() -> str:
    d = _load()
    fits = _fits(d["comparison"])
    md = _md_models(d["cross"], fits)
    reading = _reading(d["comparison"])
    name = CAPTION_NAMES
    aimd = d["aimd"].set_index("temperature_K")
    caged = sorted(_caged(d["aimd"]))
    production = C.SERIES_BY_ID[fits.loc["foundation", "series"]]["production_ps"]
    start, aimd_start = _window_start(production), _window_start(float(d["aimd"]["length_ps"].iloc[0]))

    # the two cells
    run = d["transport"][d["transport"]["series"] == fits.loc["foundation", "series"]].iloc[0]
    a_mlip = (run["volume_A3"] * int(aimd["n_atoms"].iloc[0]) / run["n_atoms"]) ** (1.0 / 3.0)
    a_aimd = float(aimd["a_A"].iloc[0])
    smaller = (1.0 - (a_aimd / a_mlip) ** 3) * 100.0
    cell = (f"AIMD: VASP, {int(aimd['n_atoms'].iloc[0])} atoms (Li24P4S20Cl4, Cl on 4a and S on 4c, as here), "
            f"a = {_sig(a_aimd, 4)} A against {_sig(a_mlip, 4)} A for the MLIP runs ({_sig(smaller, 2)} % smaller "
            f"volume), {int(aimd['time_step_fs'].iloc[0])}-fs time step, every {int(aimd['step_skip'].iloc[0])}th "
            f"step stored, {round(float(aimd['length_ps'].iloc[0]))} ps per temperature, one trajectory per "
            f"temperature; exchange-correlation functional and thermostat not recorded in the files. The MLIP runs "
            f"have {int(run['n_atoms'])} atoms and {production} ps. Both are analysed with the same estimators and "
            f"settings")
    aimd_text = "; ".join(f"{t} K, {_sci(r['kinisi_D_cm2_s'])} +/- {_sci(r['kinisi_D_std_cm2_s'])} cm2/s "
                          f"(log-log slope {_sig(r['msd_loglog_slope_fit_window'])})" for t, r in aimd.iterrows())

    def ratio(model, t, column="kinisi_D_ratio_to_aimd"):
        return _aimd_ratio(d["aimd_ratios"], fits.loc[model, "series"], t)[column]

    ratio_text = "; ".join(
        f"{name[m]} " + ", ".join(f"{_sig(ratio(m, t))} +/- {_sig(ratio(m, t) * ratio(m, t, 'kinisi_D_ratio_rel_err'))}"
                                  for t in C.TEMPERATURES) for m in MODELS)
    pym_text = "; ".join(f"{name[m]} " + ", ".join(_sig(ratio(m, t, "pymatgen_D_ratio_to_aimd"))
                                                    for t in C.TEMPERATURES) for m in MODELS)
    hot = [t for t in C.TEMPERATURES if t not in caged]
    fine = [ratio(m, t) for m in FINE_TUNED for t in hot]
    base = [ratio("foundation", t) for t in hot]
    low = d["ratios"]
    pre = "; ".join(f"{name[m]} " + ", ".join(
        f"{_sig(r['kinisi_D_ratio'])} +/- {_sig(r['kinisi_D_ratio'] * r['kinisi_D_ratio_rel_err'])}"
        for _, r in low[low["model"] == m].sort_values("temperature_K").iterrows()) for m in FINE_TUNED)

    def fit_text(model):
        f = fits.loc[model]
        return (f"{name[model]} {_sig(f['sigma300_mS_cm'])} [{_sig(f['sigma300_min_mS_cm'])}, "
                f"{_sig(f['sigma300_max_mS_cm'])}] mS/cm and Ea {_sig(f['Ea_eV'])} +/- {_sig(f['Ea_err_eV'])} eV")

    slopes = d["transport"].set_index("run_id")["msd_loglog_slope_fit_window"]
    slope_text = ", ".join(f"{name[m]} {_sig(slopes[C.run_id(fits.loc[m, 'series'], MSD_TEMPERATURES[0])])}"
                           for m in MODELS)
    soft_text = ", ".join(f"{name[m]} {_sig(_softening(d['cross'], md[m], SOFTENING_SUBSET))}" for m in MODELS)
    pressure = {m: d["thermostat"][d["thermostat"]["series"] == fits.loc[m, "series"]].set_index("temperature_K")
                ["mean_P_GPa"] for m in MODELS}
    shift = [pressure[m][t] - pressure["foundation"][t] for m in FINE_TUNED for t in C.TEMPERATURES]
    if max(shift) >= 0:
        raise ValueError("caption assumes every fine-tuned run has a lower mean pressure than the foundation")
    runs = d["thermostat"][d["thermostat"]["series"].isin(fits["series"])]
    n_blocks = sorted({int(d["blocks"][r]) for r in runs["run_id"]})
    sem = max(row.std_P_GPa / math.sqrt(d["blocks"][row.run_id]) for row in runs.itertuples())
    caged_text = " and ".join(f"{t} K" for t in caged)
    return (
        "Fig. 4 | What fine-tuning changes in Li transport. "
        "MLIP values are re-analysis values of the 416-atom, 200-ps runs at 600, 800 and 1000 K. The transport "
        "reference is AIMD of the same ordered cell at the simulated temperatures (Deng et al., Chem. Mater. 29, "
        f"281 (2017); amendment A3), compared without extrapolation. {cell}. "
        f"The Gamma-label run used the {_stage_word(md['ftG'])} model and the 2x2x2-label run the "
        f"{_stage_word(md['ftK'])} model (identified by force fingerprints, Extended Data Fig. 7a); filled markers, "
        "foundation or stage-two model; open markers, stage-one model, as in Fig. 3. "
        "**a**, Li diffusion coefficient from kinisi (+/- 1 s.d.; the estimator of the AIMD comparison) against "
        "1000/T, points joined by straight lines; the fine-tuned points are offset horizontally for clarity. The "
        "Arrhenius fits of c use the pymatgen D and are therefore not drawn here. AIMD (black squares): "
        f"{aimd_text}. At {caged_text} the AIMD Li MSD is almost flat in the fit window (open square): Li is caged "
        "and this point is not a diffusion coefficient. "
        "**b**, Top, kinisi D of each run divided by that of AIMD at the same temperature (circles; error, the two "
        "kinisi relative standard deviations in quadrature; logarithmic axis) and the pymatgen D ratio (bars). At "
        f"600, 800 and 1000 K: {ratio_text} (pymatgen: {pym_text}). The {caged_text} column (grey) compares with "
        "caged AIMD motion, not with diffusion. At the diffusive temperatures fine-tuning moves D towards AIMD: the "
        f"fine-tuned runs are {_sig(min(fine))}-{_sig(max(fine))} times AIMD, the foundation {_sig(min(base))}-"
        f"{_sig(max(base))} times. Bottom, the pre-registered comparison: kinisi D of each fine-tuned run divided "
        f"by that of the foundation run, at 600, 800 and 1000 K: {pre}. "
        "**c**, sigma(300 K) (logarithmic axis, 1-sigma interval from the fit covariance) and Ea (+/- 1 s.d.) of "
        "the weighted Arrhenius fits of ln(sigma T) against 1/T (pymatgen D, 600-1000 K): "
        f"{'; '.join(fit_text(m) for m in MODELS)}. Reading fixed in the analysis plan: f = "
        "ln(sigma_2x2x2/sigma_Gamma)/ln(sigma_MP0/sigma_Gamma), the fraction of the fine-tuning slowdown recovered "
        f"with the 2x2x2 labels (MP0, the foundation run), is {_sig(reading['f_recovered_fraction'])}, and "
        f"ln(sigma_2x2x2/sigma_Gamma) = {_sig(reading['ln_sigma_ftK_over_ftG'])} +/- "
        f"{_sig(reading['ln_sigma_ftK_over_ftG_err'])}. Since f < 0.25 the reading is '{reading['reading']}': the "
        "k-point sampling of the labels does not explain the slowdown. The two runs used models from different "
        "training stages, so this comparison is confounded with the training stage. "
        f"**d**, Li mean-squared displacement at {MSD_TEMPERATURES[0]} K (left) and {MSD_TEMPERATURES[1]} K "
        "(right) averaged over all time origins, with the AIMD MSD of the 52-atom cell (black); logarithmic axes. "
        f"Grey band, the MLIP fit window (lag >= {start:g} ps); black bar, the AIMD fit window (lag >= "
        f"{_sig(aimd_start)} ps); both start at {C.KINISI_START_FRACTION * 100:g} % of the run length. Log-log slopes "
        f"in the window at {MSD_TEMPERATURES[0]} K: {slope_text}, AIMD "
        f"{_sig(aimd.loc[MSD_TEMPERATURES[0], 'msd_loglog_slope_fit_window'])} (1 for diffusive motion). "
        "**e**, Li force softening slope (least-squares slope through the origin of model against 2x2x2 DFT Li "
        f"force components on all 27 configurations, {SOFTENING_SUBSET}: {soft_text}) against the kinisi D ratio "
        f"to AIMD at {RATIO_TEMPERATURE} K (+/- 1 s.d.). Across these three models, softer Li forces go with a "
        "larger overestimate of D; n = 3, descriptive only, no fit. "
        "**f**, Mean pressure of each MLIP run in the fixed cell (one cell for all three models) against "
        f"temperature; markers are the run means. The fine-tuned runs lie {_sig(-max(shift))} to "
        f"{_sig(-min(shift))} GPa below the foundation run at the same temperature. The standard deviation of the "
        "instantaneous pressure divided by the square root of the number of 0.5-ps blocks "
        f"({n_blocks[0]}) is at most {_sig(sem)} GPa, smaller than the markers. "
        + common.sources([TRANSPORT, AIMD_TRANSPORT, AIMD_COMPARISON, RATIOS, COMPARISON, MSD, AIMD_MSD, CROSS,
                          THERMOSTAT, TIMESERIES])
    )


CAPTION = _caption()
