"""Fig. 4: what fine-tuning changes in Li transport (model against model, re-analysis values only).

a, Arrhenius plot of the MACE-MP-0 production run and the two fine-tuned MD runs, with the weighted
fits extrapolated to 300 K; b, Li diffusion coefficient of each fine-tuned run relative to the
foundation at 600, 800 and 1000 K (pymatgen and kinisi); c, sigma(300 K) and Ea of the three runs and
the reading fixed in the analysis plan; d, Li mean-squared displacement at 600 K with the kinisi fit
window; e, Li force softening slope against the 2x2x2 labels against Ea of the MD run (n = 3,
descriptive); f, mean pressure of each run in the fixed cell. No measured value enters this figure:
the fine-tuned models are compared with the foundation model only.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import constants as C
from .. import paths
from ..transport.arrhenius import sigma_model
from . import common, style

NAME = "fig4"
WIDTH, HEIGHT = 183.0, 130.0
COMPARISON = paths.RESULTS / "transport_comparison.csv"
RATIOS = paths.RESULTS / "transport_ratios.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
TIMESERIES = paths.MD / "timeseries.csv.gz"
MODELS = ("foundation", "ftG", "ftK")
FINE_TUNED = ("ftG", "ftK")
ROW_LABELS = {"foundation": "MACE-MP-0", "ftG": "Gamma", "ftK": "2x2x2"}
SUBSCRIPTS = {"foundation": "MP0", "ftG": "Gamma", "ftK": "2x2x2"}
CAPTION_NAMES = {"foundation": "MACE-MP-0", "ftG": "Gamma-label run", "ftK": "2x2x2-label run"}
SOFTENING_SUBSET = "All27"
MSD_TEMPERATURE = 600
X300 = 1000.0 / C.T_REF_K


def _load():
    comparison = common.read(COMPARISON)
    ratios = common.read(RATIOS)
    cross = common.read(CROSS)
    thermostat = common.read(THERMOSTAT)
    transport = common.read(TRANSPORT)
    msd = common.read(MSD)
    blocks = common.read(TIMESERIES).groupby("run_id").size().rename("n_blocks")
    return comparison, ratios, cross, thermostat, transport, msd, blocks


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


def _softening(cross: pd.DataFrame, model_id: str, subset: str) -> float:
    rows = cross[(cross["model"] == model_id) & (cross["labels"] == "k222") & (cross["subset"] == subset)
                 & (cross["element"] == "Li")]
    if len(rows) != 1:
        raise ValueError(f"expected one Li row for {model_id} on {subset}")
    return float(rows["softening_slope"].iloc[0])


def _inverse(values):
    values = np.asarray(values, float)
    return np.divide(1000.0, values, out=np.full_like(values, np.inf), where=values != 0)


def _handle(model: str, md_model: str) -> Line2D:
    return Line2D([0], [0], label=style.MODEL_LABELS[model], **common.marker_kwargs(md_model, "o", 3.6))


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, fits: pd.DataFrame, transport: pd.DataFrame, md: dict[str, str]) -> Path:
    offsets = {"foundation": 0.0, "ftG": -0.07, "ftK": 0.07}
    rows = []
    ax.axvline(X300, color=style.LIGHT_GREY, lw=0.6, zorder=0)
    for model in MODELS:
        fit, color = fits.loc[model], style.MODEL_COLORS[model]
        runs = transport[transport["series"] == fit["series"]].sort_values("temperature_K")
        temps = runs["temperature_K"].to_numpy(float)
        x, sigma, err = 1000.0 / temps, runs["sigma_mS_cm"].to_numpy(), runs["kinisi_sigma_std_mS_cm"].to_numpy()
        inside = np.linspace(temps.min(), temps.max(), 41)
        outside = np.linspace(C.T_REF_K, temps.min(), 41)
        for t, ls in ((inside, "-"), (outside, (0, (3, 2)))):
            ax.plot(1000.0 / t, sigma_model(t, fit["slope"], fit["intercept"]), color=color, lw=0.8, ls=ls, zorder=2)
        ax.errorbar(x, sigma, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, zorder=3)
        ax.plot(x, sigma, **common.marker_kwargs(md[model], "o", 3.4), zorder=4)
        s300, lo, hi = fit["sigma300_mS_cm"], fit["sigma300_min_mS_cm"], fit["sigma300_max_mS_cm"]
        x300 = X300 + offsets[model]
        ax.errorbar(x300, s300, yerr=[[s300 - lo], [hi - s300]], fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=1.5, zorder=3)
        ax.plot(x300, s300, **common.marker_kwargs(md[model], "o", 3.4), zorder=4)
        rows.extend({"model": model, "series": fit["series"], "kind": "MD run", "temperature_K": t,
                     "x_inv_T_1000_per_K": 1000.0 / t, "sigma_mS_cm": s, "error_low_mS_cm": s - e,
                     "error_high_mS_cm": s + e, "fit_slope_K": fit["slope"], "fit_intercept": fit["intercept"]}
                    for t, s, e in zip(temps, sigma, err))
        rows.append({"model": model, "series": fit["series"], "kind": "fit at 300 K, 1-sigma interval",
                     "temperature_K": C.T_REF_K, "x_inv_T_1000_per_K": x300, "sigma_mS_cm": s300,
                     "error_low_mS_cm": lo, "error_high_mS_cm": hi, "fit_slope_K": fit["slope"],
                     "fit_intercept": fit["intercept"]})
    ax.set_yscale("log")
    ax.set_ylim(0.1, 1e4)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1000", "$10^4$"])
    ax.set_xlim(0.92, 3.5)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("Li conductivity (mS cm$^{-1}$)")
    top = ax.secondary_xaxis("top", functions=(_inverse, _inverse))
    top.set_xticks([1000, 800, 600, 400, 300], ["1000", "800", "600", "400", "300"])
    top.tick_params(axis="x", labelsize=style.SMALL, pad=1.0, length=2.0)
    top.set_xlabel("T (K)", fontsize=style.SMALL, labelpad=1.5)
    handles = [Line2D([0], [0], color="black", lw=0.8, label="weighted fit"),
               Line2D([0], [0], color="black", lw=0.8, ls=(0, (3, 2)), label="extrapolated to 300 K")]
    ax.legend(handles=handles, loc="lower left", fontsize=style.SMALL, handlelength=1.8, borderaxespad=0.2,
              labelspacing=0.3)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, ratios: pd.DataFrame, md: dict[str, str]) -> Path:
    offsets = {"ftG": -0.15, "ftK": 0.15}
    ax.axhline(1.0, color=style.MODEL_COLORS["foundation"], lw=0.7, zorder=1)
    ax.text(2.45, 1.02, "MACE-MP-0", ha="right", va="bottom", fontsize=style.SMALL,
            color=style.MODEL_COLORS["foundation"])
    rows = []
    for index, temperature in enumerate(C.TEMPERATURES):
        for model in FINE_TUNED:
            row = ratios[(ratios["model"] == model) & (ratios["temperature_K"] == temperature)].iloc[0]
            x, color = index + offsets[model], style.MODEL_COLORS[model]
            kinisi, err = row["kinisi_D_ratio"], row["kinisi_D_ratio"] * row["kinisi_D_ratio_rel_err"]
            ax.errorbar(x, kinisi, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, zorder=2)
            ax.plot(x, kinisi, **common.marker_kwargs(md[model], "o", 3.4), zorder=3)
            ax.plot(x, row["pymatgen_D_ratio"], marker="_", markersize=7.5, markeredgewidth=1.1,
                    color=color, linestyle="none", zorder=4)
            rows.append({"model": model, "temperature_K": temperature, "x": x,
                         "pymatgen_D_ratio": row["pymatgen_D_ratio"], "kinisi_D_ratio": kinisi,
                         "kinisi_D_ratio_err": err})
    ax.set_xticks(range(len(C.TEMPERATURES)), [str(t) for t in C.TEMPERATURES])
    ax.set_xlim(-0.5, len(C.TEMPERATURES) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("$D_{Li}$ / $D_{Li}$(MACE-MP-0)")
    handles = [Line2D([0], [0], marker="o", linestyle="-", color="black", markersize=3.0, lw=0.6,
                      label="kinisi, $\\pm$1 s.d."),
               Line2D([0], [0], marker="_", linestyle="none", color="black", markersize=7.0, markeredgewidth=1.1,
                      label="pymatgen")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=style.SMALL,
              borderaxespad=0.2, labelspacing=0.3, handlelength=1.2)
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


def _panel_d(ax, fits: pd.DataFrame, transport: pd.DataFrame, msd: pd.DataFrame) -> Path:
    production = float(C.SERIES_BY_ID[fits.loc["foundation", "series"]]["production_ps"])
    start = C.KINISI_START_FRACTION * production
    ax.axvspan(start, 250.0, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
    ax.text(math.sqrt(start * 200.0), 0.97, "kinisi fit\nwindow", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=style.SMALL, color="#555555", linespacing=1.1)
    frames, slopes = [], []
    for model in MODELS:
        run_id = C.run_id(fits.loc[model, "series"], MSD_TEMPERATURE)
        curve = msd[msd["run_id"] == run_id].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.MODEL_COLORS[model], lw=0.9, zorder=2)
        frames.append(curve[["run_id", "lag_ps", "msd_Li_A2"]].assign(model=model, kinisi_window_start_ps=start))
        slopes.append((model, transport.loc[transport["run_id"] == run_id, "msd_loglog_slope_fit_window"].iloc[0]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 200.0)
    ax.set_ylim(0.2, 200.0)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_xlabel("Lag time (ps)")
    ax.set_ylabel(f"Li MSD at {MSD_TEMPERATURE} K (Å$^2$)")
    ax.text(0.04, 0.97, f"Log-log slope, lag $\\geq$ {start:g} ps", transform=ax.transAxes, ha="left", va="top",
            fontsize=style.SMALL)
    for index, (model, slope) in enumerate(slopes):
        ax.text(0.04, 0.87 - index * 0.09, f"{ROW_LABELS[model]}  {_sig(slope)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=style.SMALL, color=style.MODEL_COLORS[model])
    source = pd.concat(frames, ignore_index=True)
    source["loglog_slope_fit_window"] = source["model"].map(dict(slopes))
    return style.save_source(source, NAME, "d")


def _panel_e(ax, fits: pd.DataFrame, cross: pd.DataFrame, md: dict[str, str]) -> Path:
    rows = []
    for model in MODELS:
        fit, color = fits.loc[model], style.MODEL_COLORS[model]
        slope = _softening(cross, md[model], SOFTENING_SUBSET)
        ax.errorbar(slope, fit["Ea_eV"], yerr=fit["Ea_err_eV"], fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=1.5, zorder=2)
        ax.plot(slope, fit["Ea_eV"], **common.marker_kwargs(md[model], "o", 3.8), zorder=3)
        rows.append({"model": model, "md_model": md[model], "series": fit["series"], "labels": "k222",
                     "element": "Li", "subset": SOFTENING_SUBSET, "softening_slope": slope, "Ea_eV": fit["Ea_eV"],
                     "Ea_err_eV": fit["Ea_err_eV"]})
    ax.set_xlim(0.83, 1.01)
    ax.set_xticks([0.85, 0.9, 0.95, 1.0])
    ax.set_ylim(0.2, 0.4)
    ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4])
    ax.set_xlabel(f"Li softening slope vs 2x2x2 ({SOFTENING_SUBSET})")
    ax.set_ylabel("$E_a$ of the MD run (eV)")
    ax.text(0.04, 0.97, "n = 3, descriptive", transform=ax.transAxes, ha="left", va="top",
            fontsize=style.SMALL, color="#555555")
    return style.save_source(pd.DataFrame(rows), NAME, "e")


def _panel_f(ax, fits: pd.DataFrame, thermostat: pd.DataFrame, blocks: pd.Series, md: dict[str, str]) -> Path:
    ax.axhline(0.0, color=style.GREY, lw=0.5, zorder=1)
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
    comparison, ratios, cross, thermostat, transport, msd, blocks = _load()
    fits = _fits(comparison)
    md = _md_models(cross, fits)
    reading = _reading(comparison)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    fig.legend(handles=[_handle(m, md[m]) for m in MODELS], loc="upper left", bbox_to_anchor=(L.x(10.0), L.y(0.8)),
               ncol=len(MODELS), fontsize=style.FONT_SIZE, handletextpad=0.3, columnspacing=2.0, borderaxespad=0.0)

    # row 1: a, b, c
    top1, h1 = 16.0, 38.0
    ax_a = L.axes(13.0, top1, 48.0, h1)
    ax_b = L.axes(79.0, top1, 36.0, h1)
    ax_cs = L.axes(139.0, top1, 17.0, 19.0)
    ax_ce = L.axes(162.0, top1, 17.0, 19.0)
    sources = [_panel_a(ax_a, fits, transport, md), _panel_b(ax_b, ratios, md), _panel_c(ax_cs, ax_ce, fits, md)]
    L.text(126.0, top1 + 28.0, _reading_text(reading, md), ha="left", va="top", fontsize=style.SMALL,
           linespacing=1.5)

    # row 2: d, e, f
    top2, h2 = 77.0, 38.0
    ax_d = L.axes(13.0, top2, 48.0, h2)
    ax_e = L.axes(79.0, top2, 40.0, h2)
    ax_f = L.axes(140.0, top2, 41.0, h2)
    sources += [_panel_d(ax_d, fits, transport, msd), _panel_e(ax_e, fits, cross, md),
                _panel_f(ax_f, fits, thermostat, blocks, md)]

    L.letter(ax_a, "a", 3.0, 9.5)
    L.letter(ax_b, "b", 68.0, 9.5)
    L.letter(ax_cs, "c", 124.0, 9.5)
    L.letter(ax_d, "d", 3.0, 71.0)
    L.letter(ax_e, "e", 68.0, 71.0)
    L.letter(ax_f, "f", 128.0, 71.0)
    return style.save(fig, "fig4_transport", "main") + sources


def _caption() -> str:
    comparison, ratios, cross, thermostat, transport, msd, blocks = _load()
    fits = _fits(comparison)
    md = _md_models(cross, fits)
    reading = _reading(comparison)
    name = CAPTION_NAMES
    start = C.KINISI_START_FRACTION * C.SERIES_BY_ID[fits.loc["foundation", "series"]]["production_ps"]

    def fit_text(model):
        f = fits.loc[model]
        return (f"{name[model]} {_sig(f['sigma300_mS_cm'])} [{_sig(f['sigma300_min_mS_cm'])}, "
                f"{_sig(f['sigma300_max_mS_cm'])}] mS/cm and Ea {_sig(f['Ea_eV'])} +/- {_sig(f['Ea_err_eV'])} eV")

    low = ratios[ratios["temperature_K"] == min(C.TEMPERATURES)].set_index("model")
    ratio_text = "; ".join(
        f"{name[m]} {_sig(low.loc[m, 'pymatgen_D_ratio'])} (pymatgen) and {_sig(low.loc[m, 'kinisi_D_ratio'])} +/- "
        f"{_sig(low.loc[m, 'kinisi_D_ratio'] * low.loc[m, 'kinisi_D_ratio_rel_err'])} (kinisi)" for m in FINE_TUNED)
    high = ratios[ratios["temperature_K"] > min(C.TEMPERATURES)]
    gap = (high["pymatgen_D_ratio"] - high["kinisi_D_ratio"]).abs().max()
    slopes = transport.set_index("run_id")["msd_loglog_slope_fit_window"]
    slope_text = ", ".join(f"{name[m]} {_sig(slopes[C.run_id(fits.loc[m, 'series'], MSD_TEMPERATURE)])}"
                           for m in MODELS)
    soft_text = ", ".join(f"{name[m]} {_sig(_softening(cross, md[m], SOFTENING_SUBSET))}" for m in MODELS)
    h3_text = ", ".join(f"{_sig(_softening(cross, md[m], 'H3'))}" for m in MODELS)
    h3_shift = max(abs(_softening(cross, md[m], "H3") - _softening(cross, md[m], SOFTENING_SUBSET)) for m in MODELS)
    pressure = {m: thermostat[thermostat["series"] == fits.loc[m, "series"]].set_index("temperature_K")["mean_P_GPa"]
                for m in MODELS}
    shift = [pressure[m][t] - pressure["foundation"][t] for m in FINE_TUNED for t in C.TEMPERATURES]
    if max(shift) >= 0:
        raise ValueError("caption assumes every fine-tuned run has a lower mean pressure than the foundation")
    runs = thermostat[thermostat["series"].isin(fits["series"])]
    n_blocks = sorted({int(blocks[r]) for r in runs["run_id"]})
    sem = max(row.std_P_GPa / math.sqrt(blocks[row.run_id]) for row in runs.itertuples())
    return (
        "Fig. 4 | What fine-tuning changes in Li transport. "
        "All values are re-analysis values of the 416-atom, 200-ps runs at 600, 800 and 1000 K, evaluated in one "
        "environment; the fine-tuned models are compared with the foundation model only. "
        f"The Gamma-label run used the {_stage_word(md['ftG'])} model and the 2x2x2-label run the "
        f"{_stage_word(md['ftK'])} model (identified by force fingerprints, Extended Data Fig. 7a); filled markers, "
        "foundation or stage-two model; open markers, stage-one model, as in Fig. 3. "
        "**a**, Arrhenius plot: per-run Nernst-Einstein conductivity from the pymatgen diffusion coefficient, with "
        "the kinisi standard deviation as error bar, and the fit of ln(sigma T) against 1/T weighted by the kinisi "
        "uncertainty (solid, 600-1000 K; dashed, extrapolated to 300 K). Markers at 300 K, sigma(300 K) with the "
        "1-sigma interval from the fit covariance (the two fine-tuned values are offset horizontally for "
        f"clarity): {'; '.join(fit_text(m) for m in MODELS)}. "
        "**b**, Li diffusion coefficient of each fine-tuned run divided by that of the foundation run at the same "
        "temperature, from pymatgen (bars) and kinisi (circles; error, the two kinisi relative standard deviations "
        f"added in quadrature). At {min(C.TEMPERATURES)} K: {ratio_text}; at 800 and 1000 K the two estimators "
        f"differ by at most {_sig(gap)}. "
        "**c**, sigma(300 K) (logarithmic axis, 1-sigma interval) and Ea (+/- 1 s.d.) of the three fits, and the "
        "reading fixed in the analysis plan: f = ln(sigma_2x2x2/sigma_Gamma)/ln(sigma_MP0/sigma_Gamma), the "
        "fraction of the fine-tuning slowdown recovered with the 2x2x2 labels (MP0, the foundation run), is "
        f"{_sig(reading['f_recovered_fraction'])}, and ln(sigma_2x2x2/sigma_Gamma) = "
        f"{_sig(reading['ln_sigma_ftK_over_ftG'])} +/- {_sig(reading['ln_sigma_ftK_over_ftG_err'])} (half-widths of "
        f"the two log intervals in quadrature). Since f < 0.25 the reading is '{reading['reading']}': the k-point "
        "sampling of the labels does not explain the slowdown. The two runs used models from different training "
        "stages, so this comparison is confounded with the training stage. "
        f"**d**, Li mean-squared displacement at {MSD_TEMPERATURE} K averaged over all time origins (logarithmic "
        f"axes); shaded, the kinisi fit window (lag >= {start:g} ps, {C.KINISI_START_FRACTION * 100:g} % of the run length). Log-log slope of "
        f"the MSD in the window: {slope_text} (1 for diffusive motion); the fine-tuned runs are not yet diffusive "
        f"at {MSD_TEMPERATURE} K, which is where the two estimators in b disagree. "
        "**e**, Li force softening slope (least-squares slope through the origin of model against 2x2x2 DFT Li "
        f"force components on all 27 configurations, {SOFTENING_SUBSET}: {soft_text}) against Ea of the MD run "
        "driven by the same model (+/- 1 s.d.); n = 3, descriptive only, no fit. Only All27 is shown; on H3 the "
        f"slopes are {h3_text} in the same order (at most {_sig(h3_shift)} from All27). "
        "**f**, Mean pressure of each run in the fixed cell (one cell for all three models) against temperature; "
        f"markers are the run means. The fine-tuned runs lie {_sig(-max(shift))} to {_sig(-min(shift))} GPa below the "
        "foundation run at the same temperature. The standard deviation of the instantaneous pressure divided by "
        f"the square root of the number of 0.5-ps blocks ({n_blocks[0]}) is at most {_sig(sem)} GPa, smaller than "
        "the markers. "
        + common.sources([TRANSPORT, COMPARISON, RATIOS, MSD, CROSS, THERMOSTAT, TIMESERIES])
    )


CAPTION = _caption()
