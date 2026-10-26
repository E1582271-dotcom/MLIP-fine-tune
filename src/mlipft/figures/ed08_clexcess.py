"""Extended Data Fig. 8: the Cl-excess series Li(6-x)PS(5-x)Cl(1+x) under MACE-MP-0.

a, Arrhenius plot of the four compositions (quoted values) with the weighted fits extrapolated to
300 K; b, sigma(300 K) against Cl content; c, enhancement of sigma(300 K) over Li6PS5Cl, with the
measured enhancement of Li5.5PS4.5Cl1.5 over Li6PS5Cl (Adeli et al. 2019); d, activation energy
against Cl content; e, Li mean-squared displacement at 600 K. The series uses a universal potential,
so the comparison with a measured value is allowed here.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import constants as C
from .. import paths
from ..transport.arrhenius import sigma_model
from . import common, style

NAME = "ed08"
WIDTH, HEIGHT = 183.0, 102.0
CLEXCESS = paths.RESULTS / "clexcess.csv"
FITS = paths.RESULTS / "fits.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
LITERATURE = paths.REFERENCES / "literature.csv"
MEASURED_REF = "adeli2019"
MSD_TEMPERATURE = 600
X300 = 1000.0 / C.T_REF_K
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _load():
    series = common.read(CLEXCESS).sort_values("cl_content").reset_index(drop=True)
    fits = common.read(FITS)
    transport = common.read(TRANSPORT)
    msd = common.read(MSD)
    literature = common.read(LITERATURE)
    return series, fits, transport, msd, literature


def _formula(composition: str) -> str:
    """Chemical formula with subscripted counts (mathtext)."""
    return re.sub(r"(\d+(?:\.\d+)?)", r"$_{\1}$", composition)


def _color(cl_content: float) -> str:
    return style.CL_COLORS[round(float(cl_content), 2)]


def _sig(value: float, digits: int = 3) -> str:
    """``value`` to ``digits`` significant figures, keeping trailing zeros (0.260, not 0.26)."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    decimals = max(digits - 1 - math.floor(math.log10(abs(value))), 0)
    return f"{value:.{decimals}f}"


def _measured(literature: pd.DataFrame) -> dict:
    """Measured enhancement over Li6PS5Cl from the reference table: Cl content, factor, citation."""
    row = literature[literature["ref_id"] == MEASURED_REF].iloc[0]
    match = re.search(r"about (\w+) times (\S+)", row["conditions"])
    if match is None or match.group(1) not in NUMBER_WORDS:
        raise ValueError(f"no enhancement factor in {MEASURED_REF}")
    cl = re.search(r"Cl(\d+(?:\.\d+)?)$", row["material"])
    return {"cl_content": float(cl.group(1)) if cl else 1.0, "enhancement": float(NUMBER_WORDS[match.group(1)]),
            "material": row["material"], "reference_material": match.group(2), "sigma_mS_cm": row["value"],
            "conditions": row["conditions"], "citation": row["citation"], "doi": row["doi"]}


def _style_cl_axis(ax) -> None:
    ax.set_xlim(0.9, 1.85)
    ax.set_xticks([1.0, 1.25, 1.5, 1.75], ["1.00", "1.25", "1.50", "1.75"])
    ax.set_xlabel("Cl per formula unit (1 + x)")


def _series_markers(ax, x, y, colors, yerr=None) -> None:
    ax.plot(x, y, color=style.GREY, lw=0.6, zorder=1)
    for xi, yi, color, err in zip(x, y, colors, yerr if yerr is not None else [None] * len(x)):
        if err is not None and np.any(np.asarray(err) > 0):
            ax.errorbar(xi, yi, yerr=err, fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=2)
        ax.plot(xi, yi, marker="o", markersize=3.6, markerfacecolor=color, markeredgecolor=color, linestyle="none",
                zorder=3)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, series: pd.DataFrame, transport: pd.DataFrame) -> Path:
    rows = []
    ax.axvline(X300, color=style.LIGHT_GREY, lw=0.6, zorder=0)
    for fit in series.itertuples():
        color = _color(fit.cl_content)
        runs = transport[transport["series"] == fit.series].sort_values("temperature_K")
        temps = runs["temperature_K"].to_numpy(float)
        sigma, err = runs["sigma_mS_cm"].to_numpy(), runs["kinisi_sigma_std_mS_cm"].to_numpy()
        inside = np.linspace(temps.min(), temps.max(), 41)
        outside = np.linspace(C.T_REF_K, temps.min(), 41)
        for t, ls in ((inside, "-"), (outside, (0, (3, 2)))):
            ax.plot(1000.0 / t, sigma_model(t, fit.slope, fit.intercept), color=color, lw=0.8, ls=ls, zorder=2)
        ax.errorbar(1000.0 / temps, sigma, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, zorder=3)
        ax.plot(1000.0 / temps, sigma, marker="o", markersize=3.2, markerfacecolor=color, markeredgecolor=color,
                linestyle="none", zorder=4)
        rows.extend({"series": fit.series, "composition": fit.composition, "cl_content": fit.cl_content,
                     "temperature_K": t, "x_inv_T_1000_per_K": 1000.0 / t, "sigma_mS_cm": s,
                     "kinisi_sigma_std_mS_cm": e, "fit_source": fit.source, "fit_slope_K": fit.slope,
                     "fit_intercept": fit.intercept} for t, s, e in zip(temps, sigma, err))
    ax.set_yscale("log")
    ax.set_ylim(3, 1e4)
    style.log_ticks(ax, "y")
    ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1000", "$10^4$"])
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


def _inverse(values):
    values = np.asarray(values, float)
    return np.divide(1000.0, values, out=np.full_like(values, np.inf), where=values != 0)


def _panel_b(ax, series: pd.DataFrame) -> Path:
    s, lo, hi = series["sigma300_mS_cm"], series["sigma300_min_mS_cm"], series["sigma300_max_mS_cm"]
    errors = [np.array([[a - b], [c - a]]) for a, b, c in zip(s, lo, hi)]
    _series_markers(ax, series["cl_content"], s, [_color(c) for c in series["cl_content"]], errors)
    ax.set_yscale("log")
    ax.set_ylim(3, 500)
    style.log_ticks(ax, "y")
    ax.set_yticks([10, 100], ["10", "100"])
    _style_cl_axis(ax)
    ax.set_ylabel("$\\sigma$(300 K) (mS cm$^{-1}$)")
    source = series[["series", "composition", "cl_content", "source", "sigma300_mS_cm", "sigma300_min_mS_cm",
                     "sigma300_max_mS_cm"]]
    return style.save_source(source, NAME, "b")


def _panel_c(ax, series: pd.DataFrame, measured: dict) -> Path:
    ax.axhline(1.0, color=style.GREY, lw=0.5, zorder=0)
    e, lo, hi = series["enhancement_over_x0"], series["enhancement_min"], series["enhancement_max"]
    errors = [np.array([[a - b], [c - a]]) for a, b, c in zip(e, lo, hi)]
    _series_markers(ax, series["cl_content"], e, [_color(c) for c in series["cl_content"]], errors)
    ax.plot(measured["cl_content"], measured["enhancement"], marker="D", markersize=3.6,
            markerfacecolor=style.EXPERIMENT_COLOR, markeredgecolor=style.EXPERIMENT_COLOR, linestyle="none",
            zorder=4)
    ax.annotate(f"measured,\nabout {measured['enhancement']:g}x", (measured["cl_content"], measured["enhancement"]),
                xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=style.SMALL,
                linespacing=1.1)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 70)
    style.log_ticks(ax, "y")
    ax.set_yticks([1, 10], ["1", "10"])
    _style_cl_axis(ax)
    ax.set_ylabel("$\\sigma$(300 K) relative to Li$_6$PS$_5$Cl")
    handles = [Line2D([0], [0], marker="o", linestyle="-", color=style.GREY, markerfacecolor=style.GREY,
                      markeredgecolor=style.GREY, markersize=3.2, lw=0.6, label="MACE-MP-0, 1-sigma interval"),
               Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=style.EXPERIMENT_COLOR,
                      markeredgecolor=style.EXPERIMENT_COLOR, markersize=3.2,
                      label=f"Measured, {_formula(measured['material'])} over\n"
                            f"{_formula(measured['reference_material'])} (Adeli et al. 2019)")]
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, borderaxespad=0.2, labelspacing=0.4,
              handlelength=1.4)
    rows = series[["series", "composition", "cl_content", "source", "enhancement_over_x0", "enhancement_min",
                   "enhancement_max"]].assign(kind="MACE-MP-0")
    measured_row = pd.DataFrame([{"series": measured["material"], "composition": measured["material"],
                                  "cl_content": measured["cl_content"], "source": MEASURED_REF,
                                  "enhancement_over_x0": measured["enhancement"], "kind": "measured"}])
    return style.save_source(pd.concat([rows, measured_row], ignore_index=True), NAME, "c")


def _panel_d(ax, series: pd.DataFrame) -> Path:
    _series_markers(ax, series["cl_content"], series["Ea_eV"], [_color(c) for c in series["cl_content"]],
                    list(series["Ea_err_eV"]))
    ax.set_ylim(0.1, 0.3)
    ax.set_yticks([0.1, 0.15, 0.2, 0.25, 0.3])
    _style_cl_axis(ax)
    ax.set_ylabel("$E_a$ (eV)")
    return style.save_source(series[["series", "composition", "cl_content", "source", "Ea_eV", "Ea_err_eV"]], NAME,
                             "d")


def _panel_e(ax, series: pd.DataFrame, transport: pd.DataFrame, msd: pd.DataFrame) -> tuple[Path, float]:
    production = {float(C.SERIES_BY_ID[s]["production_ps"]) for s in series["series"]}
    if len(production) != 1:
        raise ValueError("the Cl-excess runs differ in length")
    start = C.KINISI_START_FRACTION * production.pop()
    ax.axvspan(start, 250.0, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
    ax.text(math.sqrt(start * 150.0), 0.03, "kinisi fit\nwindow", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=style.SMALL, color="#555555", linespacing=1.1)
    frames, slopes = [], []
    for fit in series.itertuples():
        run_id = C.run_id(fit.series, MSD_TEMPERATURE)
        curve = msd[msd["run_id"] == run_id].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=_color(fit.cl_content), lw=0.9, zorder=2)
        frames.append(curve[["run_id", "lag_ps", "msd_Li_A2"]].assign(composition=fit.composition,
                                                                      cl_content=fit.cl_content,
                                                                      kinisi_window_start_ps=start))
        slopes.append((run_id, transport.loc[transport["run_id"] == run_id, "msd_loglog_slope_fit_window"].iloc[0]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 150.0)
    ax.set_ylim(0.2, 500.0)
    style.log_ticks(ax, "x")
    style.log_ticks(ax, "y")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_xlabel("Lag time (ps)")
    ax.set_ylabel(f"Li MSD at {MSD_TEMPERATURE} K (Å$^2$)")
    ax.text(0.04, 0.97, f"Log-log slope, lag $\\geq$ {start:g} ps", transform=ax.transAxes, ha="left", va="top",
            fontsize=style.SMALL)
    for index, (fit, (_, slope)) in enumerate(zip(series.itertuples(), slopes)):
        ax.text(0.04, 0.86 - index * 0.105, f"Cl {fit.cl_content:.2f}  {_sig(slope)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=style.SMALL, color=_color(fit.cl_content))
    source = pd.concat(frames, ignore_index=True)
    source["loglog_slope_fit_window"] = source["run_id"].map(dict(slopes))
    return style.save_source(source, NAME, "e"), start


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    series, fits, transport, msd, literature = _load()
    measured = _measured(literature)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    handles = [Line2D([0], [0], marker="o", linestyle="-", lw=0.8, color=_color(row.cl_content), markersize=3.4,
                      label=f"{_formula(row.composition)} (x = {row.cl_content - 1:.2f}, {row.n_atoms} atoms)")
               for row in series.itertuples()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L.x(10.0), L.y(0.8)), ncol=len(handles),
               fontsize=style.SMALL, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0, handlelength=1.4)

    top1, top2, h = 15.0, 62.0, 30.0
    ax_a = L.axes(13.0, top1, 50.0, top2 + h - top1)
    ax_b = L.axes(80.0, top1, 38.0, h)
    ax_c = L.axes(137.0, top1, 44.0, h)
    ax_d = L.axes(80.0, top2, 38.0, h)
    ax_e = L.axes(137.0, top2, 44.0, h)
    sources = [_panel_a(ax_a, series, transport), _panel_b(ax_b, series), _panel_c(ax_c, series, measured),
               _panel_d(ax_d, series), _panel_e(ax_e, series, transport, msd)[0]]

    L.letter(ax_a, "a", 3.0, 9.5)
    L.letter(ax_b, "b", 69.0, 9.5)
    L.letter(ax_c, "c", 126.0, 9.5)
    L.letter(ax_d, "d", 69.0, 56.5)
    L.letter(ax_e, "e", 126.0, 56.5)
    return style.save(fig, "ed08_clexcess", "extended") + sources


def _caption() -> str:
    series, fits, transport, msd, literature = _load()
    measured = _measured(literature)
    indexed = series.set_index("cl_content")
    start = C.KINISI_START_FRACTION * C.SERIES_BY_ID[series["series"].iloc[0]]["production_ps"]
    sources_used = ", ".join(sorted(set(series["source"])))
    pairs = fits[fits["series"].isin(series["series"])].pivot(index="series", columns="source",
                                                              values="sigma300_mS_cm")
    agreement = (pairs["reanalysis"] / pairs["published"] - 1).abs().max() * 100 if "published" in pairs else 0.0
    sigma_text = "; ".join(f"Cl {cl:.2f}, {_sig(r.sigma300_mS_cm)} [{_sig(r.sigma300_min_mS_cm)}, "
                           f"{_sig(r.sigma300_max_mS_cm)}] mS/cm" for cl, r in indexed.iterrows())
    ea_text = ", ".join(f"{_sig(r.Ea_eV)} +/- {_sig(r.Ea_err_eV)}" for _, r in indexed.iterrows())
    enh_text = "; ".join(f"Cl {cl:.2f}, {_sig(r.enhancement_over_x0)} [{_sig(r.enhancement_min)}, "
                         f"{_sig(r.enhancement_max)}]" for cl, r in indexed.iterrows() if cl > indexed.index.min())
    at = indexed.loc[measured["cl_content"]]
    over = at["enhancement_over_x0"] / measured["enhancement"]
    slopes = transport.set_index("run_id")["msd_loglog_slope_fit_window"]
    slope_text = ", ".join(f"{_sig(slopes[C.run_id(s, MSD_TEMPERATURE)])}" for s in series["series"])
    xs = ", ".join(f"{cl - 1:g}" for cl in indexed.index)
    atoms = ", ".join(str(int(n)) for n in series["n_atoms"])
    length = C.SERIES_BY_ID[series["series"].iloc[0]]["production_ps"]
    measured_conditions = measured["conditions"].split(";")[0].strip()
    return (
        "Extended Data Fig. 8 | Cl-excess series under MACE-MP-0. "
        f"Li(6-x)PS(5-x)Cl(1+x) with x = {xs} (Cl {indexed.index.min():.2f}-{indexed.index.max():.2f} per formula "
        f"unit; {atoms} atoms), MACE-MP-0, {length:g}-ps runs at 600, 800 and 1000 K. Fits are the quoted values "
        "under the "
        f"reproduction rule ({sources_used}; the re-analysis reproduces sigma(300 K) to within "
        f"{_sig(agreement)} %). "
        "**a**, Arrhenius plot: per-run Nernst-Einstein conductivity from the pymatgen diffusion coefficient with "
        "the kinisi standard deviation as error bar, and the weighted fit of ln(sigma T) against 1/T (solid, "
        "600-1000 K; dashed, extrapolated to 300 K). "
        f"**b**, sigma(300 K) with the 1-sigma interval of the fit: {sigma_text} (logarithmic axis). "
        "**c**, Enhancement of sigma(300 K) over Li6PS5Cl (x = 0), with an interval combining the log half-widths "
        f"of the two fits in quadrature: {enh_text}. Black diamond, the measured enhancement of "
        f"{measured['material']} over {measured['reference_material']} prepared the same way (about "
        f"{measured['enhancement']:g} times; {measured['sigma_mS_cm']:g} mS/cm, {measured_conditions}; Adeli et "
        f"al. 2019). The simulated enhancement at Cl {measured['cl_content']:.2f} is {_sig(over)} times the measured one. "
        f"Logarithmic axis. "
        f"**d**, Activation energy (+/- 1 s.d.): {ea_text} eV for Cl 1.00 to 1.75. "
        f"**e**, Li mean-squared displacement at {MSD_TEMPERATURE} K averaged over all time origins (logarithmic "
        f"axes); shaded, the kinisi fit window (lag >= {start:g} ps, {C.KINISI_START_FRACTION * 100:g} % of the run length). Log-log slopes in the window, Cl 1.00 to "
        f"1.75: {slope_text}. "
        "MACE-MP-0 is a universal potential, so the comparison with a measured value is made here; the fine-tuned "
        "models are not used in this figure. "
        + common.sources([CLEXCESS, FITS, TRANSPORT, MSD, LITERATURE])
    )


CAPTION = _caption()
