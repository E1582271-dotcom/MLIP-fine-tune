"""The Cl-excess series Li(6-x)PS(5-x)Cl(1+x) under MACE-MP-0 (Extended Data, registry key "clexcess").

a (the hero panel), Arrhenius plot of the four compositions (quoted values) with the weighted fits
extrapolated to 300 K; b, sigma(300 K) against Cl content; c, enhancement of sigma(300 K) over
Li6PS5Cl, with the measured enhancement of Li5.5PS4.5Cl1.5 over Li6PS5Cl (Adeli et al. 2019) as a
grey context diamond; d, activation energy against Cl content; e, Li mean-squared displacement at
600 K. Cl content is coloured in the blue tiers of ``style.CL_COLORS``; every run of the series is an
inverted triangle (the Cl-excess run type). The series uses a universal potential, so a measured
value may appear here, as context.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import constants as C
from .. import paths
from ..transport.arrhenius import sigma_model
from . import common, registry, style

KEY = "clexcess"
NAME = registry.number(KEY)
PANELS = "abcde"
WIDTH, HEIGHT = 183.0, 98.0
CLEXCESS = paths.RESULTS / "clexcess.csv"
FITS = paths.RESULTS / "fits.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
LITERATURE = paths.REFERENCES / "literature.csv"
MEASURED_REF = "adeli2019"
MSD_TEMPERATURE = 600
LABELLED_TEMPERATURES = (1000, 600, C.T_REF_K)  # written above the data columns of a (800 K has no room)
MARKER_SIZE = 3.6
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


def _marker(cl_content: float, size: float = MARKER_SIZE, **overrides) -> dict:
    """``ax.plot`` keyword arguments of a Cl-excess run (MACE-MP-0, filled): inverted triangle, colour by Cl
    content."""
    color = _color(cl_content)
    return common.model_style("foundation", "clexcess", size=size, color=color, markeredgecolor=color,
                              markerfacecolor=color, **overrides)


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


def _series_markers(ax, x, y, yerr=None) -> None:
    """One marker per composition (x = Cl content), joined by a thin grey line, with optional error bars."""
    ax.plot(x, y, color=style.GRAY, lw=0.6, zorder=1)
    for xi, yi, err in zip(x, y, yerr if yerr is not None else [None] * len(x)):
        if err is not None and np.any(np.asarray(err) > 0):
            ax.errorbar(xi, yi, yerr=err, fmt="none", ecolor=_color(xi), elinewidth=0.7, capsize=1.5, zorder=2)
        ax.plot(xi, yi, **_marker(xi, zorder=3))


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, series: pd.DataFrame, transport: pd.DataFrame) -> Path:
    rows = []
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
        ax.plot(1000.0 / temps, sigma, **_marker(fit.cl_content, size=3.4, zorder=4))
        rows.extend({"series": fit.series, "composition": fit.composition, "cl_content": fit.cl_content,
                     "temperature_K": t, "x_inv_T_1000_per_K": 1000.0 / t, "sigma_mS_cm": s,
                     "kinisi_sigma_std_mS_cm": e, "fit_source": fit.source, "fit_slope_K": fit.slope,
                     "fit_intercept": fit.intercept} for t, s, e in zip(temps, sigma, err))
    ax.set_yscale("log")
    ax.set_ylim(3, 1e4)
    style.log_ticks(ax, "y")
    ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1000", "$10^4$"])
    ax.set_xlim(0.8, 3.5)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("Li conductivity (mS cm$^{-1}$)")
    common.temperature_labels(ax, LABELLED_TEMPERATURES)
    handles = [Line2D([0], [0], color=style.BLACK, lw=0.8, label="weighted fit"),
               Line2D([0], [0], color=style.BLACK, lw=0.8, ls=(0, (3, 2)), label="extrapolated to 300 K")]
    ax.legend(handles=handles, loc="lower left", fontsize=style.SMALL, handlelength=1.8, borderaxespad=0.2,
              labelspacing=0.3)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, series: pd.DataFrame) -> Path:
    s, lo, hi = series["sigma300_mS_cm"], series["sigma300_min_mS_cm"], series["sigma300_max_mS_cm"]
    errors = [np.array([[a - b], [c - a]]) for a, b, c in zip(s, lo, hi)]
    _series_markers(ax, series["cl_content"], s, errors)
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
    ax.axhline(1.0, **style.UNITY_LINE)
    e, lo, hi = series["enhancement_over_x0"], series["enhancement_min"], series["enhancement_max"]
    errors = [np.array([[a - b], [c - a]]) for a, b, c in zip(e, lo, hi)]
    _series_markers(ax, series["cl_content"], e, errors)
    ax.plot(measured["cl_content"], measured["enhancement"], **common.context_style(size=MARKER_SIZE, zorder=4))
    ax.annotate("measured\n(context)", (measured["cl_content"], measured["enhancement"]), xytext=(4.5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=style.SMALL, linespacing=1.1,
                color=style.CONTEXT_TEXT)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 70)
    style.log_ticks(ax, "y")
    ax.set_yticks([1, 10], ["1", "10"])
    _style_cl_axis(ax)
    ax.set_ylabel("$\\sigma$(300 K) relative to Li$_6$PS$_5$Cl")
    rows = series[["series", "composition", "cl_content", "source", "enhancement_over_x0", "enhancement_min",
                   "enhancement_max"]].assign(kind="MACE-MP-0")
    measured_row = pd.DataFrame([{"series": measured["material"], "composition": measured["material"],
                                  "cl_content": measured["cl_content"], "source": MEASURED_REF,
                                  "enhancement_over_x0": measured["enhancement"], "kind": "measured"}])
    return style.save_source(pd.concat([rows, measured_row], ignore_index=True), NAME, "c")


def _panel_d(ax, series: pd.DataFrame) -> Path:
    _series_markers(ax, series["cl_content"], series["Ea_eV"], list(series["Ea_err_eV"]))
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
    ax.axvspan(start, 250.0, **style.BAND_FILL)
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
    handles = [Line2D([0], [0], color=_color(fit.cl_content), lw=0.9, label=f"{fit.cl_content:.2f}: {common.sig(slope)}")
               for fit, (_, slope) in zip(series.itertuples(), slopes)]
    handles.append(Patch(facecolor=style.BAND, edgecolor="none", label="kinisi fit window"))
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.2, handletextpad=0.5,
              labelspacing=0.25, borderaxespad=0.3, title=f"Cl content: log-log slope,\nlag $\\geq$ {start:g} ps",
              title_fontsize=style.SMALL, alignment="left")
    source = pd.concat(frames, ignore_index=True)
    source["loglog_slope_fit_window"] = source["run_id"].map(dict(slopes))
    return style.save_source(source, NAME, "e"), start


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    series, fits, transport, msd, literature = _load()
    measured = _measured(literature)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    handles = [common.handle({**_marker(row.cl_content, size=3.4), "linestyle": "-", "lw": 0.8},
                             f"{_formula(row.composition)} (x = {row.cl_content - 1:.2f}, {row.n_atoms} atoms)")
               for row in series.itertuples()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L.x(10.0), L.y(0.8)), ncol=len(handles),
               fontsize=style.SMALL, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0, handlelength=1.4)

    top1, top2, h = 11.0, 58.0, 30.0
    ax_a = L.axes(13.0, top1, 50.0, top2 + h - top1, row="hero", col="a")  # the hero panel
    ax_b = L.axes(80.0, top1, 38.0, h, row="top", col="mid")
    ax_c = L.axes(137.0, top1, 44.0, h, row="top", col="right")
    ax_d = L.axes(80.0, top2, 38.0, h, row="bottom", col="mid")
    ax_e = L.axes(137.0, top2, 44.0, h, row="bottom", col="right")
    sources = [_panel_a(ax_a, series, transport), _panel_b(ax_b, series), _panel_c(ax_c, series, measured),
               _panel_d(ax_d, series), _panel_e(ax_e, series, transport, msd)[0]]

    L.letter(ax_a, "a", 3.0, top1 - 1.5)
    L.letter(ax_b, "b", 69.0, top1 - 1.5)
    L.letter(ax_c, "c", 126.0, top1 - 1.5)
    L.letter(ax_d, "d", 69.0, top2 - 1.5)
    L.letter(ax_e, "e", 126.0, top2 - 1.5)
    return L.save(*registry.target(KEY)) + sources


def _caption() -> str:
    series, fits, transport, msd, literature = _load()
    measured = _measured(literature)
    indexed = series.set_index("cl_content")
    start = C.KINISI_START_FRACTION * C.SERIES_BY_ID[series["series"].iloc[0]]["production_ps"]
    sources_used = ", ".join(sorted(set(series["source"])))
    pairs = fits[fits["series"].isin(series["series"])].pivot(index="series", columns="source",
                                                              values="sigma300_mS_cm")
    agreement = (pairs["reanalysis"] / pairs["published"] - 1).abs().max() * 100 if "published" in pairs else 0.0
    sigma_text = "; ".join(f"Cl {cl:.2f}, {common.sig(r.sigma300_mS_cm)} [{common.sig(r.sigma300_min_mS_cm)}, "
                           f"{common.sig(r.sigma300_max_mS_cm)}] mS/cm" for cl, r in indexed.iterrows())
    ea_text = ", ".join(f"{common.sig(r.Ea_eV)} +/- {common.sig(r.Ea_err_eV)}" for _, r in indexed.iterrows())
    enh_text = "; ".join(f"Cl {cl:.2f}, {common.sig(r.enhancement_over_x0)} [{common.sig(r.enhancement_min)}, "
                         f"{common.sig(r.enhancement_max)}]" for cl, r in indexed.iterrows() if cl > indexed.index.min())
    at = indexed.loc[measured["cl_content"]]
    over = at["enhancement_over_x0"] / measured["enhancement"]
    slopes = transport.set_index("run_id")["msd_loglog_slope_fit_window"]
    slope_text = ", ".join(f"{common.sig(slopes[C.run_id(s, MSD_TEMPERATURE)])}" for s in series["series"])
    xs = ", ".join(f"{cl - 1:g}" for cl in indexed.index)
    atoms = ", ".join(str(int(n)) for n in series["n_atoms"])
    length = C.SERIES_BY_ID[series["series"].iloc[0]]["production_ps"]
    measured_conditions = measured["conditions"].split(";")[0].strip()
    return (
        registry.heading(KEY)
        + f"Li(6-x)PS(5-x)Cl(1+x) with x = {xs} (Cl {indexed.index.min():.2f}-{indexed.index.max():.2f} per formula "
        f"unit; {atoms} atoms), MACE-MP-0, {length:g}-ps runs at 600, 800 and 1000 K. Fits are the quoted values "
        "under the "
        f"reproduction rule ({sources_used}; the re-analysis reproduces sigma(300 K) to within "
        f"{common.sig(agreement)} %). "
        "Colour, Cl content (light to dark blue, 1.00 to 1.75); inverted triangles, the runs of the series. "
        "**a**, Arrhenius plot, n = 3 runs per composition (one per temperature): per-run Nernst-Einstein "
        "conductivity from the pymatgen diffusion coefficient with the kinisi standard deviation as error bar, and "
        "the weighted fit of ln(sigma T) against 1/T (solid, 600-1000 K; dashed, extrapolated to 300 K); the "
        "temperatures of the outer data columns and of the extrapolation target are written above the data. "
        f"Per-series fits with their residuals are in {registry.label('arrhenius')}. "
        f"**b**, sigma(300 K) from the fit over n = 3 temperatures; error bars, the 1-sigma interval of the fit: "
        f"{sigma_text} (logarithmic axis). "
        "**c**, Enhancement of sigma(300 K) over Li6PS5Cl (x = 0); error bars, an interval combining the log "
        f"half-widths of the two fits in quadrature: {enh_text}. Grey diamond, labelled 'measured (context)', the "
        f"measured enhancement of {measured['material']} over {measured['reference_material']} prepared the same "
        f"way (one measurement pair, no uncertainty given; about {measured['enhancement']:g} times; "
        f"{measured['sigma_mS_cm']:g} mS/cm, {measured_conditions}; Adeli et al. 2019). The simulated enhancement "
        f"at Cl {measured['cl_content']:.2f} is {common.sig(over)} times the measured one. Logarithmic axis. "
        f"**d**, Activation energy from the same fits; error bars, +/- 1 s.d. of the fit: {ea_text} eV for Cl 1.00 "
        "to 1.75. "
        f"**e**, Li mean-squared displacement of the single {MSD_TEMPERATURE} K run of each composition (n = 1, "
        f"averaged over all time origins; no error bars; logarithmic axes); shaded, the kinisi fit window (lag >= "
        f"{start:g} ps, {C.KINISI_START_FRACTION * 100:g} % of the run length). Log-log slopes in the window, Cl "
        f"1.00 to 1.75: {slope_text}. "
        "MACE-MP-0 is a universal potential, so the measured value is shown here as context; the fine-tuned "
        "models are not used in this figure. "
        + common.sources([CLEXCESS, FITS, TRANSPORT, MSD, LITERATURE])
    )


CAPTION = _caption()
