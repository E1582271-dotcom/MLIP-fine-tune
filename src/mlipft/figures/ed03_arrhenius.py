"""Extended Data Fig. 3: the Arrhenius fit of every series.

One panel per series (14), grouped by part: A, the universal potentials on Li6PS5Cl (52-atom
tiers and the 416-atom production runs); C, the two fine-tuned models; D, the Cl-excess series
under MACE-MP-0. Each panel shows the per-run conductivity with its kinisi uncertainty, the quoted
weighted fit (published value where one exists, otherwise the re-analysis) extrapolated to 300 K
with its 1-sigma band, and a residual strip in units of each point's ln-space uncertainty. No
measured value is shown: the figure contains the fine-tuned series.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import paths
from ..constants import T_REF_K, TEMPERATURES
from . import common, style

NAME = "ed03"
WIDTH, HEIGHT = 183.0, 236.0
FITS = paths.RESULTS / "fits.csv"
TRANSPORT = paths.MD / "transport.csv"
SOURCE_TEXT = {"published": "published fit", "reanalysis": "re-analysis fit"}
# Rows of the grid: (heading, series per column)
ROWS = (
    ("Part A. Universal potentials, Li$_6$PS$_5$Cl", ("mace_52_050", "mace_52_100", "mace_52_150", "mace_416_200")),
    (None, ("mattersim_52_050", "mattersim_52_100", "mattersim_52_150", "mattersim_416_200")),
    ("Part C. Fine-tuned MACE, Li$_6$PS$_5$Cl (same protocol as the 416-atom production runs)",
     ("ftG_416_200", "ftK_416_200")),
    ("Part D. Cl-excess series Li$_{6-x}$PS$_{5-x}$Cl$_{1+x}$, MACE-MP-0", ("clx000_416_150", "clx025_408_150",
                                                                    "clx050_400_150", "clx075_392_150")),
)
TITLES = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG": "Fine-tuned, Gamma labels",
          "ftK": "Fine-tuned, 2x2x2 labels"}
X_LIMITS = (0.85, 3.55)
Y_LIMITS = (0.08, 2.0e4)
Z_LIMIT = 3.6


def _sig3(value: float) -> str:
    """Three significant figures, keeping trailing zeros (0.260, 2.44, 213)."""
    return f"{value:.{max(0, 2 - int(np.floor(np.log10(abs(value)))))}f}"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = common.read(FITS)
    fits = fits[fits["quoted"]].set_index("series")
    transport = common.read(TRANSPORT)
    return fits, transport


def _order() -> list[str]:
    return [series for _, row in ROWS for series in row]


def _color(fit: pd.Series) -> str:
    if fit["part"] == "D":
        return style.CL_COLORS[float(fit["cl_content"])]
    return style.MODEL_COLORS[fit["model"]]


def _title(fit: pd.Series) -> tuple[str, str]:
    if fit["part"] == "D":
        composition, content = fit["label"].split(",")[0].split(" ", 1)
        return f"{_subscripts(composition)} {content}", f"{fit['n_atoms']} atoms, {fit['production_ps']} ps"
    return TITLES[fit["model"]], f"{fit['n_atoms']} atoms, {fit['production_ps']} ps"


def _fit_band(fit: pd.Series, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sigma(T) of ln(sigma T) = slope/T + intercept with its 1-sigma band (mS/cm).

    The slope-intercept covariance is recovered from the stored 300 K interval, so the band equals
    the quoted interval at 300 K exactly.
    """
    x0 = 1.0 / T_REF_K
    var_s, var_i = fit["slope_err"] ** 2, fit["intercept_err"] ** 2
    half = 0.5 * (np.log(fit["sigma300_max_mS_cm"]) - np.log(fit["sigma300_min_mS_cm"]))
    covariance = (half ** 2 - x0 ** 2 * var_s - var_i) / (2.0 * x0)
    x = 1.0 / temperature
    y = fit["slope"] * x + fit["intercept"]
    sd = np.sqrt(np.maximum(x ** 2 * var_s + var_i + 2.0 * x * covariance, 0.0))
    return np.exp(y) / temperature, np.exp(y - sd) / temperature, np.exp(y + sd) / temperature


def _points(transport: pd.DataFrame, fit: pd.Series, series: str) -> pd.DataFrame:
    """Per-run sigma with its kinisi error, the fitted value and the normalised residual."""
    runs = transport[transport["series"] == series].sort_values("temperature_K")
    temperature = runs["temperature_K"].to_numpy(float)
    sigma = runs["sigma_mS_cm"].to_numpy()
    error = runs["kinisi_sigma_std_mS_cm"].to_numpy()
    fitted = np.exp(fit["slope"] / temperature + fit["intercept"]) / temperature
    ln_error = error / sigma
    return pd.DataFrame({"series": series, "run_id": runs["run_id"].to_numpy(), "temperature_K": temperature,
                         "inv_temperature_1000_per_K": 1000.0 / temperature, "sigma_mS_cm": sigma,
                         "kinisi_sigma_std_mS_cm": error, "fit_sigma_mS_cm": fitted, "ln_sigma_error": ln_error,
                         "residual_normalised": (np.log(sigma) - np.log(fitted)) / ln_error})


# ----------------------------------------------------------------------------------------------- panels


def _panel(ax, res_ax, fit: pd.Series, points: pd.DataFrame, first_column: bool) -> pd.DataFrame:
    color = _color(fit)
    x300 = 1000.0 / T_REF_K
    x_split = 1000.0 / min(TEMPERATURES)
    x = np.concatenate([np.linspace(1000.0 / max(TEMPERATURES), x_split, 30)[:-1], np.linspace(x_split, x300, 60)])
    temperature = 1000.0 / x
    sigma, low, high = _fit_band(fit, temperature)
    ax.fill_between(x, low, high, color=color, alpha=0.18, lw=0, zorder=1)
    ax.plot(x[x <= x_split], sigma[x <= x_split], color=color, lw=0.8, zorder=3)
    ax.plot(x[x >= x_split], sigma[x >= x_split], color=color, lw=0.8, ls=(0, (3, 1.5)), zorder=3)
    ax.errorbar(points["inv_temperature_1000_per_K"], points["sigma_mS_cm"], yerr=points["kinisi_sigma_std_mS_cm"],
                fmt="o", color=color, markersize=2.8, markeredgewidth=0.0, elinewidth=0.6, capsize=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1000", "10000"])
    ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
    ax.tick_params(labelbottom=False, labelleft=first_column)
    if first_column:
        ax.set_ylabel(r"$\sigma$ (mS cm$^{-1}$)", labelpad=1.5)
    ax.text(0.04, 0.04, f"$\\sigma$(300 K) {_sig3(fit['sigma300_mS_cm'])}\n"
                        f"[{_sig3(fit['sigma300_min_mS_cm'])}, {_sig3(fit['sigma300_max_mS_cm'])}] mS cm$^{{-1}}$\n"
                        f"$E_a$ {fit['Ea_eV']:.3f} $\\pm$ {fit['Ea_err_eV']:.3f} eV\n{SOURCE_TEXT[fit['source']]}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=style.SMALL, linespacing=1.3)

    res_ax.axhspan(-1, 1, color=style.LIGHT_GREY, alpha=0.6, lw=0, zorder=0)
    res_ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    res_ax.vlines(points["inv_temperature_1000_per_K"], 0, points["residual_normalised"], color=color, lw=0.6,
                  zorder=2)
    res_ax.plot(points["inv_temperature_1000_per_K"], points["residual_normalised"], "o", color=color,
                markersize=2.6, markeredgewidth=0.0, zorder=3)
    res_ax.set_xlim(*X_LIMITS)
    res_ax.set_ylim(-Z_LIMIT, Z_LIMIT)
    res_ax.set_yticks([-2, 0, 2])
    res_ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5], ["1", "1.5", "2", "2.5", "3", "3.5"])
    res_ax.tick_params(axis="x", pad=1.5)
    res_ax.set_xlabel("1000/T (K$^{-1}$)", labelpad=1.0)
    res_ax.tick_params(labelleft=first_column)
    if first_column:
        res_ax.set_ylabel("Residual\n(s.d.)", labelpad=1.5, linespacing=1.0)

    curve = pd.DataFrame({"series": fit.name, "run_id": "fit", "temperature_K": temperature,
                          "inv_temperature_1000_per_K": x, "fit_sigma_mS_cm": sigma, "band_low_mS_cm": low,
                          "band_high_mS_cm": high})
    return pd.concat([points.assign(kind="run"), curve.assign(kind="fit")], ignore_index=True)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    fits, transport = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)
    left, width, gap = 14.0, 38.5, 4.0
    main_h, res_h, res_gap = 30.0, 8.5, 1.2
    title_h, heading_h, row_gap = 6.5, 5.0, 9.0
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    written = []
    top = 1.0
    for heading, row in ROWS:
        if heading is not None:
            L.text(1.0, top, heading, ha="left", va="top", fontsize=style.FONT_SIZE, fontweight="bold")
            top += heading_h
        panel_top = top + title_h
        for column, series in enumerate(row):
            fit = fits.loc[series]
            x0 = left + column * (width + gap)
            ax = L.axes(x0, panel_top, width, main_h)
            res_ax = L.axes(x0, panel_top + main_h + res_gap, width, res_h)
            frame = _panel(ax, res_ax, fit, _points(transport, fit, series), column == 0)
            name, detail = _title(fit)
            L.text(x0 + 1.0, panel_top - 3.3, name, ha="left", va="bottom", fontsize=style.FONT_SIZE,
                   color=_color(fit), fontweight="bold")
            L.text(x0 + 1.0, panel_top - 0.6, detail, ha="left", va="bottom", fontsize=style.SMALL)
            letter = next(letters)
            L.letter(ax, letter, x0 - 0.8 if column else 4.5, panel_top - 2.6)
            written.append(_save_panel(frame, fit, letter))
        if len(row) < 4:
            _key(L.axes(left + len(row) * (width + gap) + 6.0, panel_top, 2 * width, main_h + res_gap + res_h))
        top = panel_top + main_h + res_gap + res_h + row_gap
    return style.save(fig, "ed03_arrhenius", "extended") + written


def _key(ax) -> None:
    """What the marks of every panel mean (drawn in the empty part of the part C row)."""
    ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="black", markersize=2.8, markeredgewidth=0.0, linestyle="none",
                      label="one run at 600, 800 or 1000 K (pymatgen); bar, kinisi 1-sigma"),
               Line2D([0], [0], color="black", lw=0.8, label="quoted fit of ln($\\sigma$T) against 1/T"),
               Line2D([0], [0], color="black", lw=0.8, ls=(0, (3, 1.5)), label="the same fit extrapolated to 300 K"),
               Patch(facecolor="black", alpha=0.18, edgecolor="none", label="1-sigma band of the fit"),
               Patch(facecolor=style.LIGHT_GREY, alpha=0.6, edgecolor="none",
                     label="residual strips: +/-1 s.d. of the point")]
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.6, handletextpad=0.6,
              labelspacing=0.7, borderaxespad=0.0, title="Every panel", title_fontsize=style.FONT_SIZE,
              alignment="left")


def _subscripts(formula: str) -> str:
    """Chemical formula with its numbers as mathtext subscripts."""
    return re.sub(r"(\d+(?:\.\d+)?)", r"$_{\1}$", formula)


def _save_panel(frame: pd.DataFrame, fit: pd.Series, letter: str) -> Path:
    for column in ("label", "part", "source", "Ea_eV", "Ea_err_eV", "sigma300_mS_cm", "sigma300_min_mS_cm",
                   "sigma300_max_mS_cm", "slope", "intercept"):
        frame[column] = fit[column]
    columns = ["series", "label", "part", "source", "kind", "run_id", "temperature_K", "inv_temperature_1000_per_K",
               "sigma_mS_cm", "kinisi_sigma_std_mS_cm", "fit_sigma_mS_cm", "band_low_mS_cm", "band_high_mS_cm",
               "ln_sigma_error", "residual_normalised", "Ea_eV", "Ea_err_eV", "sigma300_mS_cm", "sigma300_min_mS_cm",
               "sigma300_max_mS_cm", "slope", "intercept"]
    return style.save_source(frame[columns], NAME, letter)


def _caption() -> str:
    fits, transport = _load()
    order = _order()
    residuals = pd.concat([_points(transport, fits.loc[s], s) for s in order], ignore_index=True)
    worst = residuals.loc[residuals["residual_normalised"].abs().idxmax()]
    reanalysed = [fits.loc[s, "label"] for s in order if fits.loc[s, "source"] == "reanalysis"]
    letters = "abcdefghijklmn"
    return (
        "Extended Data Fig. 3 | Arrhenius fits of every series. "
        f"**a-{letters[len(order) - 1]}**, One panel per MD series, grouped by part: **a-h**, the universal potentials on "
        "Li6PS5Cl (part A; MACE-MP-0 top row, MatterSim second row; 52-atom cells run for 50, 100 and 150 ps, which "
        "are independent trajectories, and the 416-atom, 200-ps production runs); **i**, **j**, the two fine-tuned models "
        "under the production protocol (part C); **k-n**, the Cl-excess series Li(6-x)PS(5-x)Cl(1+x) under MACE-MP-0 "
        "with x = 0, 0.25, 0.50 and 0.75 (part D). "
        "Markers, conductivity of each run at 600, 800 and 1000 K from the pymatgen diffusion coefficient (Nernst-"
        "Einstein, no Haven ratio); error bars, the kinisi 1-sigma uncertainty, whose ratio to sigma weights the fit "
        "of ln(sigma T) against 1/T. Line, the quoted fit (solid over the simulated range, dashed where extrapolated "
        "to 300 K) with its 1-sigma band from the fit covariance; the value printed in each panel is sigma(300 K) "
        "with its 1-sigma interval and Ea with its 1-sigma error. The quoted fit is the published one for every "
        "series analysed in July 2026 and the re-analysis otherwise ("
        + "; ".join(reanalysed) + "). "
        "Strips, residual ln sigma minus the quoted fit at each temperature in units of that point's ln-space "
        "uncertainty (kinisi standard deviation over sigma, from the re-analysis also for the published fits); grey "
        f"band, +/-1. With three temperatures and two parameters each fit has one degree of freedom; the largest "
        f"residual is {worst['residual_normalised']:+.2f} ({fits.loc[worst['series'], 'label']}, "
        f"{worst['temperature_K']:.0f} K). Shared logarithmic "
        "conductivity axis in all panels. "
        + common.sources([FITS, TRANSPORT])
    )


CAPTION = _caption()
