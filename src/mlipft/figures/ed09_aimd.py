"""Extended Data Fig. 9: the AIMD reference (amendment A3).

a, mean-squared displacement per species of the AIMD of Deng et al. (2017) at 600-1200 K, with the
kinisi fit window; b, AIMD Li D from pymatgen and kinisi against 1000/T, with the log-log slope of
the Li MSD in the fit window; c, the scatter of the AIMD kinisi D over seeds 0-10; d, kinisi D of the
52-atom MLIP runs, which share the AIMD cell size, relative to AIMD at 600, 800 and 1000 K; e, why an
extrapolated sigma(300 K) is not a usable reference: published AIMD extrapolations for the ordered
cell, this study's MLIP extrapolations, and measured values of disordered polycrystals as context
only (separate rows, never a ratio with a fine-tuned model).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import constants as C
from .. import paths
from ..ingest import aimd
from . import common, style

NAME = "ed09"
WIDTH, HEIGHT = 183.0, 160.0
FOLDER = paths.REFERENCES / "aimd_deng2017"
AIMD_TRANSPORT = FOLDER / "transport.csv"
AIMD_SEEDS = FOLDER / "kinisi_seeds.csv"
AIMD_MSD = FOLDER / "msd.csv.gz"
COMPARISON = paths.RESULTS / "aimd_comparison.csv"
FITS = paths.RESULTS / "fits.csv"
LITERATURE = paths.REFERENCES / "literature.csv"
MD_TRANSPORT = paths.MD / "transport.csv"
AIMD_COLOR = style.OKABE_ITO["black"]
CONTEXT_COLOR = style.GREY
SMALL_SERIES = ("mace_52_050", "mace_52_100", "mace_52_150", "mattersim_52_050", "mattersim_52_100",
                "mattersim_52_150")
SMALL_MODELS = {"mace": "foundation", "mattersim": "mattersim"}
TIER_MARKERS = {50: "o", 100: "s", 150: "^"}
PRODUCTION_LABELS = {"mace_416_200": "MACE-MP-0", "mattersim_416_200": "MatterSim",
                     "ftG_416_200": "Fine-tuned, Gamma labels", "ftK_416_200": "Fine-tuned, 2x2x2 labels"}
PRODUCTION_MODELS = {"mace_416_200": "foundation", "mattersim_416_200": "mattersim", "ftG_416_200": "ftG",
                     "ftK_416_200": "ftK"}
AIMD_ROWS = (("deng2017_sigma300", "Deng et al. 2017, as quoted by Baktash et al. 2020"),
             ("deklerk2016_sigma300", "de Klerk et al. 2016"),
             ("baktash2020", "Baktash et al. 2020"))
MEASURED_ROWS = (("wang2018", "Wang et al. 2018"), ("li2025_compiled", "compiled range (Li et al. 2025)"))


def _load():
    transport = common.read(AIMD_TRANSPORT)
    seeds = common.read(AIMD_SEEDS)
    msd = common.read(AIMD_MSD)
    comparison = common.read(COMPARISON)
    fits = common.read(FITS)
    literature = pd.read_csv(LITERATURE, dtype=str, keep_default_na=False)
    md = common.read(MD_TRANSPORT)
    return transport, seeds, msd, comparison, fits, literature, md


def _sig3(value: float) -> str:
    """Three significant figures with trailing zeros kept (0.260, 5.55, 290)."""
    text = f"{value:#.3g}"
    return text[:-1] if text.endswith(".") else text


def _fit_start_ps(transport: pd.DataFrame) -> pd.Series:
    """Start of the kinisi fit window (lag, ps) per temperature: the MLIP setting, 30 % of the run."""
    return transport.set_index("temperature_K")["length_ps"] * C.KINISI_START_FRACTION


def _quoted(fits: pd.DataFrame, series: str) -> pd.Series:
    rows = fits[(fits["series"] == series) & (fits["quoted"].astype(str).str.lower() == "true")]
    if len(rows) != 1:
        raise ValueError(f"expected one quoted fit for {series}")
    return rows.iloc[0]


def _literature(literature: pd.DataFrame, ref_id: str) -> pd.Series:
    rows = literature[literature["ref_id"] == ref_id]
    if len(rows) != 1:
        raise ValueError(f"expected one literature row {ref_id}")
    return rows.iloc[0]


def _literature_value(row: pd.Series) -> tuple[float, float, float, str]:
    """(value or geometric centre, low, high, text as written in the table)."""
    if row["value"]:
        value = float(row["value"])
        return value, value, value, row["value"]
    low, high = float(row["value_low"]), float(row["value_high"])
    return float(np.sqrt(low * high)), low, high, f"{row['value_low']}-{row['value_high']}"


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(axes, msd: pd.DataFrame, transport: pd.DataFrame) -> None:
    start = _fit_start_ps(transport)
    rows = []
    for index, (ax, temperature) in enumerate(zip(axes, aimd.TEMPERATURES)):
        frame = msd[msd["temperature_K"] == temperature].sort_values("lag_ps")
        ax.axvspan(start[temperature], frame["lag_ps"].max(), color=style.LIGHT_GREY, lw=0, zorder=0)
        for element in common.ELEMENTS:
            ax.plot(frame["lag_ps"], frame[f"msd_{element}_A2"], color=style.ELEMENT_COLORS[element], lw=0.9,
                    zorder=3 if element == "Li" else 2)
        rows.extend({"temperature_K": temperature, "lag_ps": lag, "msd_Li_A2": li, "msd_P_A2": p, "msd_S_A2": s,
                     "msd_Cl_A2": cl, "in_kinisi_fit_window": lag >= start[temperature]}
                    for lag, li, p, s, cl in zip(frame["lag_ps"], frame["msd_Li_A2"], frame["msd_P_A2"],
                                                 frame["msd_S_A2"], frame["msd_Cl_A2"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.015, 130)
        ax.set_ylim(0.008, 800)
        style.log_ticks(ax, "x")
        style.log_ticks(ax, "y")
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_title(f"{temperature} K", fontsize=style.SMALL, pad=2.0)
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
            ax.set_ylabel("AIMD MSD (Å$^2$)")
    left, right = axes[1].get_position(), axes[2].get_position()
    axes[0].figure.text((left.x1 + right.x0) / 2, left.y0 - 0.036, "Lag time (ps)", ha="center", va="top",
                        fontsize=style.FONT_SIZE)
    handles = [Line2D([0], [0], color=style.ELEMENT_COLORS[e], lw=1.0, label=e) for e in common.ELEMENTS]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=style.LIGHT_GREY, edgecolor="none", label="kinisi fit"))
    axes[0].legend(handles=handles, loc="upper left", ncol=2, fontsize=style.SMALL, handlelength=1.1,
                   handletextpad=0.3, columnspacing=0.5, labelspacing=0.25, borderaxespad=0.2)
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, transport: pd.DataFrame) -> None:
    x = 1000.0 / transport["temperature_K"]
    ax.errorbar(x, transport["kinisi_D_cm2_s"], yerr=transport["kinisi_D_std_cm2_s"], fmt="o", color=AIMD_COLOR,
                markersize=3.4, elinewidth=0.7, capsize=1.5, zorder=3, label="kinisi (seed 0) ± 1 s.d.")
    ax.plot(x, transport["D_cm2_s"], marker="s", linestyle="none", markersize=3.4, markeredgewidth=0.7,
            markeredgecolor=AIMD_COLOR, markerfacecolor="white", zorder=2, label="pymatgen")
    tops = np.maximum(transport["D_cm2_s"], transport["kinisi_D_cm2_s"] + transport["kinisi_D_std_cm2_s"])
    for xi, top, temperature, slope in zip(x, tops, transport["temperature_K"],
                                           transport["msd_loglog_slope_fit_window"]):
        ax.text(xi, top * 1.35, f"{temperature} K\nslope {_sig3(slope)}", ha="center", va="bottom",
                fontsize=style.SMALL, linespacing=1.1, color="black" if slope > 0.5 else style.OKABE_ITO["vermillion"])
    ax.set_yscale("log")
    ax.set_ylim(2e-7, 4e-4)
    style.log_ticks(ax, "y")
    ax.set_xlim(0.64, 1.82)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6], ["0.8", "1.0", "1.2", "1.4", "1.6"])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("AIMD Li D (cm$^2$ s$^{-1}$)")
    ax.legend(loc="lower left", fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    source = transport[["temperature_K", "D_cm2_s", "kinisi_D_cm2_s", "kinisi_D_std_cm2_s",
                        "msd_loglog_slope_fit_window", "length_ps", "a_A"]].copy()
    source.insert(1, "inverse_T_1000_per_K", x)
    style.save_source(source, NAME, "b")


def _panel_c(ax, seeds: pd.DataFrame, transport: pd.DataFrame) -> None:
    rows = []
    info = transport.set_index("temperature_K")
    ax.axhline(0, color=AIMD_COLOR, lw=0.6, zorder=1)
    for index, temperature in enumerate(aimd.TEMPERATURES):
        frame = seeds[seeds["temperature_K"] == temperature].sort_values("seed")
        primary = frame.loc[frame["seed"] == 0, "D_cm2_s"].iloc[0]
        others = frame[frame["seed"] != 0]
        deviation = (others["D_cm2_s"] / primary - 1.0) * 100.0
        offsets = common.jitter(len(others), 0.45, seed=index)
        ax.scatter(index + offsets, deviation, s=5, facecolors=style.GREY, edgecolors="none", zorder=2)
        ax.plot([index - 0.3, index + 0.3], [0, 0], color=AIMD_COLOR, lw=1.4, solid_capstyle="butt", zorder=3)
        scatter = info.loc[temperature, "seed_scatter_D_rel"] * 100.0
        posterior = info.loc[temperature, "kinisi_D_std_cm2_s"] / info.loc[temperature, "kinisi_D_cm2_s"] * 100.0
        ax.text(index, 4.1, f"s$_{{seed}}$ {_sig3(scatter)} %\npost. {_sig3(posterior)} %", ha="center", va="bottom",
                fontsize=style.SMALL, linespacing=1.15)
        rows.append({"temperature_K": temperature, "seed": 0, "x": float(index), "D_cm2_s": primary,
                     "deviation_from_seed0_percent": 0.0, "s_seed_percent_seeds_1_10": scatter,
                     "posterior_rel_std_percent": posterior})
        rows.extend({"temperature_K": temperature, "seed": int(s), "x": index + o, "D_cm2_s": d,
                     "deviation_from_seed0_percent": v, "s_seed_percent_seeds_1_10": scatter,
                     "posterior_rel_std_percent": posterior}
                    for s, o, d, v in zip(others["seed"], offsets, others["D_cm2_s"], deviation))
    ax.set_xlim(-0.6, len(aimd.TEMPERATURES) - 0.4)
    ax.set_xticks(range(len(aimd.TEMPERATURES)), [f"{t} K" for t in aimd.TEMPERATURES])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylim(-3.5, 7.0)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_ylabel("kinisi D vs seed 0 (%)")
    handles = [Line2D([0], [0], color=AIMD_COLOR, lw=1.4, label="seed 0 (primary)"),
               Line2D([0], [0], marker="o", linestyle="none", markersize=2.2, markeredgewidth=0,
                      color=style.GREY, label="seeds 1-10")]
    ax.legend(handles=handles, loc="lower right", fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.2,
              labelspacing=0.25, handlelength=1.2)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, comparison: pd.DataFrame, transport: pd.DataFrame) -> None:
    temperatures = [t for t in C.TEMPERATURES if t in set(comparison["temperature_K"])]
    slope = transport.set_index("temperature_K")["msd_loglog_slope_fit_window"]
    ax.axvspan(-0.5, 0.5, color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.text(0.0, 0.97, f"AIMD sub-diffusive\n(MSD slope {_sig3(slope[temperatures[0]])})",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=style.SMALL, linespacing=1.1)
    ax.axhline(1.0, color=AIMD_COLOR, lw=0.7, zorder=1)
    offsets = dict(zip(SMALL_SERIES, (-0.33, -0.22, -0.11, 0.11, 0.22, 0.33)))
    rows = []
    for series in SMALL_SERIES:
        model = SMALL_MODELS[series.split("_")[0]]
        tier = int(C.SERIES_BY_ID[series]["production_ps"])
        part = comparison[comparison["series"] == series].set_index("temperature_K").loc[temperatures]
        x = np.arange(len(temperatures)) + offsets[series]
        ratio = part["kinisi_D_ratio_to_aimd"].to_numpy()
        rel = part["kinisi_D_ratio_rel_err"].to_numpy()
        color = style.MODEL_COLORS[model]
        ax.errorbar(x, ratio, yerr=ratio * rel, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.0,
                    capthick=0.6, zorder=2)
        ax.plot(x, ratio, marker=TIER_MARKERS[tier], linestyle="none", markersize=3.3, markeredgewidth=0.6,
                markeredgecolor=color, markerfacecolor=color, zorder=3)
        rows.extend({"series": series, "model": model, "production_ps": tier, "temperature_K": t, "x": xi,
                     "kinisi_D_ratio_to_aimd": r, "kinisi_D_ratio_rel_err": e, "pymatgen_D_ratio_to_aimd": p,
                     "mlip_msd_loglog_slope": m, "aimd_msd_loglog_slope": a}
                    for t, xi, r, e, p, m, a in zip(temperatures, x, ratio, rel, part["pymatgen_D_ratio_to_aimd"],
                                                    part["msd_loglog_slope"], part["aimd_msd_loglog_slope"]))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 40)
    style.log_ticks(ax, "y")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(-0.5, len(temperatures) - 0.5)
    ax.set_xticks(range(len(temperatures)), [f"{t} K" for t in temperatures])
    ax.tick_params(axis="x", length=0, pad=2.5)
    for index in range(1, len(temperatures)):
        ax.axvline(index - 0.5, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    ax.set_ylabel("kinisi D, 52-atom MLIP run / AIMD")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.3, markeredgewidth=0,
                      color=style.MODEL_COLORS[m], label=style.MODEL_LABELS[m]) for m in SMALL_MODELS.values()]
    handles += [Line2D([0], [0], marker=TIER_MARKERS[t], linestyle="none", markersize=3.3, markeredgewidth=0.6,
                       markeredgecolor="black", markerfacecolor="black", label=f"{t} ps")
                for t in TIER_MARKERS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=style.SMALL,
              handletextpad=0.3, borderaxespad=0.0, labelspacing=0.35, title="52 atoms",
              title_fontsize=style.SMALL, alignment="left")
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _panel_e(ax, label_ax, value_ax, fits: pd.DataFrame, literature: pd.DataFrame) -> None:
    entries = []  # (group, label, centre, low, high, text, colour, marker, kind)
    for ref_id, label in AIMD_ROWS:
        centre, low, high, text = _literature_value(_literature(literature, ref_id))
        entries.append(("aimd", label, centre, low, high, text, AIMD_COLOR, "o", "published AIMD extrapolation",
                        ref_id))
    for series, label in PRODUCTION_LABELS.items():
        fit = _quoted(fits, series)
        text = (f"{_sig3(fit['sigma300_mS_cm'])} [{_sig3(fit['sigma300_min_mS_cm'])}, "
                f"{_sig3(fit['sigma300_max_mS_cm'])}]")
        entries.append(("mlip", label, fit["sigma300_mS_cm"], fit["sigma300_min_mS_cm"], fit["sigma300_max_mS_cm"],
                        text, style.MODEL_COLORS[PRODUCTION_MODELS[series]], "o",
                        f"MLIP extrapolation, this study ({fit['source']})", series))
    for ref_id, label in MEASURED_ROWS:
        centre, low, high, text = _literature_value(_literature(literature, ref_id))
        entries.append(("measured", label, centre, low, high, text, CONTEXT_COLOR, "D",
                        "measured, disordered polycrystal (context only)", ref_id))
    headers = {"aimd": "Published AIMD extrapolations, ordered cell",
               "mlip": "This study: MLIP extrapolations, ordered cell, 416 atoms",
               "measured": "Measured, disordered polycrystals (context)"}
    y, positions, header_positions, previous = 0.0, [], {}, None
    for entry in entries:
        if entry[0] != previous:
            if previous is not None:
                y += 0.5
            header_positions[entry[0]] = y
            y += 1.0
            previous = entry[0]
        positions.append(y)
        y += 1.0
    bottom = y - 0.4
    lo, hi = 1e-3, 1e3
    rows = []
    for entry, yi in zip(entries, positions):
        group, label, centre, low, high, text, color, marker, kind, ref = entry
        if high > low:
            ax.plot([low, high], [yi, yi], color=color, lw=1.6 if group != "mlip" else 0.9,
                    solid_capstyle="butt" if group != "mlip" else "round", zorder=2)
        if group == "mlip" or low == high:
            ax.plot(centre, yi, marker=marker, linestyle="none", markersize=3.6, markeredgewidth=0.7,
                    markeredgecolor=color, markerfacecolor=color, zorder=3)
        label_ax.text(1.0, yi, label, ha="right", va="center", fontsize=style.SMALL, linespacing=1.05,
                      color="black" if group != "measured" else CONTEXT_COLOR)
        value_ax.text(0.0, yi, text, ha="left", va="center", fontsize=style.SMALL,
                      color="black" if group != "measured" else CONTEXT_COLOR)
        rows.append({"group": kind, "row": label.replace("\n", " "), "reference": ref, "y": yi,
                     "sigma300_mS_cm": centre if (group == "mlip" or low == high) else np.nan,
                     "sigma300_low_mS_cm": low, "sigma300_high_mS_cm": high})
    for group, yi in header_positions.items():
        label_ax.text(0.0, yi, headers[group], ha="left", va="center", fontsize=style.SMALL, style="italic",
                      color="black" if group != "measured" else CONTEXT_COLOR)
        if yi > 0:
            ax.axhline(yi - 0.75, color=style.LIGHT_GREY, lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    style.log_ticks(ax, "x")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], ["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_ylim(bottom, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("σ(300 K), Arrhenius extrapolation or measured (mS cm$^{-1}$)")
    for axis in (label_ax, value_ax):
        axis.set_ylim(bottom, -0.6)
        axis.set_xlim(0, 1)
        axis.axis("off")
    value_ax.text(0.0, -0.6, "mS cm$^{-1}$", ha="left", va="bottom", fontsize=style.SMALL, color=style.GREY)
    style.save_source(pd.DataFrame(rows), NAME, "e")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    transport, seeds, msd, comparison, fits, literature, md = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    width, gap = 23.0, 2.5
    ax_a = [L.axes(15.0 + i * (width + gap), 10.0, width, 33.0) for i in range(4)]
    _panel_a(ax_a, msd, transport)
    ax_b = L.axes(136.0, 10.0, 45.0, 33.0)
    _panel_b(ax_b, transport)

    ax_c = L.axes(15.0, 62.0, 56.0, 35.0)
    _panel_c(ax_c, seeds, transport)
    ax_d = L.axes(88.0, 62.0, 66.0, 35.0)
    _panel_d(ax_d, comparison, transport)

    ax_e_labels = L.axes(4.0, 113.0, 60.0, 37.0)
    ax_e = L.axes(66.0, 113.0, 88.0, 37.0)
    ax_e_values = L.axes(156.0, 113.0, 26.0, 37.0)
    _panel_e(ax_e, ax_e_labels, ax_e_values, fits, literature)

    L.letter(ax_a[0], "a", 3.0, 7.0)
    L.letter(ax_b, "b", 124.0, 7.0)
    L.letter(ax_c, "c", 3.0, 58.0)
    L.letter(ax_d, "d", 76.0, 58.0)
    L.letter(ax_e, "e", 3.0, 109.0)
    return style.save(fig, "ed09_aimd", "extended")


def _caption() -> str:
    transport, seeds, msd, comparison, fits, literature, md = _load()
    info = transport.set_index("temperature_K")
    a_aimd = float(info["a_A"].iloc[0])
    small_md = md[md["n_atoms"] == 52]
    a_mlip = float(small_md["volume_A3"].iloc[0] ** (1.0 / 3.0))
    volume_change = (a_aimd / a_mlip) ** 3 - 1.0
    start = _fit_start_ps(transport)
    t0 = aimd.TEMPERATURES[0]
    m0 = msd[(msd["temperature_K"] == t0) & (msd["lag_ps"] >= start[t0])]
    tails = msd.sort_values("lag_ps").groupby("temperature_K").tail(1).set_index("temperature_K")
    framework_max = {e: float(tails[f"msd_{e}_A2"].max()) for e in ("P", "S", "Cl")}
    slopes = ", ".join(f"{_sig3(info.loc[t, 'msd_loglog_slope_fit_window'])} at {t} K" for t in aimd.TEMPERATURES)
    kinisi = ", ".join(f"{info.loc[t, 'kinisi_D_cm2_s']:.2e} +/- {info.loc[t, 'kinisi_D_std_cm2_s']:.2e}"
                       for t in aimd.TEMPERATURES)
    pymatgen = ", ".join(f"{info.loc[t, 'D_cm2_s']:.2e}" for t in aimd.TEMPERATURES)
    scatter = ", ".join(_sig3(100 * info.loc[t, "seed_scatter_D_rel"]) for t in aimd.TEMPERATURES)
    share = info["seed_scatter_D_rel"] / (info["kinisi_D_std_cm2_s"] / info["kinisi_D_cm2_s"])
    seeds_dev = seeds.assign(dev=seeds.groupby("temperature_K")["D_cm2_s"].transform(lambda d: d / d.iloc[0] - 1))
    small = comparison[comparison["series"].isin(SMALL_SERIES)]
    ranges = small.groupby("temperature_K")["kinisi_D_ratio_to_aimd"].agg(["min", "max"])
    by_model = {m: small[small["series"].str.startswith(m + "_")].groupby("temperature_K")["kinisi_D_ratio_to_aimd"]
                .agg(["min", "max"]) for m in SMALL_MODELS}
    d_text = "; ".join(
        f"{t} K: MACE-MP-0 {_sig3(by_model['mace'].loc[t, 'min'])}-{_sig3(by_model['mace'].loc[t, 'max'])}, "
        f"MatterSim {_sig3(by_model['mattersim'].loc[t, 'min'])}-{_sig3(by_model['mattersim'].loc[t, 'max'])}"
        for t in ranges.index)
    production = comparison[comparison["series"].isin(PRODUCTION_LABELS)]
    prod_text = "; ".join(
        f"{PRODUCTION_LABELS[s]} " + ", ".join(
            _sig3(production[(production["series"] == s) & (production["temperature_K"] == t)]
                  ["kinisi_D_ratio_to_aimd"].iloc[0])
            for t in C.TEMPERATURES) for s in PRODUCTION_LABELS)
    aimd_values = [_literature_value(_literature(literature, r)) for r, _ in AIMD_ROWS]
    aimd_low = min(v[1] for v in aimd_values)
    aimd_high = max(v[2] for v in aimd_values)
    mlip = {s: _quoted(fits, s) for s in PRODUCTION_LABELS}
    mlip_low = min(f["sigma300_mS_cm"] for f in mlip.values())
    mlip_high = max(f["sigma300_mS_cm"] for f in mlip.values())
    decades = np.log10(aimd_high / aimd_low)
    return (
        "Extended Data Fig. 9 | The AIMD reference. Li transport of every model is compared with ab initio MD (AIMD) "
        "of the same ordered Li6PS5Cl cell (Li24P4S20Cl4, Cl on 4a, S on 4c; amendment A3): the trajectories of "
        "Deng et al. (Chem. Mater. 29, 281-288, 2017), published with the paper's notebooks (repository "
        f"{aimd.REPOSITORY}, commit {aimd.COMMIT[:12]}, BSD-3-Clause license, Copyright (c) 2017 Materials Virtual "
        f"Lab) and re-analysed here with the estimators and settings of the MLIP runs. From the files: VASP, 52 "
        f"atoms, time step {int(info['time_step_fs'].iloc[0])} fs, every {int(info['step_skip'].iloc[0])}th step "
        f"stored, {info['length_ps'].iloc[0]:.0f} ps at each of "
        f"{', '.join(str(t) for t in aimd.TEMPERATURES[:-1])} and {aimd.TEMPERATURES[-1]} K; the exchange-"
        "correlation functional and the thermostat are not recorded. Differences from the MLIP runs: lattice "
        f"parameter {a_aimd:.2f} A against {a_mlip:.2f} A ({_sig3(-100 * volume_change)} % smaller volume), 52 atoms "
        "against "
        "416 for the production runs (the 52-atom MLIP runs match the AIMD cell), one trajectory per temperature in "
        "both. "
        f"**a**, Mean-squared displacement of Li, P, S and Cl (all time origins, centre-of-mass drift removed); grey, "
        f"the kinisi fit window (lags from {100 * C.KINISI_START_FRACTION:.0f} % of the run). The framework stays in "
        f"place: at the longest lag P, S and Cl reach at most {_sig3(framework_max['P'])}, "
        f"{_sig3(framework_max['S'])} and {_sig3(framework_max['Cl'])} A^2 at any temperature. At {t0} K the Li MSD is caged at "
        f"{m0['msd_Li_A2'].min():.1f}-{m0['msd_Li_A2'].max():.1f} A^2 across the fit window. "
        f"**b**, AIMD Li D against 1000/T: kinisi (seed 0, filled; error bars, posterior standard deviation; "
        f"{kinisi} cm^2/s) and pymatgen (open; {pymatgen} cm^2/s), in the order {', '.join(map(str, aimd.TEMPERATURES))} "
        f"K. Labels give the log-log slope of the Li MSD in the fit window ({slopes}); a slope near 1 is diffusive. "
        f"At {t0} K (slope {_sig3(info.loc[t0, 'msd_loglog_slope_fit_window'])}, red label) neither estimator measures "
        "a diffusion coefficient, so no 600 K ratio below is a ratio of diffusion coefficients, and the 1200 K run "
        "has no MLIP counterpart. "
        f"**c**, kinisi D for seeds 1-10 relative to seed 0 (seeds 1-10 span "
        f"{100 * seeds_dev[seeds_dev['seed'] != 0]['dev'].min():+.2f} to "
        f"{100 * seeds_dev[seeds_dev['seed'] != 0]['dev'].max():+.2f} %). s_seed, their relative standard deviation "
        f"({scatter} %), is {_sig3(100 * share.min())}-{_sig3(100 * share.max())} % of the posterior width (post.). "
        "**d**, kinisi D of the 52-atom MLIP runs (50, 100 and 150 ps) divided by that of AIMD at the same "
        "temperature; error bars, the two relative standard deviations added in quadrature; logarithmic axis. "
        f"{d_text}. The 416-atom production runs give {prod_text} at 600, 800 and 1000 K. "
        f"**e**, Extrapolated sigma(300 K) of the ordered cell. Published AIMD extrapolations span {_sig3(decades)} "
        f"orders of magnitude ({_literature(literature, 'deng2017_sigma300')['value']} to "
        f"{_literature(literature, 'deklerk2016_sigma300')['value']} mS/cm; the first two as quoted in Table 2 of "
        f"Baktash et al., npj Comput. Mater. 6, 162, 2020), and this study's MLIP extrapolations of the 416-atom "
        f"production runs (quoted fits; bars, 1-sigma interval) give {_sig3(mlip_low)}-{_sig3(mlip_high)} mS/cm. An "
        "extrapolated value is therefore not a usable reference, and models are compared through D at the simulated "
        "temperatures (b-d). Grey rows: measured room-temperature conductivities of S/Cl-disordered polycrystals "
        "(Wang et al. 2018; range compiled by Li et al. 2025), shown only as context on the same axis; no ratio is "
        "formed with them, and the ordered cell is about two orders of magnitude less conductive than the "
        "disordered one in AIMD (Baktash et al. 2020). Literature values are printed as given in the sources. "
        + common.sources([AIMD_TRANSPORT, AIMD_SEEDS, AIMD_MSD, COMPARISON, FITS, LITERATURE, MD_TRANSPORT])
    )


CAPTION = _caption()
