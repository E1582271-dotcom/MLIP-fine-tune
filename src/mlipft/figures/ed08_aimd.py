"""The AIMD reference, amendment A3 (Extended Data, registry key "aimd").

a, mean-squared displacement per species of the AIMD of Deng et al. (2017) at 600-1200 K, with the
kinisi fit window (the hero panel); b, AIMD Li D from kinisi and pymatgen against 1000/T, with the
log-log slope of the Li MSD in the fit window; c, the scatter of the AIMD kinisi D over seeds 0-10;
d, why an extrapolated sigma(300 K) is not a usable reference: published AIMD extrapolations for the
ordered cell (black), this study's extrapolations for the two universal potentials, and measured
values of disordered polycrystals as grey context only (separate rows, no ratio; the fine-tuned
models are not in this panel). The ratios of the 52-atom MLIP runs to AIMD are in the sampling
figure (registry key "sampling").
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
from . import common, registry, style

KEY = "aimd"
NAME = registry.number(KEY)
PANELS = "abcd"
WIDTH, HEIGHT = 183.0, 158.0
FOLDER = paths.REFERENCES / "aimd_deng2017"
AIMD_TRANSPORT = FOLDER / "transport.csv"
AIMD_SEEDS = FOLDER / "kinisi_seeds.csv"
AIMD_MSD = FOLDER / "msd.csv.gz"
FITS = paths.RESULTS / "fits.csv"
LITERATURE = paths.REFERENCES / "literature.csv"
MD_TRANSPORT = paths.MD / "transport.csv"
# This study's extrapolations in d: the universal potentials only (series -> model)
UNIVERSAL_SERIES = {"mace_416_200": "foundation", "mattersim_416_200": "mattersim"}
PYMATGEN_OFFSET = 0.035  # 1000/T units: the pymatgen dash is drawn just right of the kinisi square
AIMD_ROWS = (("deng2017_sigma300", "Deng et al. 2017 (quoted by Baktash et al. 2020)"),
             ("deklerk2016_sigma300", "de Klerk et al. 2016 (quoted by Baktash et al. 2020)"),
             ("baktash2020", "Baktash et al. 2020"))
MEASURED_ROWS = (("wang2018", "Wang et al. 2018"), ("li2025_compiled", "compiled range (Li et al. 2025)"))


def _load():
    transport = common.read(AIMD_TRANSPORT)
    seeds = common.read(AIMD_SEEDS)
    msd = common.read(AIMD_MSD)
    fits = common.read(FITS)
    literature = pd.read_csv(LITERATURE, dtype=str, keep_default_na=False)
    md = common.read(MD_TRANSPORT)
    return transport, seeds, msd, fits, literature, md


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


def _panel_a(L: common.Layout, axes, msd: pd.DataFrame, transport: pd.DataFrame, label_top: float) -> None:
    start = _fit_start_ps(transport)
    rows = []
    for index, (ax, temperature) in enumerate(zip(axes, aimd.TEMPERATURES)):
        frame = msd[msd["temperature_K"] == temperature].sort_values("lag_ps")
        ax.axvspan(start[temperature], frame["lag_ps"].max(), **style.BAND_FILL)
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
        ax.set_title(f"{temperature} K", fontsize=style.FONT_SIZE, pad=2.0)
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
            ax.set_ylabel("AIMD MSD (Å$^2$)")
    left, right = axes[0].get_position(), axes[-1].get_position()
    L.text(L.width * (left.x0 + right.x1) / 2, label_top, "Lag time (ps)", ha="center", va="top")
    handles = [Line2D([0], [0], color=style.ELEMENT_COLORS[e], lw=1.0, label=e) for e in common.ELEMENTS]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none", label="kinisi fit window"))
    axes[0].legend(handles=handles, loc="upper left", ncol=2, fontsize=style.SMALL, handlelength=1.1,
                   handletextpad=0.5, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.3)
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, transport: pd.DataFrame) -> None:
    x = 1000.0 / transport["temperature_K"]
    caged = common.caged_temperatures(transport)
    ax.errorbar(x, transport["kinisi_D_cm2_s"], yerr=transport["kinisi_D_std_cm2_s"], fmt="none",
                ecolor=style.AIMD_COLOR, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.plot(x, transport["kinisi_D_cm2_s"], **common.aimd_style(zorder=4))
    pymatgen = {"color": style.AIMD_COLOR, "marker": "_", "markersize": 5.0, "markeredgewidth": 1.0,
                "linestyle": "none"}
    ax.plot(x + PYMATGEN_OFFSET, transport["D_cm2_s"], **pymatgen, zorder=2)
    tops = np.maximum(transport["D_cm2_s"], transport["kinisi_D_cm2_s"] + transport["kinisi_D_std_cm2_s"])
    bottoms = np.minimum(transport["D_cm2_s"], transport["kinisi_D_cm2_s"] - transport["kinisi_D_std_cm2_s"])
    below = transport["temperature_K"].max()  # its label goes under the point, clear of the next label
    for xi, top, bottom, temperature, slope in zip(x, tops, bottoms, transport["temperature_K"],
                                                   transport["msd_loglog_slope_fit_window"]):
        text = f"{temperature} K\nslope {common.sig(slope)}" + ("\ncaged" if temperature in caged else "")
        ax.text(xi, bottom / 1.4 if temperature == below else top * 1.4, text, ha="center",
                va="top" if temperature == below else "bottom", fontsize=style.SMALL, linespacing=1.1,
                color=style.BLACK, style="italic" if temperature in caged else "normal")
    ax.set_yscale("log")
    ax.set_ylim(2e-7, 6e-4)
    style.log_ticks(ax, "y")
    ax.set_xlim(0.66, 1.80)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6], ["0.8", "1.0", "1.2", "1.4", "1.6"])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("AIMD Li D (cm$^2$ s$^{-1}$)")
    handles = [common.handle(common.aimd_style(), "kinisi (seed 0) +/- 1 s.d."), common.handle(pymatgen, "pymatgen")]
    ax.legend(handles=handles, loc="lower left", fontsize=style.SMALL, handletextpad=0.5, borderaxespad=0.3,
              labelspacing=0.3)
    source = transport[["temperature_K", "D_cm2_s", "kinisi_D_cm2_s", "kinisi_D_std_cm2_s",
                        "msd_loglog_slope_fit_window", "length_ps", "a_A"]].copy()
    source.insert(1, "inverse_T_1000_per_K", x)
    source["pymatgen_plot_x"] = x + PYMATGEN_OFFSET
    source["caged"] = source["temperature_K"].isin(caged)
    style.save_source(source, NAME, "b")


def _panel_c(ax, seeds: pd.DataFrame, transport: pd.DataFrame) -> None:
    rows = []
    info = transport.set_index("temperature_K")
    ax.axhline(0, **style.UNITY_LINE)
    others_style = common.aimd_style(size=2.2, edge=0.0, color=style.GRAY, markerfacecolor=style.GRAY,
                                     markeredgecolor=style.GRAY)
    for index, temperature in enumerate(aimd.TEMPERATURES):
        frame = seeds[seeds["temperature_K"] == temperature].sort_values("seed")
        primary = frame.loc[frame["seed"] == 0, "D_cm2_s"].iloc[0]
        others = frame[frame["seed"] != 0]
        deviation = (others["D_cm2_s"] / primary - 1.0) * 100.0
        offsets = common.jitter(len(others), 0.45, seed=index)
        ax.plot(index + offsets, deviation, **others_style, zorder=2)
        ax.plot([index - 0.3, index + 0.3], [0, 0], color=style.AIMD_COLOR, lw=1.4, solid_capstyle="butt", zorder=3)
        scatter = info.loc[temperature, "seed_scatter_D_rel"] * 100.0
        posterior = info.loc[temperature, "kinisi_D_std_cm2_s"] / info.loc[temperature, "kinisi_D_cm2_s"] * 100.0
        ax.text(index, 4.1, f"s$_{{seed}}$ {common.sig(scatter)} %\npost. {common.sig(posterior)} %", ha="center",
                va="bottom", fontsize=style.SMALL, linespacing=1.15)
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
    ax.set_ylabel("kinisi D relative to seed 0 (%)")
    handles = [Line2D([0], [0], color=style.AIMD_COLOR, lw=1.4, label="seed 0 (primary)"),
               common.handle(others_style, "seeds 1-10")]
    ax.legend(handles=handles, loc="lower right", fontsize=style.SMALL, handletextpad=0.6, borderaxespad=0.3,
              labelspacing=0.25, handlelength=1.4)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _strip_entries(fits: pd.DataFrame, literature: pd.DataFrame) -> list[dict]:
    """Rows of panel d: published AIMD extrapolations, this study's universal potentials, measured context."""
    entries = []
    for ref_id, label in AIMD_ROWS:
        centre, low, high, text = _literature_value(_literature(literature, ref_id))
        entries.append({"group": "aimd", "label": label, "centre": centre, "low": low, "high": high, "text": text,
                        "reference": ref_id, "kind": "published AIMD extrapolation, ordered cell"})
    for series, model in UNIVERSAL_SERIES.items():
        fit = _quoted(fits, series)
        text = (f"{common.sig(fit['sigma300_mS_cm'])} [{common.sig(fit['sigma300_min_mS_cm'])}, "
                f"{common.sig(fit['sigma300_max_mS_cm'])}]")
        entries.append({"group": "mlip", "label": style.MODEL_LABELS[model], "centre": fit["sigma300_mS_cm"],
                        "low": fit["sigma300_min_mS_cm"], "high": fit["sigma300_max_mS_cm"], "text": text,
                        "reference": series, "model": model,
                        "kind": f"universal-potential extrapolation, this study ({fit['source']} fit)"})
    for ref_id, label in MEASURED_ROWS:
        centre, low, high, text = _literature_value(_literature(literature, ref_id))
        entries.append({"group": "measured", "label": label, "centre": centre, "low": low, "high": high,
                        "text": text, "reference": ref_id, "kind": "measured, disordered polycrystal (context only)"})
    return entries


def _panel_d(ax, label_ax, value_ax, fits: pd.DataFrame, literature: pd.DataFrame) -> None:
    entries = _strip_entries(fits, literature)
    headers = {"aimd": "Published AIMD extrapolations, ordered cell",
               "mlip": "This study: universal potentials, ordered cell, 416 atoms",
               "measured": "Measured, disordered polycrystals (context)"}
    y, header_positions, previous = 0.0, {}, None
    for entry in entries:
        if entry["group"] != previous:
            if previous is not None:
                y += 0.5
            header_positions[entry["group"]] = y
            y += 1.0
            previous = entry["group"]
        entry["y"] = y
        y += 1.0
    bottom = y - 0.4
    rows = []
    for entry in entries:
        group, yi, low, high, centre = entry["group"], entry["y"], entry["low"], entry["high"], entry["centre"]
        single = low == high
        if group == "mlip":
            marker = common.model_style(entry["model"], size=3.6, zorder=3)
            bar = {"color": style.MODEL_COLORS[entry["model"]], "lw": 0.9, "solid_capstyle": "butt"}
        elif group == "aimd":
            marker = common.aimd_style(size=3.6, zorder=3)
            bar = {"color": style.AIMD_COLOR, "lw": 1.6, "solid_capstyle": "butt"}
        else:
            marker = common.context_style(size=3.8, zorder=3)
            bar = {"color": style.CONTEXT_COLOR, "lw": 1.6, "solid_capstyle": "butt"}
        if not single:
            ax.plot([low, high], [yi, yi], zorder=2, **bar)
        if group == "mlip" or single:
            ax.plot(centre, yi, **marker)
        color = style.CONTEXT_TEXT if group == "measured" else style.BLACK
        label_ax.text(1.0, yi, entry["label"], ha="right", va="center", fontsize=style.SMALL, color=color)
        value_ax.text(0.0, yi, entry["text"], ha="left", va="center", fontsize=style.SMALL, color=color)
        rows.append({"group": entry["kind"], "row": entry["label"], "reference": entry["reference"], "y": yi,
                     "sigma300_mS_cm": centre if (group == "mlip" or single) else np.nan,
                     "sigma300_low_mS_cm": low, "sigma300_high_mS_cm": high})
    for group, yi in header_positions.items():
        label_ax.text(0.0, yi, headers[group], ha="left", va="center", fontsize=style.SMALL, style="italic",
                      color=style.CONTEXT_TEXT if group == "measured" else style.BLACK)
        if yi > 0:
            ax.axhline(yi - 0.75, color=style.LIGHT_GRAY, lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e3)
    style.log_ticks(ax, "x")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], ["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_ylim(bottom, -0.6)
    ax.set_yticks([])
    ax.set_xlabel("$\\sigma$(300 K), Arrhenius extrapolation or measured (mS cm$^{-1}$)")
    for axis in (label_ax, value_ax):
        axis.set_ylim(bottom, -0.6)
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
    value_ax.text(0.0, -0.6, "mS cm$^{-1}$", ha="left", va="bottom", fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    style.save_source(pd.DataFrame(rows), NAME, "d")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    transport, seeds, msd, fits, literature, md = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    left, right, gap, top_a, height_a = 15.0, 181.0, 2.5, 10.0, 34.0
    width = (right - left - 3 * gap) / 4
    lefts = [left + i * (width + gap) for i in range(4)]
    ax_a = [L.axes(x0, top_a, width, height_a, row="a", col=f"a{i}") for i, x0 in enumerate(lefts)]  # the hero
    _panel_a(L, ax_a, msd, transport, top_a + height_a + 5.2)

    top_b, height_b = 63.0, 36.0
    ax_b = L.axes(lefts[0], top_b, 65.0, height_b, row="bc", col="a0")
    _panel_b(ax_b, transport)
    ax_c = L.axes(lefts[2], top_b, right - lefts[2], height_b, row="bc", col="a2")
    _panel_c(ax_c, seeds, transport)

    top_d, height_d = 116.0, 30.0
    ax_d_labels = L.axes(4.0, top_d, lefts[1] - 6.0, height_d, frame="none")
    ax_d = L.axes(lefts[1], top_d, 93.0, height_d, row="d", col="a1")
    ax_d_values = L.axes(lefts[1] + 95.0, top_d, right + 1.0 - lefts[1] - 95.0, height_d, frame="none")
    _panel_d(ax_d, ax_d_labels, ax_d_values, fits, literature)

    L.letter(ax_a[0], "a", 3.0, 7.0)
    L.letter(ax_b, "b", 3.0, 59.0)
    L.letter(ax_c, "c", lefts[2] - 13.0, 59.0)
    L.letter(ax_d, "d", 3.0, 112.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    transport, seeds, msd, fits, literature, md = _load()
    info = transport.set_index("temperature_K")
    a_aimd = float(info["a_A"].iloc[0])
    small_md = md[md["n_atoms"] == 52]
    a_mlip = float(small_md["volume_A3"].iloc[0] ** (1.0 / 3.0))
    volume_change = (a_aimd / a_mlip) ** 3 - 1.0
    start = _fit_start_ps(transport)
    caged = common.caged_temperatures(transport)
    if len(caged) != 1:
        raise ValueError(f"expected one caged AIMD temperature, got {caged}")
    t0 = caged[0]
    m0 = msd[(msd["temperature_K"] == t0) & (msd["lag_ps"] >= start[t0])]
    tails = msd.sort_values("lag_ps").groupby("temperature_K").tail(1).set_index("temperature_K")
    framework_max = {e: float(tails[f"msd_{e}_A2"].max()) for e in ("P", "S", "Cl")}
    slopes = ", ".join(f"{common.sig(info.loc[t, 'msd_loglog_slope_fit_window'])} at {t} K"
                       for t in aimd.TEMPERATURES)
    kinisi = ", ".join(f"{info.loc[t, 'kinisi_D_cm2_s']:.2e} +/- {info.loc[t, 'kinisi_D_std_cm2_s']:.2e}"
                       for t in aimd.TEMPERATURES)
    pymatgen = ", ".join(f"{info.loc[t, 'D_cm2_s']:.2e}" for t in aimd.TEMPERATURES)
    scatter = ", ".join(common.sig(100 * info.loc[t, "seed_scatter_D_rel"]) for t in aimd.TEMPERATURES)
    share = info["seed_scatter_D_rel"] / (info["kinisi_D_std_cm2_s"] / info["kinisi_D_cm2_s"])
    seeds_dev = seeds.assign(dev=seeds.groupby("temperature_K")["D_cm2_s"].transform(lambda d: d / d.iloc[0] - 1))
    n_seeds = int((seeds_dev["seed"] != 0).groupby(seeds_dev["temperature_K"]).sum().min())
    aimd_values = [_literature_value(_literature(literature, r)) for r, _ in AIMD_ROWS]
    aimd_low = min(v[1] for v in aimd_values)
    aimd_high = max(v[2] for v in aimd_values)
    universal = {s: _quoted(fits, s) for s in UNIVERSAL_SERIES}
    mlip_text = " and ".join(f"{style.MODEL_LABELS[m]} {common.sig(universal[s]['sigma300_mS_cm'])} "
                             f"[{common.sig(universal[s]['sigma300_min_mS_cm'])}, "
                             f"{common.sig(universal[s]['sigma300_max_mS_cm'])}]"
                             for s, m in UNIVERSAL_SERIES.items())
    decades = np.log10(aimd_high / aimd_low)
    below_text = f"{transport['temperature_K'].max():.0f}"
    return (
        registry.heading(KEY) + "Li transport of every model is compared with ab initio MD (AIMD) "
        "of the same ordered Li6PS5Cl cell (Li24P4S20Cl4, Cl on 4a, S on 4c; amendment A3): the trajectories of "
        "Deng et al. (Chem. Mater. 29, 281-288, 2017), published with the paper's notebooks (repository "
        f"{aimd.REPOSITORY}, commit {aimd.COMMIT[:12]}, BSD-3-Clause license, Copyright (c) 2017 Materials Virtual "
        f"Lab) and re-analysed here with the estimators and settings of the MLIP runs. From the files: VASP, 52 "
        f"atoms, time step {int(info['time_step_fs'].iloc[0])} fs, every {int(info['step_skip'].iloc[0])}th step "
        f"stored, {info['length_ps'].iloc[0]:.0f} ps at each of "
        f"{', '.join(str(t) for t in aimd.TEMPERATURES[:-1])} and {aimd.TEMPERATURES[-1]} K; the exchange-"
        "correlation functional and the thermostat are not recorded. Differences from the MLIP runs: lattice "
        f"parameter {a_aimd:.2f} A against {a_mlip:.2f} A ({common.sig(-100 * volume_change)} % smaller volume), "
        "52 atoms against 416 for the production runs (the 52-atom MLIP runs match the AIMD cell; their ratios to "
        f"AIMD are in {registry.label('sampling')}), one trajectory per temperature in both. The comparison of the "
        f"four production models with this reference is {registry.label('diffusion')}. "
        f"**a**, Mean-squared displacement of Li, P, S and Cl from the single trajectory at each temperature "
        f"(n = 1; averaged over all time origins, centre-of-mass drift removed; one curve per trajectory, no error "
        f"bars); grey, the kinisi "
        f"fit window (lags from {100 * C.KINISI_START_FRACTION:.0f} % of the run). The framework stays in place: "
        f"at the longest lag P, S and Cl reach at most {common.sig(framework_max['P'])}, "
        f"{common.sig(framework_max['S'])} and {common.sig(framework_max['Cl'])} A^2 at any temperature. At {t0} K "
        f"the Li MSD is caged at {m0['msd_Li_A2'].min():.1f}-{m0['msd_Li_A2'].max():.1f} A^2 across the fit window. "
        f"**b**, AIMD Li D against 1000/T, one trajectory per temperature: kinisi (seed 0; squares; error bars, the "
        f"posterior standard deviation; {kinisi} cm^2/s) and pymatgen (dashes, drawn just right of the "
        f"squares; no error estimate; {pymatgen} cm^2/s), in the order {', '.join(map(str, aimd.TEMPERATURES))} K. Labels (above each point, below it at "
        f"{below_text} K) give the temperature and the log-log slope of the Li MSD in the fit window ({slopes}); a slope near 1 is diffusive. At {t0} K (slope "
        f"{common.sig(info.loc[t0, 'msd_loglog_slope_fit_window'])}, italic label 'caged') neither estimator "
        "measures a diffusion coefficient, so a 600 K ratio to AIMD is not a ratio of diffusion coefficients; the "
        "1200 K run has no MLIP counterpart. "
        f"**c**, kinisi D for seeds 1-10 (n = {n_seeds} per temperature; grey squares, jittered horizontally; "
        f"single values, no error bars) "
        f"relative to seed 0 (black bar); they span {100 * seeds_dev[seeds_dev['seed'] != 0]['dev'].min():+.2f} to "
        f"{100 * seeds_dev[seeds_dev['seed'] != 0]['dev'].max():+.2f} %. s_seed, their relative standard deviation "
        f"({scatter} %), is {common.sig(100 * share.min())}-{common.sig(100 * share.max())} % of the posterior "
        "width (post., the relative posterior standard deviation of seed 0). "
        f"**d**, Extrapolated sigma(300 K) of the ordered cell (logarithmic axis). Published AIMD extrapolations "
        f"(black; squares, single values; bar, a published range) span {common.sig(decades)} orders of magnitude "
        f"({_literature(literature, 'deng2017_sigma300')['value']} to "
        f"{_literature(literature, 'deklerk2016_sigma300')['value']} mS/cm; the first two as quoted in Table 2 of "
        "Baktash et al., npj Comput. Mater. 6, 162, 2020). This study's extrapolations for the two universal "
        "potentials (416-atom production runs, quoted fits over n = 3 temperatures; bars, the 1-sigma interval of "
        f"the fit) give {mlip_text} mS/cm. An extrapolated value is therefore not a usable reference, and models are "
        "compared through D at the simulated temperatures. Grey rows: measured room-temperature conductivities of "
        "S/Cl-disordered polycrystals (Wang et al. 2018, diamond; range compiled by Li et al. 2025, bar), shown "
        "only as context on the same axis; no ratio is formed with them, and the ordered cell is about two orders "
        "of magnitude less conductive than the disordered one in AIMD (Baktash et al. 2020). Literature values are "
        "printed as given in the sources. "
        + common.sources([AIMD_TRANSPORT, AIMD_SEEDS, AIMD_MSD, FITS, LITERATURE, MD_TRANSPORT])
    )


CAPTION = _caption()
