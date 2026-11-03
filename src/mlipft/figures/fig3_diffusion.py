"""Li diffusion of the four MLIPs against AIMD of the same ordered cell (amendment A3).

a (hero), kinisi Li D against 1000/T of the 416-atom, 200-ps production runs of MACE-MP-0, MatterSim and
the two fine-tunes, with AIMD (Deng et al. 2017) at 600-1200 K; b, kinisi D of each run divided by the
AIMD value at 600, 800 and 1000 K, with the pymatgen ratio; c, Li mean-squared displacement at 600 K and
1000 K with the two fit windows; d, Li force softening slope on H3 (against the 2x2x2 labels) of the three
MACE-family MD models against their D ratio to AIMD at 1000 K (n = 3). The 52-atom runs and the second
416-atom MACE-MP-0 run are in the sampling Extended Data figure; no measured value appears here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import constants as C
from .. import paths
from . import common, registry, style

KEY = "diffusion"
NAME = registry.number(KEY)
PANELS = "abcd"
WIDTH, HEIGHT = 183.0, 145.0

AIMD_COMPARISON = paths.RESULTS / "aimd_comparison.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
TRANSPORT = paths.MD / "transport.csv"
MSD = paths.MD / "msd.csv.gz"
AIMD = paths.REFERENCES / "aimd_deng2017" / "transport.csv"
AIMD_MSD = paths.REFERENCES / "aimd_deng2017" / "msd.csv.gz"

FAMILIES = ("foundation", "mattersim", "ftG", "ftK")  # run families in drawing order
CAPTION_NAMES = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG": "Gamma fine-tune",
                 "ftK": "2x2x2 fine-tune"}
# Panel a: horizontal offsets (1000/T units) of the two fine-tunes, whose markers would otherwise overlap those of
# MatterSim and the AIMD error bars; MACE-MP-0, MatterSim and AIMD sit at their true 1000/T.
A_OFFSET = {"ftG": -0.02, "ftK": 0.02}
A_MARKER = 3.4  # marker diameter (pt) in panel a
A_ERROR = {"elinewidth": 0.6, "capsize": 1.0, "capthick": 0.6}
B_OFFSET = {"foundation": -0.3, "mattersim": -0.1, "ftG": 0.1, "ftK": 0.3}  # category units, panel b
PYMATGEN_SHIFT = 0.07  # pymatgen bar to the right of the kinisi circle, category units, panel b
C_ORDER = ("foundation", "mattersim", "ftK", "ftG")  # MSD curves, drawn in this order: fine-tunes on top
D_OFFSET = {"ftG": -0.003, "ftK": 0.003}  # panel d: vertical offset of the two fine-tunes (equal slopes, 0.992)
MSD_TEMPERATURES = (600, 1000)
RATIO_TEMPERATURE = 1000  # D ratio of panel d
SOFTENING_LABELS, SOFTENING_SUBSET = "k222", "H3"
LINE_WIDTH = 0.8
ERROR = {"elinewidth": 0.6, "capsize": 1.3, "capthick": 0.6}


# ------------------------------------------------------------------------------------------- data


def _load() -> dict:
    return {"comparison": common.read(AIMD_COMPARISON), "cross": common.read(CROSS),
            "transport": common.read(TRANSPORT), "msd": common.read(MSD),
            "aimd": common.read(AIMD).sort_values("temperature_K").reset_index(drop=True),
            "aimd_msd": common.read(AIMD_MSD)}


def _production() -> dict[str, str]:
    """Run family -> its 416-atom, 200-ps production series (constants.PRODUCTION_SERIES)."""
    out = {C.SERIES_BY_ID[series]["model"]: series for series in C.PRODUCTION_SERIES
           if C.SERIES_BY_ID[series]["model"] in FAMILIES}
    if set(out) != set(FAMILIES):
        raise ValueError(f"expected one production series per family {FAMILIES}, got {sorted(out)}")
    return {family: out[family] for family in FAMILIES}


def _style_ids(cross: pd.DataFrame) -> dict[str, str]:
    """Run family -> model id whose colour and fill draw it (MD models from ``drove_md``; MatterSim itself)."""
    md = common.md_models(cross)
    return {family: md.get(family, family) for family in FAMILIES}


def _ratio(comparison: pd.DataFrame, series: str, temperature: int) -> pd.Series:
    rows = comparison[(comparison["series"] == series) & (comparison["temperature_K"] == temperature)]
    if len(rows) != 1:
        raise ValueError(f"expected one AIMD comparison row for {series} at {temperature} K")
    return rows.iloc[0]


def _softening(cross: pd.DataFrame, model_id: str) -> pd.Series:
    rows = cross[(cross["model"] == model_id) & (cross["labels"] == SOFTENING_LABELS)
                 & (cross["subset"] == SOFTENING_SUBSET) & (cross["element"] == "Li")]
    if len(rows) != 1:
        raise ValueError(f"expected one Li row for {model_id} on {SOFTENING_SUBSET}")
    return rows.iloc[0]


def _window_start(length_ps: float) -> float:
    return C.KINISI_START_FRACTION * length_ps


def _log_ticks(ax, ticks: list[float], labels: list[str], axis: str = "y") -> None:
    style.log_ticks(ax, axis)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks, labels)


# ----------------------------------------------------------------------------------------- panels


def _panel_a(ax, d: dict, production: dict[str, str], ids: dict[str, str]) -> Path:
    rows = []
    transport = d["transport"]
    for family in FAMILIES:
        runs = transport[transport["series"] == production[family]].sort_values("temperature_K")
        temps = runs["temperature_K"].to_numpy(float)
        x = 1000.0 / temps + A_OFFSET.get(family, 0.0)
        y, err = runs["kinisi_D_cm2_s"].to_numpy(), runs["kinisi_D_std_cm2_s"].to_numpy()
        color = style.MODEL_COLORS[family]
        ax.plot(x, y, color=color, lw=LINE_WIDTH, zorder=2)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, zorder=3, **{**A_ERROR, "capsize": 0.0})
        ax.plot(x, y, **common.model_style(ids[family], size=A_MARKER), zorder=4)
        rows.extend({"source": "MLIP MD", "model": family, "md_model": ids[family], "series": production[family],
                     "temperature_K": int(t), "inv_T_1000_per_K": 1000.0 / t, "x_plot": xi, "kinisi_D_cm2_s": yi,
                     "kinisi_D_std_cm2_s": ei, "pymatgen_D_cm2_s": p, "msd_loglog_slope_fit_window": s,
                     "caged": False}
                    for t, xi, yi, ei, p, s in zip(temps, x, y, err, runs["D_cm2_s"],
                                                   runs["msd_loglog_slope_fit_window"]))
    aimd = d["aimd"]
    caged = common.caged_temperatures(aimd)
    temps = aimd["temperature_K"].to_numpy(float)
    x, y = 1000.0 / temps, aimd["kinisi_D_cm2_s"].to_numpy()
    err = aimd["kinisi_D_std_cm2_s"].to_numpy()
    for i in range(len(temps) - 1):  # dotted where a segment ends on a caged point
        dotted = int(temps[i]) in caged or int(temps[i + 1]) in caged
        ax.plot(x[i:i + 2], y[i:i + 2], color=style.AIMD_COLOR, lw=LINE_WIDTH, zorder=2,
                ls=(0, (1.2, 1.4)) if dotted else "-")
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=style.AIMD_COLOR, zorder=3, **A_ERROR)
    ax.plot(x, y, **common.aimd_style(size=A_MARKER), zorder=5)
    rows.extend({"source": "AIMD (Deng et al. 2017)", "model": "AIMD", "md_model": "", "series": "aimd_deng2017",
                 "temperature_K": int(t), "inv_T_1000_per_K": xi, "x_plot": xi, "kinisi_D_cm2_s": yi,
                 "kinisi_D_std_cm2_s": ei, "pymatgen_D_cm2_s": p, "msd_loglog_slope_fit_window": s,
                 "caged": int(t) in caged}
                for t, xi, yi, ei, p, s in zip(temps, x, y, err, aimd["D_cm2_s"], aimd["msd_loglog_slope_fit_window"]))
    slopes = aimd.set_index("temperature_K")["msd_loglog_slope_fit_window"]
    for temperature in caged:
        index = int(np.flatnonzero(temps == temperature)[0])
        ax.text(x[index] - 0.02, (y[index] - err[index]) / 1.25,
                f"AIMD caged at {temperature} K\n(MSD slope {common.sig(slopes[temperature], 1)})",
                ha="right", va="top", fontsize=style.SMALL, color=style.AIMD_COLOR, linespacing=1.1)
    ax.set_yscale("log")
    ax.set_ylim(1.5e-7, 1.5e-4)
    _log_ticks(ax, [1e-6, 1e-5, 1e-4], ["10$^{-6}$", "10$^{-5}$", "10$^{-4}$"])
    ax.set_xlim(0.77, 1.73)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("Li D, kinisi (cm$^2$ s$^{-1}$)")
    common.temperature_labels(ax, sorted(set(temps.astype(int))), y=0.975)
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, d: dict, production: dict[str, str], ids: dict[str, str]) -> Path:
    temps = list(C.TEMPERATURES)
    caged = common.caged_temperatures(d["aimd"])
    rows = []
    for index, temperature in enumerate(temps):
        if temperature in caged:
            ax.axvspan(index - 0.5, index + 0.5, **style.BAND_FILL)
            ax.text(index, 0.975, "AIMD caged", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=style.SMALL, color=style.CONTEXT_TEXT)
        for family in FAMILIES:
            row = _ratio(d["comparison"], production[family], temperature)
            ratio, rel = row["kinisi_D_ratio_to_aimd"], row["kinisi_D_ratio_rel_err"]
            xp = index + B_OFFSET[family]
            color = style.MODEL_COLORS[family]
            ax.errorbar(xp, ratio, yerr=ratio * rel, fmt="none", ecolor=color, zorder=2,
                        **{**ERROR, "capsize": 0.0})
            ax.plot(xp, ratio, **common.model_style(ids[family], size=3.4), zorder=4)
            ax.plot(xp + PYMATGEN_SHIFT, row["pymatgen_D_ratio_to_aimd"], marker="_", markersize=4.0,
                    markeredgewidth=1.0, color=color, linestyle="none", zorder=3)
            rows.append({"model": family, "md_model": ids[family], "series": production[family],
                         "temperature_K": temperature, "x_plot": xp, "x_plot_pymatgen": xp + PYMATGEN_SHIFT,
                         "kinisi_D_ratio_to_aimd": ratio,
                         "kinisi_D_ratio_rel_err": rel, "kinisi_D_ratio_err_abs": ratio * rel,
                         "pymatgen_D_ratio_to_aimd": row["pymatgen_D_ratio_to_aimd"],
                         "msd_loglog_slope": row["msd_loglog_slope"],
                         "aimd_msd_loglog_slope": row["aimd_msd_loglog_slope"], "aimd_caged": temperature in caged})
    ax.axhline(1.0, **style.UNITY_LINE)
    ax.text(len(temps) - 0.55, 0.97, "AIMD", ha="right", va="top", fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 20)
    _log_ticks(ax, [0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(-0.5, len(temps) - 0.5)
    ax.set_xticks(range(len(temps)), [str(t) for t in temps])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Li D / D(AIMD)")
    key = style.DARK_GRAY
    handles = [Line2D([0], [0], marker=style.MLIP_MARKER, markersize=3.4, markeredgewidth=0.7, color=key,
                      linestyle="none", label="kinisi, $\\pm$1 s.d."),
               Line2D([0], [0], marker="_", markersize=4.0, markeredgewidth=1.0, color=key, linestyle="none",
                      label="pymatgen")]
    ax.legend(handles=handles, loc="upper right", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.4)
    return style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(axes, d: dict, production: dict[str, str]) -> Path:
    length = float(C.SERIES_BY_ID[production["foundation"]]["production_ps"])
    aimd_length = float(d["aimd"]["length_ps"].iloc[0])
    start, aimd_start = _window_start(length), _window_start(aimd_length)
    slopes = d["transport"].set_index("run_id")["msd_loglog_slope_fit_window"]
    aimd_slopes = d["aimd"].set_index("temperature_K")["msd_loglog_slope_fit_window"]
    frames = []
    for index, (ax, temperature) in enumerate(zip(axes, MSD_TEMPERATURES)):
        ax.axvspan(start, 1000.0, **style.BAND_FILL)
        ax.plot([aimd_start, aimd_length], [0.95, 0.95], color=style.AIMD_COLOR, lw=1.4,
                solid_capstyle="butt", transform=ax.get_xaxis_transform(), zorder=5)
        for family in C_ORDER:
            run = C.run_id(production[family], temperature)
            curve = d["msd"][d["msd"]["run_id"] == run].sort_values("lag_ps")
            ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.MODEL_COLORS[family], lw=LINE_WIDTH,
                    zorder=2)
            frames.append(curve[["lag_ps", "msd_Li_A2"]].assign(model=family, run_id=run, temperature_K=temperature,
                                                                fit_window_start_ps=start))
        curve = d["aimd_msd"][d["aimd_msd"]["temperature_K"] == temperature].sort_values("lag_ps")
        ax.plot(curve["lag_ps"], curve["msd_Li_A2"], color=style.AIMD_COLOR, lw=0.9, zorder=3)
        frames.append(curve[["lag_ps", "msd_Li_A2"]].assign(model="AIMD", run_id=f"aimd_deng2017_{temperature}K",
                                                            temperature_K=temperature,
                                                            fit_window_start_ps=aimd_start))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.012, 300.0)
        ax.set_ylim(0.04, 4000.0)
        _log_ticks(ax, [0.1, 1, 10, 100], ["0.1", "1", "10", "100"], "x")
        _log_ticks(ax, [0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
        ax.set_xlabel("Lag time (ps)")
        ax.set_title(f"{temperature} K", fontsize=style.FONT_SIZE, fontweight="bold", pad=2.0)
        slope_handles = [Line2D([0], [0], color=style.MODEL_COLORS[f], lw=1.2,
                                label=f"{slopes[C.run_id(production[f], temperature)]:.2f}") for f in FAMILIES]
        slope_handles.append(Line2D([0], [0], color=style.AIMD_COLOR, lw=1.2,
                                    label=f"{aimd_slopes[temperature]:.2f}"))
        slope_key = dict(fontsize=style.SMALL, title_fontsize=style.SMALL, handlelength=1.2, handletextpad=0.4,
                         labelspacing=0.25, alignment="left", title="Slope in fit window")
        if index == 0:
            ax.set_ylabel("Li MSD (Å$^2$)")
            handles = [Patch(facecolor=style.BAND, edgecolor="none", label="MLIP fit window"),
                       Line2D([0], [0], color=style.AIMD_COLOR, lw=1.4, solid_capstyle="butt",
                              label="AIMD fit window")]
            windows = ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, handlelength=1.2,
                                handletextpad=0.4, labelspacing=0.35, borderaxespad=0.4)
            ax.add_artist(windows)
            ax.legend(handles=slope_handles, loc="upper left", bbox_to_anchor=(0.0, 0.83), borderaxespad=0.4,
                      **slope_key)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.legend(handles=slope_handles, loc="upper left", borderaxespad=0.4, **slope_key)
    source = pd.concat(frames, ignore_index=True)
    source["msd_loglog_slope_fit_window"] = [aimd_slopes[t] if m == "AIMD" else slopes[r]
                                             for m, r, t in zip(source["model"], source["run_id"],
                                                                source["temperature_K"])]
    source = source[["model", "run_id", "temperature_K", "lag_ps", "msd_Li_A2", "fit_window_start_ps",
                     "msd_loglog_slope_fit_window"]]
    return style.save_source(source, NAME, "c")


def _panel_d(ax, d: dict, production: dict[str, str], md: dict[str, str]) -> Path:
    """Li softening slope (y) against the kinisi D ratio to AIMD (x, logarithmic) of the three MD models."""
    rows = []
    ax.axvline(1.0, **style.UNITY_LINE)
    for family, model_id in md.items():
        soft = _softening(d["cross"], model_id)
        row = _ratio(d["comparison"], production[family], RATIO_TEMPERATURE)
        ratio, rel = row["kinisi_D_ratio_to_aimd"], row["kinisi_D_ratio_rel_err"]
        color = style.MODEL_COLORS[model_id]
        y = soft["softening_slope"] + D_OFFSET.get(family, 0.0)
        ax.errorbar(ratio, y, xerr=ratio * rel, fmt="none", ecolor=color, zorder=2, **ERROR)
        ax.plot(ratio, y, **common.model_style(model_id, size=3.4), zorder=3)
        rows.append({"model": family, "md_model": model_id, "series": production[family],
                     "labels": SOFTENING_LABELS, "subset": SOFTENING_SUBSET, "element": "Li",
                     "n_configs": int(soft["n_configs"]), "n_components": int(soft["n_components"]),
                     "softening_slope": soft["softening_slope"], "y_plot": y, "temperature_K": RATIO_TEMPERATURE,
                     "kinisi_D_ratio_to_aimd": ratio, "kinisi_D_ratio_rel_err": rel,
                     "kinisi_D_ratio_err_abs": ratio * rel})
    ax.text(1.04, 0.04, "AIMD", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    ax.text(0.97, 0.96, f"n = {len(rows)}", transform=ax.transAxes, ha="right", va="top", fontsize=style.SMALL,
            color=style.CONTEXT_TEXT)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 4.5)
    _log_ticks(ax, [1, 2, 3, 4], ["1", "2", "3", "4"], "x")
    ax.set_ylim(0.84, 1.005)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    ax.set_xlabel(f"Li D / D(AIMD), {RATIO_TEMPERATURE} K")
    ax.set_ylabel(f"Li softening slope, {SOFTENING_SUBSET} vs 2x2x2")
    return style.save_source(pd.DataFrame(rows), NAME, "d")


def _legend(ax, ids: dict[str, str]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    labels = {"foundation": style.MODEL_LABELS["foundation"], "mattersim": style.MODEL_LABELS["mattersim"],
              "ftG": f"{style.MODEL_LABELS['ftG']} (stage 2)", "ftK": f"{style.MODEL_LABELS['ftK']} (stage 1)"}
    handles = [common.handle(common.model_style(ids[f], size=3.6, linestyle="-", linewidth=LINE_WIDTH), labels[f])
               for f in FAMILIES]
    handles.append(common.handle(common.aimd_style(size=3.6, linestyle="-", linewidth=LINE_WIDTH),
                                 "AIMD (Deng et al. 2017)"))
    ax.legend(handles=handles, loc="center left", ncol=len(handles), fontsize=style.FONT_SIZE, handlelength=2.0,
              handletextpad=0.5, columnspacing=2.2, borderaxespad=0.0, frameon=False)


# ----------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    d = _load()
    production = _production()
    ids = _style_ids(d["cross"])
    md = common.md_models(d["cross"])
    fig, L = common.new_figure(WIDTH, HEIGHT)

    _legend(L.axes(14.0, 1.5, 167.0, 6.0, frame="none"), ids)

    top1, h1 = 14.0, 64.0
    ax_a = L.axes(14.0, top1, 96.0, h1, row="1", col="left")
    ax_b = L.axes(126.0, top1, 55.0, h1, row="1", col="right")
    sources = [_panel_a(ax_a, d, production, ids), _panel_b(ax_b, d, production, ids)]

    top2, h2 = 93.0, 42.0
    ax_c = [L.axes(14.0, top2, 46.5, h2, row="2", col="left"), L.axes(63.5, top2, 46.5, h2, row="2")]
    ax_d = L.axes(126.0, top2, 55.0, h2, row="2", col="right")
    sources += [_panel_c(ax_c, d, production), _panel_d(ax_d, d, production, md)]

    L.letter(ax_a, "a", 4.0, top1 - 1.0)
    L.letter(ax_b, "b", 115.0, top1 - 1.0)
    L.letter(ax_c[0], "c", 4.0, top2 - 1.0)
    L.letter(ax_d, "d", 115.0, top2 - 1.0)
    return L.save(*registry.target(KEY)) + sources


# ---------------------------------------------------------------------------------------- caption


def _caption() -> str:
    d = _load()
    production = _production()
    md = common.md_models(d["cross"])
    aimd = d["aimd"].set_index("temperature_K")
    caged = common.caged_temperatures(d["aimd"])
    transport = d["transport"].set_index("run_id")
    first = transport.loc[C.run_id(production["foundation"], C.TEMPERATURES[0])]
    n_aimd = int(aimd["n_atoms"].iloc[0])
    a_aimd = float(aimd["a_A"].iloc[0])
    a_mlip = (float(first["volume_A3"]) * n_aimd / float(first["n_atoms"])) ** (1.0 / 3.0)
    length = int(C.SERIES_BY_ID[production["foundation"]]["production_ps"])
    aimd_length = round(float(aimd["length_ps"].iloc[0]))
    start, aimd_start = _window_start(length), _window_start(aimd_length)
    hot = [t for t in C.TEMPERATURES if t not in caged]
    names = CAPTION_NAMES
    caged_text = " and ".join(f"{t} K" for t in caged)

    def ratio(family: str, temperature: int) -> float:
        return float(_ratio(d["comparison"], production[family], temperature)["kinisi_D_ratio_to_aimd"])

    key_numbers = ", ".join(f"{names[f]} " + " / ".join(common.sig(ratio(f, t), 2) for t in hot) for f in FAMILIES)
    aimd_d = "; ".join(f"{t} K, {common.sci(r['kinisi_D_cm2_s'])} +/- {common.sci(r['kinisi_D_std_cm2_s'])}"
                       for t, r in aimd.iterrows())
    slope_text = {t: ", ".join(
        f"{names[f]} {common.sig(transport.loc[C.run_id(production[f], t), 'msd_loglog_slope_fit_window'])}"
        for f in FAMILIES) + f", AIMD {common.sig(aimd.loc[t, 'msd_loglog_slope_fit_window'])}"
        for t in MSD_TEMPERATURES}
    soft = {f: _softening(d["cross"], m) for f, m in md.items()}
    soft_text = ", ".join(f"{names[f]} {common.sig(soft[f]['softening_slope'])} and "
                          f"{common.sig(ratio(f, RATIO_TEMPERATURE))}" for f in md)
    n_configs, n_components = int(soft["foundation"]["n_configs"]), int(soft["foundation"]["n_components"])
    stage = {"s1": "stage-1", "s2": "stage-2"}
    return (
        registry.heading(KEY)
        + "The reference is AIMD of the same ordered Li6PS5Cl cell (Cl on 4a, free S on 4c) by Deng et al., Chem. "
        "Mater. 29, 281 (2017), re-analysed with the estimators and settings of the MLIP runs "
        f"({registry.label('aimd')}). MLIP runs: {int(first['n_atoms'])} atoms, {length} ps per temperature, "
        f"a = {a_mlip:.2f} A; AIMD: {n_aimd} atoms, {aimd_length} ps, a = {a_aimd:.2f} A. One trajectory per model "
        "and temperature. Every D is the kinisi estimate (seed 0, fitted from a lag of "
        f"{C.KINISI_START_FRACTION * 100:g} % of each run onwards); error bars, +/- 1 posterior s.d. The Gamma-label "
        f"fine-tune ran MD with its {stage[md['ftG'].split('_')[1]]} model (filled markers) and the 2x2x2-label "
        f"fine-tune with its {stage[md['ftK'].split('_')[1]]} model (open markers). "
        f"**a**, Li D against 1000/T of the production runs at {', '.join(map(str, C.TEMPERATURES[:-1]))} and "
        f"{C.TEMPERATURES[-1]} K (circles joined by lines; so that markers do not overlap, the Gamma and 2x2x2 "
        f"fine-tuned points are offset horizontally by {A_OFFSET['ftG']:+g} and {A_OFFSET['ftK']:+g} in 1000/T, "
        "while MACE-MP-0, MatterSim and AIMD points sit at their true 1000/T) "
        f"and of AIMD at {int(aimd.index.min())}-{int(aimd.index.max())} K (black squares; {aimd_d} cm2/s). "
        f"At {caged_text} the AIMD Li MSD is almost flat in the fit window (log-log slope "
        f"{common.sig(aimd.loc[caged[0], 'msd_loglog_slope_fit_window'], 2)}): that point measures caged motion, not "
        "long-range diffusion (dotted segment). "
        "**b**, kinisi D of each production run divided by the AIMD D at the same temperature (circles; error bars, "
        "the two relative standard deviations added in quadrature) and the pymatgen D ratio (short bars to the right "
        "of each circle). At "
        f"{' / '.join(map(str, hot))} K: {key_numbers}. Grey column, {caged_text}, where AIMD is caged. "
        f"**c**, Li mean-squared displacement against lag time at {' and '.join(map(str, MSD_TEMPERATURES))} K, "
        f"averaged over all time origins, one trajectory each (single curves, no error bars); grey band, the MLIP "
        f"fit window (lag >= {start:g} ps); black bar, the AIMD fit window (lag >= {aimd_start:g} ps). The key in "
        "each panel gives the log-log slope in the fit window (1 for diffusive motion); at "
        f"{MSD_TEMPERATURES[0]} K: {slope_text[MSD_TEMPERATURES[0]]}; at {MSD_TEMPERATURES[1]} K: "
        f"{slope_text[MSD_TEMPERATURES[1]]}. "
        "**d**, Li force softening slope of the three MACE-family models that ran MD (least-squares slope through "
        "the origin of model against 2x2x2 DFT Li force components on the "
        f"{n_configs} configurations not trained on by either fine-tune, {SOFTENING_SUBSET}, which are the "
        f"validation configurations of both; {n_components} components) against their kinisi D ratio to AIMD at "
        f"{RATIO_TEMPERATURE} K (logarithmic axis; error bars as in b; the slope is a single value without error "
        f"bar). Slope and ratio: {soft_text}. The two fine-tuned points have the same slope to two decimals and are "
        f"offset vertically by {D_OFFSET['ftG']:+g} (Gamma) and {D_OFFSET['ftK']:+g} (2x2x2) so that their error "
        f"bars do not cross. n = {len(md)} models, no fit. "
        f"The 52-atom runs and a second 416-atom MACE-MP-0 run are in {registry.label('sampling')}. "
        + common.sources([TRANSPORT, MSD, AIMD, AIMD_MSD, AIMD_COMPARISON, CROSS])
    )


CAPTION = _caption()
