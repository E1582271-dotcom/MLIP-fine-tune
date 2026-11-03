"""What fine-tuning changes: the pre-registered transport reading and the two effects behind it.

a, sigma(300 K) and Ea of the foundation run and the two fine-tuned MD runs, and the reading fixed
in the analysis plan (f, the fraction of the fine-tuning slowdown recovered by the 2x2x2 labels);
b, kinisi D of each fine-tuned run divided by that of the foundation run at 600, 800 and 1000 K;
c, H3 force RMSE against the 2x2x2 labels by training stage, for both label sets: the two MD runs
used models of different stages, and the stage moves the error five times more than the labels;
d, the pressure offset of each fine-tuned model from the foundation in the DFT single points
against the same offset in MD. Every comparison is model against DFT or model against model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from .. import constants as C
from .. import paths
from . import common, registry, style

KEY = "finetune"
NAME = registry.number(KEY)
PANELS = "abcd"
WIDTH, HEIGHT = 183.0, 120.0
COMPARISON = paths.RESULTS / "transport_comparison.csv"
RATIOS = paths.RESULTS / "transport_ratios.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
TIMESERIES = paths.MD / "timeseries.csv.gz"
CONFIGS = paths.DFT / "configs.csv"

MODELS = ("foundation", "ftG", "ftK")  # run families of the MD comparison
FINE_TUNED = ("ftG", "ftK")
STAGES = ("s1", "s2")
STAGE_NUMBER = {"s1": 1, "s2": 2}
F_THRESHOLDS = (0.25, 0.75)  # analysis plan, part C: f < 0.25 not explained, f > 0.75 explains most
SUBSCRIPTS = {"foundation": "MP0", "ftG": "Gamma", "ftK": "2x2x2"}
CAPTION_NAMES = {"foundation": "MACE-MP-0", "ftG": "Gamma-label run", "ftK": "2x2x2-label run"}
STAGE_SUBSET, STAGE_LABELS = "H3", "k222"  # panel c: held-out H3 configurations against the 2x2x2 labels
PRESSURE_LABELS = "k222"  # panel d: the DFT term cancels in the paired difference, either label set gives it
KBAR_PER_GPA = 10.0
B_OFFSETS = {"ftG": -0.18, "ftK": 0.18}  # horizontal offsets in panel b (category units)
PYMATGEN_SHIFT = 0.14  # pymatgen bar to the right of the kinisi circle in panel b (category units)
D_BAND_ALPHA = {"ftG": 0.24, "ftK": 0.09}  # panel d bands: the two blues differ in lightness as well
D_OFFSETS = {600: -0.012, 800: 0.0, 1000: 0.012}  # horizontal offsets in panel d (GPa), for clarity only
MARKER = 3.6
RING = 7.2  # diameter of the ring around the MD models in panel c (pt)
LINE_WIDTH = 0.8


def _load() -> dict:
    timeseries = common.read(TIMESERIES)
    pressure = timeseries.groupby("run_id")["P_GPa"]
    blocks = pd.DataFrame({"mean_P_GPa": pressure.mean(), "block_sd_P_GPa": pressure.std(ddof=1),
                           "n_blocks": pressure.size()})
    blocks["block_sem_P_GPa"] = blocks["block_sd_P_GPa"] / np.sqrt(blocks["n_blocks"])
    ratios = common.read(RATIOS)
    ratios = ratios[ratios["reference"] == "foundation"].reset_index(drop=True)  # D ratios to the foundation
    return {"comparison": common.read(COMPARISON), "ratios": ratios, "cross": common.read(CROSS),
            "cross_configs": common.read(CROSS_CONFIGS), "thermostat": common.read(THERMOSTAT), "blocks": blocks,
            "configs": common.read(CONFIGS)}


def _fits(comparison: pd.DataFrame) -> pd.DataFrame:
    """One fit row per run family (foundation, ftG, ftK), indexed by family."""
    return comparison[comparison["model"].isin(MODELS)].set_index("model").loc[list(MODELS)]


def _reading(comparison: pd.DataFrame) -> pd.Series:
    rows = comparison[comparison["model"] == "reading"]
    if len(rows) != 1:
        raise ValueError("expected one reading row in transport_comparison.csv")
    return rows.iloc[0]


def _stage(model_id: str) -> str:
    return model_id.split("_")[1]


def _stage_style(family: str, stage: str, **overrides) -> dict:
    """Markers of a fine-tune by label set (colour of the family) and stage (fill): panel c shows all four."""
    color = style.MODEL_COLORS[family]
    return common.model_style(family, markerfacecolor=color if style.STAGE_FILLED[stage] else style.WHITE,
                              **overrides)


def _sigma_at(fits: pd.DataFrame, f: float) -> float:
    """sigma(300 K) of the 2x2x2 run that would give the reading value ``f``."""
    g, base = fits.loc["ftG", "sigma300_mS_cm"], fits.loc["foundation", "sigma300_mS_cm"]
    return float(g * (base / g) ** f)


def _stage_rmse(cross: pd.DataFrame) -> pd.DataFrame:
    rows = cross[(cross["labels"] == STAGE_LABELS) & (cross["subset"] == STAGE_SUBSET) & (cross["element"] == "all")
                 & cross["model"].str.startswith("ft")]
    if len(rows) != 4:
        raise ValueError(f"expected four fine-tuned rows on {STAGE_SUBSET} against {STAGE_LABELS}")
    return rows.set_index("model")


def _pressure_offsets(cross_configs: pd.DataFrame, md: dict[str, str]) -> pd.DataFrame:
    """DFT single-point pressure of each fine-tuned MD model minus that of the foundation, paired per configuration."""
    table = cross_configs[cross_configs["labels"] == PRESSURE_LABELS].pivot(
        index="config_id", columns="model", values="pressure_error_kbar")
    rows = []
    for family in FINE_TUNED:
        diff = (table[md[family]] - table["foundation"]).to_numpy() / KBAR_PER_GPA
        rows.append({"family": family, "model": md[family], "n_configs": len(diff),
                     "dft_offset_GPa": float(diff.mean()),
                     "dft_offset_sem_GPa": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                     "dft_pressure_error_GPa": float(table[md[family]].mean() / KBAR_PER_GPA),
                     "foundation_dft_pressure_error_GPa": float(table["foundation"].mean() / KBAR_PER_GPA)})
    return pd.DataFrame(rows).set_index("family")


def _md_offsets(d: dict, fits: pd.DataFrame) -> pd.DataFrame:
    """Mean MD pressure of each fine-tuned run minus the foundation run at the same temperature."""
    thermostat = d["thermostat"].set_index("run_id")
    blocks = d["blocks"]
    rows = []
    for family in FINE_TUNED:
        for temperature in C.TEMPERATURES:
            run = C.run_id(fits.loc[family, "series"], temperature)
            base = C.run_id(fits.loc["foundation", "series"], temperature)
            rows.append({"family": family, "temperature_K": temperature, "run_id": run, "foundation_run_id": base,
                         "md_mean_P_GPa": thermostat.loc[run, "mean_P_GPa"],
                         "foundation_md_mean_P_GPa": thermostat.loc[base, "mean_P_GPa"],
                         "md_offset_GPa": thermostat.loc[run, "mean_P_GPa"] - thermostat.loc[base, "mean_P_GPa"],
                         "md_block_sem_P_GPa": blocks.loc[run, "block_sem_P_GPa"],
                         "foundation_md_block_sem_P_GPa": blocks.loc[base, "block_sem_P_GPa"],
                         "md_offset_sem_GPa": math.hypot(blocks.loc[run, "block_sem_P_GPa"],
                                                         blocks.loc[base, "block_sem_P_GPa"]),
                         "n_blocks": int(blocks.loc[run, "n_blocks"]),
                         "foundation_n_blocks": int(blocks.loc[base, "n_blocks"])})
    return pd.DataFrame(rows)


def _errorbar_handle(label: str) -> ErrorbarContainer:
    """Legend handle of a marker with a vertical error bar (drawn like the kinisi points), in a neutral grey."""
    marker = Line2D([0], [0], marker=style.MLIP_MARKER, markersize=3.0, markeredgewidth=0.7, color=style.DARK_GRAY,
                    linestyle="none")
    bars = LineCollection([[(0.0, 0.0), (0.0, 1.0)]], colors=style.DARK_GRAY, linewidths=0.6)
    return ErrorbarContainer((marker, (), (bars,)), has_xerr=False, has_yerr=True, label=label)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax_s, ax_e, ax_t, fits: pd.DataFrame, reading: pd.Series, md: dict[str, str]) -> Path:
    """Forest plot (sigma(300 K) left, Ea right, one row per MD model) and the pre-registered reading."""
    rows = []
    for index, family in enumerate(MODELS):
        fit, color = fits.loc[family], style.MODEL_COLORS[family]
        s300, lo, hi = fit["sigma300_mS_cm"], fit["sigma300_min_mS_cm"], fit["sigma300_max_mS_cm"]
        ax_s.errorbar(s300, index, xerr=[[s300 - lo], [hi - s300]], fmt="none", ecolor=color, elinewidth=0.7,
                      capsize=1.6, zorder=2)
        ax_s.plot(s300, index, **common.model_style(md[family], size=MARKER + 0.6, zorder=3))
        ax_s.annotate(common.sig(s300), (s300, index), xytext=(0, 4.2), textcoords="offset points", ha="center",
                      va="bottom", fontsize=style.SMALL, color=color)
        ax_e.errorbar(fit["Ea_eV"], index, xerr=fit["Ea_err_eV"], fmt="none", ecolor=color, elinewidth=0.7,
                      capsize=1.6, zorder=2)
        ax_e.plot(fit["Ea_eV"], index, **common.model_style(md[family], size=MARKER + 0.6, zorder=3))
        ax_e.annotate(common.sig(fit["Ea_eV"]), (fit["Ea_eV"], index), xytext=(0, 4.2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=style.SMALL, color=color)
        rows.append({"row": "fit", "model": family, "series": fit["series"], "md_model": md[family],
                     "stage": "" if family == "foundation" else STAGE_NUMBER[_stage(md[family])],
                     "sigma300_mS_cm": s300, "sigma300_min_mS_cm": lo, "sigma300_max_mS_cm": hi,
                     "Ea_eV": fit["Ea_eV"], "Ea_err_eV": fit["Ea_err_eV"]})

    # The pre-registered zones of the reading, drawn where sigma_2x2x2 would have to lie.
    row = MODELS.index("ftK")
    low, high = (_sigma_at(fits, f) for f in F_THRESHOLDS)
    right = 20.0
    ax_s.fill_between([low, high], row - 0.42, row + 0.42, **style.BAND_FILL)
    ax_s.fill_between([high, right], row - 0.42, row + 0.42, color=style.LIGHT_GRAY, linewidth=0, zorder=0)
    for text, x in (("partial", math.sqrt(low * high)), ("explains\nmost", math.sqrt(high * right))):
        ax_s.text(x, row + 0.36, text, ha="center", va="bottom", fontsize=style.SMALL, color=style.DARK_GRAY,
                  linespacing=1.0)
    ax_s.text(0.112, row + 0.36, "not explained", ha="left", va="bottom", fontsize=style.SMALL,
              color=style.DARK_GRAY)
    rows.append({"row": "reading", "model": "reading", "f_recovered_fraction": reading["f_recovered_fraction"],
                 "ln_sigma_ftK_over_ftG": reading["ln_sigma_ftK_over_ftG"],
                 "ln_sigma_ftK_over_ftG_err": reading["ln_sigma_ftK_over_ftG_err"], "reading": reading["reading"],
                 "sigma300_at_f_0.25_mS_cm": low, "sigma300_at_f_0.75_mS_cm": high})

    for axis in (ax_s, ax_e):
        axis.set_ylim(len(MODELS) - 0.5, -0.5)
        axis.tick_params(axis="y", length=0, pad=3.0)
    ax_s.set_yticks(range(len(MODELS)), [common.SHORT_LABELS[md[m]].replace(", ", "\n") for m in MODELS])
    for label, family in zip(ax_s.get_yticklabels(), MODELS):
        label.set_color(style.MODEL_COLORS[family])
        label.set_linespacing(1.1)
    ax_e.set_yticks(range(len(MODELS)), [])
    ax_s.set_xscale("log")
    ax_s.set_xlim(0.1, right)
    style.log_ticks(ax_s, "x")
    ax_s.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax_s.set_xlabel("$\\sigma$(300 K) (mS cm$^{-1}$)")
    ax_e.set_xlim(0.22, 0.40)
    ax_e.set_xticks([0.25, 0.30, 0.35, 0.40], ["0.25", "0.30", "0.35", "0.40"])
    ax_e.set_xlabel("$E_a$ (eV)")

    g, k, f0 = SUBSCRIPTS["ftG"], SUBSCRIPTS["ftK"], SUBSCRIPTS["foundation"]
    ratio = f"ln($\\sigma_{{{k}}}$/$\\sigma_{{{g}}}$)"
    stages = {f: STAGE_NUMBER[_stage(md[f])] for f in FINE_TUNED}
    blocks = [
        [("Fixed in the analysis plan", {"fontweight": "bold"}),
         (f"$f$ = {ratio} / ln($\\sigma_{{{f0}}}$/$\\sigma_{{{g}}}$)", {}),
         (f"    = {common.sig(reading['f_recovered_fraction'])}", {}),
         (f"{ratio} = {common.sig(reading['ln_sigma_ftK_over_ftG'])}"
          f" $\\pm$ {common.sig(reading['ln_sigma_ftK_over_ftG_err'])}", {})],
        [(f"Pre-registered reading: {reading['reading']}", {"fontweight": "bold"}),
         (f"($f$ < {F_THRESHOLDS[0]:g}; partial up to {F_THRESHOLDS[1]:g})", {})],
        [(f"MD models: Gamma stage {stages['ftG']}, 2x2x2 stage {stages['ftK']}", {}),
         ("so the comparison is confounded with the", {}),
         ("training stage (c)", {})],
    ]
    y = 0.97
    for block in blocks:
        for text, kwargs in block:
            ax_t.text(0.0, y, text, transform=ax_t.transAxes, ha="left", va="top",
                      **{"fontsize": style.FONT_SIZE, **kwargs})
            y -= 0.095
        y -= 0.06
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    return style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, d: dict, md: dict[str, str]) -> Path:
    temps = list(C.TEMPERATURES)
    rows = []
    for index, temperature in enumerate(temps):
        for family in FINE_TUNED:
            row = d["ratios"][(d["ratios"]["model"] == family) & (d["ratios"]["temperature_K"] == temperature)]
            if len(row) != 1:
                raise ValueError(f"expected one ratio row for {family} at {temperature} K")
            row = row.iloc[0]
            x, kinisi, pymatgen = index + B_OFFSETS[family], row["kinisi_D_ratio"], row["pymatgen_D_ratio"]
            err = kinisi * row["kinisi_D_ratio_rel_err"]
            color = style.MODEL_COLORS[family]
            ax.errorbar(x, kinisi, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=0.0, zorder=2)
            ax.plot(x, kinisi, **common.model_style(md[family], size=MARKER, zorder=3))
            ax.plot(x + PYMATGEN_SHIFT, pymatgen, marker="_", markersize=4.0, markeredgewidth=1.0, color=color,
                    linestyle="none", zorder=4)
            rows.append({"model": family, "md_model": md[family], "reference": "foundation",
                         "temperature_K": temperature, "x": x, "x_pymatgen": x + PYMATGEN_SHIFT,
                         "kinisi_D_ratio": kinisi, "kinisi_D_ratio_err": err, "pymatgen_D_ratio": pymatgen})
    ax.axhline(1.0, **style.UNITY_LINE)
    ax.text(len(temps) - 0.58, 1.04, style.MODEL_LABELS["foundation"], ha="right", va="bottom",
            fontsize=style.SMALL, color=style.MODEL_COLORS["foundation"])
    ax.set_xlim(-0.5, len(temps) - 0.5)
    ax.set_xticks(range(len(temps)), [str(t) for t in temps])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Li D, fine-tuned / MACE-MP-0")
    handles = [_errorbar_handle("kinisi, $\\pm$1 s.d."),
               Line2D([0], [0], marker="_", linestyle="none", color=style.DARK_GRAY, markersize=4.0,
                      markeredgewidth=1.0, label="pymatgen")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.79), fontsize=style.SMALL,
              borderaxespad=0.4, labelspacing=0.35, handlelength=1.2)
    return style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(ax, d: dict, md: dict[str, str]) -> Path:
    rmse = _stage_rmse(d["cross"])
    value = {model: float(rmse.loc[model, "force_rmse_meV_A"]) for model in rmse.index}
    rows = []
    for family in FINE_TUNED:
        color = style.MODEL_COLORS[family]
        ids = [f"{family}_{stage}" for stage in STAGES]
        xs = [STAGE_NUMBER[stage] for stage in STAGES]
        ax.plot(xs, [value[m] for m in ids], color=color, lw=LINE_WIDTH, zorder=2)
        for model, stage, x in zip(ids, STAGES, xs):
            ax.plot(x, value[model], **_stage_style(family, stage, size=MARKER + 0.4, zorder=4))
            rows.append({"model": model, "family": family, "stage": STAGE_NUMBER[stage],
                         "drove_md": model == md[family], "labels": STAGE_LABELS, "subset": STAGE_SUBSET,
                         "element": "all", "n_configs": int(rmse.loc[model, "n_configs"]),
                         "n_components": int(rmse.loc[model, "n_components"]), "x": x,
                         "force_rmse_meV_A": value[model]})
    # The two models that drove MD, ringed and joined: the MD comparison crosses label set and stage at once.
    mx = [STAGE_NUMBER[_stage(md[f])] for f in FINE_TUNED]
    my = [value[md[f]] for f in FINE_TUNED]
    ax.plot(mx, my, color=style.BLACK, lw=0.6, linestyle=(0, (2.5, 1.5)), zorder=3)
    ring = {"marker": "o", "markersize": RING, "markerfacecolor": "none", "markeredgecolor": style.BLACK,
            "markeredgewidth": 0.6, "linestyle": "none"}
    common.allow_overlap(ax.plot(mx, my, zorder=5, **ring))  # the rings encircle the two MD markers on purpose
    ax.legend(handles=[Line2D([0], [0], color=style.BLACK, lw=0.6, label="MD comparison",
                              **{**ring, "linestyle": (0, (2.5, 1.5))})],
              loc="upper right", fontsize=style.SMALL, handlelength=2.4, borderaxespad=0.4)

    # Within-stage differences (label set) as brackets, between-stage differences on the lines.
    within = {stage: value[f"ftG_{stage}"] - value[f"ftK_{stage}"] for stage in STAGES}
    for stage, x, side in (("s1", 0.83, "right"), ("s2", 2.17, "left")):
        top, bottom = value[f"ftG_{stage}"], value[f"ftK_{stage}"]
        tick = 0.035 if side == "left" else -0.035
        ax.plot([x - tick, x, x, x - tick], [top, top, bottom, bottom], color=style.DARK_GRAY, lw=0.5, zorder=1)
        ax.text(x + (0.05 if side == "left" else -0.05), (top + bottom) / 2, common.sig(within[stage], 2),
                ha=side, va="center", fontsize=style.SMALL, color=style.DARK_GRAY)
    between = {family: value[f"{family}_s2"] - value[f"{family}_s1"] for family in FINE_TUNED}
    for family, x, y, va, ha in (("ftG", 1.53, (value["ftG_s1"] + value["ftG_s2"]) / 2 + 1.6, "bottom", "left"),
                                 ("ftK", 1.47, (value["ftK_s1"] + value["ftK_s2"]) / 2 - 1.6, "top", "right")):
        ax.text(x, y, common.sig(between[family], 3), ha=ha, va=va, fontsize=style.SMALL,
                color=style.MODEL_COLORS[family])
    ax.text(0.04, 0.04, "Brackets: Gamma minus 2x2x2\nOn the lines: stage 2 minus stage 1", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=style.SMALL, color=style.DARK_GRAY, linespacing=1.15)
    ax.set_xlim(0.45, 2.55)
    ax.set_xticks([1, 2], ["1", "2"])
    ax.set_ylim(50, 80)
    ax.set_yticks([50, 60, 70, 80])
    ax.set_xlabel("Training stage")
    ax.set_ylabel(f"{STAGE_SUBSET} force RMSE vs 2x2x2 (meV Å$^{{-1}}$)")
    for row in rows:
        row.update({"within_stage_diff_ftG_minus_ftK_meV_A": within[f"s{row['stage']}"],
                    "between_stage_diff_s2_minus_s1_meV_A": between[row["family"]]})
    return style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    dft = _pressure_offsets(d["cross_configs"], md)
    mdp = _md_offsets(d, fits)
    low, high = -0.545, -0.325
    ax.plot([low, high], [low, high], **style.IDENTITY_LINE)
    rows = []
    for family in FINE_TUNED:
        color = style.MODEL_COLORS[family]
        x0, sem = dft.loc[family, "dft_offset_GPa"], dft.loc[family, "dft_offset_sem_GPa"]
        ax.axvspan(x0 - sem, x0 + sem, color=color, alpha=D_BAND_ALPHA[family], linewidth=0, zorder=0)
        for run in mdp[mdp["family"] == family].itertuples():
            x = x0 + D_OFFSETS[run.temperature_K]
            ax.errorbar(x, run.md_offset_GPa, yerr=run.md_offset_sem_GPa, fmt="none", ecolor=color, elinewidth=0.6,
                        capsize=0.0, zorder=2)
            ax.plot(x, run.md_offset_GPa, **common.model_style(md[family], size=MARKER, zorder=3))
            rows.append({"model": family, "md_model": md[family], "temperature_K": run.temperature_K, "x": x,
                         "dft_offset_GPa": x0, "dft_offset_sem_GPa": sem,
                         "n_configs": int(dft.loc[family, "n_configs"]),
                         "dft_pressure_error_GPa": dft.loc[family, "dft_pressure_error_GPa"],
                         "foundation_dft_pressure_error_GPa": dft.loc[family, "foundation_dft_pressure_error_GPa"],
                         **{k: getattr(run, k) for k in ("md_offset_GPa", "md_offset_sem_GPa", "run_id",
                                                         "md_mean_P_GPa", "md_block_sem_P_GPa", "foundation_run_id",
                                                         "foundation_md_mean_P_GPa",
                                                         "foundation_md_block_sem_P_GPa", "n_blocks",
                                                         "foundation_n_blocks")}})
    order = ", ".join(str(t) for t in sorted(D_OFFSETS, key=D_OFFSETS.get))
    ax.text(0.96, 0.04, f"Left to right:\n{order} K", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.SMALL, color=style.BLACK, linespacing=1.15)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ticks = [-0.50, -0.45, -0.40, -0.35]
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.set_xlabel("Single-point offset from MACE-MP-0 (GPa)")
    ax.set_ylabel("MD offset from MACE-MP-0 (GPa)")
    return style.save_source(pd.DataFrame(rows), NAME, "d")


def _legend(ax, md: dict[str, str]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    labels = {"foundation": f"{style.MODEL_LABELS['foundation']} (foundation)",
              "ftG": f"{style.MODEL_LABELS['ftG']} (stage {STAGE_NUMBER[_stage(md['ftG'])]})",
              "ftK": f"{style.MODEL_LABELS['ftK']} (stage {STAGE_NUMBER[_stage(md['ftK'])]})"}
    handles = [common.handle(common.model_style(md[m], size=MARKER + 0.2), labels[m]) for m in MODELS]
    handles += [common.handle(common.model_style("ftG", size=MARKER + 0.2, color=style.GRAY,
                                                 markeredgecolor=style.GRAY, markerfacecolor=style.GRAY),
                              "filled: foundation or stage 2"),
                common.handle(common.model_style("ftK", size=MARKER + 0.2, color=style.GRAY,
                                                 markeredgecolor=style.GRAY), "open: stage 1")]
    ax.legend(handles=handles, loc="center left", ncol=len(handles), fontsize=style.FONT_SIZE, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.8, borderaxespad=0.0, frameon=False)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    d = _load()
    fits = _fits(d["comparison"])
    md = common.md_models(d["cross"])
    reading = _reading(d["comparison"])
    fig, L = common.new_figure(WIDTH, HEIGHT)

    _legend(L.axes(13.0, 1.5, 168.0, 6.0, frame="none"), md)

    top1, h1 = 15.0, 40.0
    ax_s = L.axes(26.0, top1, 60.0, h1, row="1", col="left")
    ax_e = L.axes(92.0, top1, 36.0, h1, row="1")
    ax_t = L.axes(135.0, top1, 46.0, h1, frame="none")
    sources = [_panel_a(ax_s, ax_e, ax_t, fits, reading, md)]

    top2, h2 = 74.0, 36.0
    ax_b = L.axes(26.0, top2, 40.0, h2, row="2", col="left")
    ax_c = L.axes(84.0, top2, 40.0, h2, row="2")
    ax_d = L.axes(141.0, top2, 40.0, h2, row="2")
    sources += [_panel_b(ax_b, d, md), _panel_c(ax_c, d, md), _panel_d(ax_d, d, fits, md)]

    L.letter(ax_s, "a", 4.0, top1 - 1.5)
    L.letter(ax_b, "b", 4.0, top2 - 1.5)
    L.letter(ax_c, "c", 72.0, top2 - 1.5)
    L.letter(ax_d, "d", 129.0, top2 - 1.5)
    return L.save(*registry.target(KEY)) + sources


def _caption() -> str:
    d = _load()
    fits = _fits(d["comparison"])
    md = common.md_models(d["cross"])
    reading = _reading(d["comparison"])
    name = CAPTION_NAMES
    stage = {f: STAGE_NUMBER[_stage(md[f])] for f in FINE_TUNED}
    md_name = {"foundation": "MACE-MP-0", "ftG": f"Gamma, stage {stage['ftG']}", "ftK": f"2x2x2, stage {stage['ftK']}"}

    # a
    def fit_text(family):
        f = fits.loc[family]
        return (f"{name[family]} {common.sig(f['sigma300_mS_cm'])} [{common.sig(f['sigma300_min_mS_cm'])}, "
                f"{common.sig(f['sigma300_max_mS_cm'])}] mS/cm and Ea {common.sig(f['Ea_eV'])} +/- "
                f"{common.sig(f['Ea_err_eV'], 2)} eV")

    f_value, gap, gap_err = (reading[k] for k in ("f_recovered_fraction", "ln_sigma_ftK_over_ftG",
                                                  "ln_sigma_ftK_over_ftG_err"))
    low, high = (_sigma_at(fits, f) for f in F_THRESHOLDS)
    direction = ("sigma_2x2x2 lies below sigma_Gamma, on the side away from the foundation" if gap < 0 else
                 "sigma_2x2x2 lies between sigma_Gamma and the foundation")
    separable = ("exceeds its 1-sigma uncertainty, so the two fine-tunes are not indistinguishable"
                 if abs(gap) >= gap_err else "is below its 1-sigma uncertainty")

    # b
    ratios = d["ratios"].set_index(["model", "temperature_K"])

    def ratio(family, t, column="kinisi_D_ratio"):
        return float(ratios.loc[(family, t), column])

    def ratio_err(family, t):
        return ratio(family, t) * ratio(family, t, "kinisi_D_ratio_rel_err")

    ratio_text = "; ".join(f"{name[m]} " + ", ".join(f"{common.sig(ratio(m, t))} +/- {common.sig(ratio_err(m, t), 2)}"
                                                      for t in C.TEMPERATURES) for m in FINE_TUNED)
    pym_text = "; ".join(f"{name[m]} " + ", ".join(common.sig(ratio(m, t, "pymatgen_D_ratio")) for t in C.TEMPERATURES)
                         for m in FINE_TUNED)
    z = max(abs(ratio("ftG", t) - ratio("ftK", t)) / math.hypot(ratio_err("ftG", t), ratio_err("ftK", t))
            for t in C.TEMPERATURES)
    kinisi_all = [ratio(m, t) for m in FINE_TUNED for t in C.TEMPERATURES]

    # c
    rmse = _stage_rmse(d["cross"])["force_rmse_meV_A"]
    within = [rmse[f"ftG_{s}"] - rmse[f"ftK_{s}"] for s in STAGES]
    between = [rmse[f"{f}_s2"] - rmse[f"{f}_s1"] for f in FINE_TUNED]
    n_components = int(_stage_rmse(d["cross"])["n_components"].iloc[0])
    n_h3 = int(_stage_rmse(d["cross"])["n_configs"].iloc[0])
    rmse_text = "; ".join(f"{'Gamma' if f == 'ftG' else '2x2x2'} {common.sig(rmse[f + '_s1'])} (stage 1) and "
                          f"{common.sig(rmse[f + '_s2'])} (stage 2)" for f in FINE_TUNED)

    # d
    dft = _pressure_offsets(d["cross_configs"], md)
    mdp = _md_offsets(d, fits)
    n_configs = int(dft["n_configs"].iloc[0])
    dft_text = "; ".join(f"{md_name[f]} {common.sig(dft.loc[f, 'dft_offset_GPa'])} +/- "
                         f"{common.sig(dft.loc[f, 'dft_offset_sem_GPa'], 2)} GPa" for f in FINE_TUNED)
    error_text = "; ".join([f"MACE-MP-0 {dft['foundation_dft_pressure_error_GPa'].iloc[0]:+.3f}"]
                           + [f"{md_name[f]} {dft.loc[f, 'dft_pressure_error_GPa']:+.3f}" for f in FINE_TUNED])
    md_offset_text = "; ".join(f"{md_name[f]} " + ", ".join(
        common.sig(v) for v in mdp[mdp["family"] == f].sort_values("temperature_K")["md_offset_GPa"])
        for f in FINE_TUNED)
    absolute = {"foundation": mdp[mdp["family"] == "ftG"].sort_values("temperature_K")["foundation_md_mean_P_GPa"]}
    absolute.update({f: mdp[mdp["family"] == f].sort_values("temperature_K")["md_mean_P_GPa"] for f in FINE_TUNED})
    absolute_text = "; ".join(f"{md_name[m]} " + ", ".join(f"{v:+.3f}" for v in absolute[m]) for m in MODELS)
    n_blocks = sorted(set(mdp["n_blocks"]) | set(mdp["foundation_n_blocks"]))
    md_mean = mdp.groupby("family")["md_offset_GPa"].mean()
    order_dft = dft["dft_offset_GPa"].idxmin()
    order_md = md_mean.idxmin()
    prose = {f: f"{'Gamma' if f == 'ftG' else '2x2x2'} (stage {stage[f]})" for f in FINE_TUNED}
    order_text = (f"the order of the two fine-tunes is reversed: the {prose[order_dft]} model lowers the "
                  f"single-point pressure more, the {prose[order_md]} model the MD pressure"
                  if order_dft != order_md else
                  f"the {prose[order_dft]} model lowers both more")

    return (
        registry.heading(KEY)
        + "MD values are re-analysis values of the 416-atom, 200-ps runs at 600, 800 and 1000 K of MACE-MP-0 small "
        "(the foundation) and of its two fine-tunes on 27 DFT configurations. "
        + common.naming_sentence(d["configs"])
        + f" The Gamma-label MD run used the stage-{stage['ftG']} model and the 2x2x2-label run the "
        f"stage-{stage['ftK']} model (identified by force fingerprints, {registry.label('models', 'a', short=True)}); "
        "filled markers, foundation or stage 2; open markers, stage 1. Every comparison here is model against model "
        f"or model against DFT; the comparison of the same runs with AIMD is in {registry.label('diffusion')}. "
        "**a**, sigma(300 K) (logarithmic axis; bars, 1-sigma interval from the fit covariance) and Ea (+/- 1 s.d.) "
        "of the weighted Arrhenius fits of ln(sigma T) against 1/T (pymatgen D, n = 3 temperatures, 600-1000 K): "
        f"{'; '.join(fit_text(m) for m in MODELS)}. Right, the reading fixed in the analysis plan before the "
        "analysis was run (protocol/analysis_plan.md, part C): f = ln(sigma_2x2x2/sigma_Gamma)/"
        "ln(sigma_MP0/sigma_Gamma) is the fraction of the fine-tuning slowdown recovered with the 2x2x2 labels (MP0, "
        f"the foundation run); f < {F_THRESHOLDS[0]:g} reads 'not explained', f > {F_THRESHOLDS[1]:g} 'explains most', "
        "otherwise 'partial', and the two fine-tunes are 'indistinguishable' if |ln(sigma_2x2x2/sigma_Gamma)| is "
        "below its 1-sigma uncertainty (the half-widths of the two log intervals in quadrature). "
        f"f = {common.sig(f_value)} and ln(sigma_2x2x2/sigma_Gamma) = {common.sig(gap)} +/- {common.sig(gap_err)}: "
        f"{direction}. |ln(sigma_2x2x2/sigma_Gamma)| {separable}. The reading is '{reading['reading']}'. "
        "Grey zones in the 2x2x2 row: where sigma_2x2x2 would have had to lie for "
        f"'partial' ({common.sig(low)}-{common.sig(high)} mS/cm) or 'explains most' (above {common.sig(high)} mS/cm). "
        "The two MD runs used models of different training stages, so this reading is confounded with the "
        "training stage (c). "
        "**b**, kinisi D of each fine-tuned run divided by that of the foundation run at the same temperature "
        "(circles; error, the two kinisi relative standard deviations in quadrature; one run per model and "
        f"temperature) and the pymatgen D ratio (short bars to the right of the circles). At 600, 800 and 1000 K: "
        f"{ratio_text} (pymatgen: {pym_text}). "
        f"Fine-tuning lowers the kinisi D to {common.sig(min(kinisi_all), 2)}-{common.sig(max(kinisi_all), 2)} of the "
        "foundation's; the two fine-tuned runs differ by at most "
        f"{common.sig(z, 2)} combined standard deviations at any temperature. "
        f"**c**, Force RMSE on the {n_h3} H3 configurations ({n_components} force components, all atoms), which "
        "neither fine-tune trained on (they are validation configurations of both and so were used to select the "
        "epoch), against the 2x2x2 labels for the four fine-tuned models, by training stage; colour, label set; "
        "fill, stage. "
        f"{rmse_text} meV/A. Ringed and joined by the dashed line, the two models that drove MD. Within a stage the "
        f"label set changes the error by {common.sig(within[0], 2)} (stage 1) and {common.sig(within[1], 2)} "
        f"(stage 2) meV/A (brackets); from stage 1 to stage 2 it changes by {common.sig(between[0])} (Gamma) and "
        f"{common.sig(between[1])} (2x2x2) meV/A (numbers on the lines). The MD comparison of a and b therefore "
        "crosses a label-set difference of about 2 meV/A with a stage difference of about 13 meV/A. Single values, "
        f"no error bars: with {n_h3} configurations no confidence intervals are given "
        f"({registry.label('forces', 'b')} shows the single configurations). "
        "**d**, Pressure offset of each fine-tuned MD model from the foundation. Horizontal, the DFT single points: "
        "model pressure minus MACE-MP-0 pressure on the same configuration, averaged over all "
        f"{n_configs} configurations (paired; shaded bands in the model colour, mean +/- standard error of the paired "
        f"differences, n = {n_configs}): {dft_text}. The DFT pressure cancels in the difference; the mean pressure errors against "
        f"the 2x2x2 labels are {error_text} GPa. Vertical, MD: mean pressure of the fine-tuned run minus that of the "
        "foundation run at the same temperature, with the standard errors of the two run means from 0.5-ps "
        f"block averages (n = {', '.join(str(n) for n in n_blocks)} blocks per run) added in quadrature (at most "
        f"{common.sig(mdp['md_offset_sem_GPa'].max(), 2)} GPa, mostly hidden by the markers): {md_offset_text} GPa at "
        "600, 800 and 1000 K. The three temperatures of one model share their horizontal value and are drawn left "
        f"to right in that order, offset by +/- {max(D_OFFSETS.values()):g} GPa for clarity. Dashed line, identity. Absolute mean MD pressures at "
        f"600, 800 and 1000 K (GPa): {absolute_text}. Both fine-tunes lower the pressure, in the single points and "
        f"in MD, but {order_text}. No model was trained on stress, so pressures are diagnostic; the DFT "
        "configurations come from the 52-atom MACE-MP-0 MD at 600, 800 and 1000 K, in the cell of which the "
        "416-atom cell is the 2x2x2 supercell, and every MD run keeps that cell fixed "
        f"(pressure time series in {registry.label('stability')}). "
        + common.sources([COMPARISON, RATIOS, CROSS, CROSS_CONFIGS, THERMOSTAT, TIMESERIES, CONFIGS])
    )


CAPTION = _caption()
