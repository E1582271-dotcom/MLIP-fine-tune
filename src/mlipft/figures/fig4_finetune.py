"""What fine-tuning changes: the transport reading at equal training stage, the stage effect, the pressure.

a, sigma(300 K) and Ea of the foundation run and of the two stage-two fine-tuned MD runs (Gamma labels;
2x2x2 labels, the run of amendment A5), and the reading of the analysis plan at equal training stage (f2,
the fraction of the fine-tuning slowdown recovered by the 2x2x2 labels; the primary Part C reading of A5),
with the pre-registered reading of the July 2x2x2 run, whose model was the stage-one one; b, kinisi D of
each stage-two fine-tuned run divided by that of the foundation run at 600, 800 and 1000 K; c, the training
stage at equal labels (A5, reading 2): kinisi D of the stage-two 2x2x2 run divided by that of the stage-one
run, and their sigma(300 K) ratio, the only panel with a stage-one model (open); d, the pressure offset of
each stage-two fine-tune from the foundation in the DFT single points against the same offset in MD. Every
comparison is model against DFT or model against model; the force errors of both stages are in the models
Extended Data figure.
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
TRANSPORT = paths.MD / "transport.csv"
CONFIGS = paths.DFT / "configs.csv"

MODELS = ("foundation", "ftG", "ftK2")  # run families of a, b and d: the foundation and the stage-two fine-tunes
FINE_TUNED = ("ftG", "ftK2")
STAGE_RUNS = ("ftK", "ftK2")  # panel c: the 2x2x2 runs of stage one (July, the reference) and stage two (A5)
STAGE_NUMBER = {"s1": 1, "s2": 2}
F_THRESHOLDS = (0.25, 0.75)  # analysis plan, part C: f < 0.25 not explained, f > 0.75 explains most
SUBSCRIPTS = {"foundation": "MP0", "ftG": "Gamma", "ftK2": "2x2x2"}
CAPTION_NAMES = {"foundation": "MACE-MP-0", "ftG": "Gamma-label run", "ftK2": "2x2x2-label run"}
PRESSURE_LABELS = "k222"  # panel d: the DFT term cancels in the paired difference, either label set gives it
KBAR_PER_GPA = 10.0
B_OFFSETS = {"ftG": -0.18, "ftK2": 0.18}  # horizontal offsets in panel b (category units)
C_OFFSETS = {"ftK": -0.17, "ftK2": 0.05}  # panel c: stage one (on the unity line) left of stage two
PYMATGEN_SHIFT = 0.14  # pymatgen bar to the right of the kinisi circle in panels b and c (category units)
D_BAND_ALPHA = {"ftG": 0.24, "ftK2": 0.09}  # panel d bands: the two blues differ in lightness as well
D_OFFSETS = {600: -0.01, 800: 0.0, 1000: 0.01}  # horizontal offsets in panel d (GPa), for clarity only
D_NAMES = {"ftG": ("Gamma", "right"), "ftK2": ("2x2x2", "below")}  # panel d: each model's points named in its colour
MARKER = 3.6


def _load() -> dict:
    timeseries = common.read(TIMESERIES)
    pressure = timeseries.groupby("run_id")["P_GPa"]
    blocks = pd.DataFrame({"mean_P_GPa": pressure.mean(), "block_sd_P_GPa": pressure.std(ddof=1),
                           "n_blocks": pressure.size()})
    blocks["block_sem_P_GPa"] = blocks["block_sd_P_GPa"] / np.sqrt(blocks["n_blocks"])
    ratios = common.read(RATIOS)
    stage_one, stage_two = STAGE_RUNS
    return {"comparison": common.read(COMPARISON), "cross": common.read(CROSS),
            # D ratios to the foundation (b), and of the stage-two 2x2x2 run to the stage-one one (c)
            "ratios": ratios[ratios["reference"] == "foundation"].reset_index(drop=True),
            "stage_ratios": ratios[(ratios["reference"] == stage_one)
                                   & (ratios["model"] == stage_two)].reset_index(drop=True),
            "cross_configs": common.read(CROSS_CONFIGS), "thermostat": common.read(THERMOSTAT), "blocks": blocks,
            "configs": common.read(CONFIGS)}


def _fits(comparison: pd.DataFrame, families: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """One fit row per run family, indexed by family."""
    return comparison[comparison["model"].isin(families)].set_index("model").loc[list(families)]


def _row(comparison: pd.DataFrame, label: str) -> pd.Series:
    """A summary row of transport_comparison.csv: reading, reading_A5 or stage_A5."""
    rows = comparison[comparison["model"] == label]
    if len(rows) != 1:
        raise ValueError(f"expected one {label} row in transport_comparison.csv")
    return rows.iloc[0]


def _f_reading(f: float) -> tuple[str, str]:
    """The reading of ``f`` by the thresholds alone (without the indistinguishability rule), and its condition."""
    low, high = F_THRESHOLDS
    if f < low:
        return "not explained", f"< {low:g}"
    return ("explains most", f"> {high:g}") if f > high else ("partial", f"between {low:g} and {high:g}")


def _stage(model_id: str) -> str:
    return model_id.split("_")[1]


def _sigma_at(fits: pd.DataFrame, f: float) -> float:
    """sigma(300 K) of the 2x2x2 run that would give the reading value ``f``."""
    g, base = fits.loc["ftG", "sigma300_mS_cm"], fits.loc["foundation", "sigma300_mS_cm"]
    return float(g * (base / g) ** f)


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


def _errorbar_handle(label: str, **marker) -> ErrorbarContainer:
    """Legend handle of a marker with a vertical error bar (drawn like the kinisi points), neutral grey by default."""
    marker = {"marker": style.MLIP_MARKER, "markersize": 3.0, "markeredgewidth": 0.7, "color": style.DARK_GRAY,
              "linestyle": "none", **marker}
    bars = LineCollection([[(0.0, 0.0), (0.0, 1.0)]], colors=marker["color"], linewidths=0.6)
    return ErrorbarContainer((Line2D([0], [0], **marker), (), (bars,)), has_xerr=False, has_yerr=True, label=label)


def _ratio_points(ax, x: float, kinisi: float, err: float, pymatgen: float, model: str) -> None:
    """A kinisi D ratio (circle, +/- 1 s.d.) and the pymatgen ratio (short bar to its right), as in b and c."""
    color = style.MODEL_COLORS[model]
    ax.errorbar(x, kinisi, yerr=err, fmt="none", ecolor=color, elinewidth=0.6, capsize=0.0, zorder=2)
    ax.plot(x, kinisi, **common.model_style(model, size=MARKER, zorder=3))
    ax.plot(x + PYMATGEN_SHIFT, pymatgen, marker="_", markersize=4.0, markeredgewidth=1.0, color=color,
            linestyle="none", zorder=4)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax_s, ax_e, ax_t, fits: pd.DataFrame, readings: dict[str, pd.Series], md: dict[str, str]) -> Path:
    """Forest plot (sigma(300 K) left, Ea right, one row per MD model) and the reading at equal training stage."""
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

    # The zones of the reading, drawn where sigma_2x2x2 would have to lie.
    row = MODELS.index("ftK2")
    low, high = (_sigma_at(fits, f) for f in F_THRESHOLDS)
    right = 20.0
    ax_s.fill_between([low, high], row - 0.42, row + 0.42, **style.BAND_FILL)
    ax_s.fill_between([high, right], row - 0.42, row + 0.42, color=style.LIGHT_GRAY, linewidth=0, zorder=0)
    for text, x in (("partial", math.sqrt(low * high)), ("explains\nmost", math.sqrt(high * right))):
        ax_s.text(x, row + 0.36, text, ha="center", va="bottom", fontsize=style.SMALL, color=style.DARK_GRAY,
                  linespacing=1.0)
    ax_s.text(0.112, row + 0.36, "not explained", ha="left", va="bottom", fontsize=style.SMALL,
              color=style.DARK_GRAY)
    for label, reading in readings.items():
        rows.append({"row": label, "model": label, "series": reading["series"],
                     "primary": label == "reading_A5", "f_recovered_fraction": reading["f_recovered_fraction"],
                     "ln_sigma_ftK_over_ftG": reading["ln_sigma_ftK_over_ftG"],
                     "ln_sigma_ftK_over_ftG_err": reading["ln_sigma_ftK_over_ftG_err"], "reading": reading["reading"],
                     "sigma300_at_f_0.25_mS_cm": low, "sigma300_at_f_0.75_mS_cm": high})

    for axis in (ax_s, ax_e):
        axis.set_ylim(len(MODELS) - 0.5, -0.5)
        axis.tick_params(axis="y", length=0, pad=3.0)
    ax_s.set_yticks(range(len(MODELS)), [common.MAIN_SHORT_LABELS[m].replace(", ", "\n") for m in MODELS])
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

    a5, pre = readings["reading_A5"], readings["reading"]
    g, k, f0 = SUBSCRIPTS["ftG"], SUBSCRIPTS["ftK2"], SUBSCRIPTS["foundation"]
    ratio = f"ln($\\sigma_{{{k}}}$/$\\sigma_{{{g}}}$)"
    side = "below" if abs(a5["ln_sigma_ftK_over_ftG"]) < a5["ln_sigma_ftK_over_ftG_err"] else "above"
    by_f, condition = _f_reading(a5["f_recovered_fraction"])
    blocks = [
        [("Equal training stage (amendment A5)", {"fontweight": "bold"}),
         (f"$f_2$ = {ratio} / ln($\\sigma_{{{f0}}}$/$\\sigma_{{{g}}}$)", {}),
         (f"     = {common.sig(a5['f_recovered_fraction'], 2)}", {}),
         (f"{ratio} = {common.sig(a5['ln_sigma_ftK_over_ftG'], 2)}"
          f" $\\pm$ {common.sig(a5['ln_sigma_ftK_over_ftG_err'], 2)}", {})],
        [(f"Reading: {a5['reading']}", {"fontweight": "bold"}),
         (f"(|ln ratio| {side} its 1$\\sigma$ uncertainty;", {}),
         (f"by $f_2$ alone: {by_f}, $f_2$ {condition})", {})],
        [(f"Pre-registered, July 2x2x2 run (stage {STAGE_NUMBER[_stage(md['ftK'])]}):", {}),
         (f"$f$ = {common.sig(pre['f_recovered_fraction'], 2)}, {pre['reading']}; stage effect in c", {})],
    ]
    y = 0.97
    for block in blocks:
        for text, kwargs in block:
            ax_t.text(0.0, y, text, transform=ax_t.transAxes, ha="left", va="top",
                      **{"fontsize": style.FONT_SIZE, **kwargs})
            y -= 0.092
        y -= 0.05
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
            _ratio_points(ax, x, kinisi, err, pymatgen, md[family])
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


def _panel_c(ax, d: dict, stage: pd.Series, md: dict[str, str]) -> Path:
    """Training stage at equal labels: D of the stage-two 2x2x2 run over the stage-one run (open, at 1)."""
    one, two = STAGE_RUNS
    temps = list(C.TEMPERATURES)
    ratios = d["stage_ratios"].set_index("temperature_K")
    rows = []
    for index, temperature in enumerate(temps):
        row = ratios.loc[temperature]
        kinisi, pymatgen = row["kinisi_D_ratio"], row["pymatgen_D_ratio"]
        err = kinisi * row["kinisi_D_ratio_rel_err"]
        x_one, x_two = index + C_OFFSETS[one], index + C_OFFSETS[two]
        ax.plot(x_one, 1.0, **common.model_style(md[one], size=MARKER, zorder=3))
        _ratio_points(ax, x_two, kinisi, err, pymatgen, md[two])
        rows.append({"row": "D", "model": two, "md_model": md[two], "reference": one, "reference_md_model": md[one],
                     "temperature_K": temperature, "x": x_two, "x_pymatgen": x_two + PYMATGEN_SHIFT,
                     "x_reference": x_one, "kinisi_D_ratio": kinisi, "kinisi_D_ratio_err": err,
                     "pymatgen_D_ratio": pymatgen})
    ax.axhline(1.0, **style.UNITY_LINE)

    ratio, ln_err = stage["sigma300_ratio_stage2_over_stage1"], stage["ln_sigma_stage2_over_stage1_err"]
    lo, hi = (ratio * math.exp(sign * ln_err) for sign in (-1, 1))
    ax.text(0.96, 0.04, f"$\\sigma$(300 K), stage 2 / stage 1:\n{common.sig(ratio)}, 1$\\sigma$ range "
            f"{lo:.2f}-{hi:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.SMALL, color=style.BLACK, linespacing=1.15)
    rows.append({"row": "sigma300", "model": two, "md_model": md[two], "reference": one,
                 "reference_md_model": md[one], "sigma300_ratio_stage2_over_stage1": ratio,
                 "ln_sigma_stage2_over_stage1_err": ln_err, "sigma300_ratio_min": lo, "sigma300_ratio_max": hi})

    # The key of fill: the only panel of the figure with a stage-one model.
    handles = [common.handle(common.model_style(md[one], size=MARKER), "2x2x2, stage 1 (July run)"),
               _errorbar_handle("2x2x2, stage 2", **{k: v for k, v in common.model_style(md[two], size=MARKER).items()
                                                      if k != "linestyle"})]
    ax.legend(handles=handles, loc="upper left", fontsize=style.SMALL, borderaxespad=0.4, labelspacing=0.35,
              handlelength=1.2)
    ax.set_xlim(-0.5, len(temps) - 0.5)
    ax.set_xticks(range(len(temps)), [str(t) for t in temps])
    ax.set_ylim(0.4, 1.6)
    ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5], ["0.50", "0.75", "1.00", "1.25", "1.50"])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Li D, stage 2 / stage 1")
    return style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, d: dict, fits: pd.DataFrame, md: dict[str, str]) -> Path:
    dft = _pressure_offsets(d["cross_configs"], md)
    mdp = _md_offsets(d, fits)
    low, high = -0.57, -0.33
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
        text, where = D_NAMES[family]
        offsets = mdp.loc[mdp["family"] == family, "md_offset_GPa"]
        if where == "right":
            ax.text(x0 + max(D_OFFSETS.values()) + 0.008, offsets.mean(), text, ha="left", va="center",
                    fontsize=style.SMALL, color=color)
        else:
            ax.text(x0, offsets.min() - 0.012, text, ha="center", va="top", fontsize=style.SMALL, color=color)
    order = ", ".join(str(t) for t in sorted(D_OFFSETS, key=D_OFFSETS.get))
    ax.text(0.96, 0.04, f"Left to right:\n{order} K", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.SMALL, color=style.BLACK, linespacing=1.15)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ticks = [-0.55, -0.50, -0.45, -0.40, -0.35]
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.set_xlabel("Single-point offset from MACE-MP-0 (GPa)")
    ax.set_ylabel("MD offset from MACE-MP-0 (GPa)")
    return style.save_source(pd.DataFrame(rows), NAME, "d")


def _legend(ax, md: dict[str, str]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [common.handle(common.model_style(md[m], size=MARKER + 0.2), style.MODEL_LABELS[m]) for m in MODELS]
    ax.legend(handles=handles, loc="center left", ncol=len(handles), fontsize=style.FONT_SIZE, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.8, borderaxespad=0.0, frameon=False)


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    d = _load()
    fits = _fits(d["comparison"])
    md = common.md_models(d["cross"])
    readings = {label: _row(d["comparison"], label) for label in ("reading_A5", "reading")}
    fig, L = common.new_figure(WIDTH, HEIGHT)

    _legend(L.axes(13.0, 1.5, 168.0, 6.0, frame="none"), md)

    top1, h1 = 15.0, 40.0
    ax_s = L.axes(26.0, top1, 60.0, h1, row="1", col="left")
    ax_e = L.axes(92.0, top1, 36.0, h1, row="1")
    ax_t = L.axes(135.0, top1, 46.0, h1, frame="none")
    sources = [_panel_a(ax_s, ax_e, ax_t, fits, readings, md)]

    top2, h2 = 74.0, 36.0
    ax_b = L.axes(26.0, top2, 40.0, h2, row="2", col="left")
    ax_c = L.axes(84.0, top2, 40.0, h2, row="2")
    ax_d = L.axes(141.0, top2, 40.0, h2, row="2")
    sources += [_panel_b(ax_b, d, md), _panel_c(ax_c, d, _row(d["comparison"], "stage_A5"), md),
                _panel_d(ax_d, d, fits, md)]

    L.letter(ax_s, "a", 4.0, top1 - 1.5)
    L.letter(ax_b, "b", 4.0, top2 - 1.5)
    L.letter(ax_c, "c", 72.0, top2 - 1.5)
    L.letter(ax_d, "d", 129.0, top2 - 1.5)
    return L.save(*registry.target(KEY)) + sources


def _caption() -> str:
    d = _load()
    fits = _fits(d["comparison"], MODELS + ("ftK",))
    md = common.md_models(d["cross"])
    a5, pre, stage_row = (_row(d["comparison"], label) for label in ("reading_A5", "reading", "stage_A5"))
    name = CAPTION_NAMES
    stage = {f: STAGE_NUMBER[_stage(md[f])] for f in FINE_TUNED + ("ftK",)}
    md_name = {"foundation": "MACE-MP-0", "ftG": "Gamma", "ftK2": "2x2x2"}

    # a
    def fit_text(family, label):
        f = fits.loc[family]
        return (f"{label} {common.sig(f['sigma300_mS_cm'])} [{common.sig(f['sigma300_min_mS_cm'])}, "
                f"{common.sig(f['sigma300_max_mS_cm'])}] mS/cm and Ea {common.sig(f['Ea_eV'])} +/- "
                f"{common.sig(f['Ea_err_eV'], 2)} eV")

    def reading_numbers(reading, symbol):
        return (f"{symbol} = {common.sig(reading['f_recovered_fraction'], 2)} and ln(sigma_2x2x2/sigma_Gamma) = "
                f"{common.sig(reading['ln_sigma_ftK_over_ftG'], 2)} +/- "
                f"{common.sig(reading['ln_sigma_ftK_over_ftG_err'], 2)}")

    f2, gap, gap_err = (a5[k] for k in ("f_recovered_fraction", "ln_sigma_ftK_over_ftG", "ln_sigma_ftK_over_ftG_err"))
    low, high = (_sigma_at(fits, f) for f in F_THRESHOLDS)
    direction = ("sigma_2x2x2 lies below sigma_Gamma, on the side away from the foundation" if gap < 0 else
                 "sigma_2x2x2 lies between sigma_Gamma and the foundation")
    separable = ("is below its 1-sigma uncertainty" if abs(gap) < gap_err else
                 "exceeds its 1-sigma uncertainty, so the two fine-tunes are not indistinguishable")
    by_f, condition = _f_reading(f2)
    reading_text = (f"The reading is '{a5['reading']}'"
                    + (f" (f2 {condition} alone would read '{by_f}')" if a5["reading"] != by_f else ""))

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
    kinisi_all = [ratio(m, t) for m in FINE_TUNED for t in C.TEMPERATURES]
    # the two fine-tuned runs against each other, directly: the foundation run, common to both ratios, cancels
    transport = common.read(TRANSPORT).set_index("run_id")
    series = {C.SERIES_BY_ID[s]["model"]: s for s in C.PRODUCTION_SERIES}

    def pair(t):
        k, g = (transport.loc[C.run_id(series[f], t)] for f in ("ftK2", "ftG"))
        value = k["kinisi_D_cm2_s"] / g["kinisi_D_cm2_s"]
        return value, value * math.hypot(k["kinisi_D_std_cm2_s"] / k["kinisi_D_cm2_s"],
                                          g["kinisi_D_std_cm2_s"] / g["kinisi_D_cm2_s"])

    pairs = {t: pair(t) for t in C.TEMPERATURES}
    pair_text = ", ".join(f"{common.sig(v)} +/- {common.sig(e, 2)}" for v, e in pairs.values())
    z_t = max(C.TEMPERATURES, key=lambda t: abs(pairs[t][0] - 1.0) / pairs[t][1])
    z = abs(pairs[z_t][0] - 1.0) / pairs[z_t][1]

    # c
    stage_d = d["stage_ratios"].set_index("temperature_K")
    stage_err = stage_d["kinisi_D_ratio"] * stage_d["kinisi_D_ratio_rel_err"]
    stage_text = ", ".join(f"{common.sig(stage_d.loc[t, 'kinisi_D_ratio'])} +/- {common.sig(stage_err[t], 2)}"
                           for t in C.TEMPERATURES)
    stage_pym = ", ".join(common.sig(stage_d.loc[t, "pymatgen_D_ratio"]) for t in C.TEMPERATURES)
    stage_z = float(((stage_d["kinisi_D_ratio"] - 1.0).abs() / stage_err).max())
    s_ratio, s_err = stage_row["sigma300_ratio_stage2_over_stage1"], stage_row["ln_sigma_stage2_over_stage1_err"]
    s_side = "below" if abs(math.log(s_ratio)) < s_err else "above"

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
    order_text = (f"but the order of the two fine-tunes is reversed: the {md_name[order_dft]} model lowers the "
                  f"single-point pressure more, the {md_name[order_md]} model the MD pressure"
                  if order_dft != order_md else
                  f"the {md_name[order_dft]} model lowers both more")

    return (
        registry.heading(KEY)
        + "MD values are re-analysis values of the 416-atom, 200-ps runs at 600, 800 and 1000 K of MACE-MP-0 small "
        "(the foundation) and of its fine-tunes on DFT labels of the 27 configurations. "
        + common.naming_sentence(d["configs"])
        + f" In a, b and d the two fine-tuned runs are those of the stage-{stage['ftG']} Gamma model and the "
        f"stage-{stage['ftK2']} 2x2x2 model (the run of amendment A5, protocol/amendments.md), so the two fine-tunes "
        "differ only in their labels; c adds the July 2x2x2-label run, which used the "
        f"stage-{stage['ftK']} 2x2x2 model (models identified by force fingerprints, "
        f"{registry.label('models', 'a', short=True)}). Filled markers, foundation or stage 2; open markers, stage 1. "
        "Every comparison here is model against model or model against DFT; the comparison of the same runs with "
        f"AIMD is in {registry.label('diffusion')}, and the force errors of the models of both stages are in "
        f"{registry.label('models', 'b', short=True)}. "
        "**a**, sigma(300 K) (logarithmic axis; bars, 1-sigma interval from the fit covariance) and Ea (+/- 1 s.d.) "
        "of the weighted Arrhenius fits of ln(sigma T) against 1/T (pymatgen D, n = 3 temperatures, 600-1000 K): "
        f"{'; '.join(fit_text(m, name[m]) for m in MODELS)}. Right, the reading fixed in the analysis plan "
        "(protocol/analysis_plan.md, part C) taken at equal training stage, which amendment A5 made the primary "
        "reading: f2 = ln(sigma_2x2x2/sigma_Gamma)/ln(sigma_MP0/sigma_Gamma) is the fraction of the fine-tuning "
        "slowdown recovered with the 2x2x2 labels (MP0, the foundation run); "
        f"f2 < {F_THRESHOLDS[0]:g} reads 'not explained', f2 > {F_THRESHOLDS[1]:g} 'explains most', otherwise "
        "'partial', and the two fine-tunes are 'indistinguishable' if |ln(sigma_2x2x2/sigma_Gamma)| is below its "
        "1-sigma uncertainty (the half-widths of the two log intervals in quadrature). "
        f"{reading_numbers(a5, 'f2')}: {direction}, and |ln(sigma_2x2x2/sigma_Gamma)| {separable}. {reading_text}. "
        "Grey zones in the 2x2x2 row: where sigma_2x2x2 would have had to lie for "
        f"'partial' ({common.sig(low)}-{common.sig(high)} mS/cm) or 'explains most' (above {common.sig(high)} mS/cm). "
        "The pre-registered reading, made with the July 2x2x2-label run of the "
        f"stage-{stage['ftK']} model, stays reported as such: {reading_numbers(pre, 'f')}, "
        f"'{pre['reading']}'; it compares fine-tunes of different training stages, whose effect is shown in c. "
        "**b**, kinisi D of each fine-tuned run divided by that of the foundation run at the same temperature "
        "(circles; error, the two kinisi relative standard deviations in quadrature; one run per model and "
        f"temperature) and the pymatgen D ratio (short bars to the right of the circles). At 600, 800 and 1000 K: "
        f"{ratio_text} (pymatgen: {pym_text}). "
        f"Fine-tuning lowers the kinisi D to {common.sig(min(kinisi_all), 2)}-{common.sig(max(kinisi_all), 2)} of the "
        "foundation's. Compared directly (the foundation cancels), kinisi D of the 2x2x2 run divided by that of the "
        f"Gamma run is {pair_text} at 600, 800 and 1000 K, at most {common.sig(z, 2)} standard deviations from 1 "
        f"(at {z_t} K). "
        "**c**, Training stage at equal labels (amendment A5, descriptive): kinisi D of the "
        f"stage-{stage['ftK2']} 2x2x2-label run divided by that of the July stage-{stage['ftK']} run at the same "
        "temperature (filled circles; error, the two kinisi relative standard deviations in quadrature; open "
        "circles, the stage-1 run, 1 by construction) and the pymatgen D ratio (short bars): "
        f"{stage_text} at 600, 800 and 1000 K (pymatgen: {stage_pym}); the kinisi ratios differ from 1 by at most "
        f"{common.sig(stage_z, 2)} standard deviations. The stage-1 run gives "
        f"{fit_text('ftK', 'sigma(300 K)')}; the ratio of sigma(300 K), stage 2 over stage 1 (bottom right), is "
        f"{common.sig(s_ratio)} (1-sigma range {s_ratio * math.exp(-s_err):.2f}-{s_ratio * math.exp(s_err):.2f}), "
        f"and its logarithm, {common.sig(math.log(s_ratio), 2)}, lies {s_side} its 1-sigma uncertainty of "
        f"{common.sig(s_err, 2)} (the half-widths of the two log intervals in quadrature). "
        "**d**, Pressure offset of each fine-tuned MD model from the foundation. Horizontal, the DFT single points: "
        "model pressure minus MACE-MP-0 pressure on the same configuration, averaged over all "
        f"{n_configs} configurations (paired; shaded bands in the model colour, mean +/- standard error of the paired "
        f"differences, n = {n_configs}): {dft_text}. The DFT pressure cancels in the difference; the mean pressure "
        f"errors against the 2x2x2 labels are {error_text} GPa. Vertical, MD: mean pressure of the fine-tuned run "
        "minus that of the foundation run at the same temperature, with the standard errors of the two run means from "
        f"0.5-ps block averages (n = {', '.join(str(n) for n in n_blocks)} blocks per run) added in quadrature (at "
        f"most {common.sig(mdp['md_offset_sem_GPa'].max(), 2)} GPa, mostly hidden by the markers): {md_offset_text} "
        "GPa at 600, 800 and 1000 K. The three temperatures of one model share their horizontal value and are drawn "
        f"left to right in that order, offset by +/- {max(D_OFFSETS.values()):g} GPa for clarity. Dashed line, "
        f"identity. Absolute mean MD pressures at 600, 800 and 1000 K (GPa): {absolute_text}. Both fine-tunes lower "
        f"the pressure, in the single points and in MD; {order_text}. No model was trained on stress, so "
        "pressures are diagnostic; the DFT configurations come from the 52-atom MACE-MP-0 MD at 600, 800 and 1000 K, "
        "in the cell of which the 416-atom cell is the 2x2x2 supercell, and every MD run keeps that cell fixed "
        f"(pressure time series in {registry.label('stability')}). "
        + common.sources([COMPARISON, RATIOS, TRANSPORT, CROSS, CROSS_CONFIGS, THERMOSTAT, TIMESERIES, CONFIGS])
    )


CAPTION = _caption()
