"""Extended Data Fig. 7: model identity and the full cross-evaluation.

a, force fingerprints: RMS residual between the forces stored in each MD run and the forces of
every candidate model, and the model each run is assigned to; b, force RMSE of all five models
against both label sets on every subset; c, softening slope per element; d, pressure error per
configuration (a diagnostic: no model was trained on stress).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from ..ingest.models import FINGERPRINT_RATIO, FINGERPRINT_MATCH_meV_A
from . import common, style

NAME = "ed07"
WIDTH, HEIGHT = 183.0, 140.0
FINGERPRINTS = paths.MODELS / "fingerprints.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
CONFIGS = paths.DFT / "configs.csv"
SERIES_ORDER = ("mace_416_200", "ftG_416_200", "ftK_416_200")
SERIES_LABELS = {"mace_416_200": "MACE-MP-0 run", "ftG_416_200": "Gamma fine-tune run",
                 "ftK_416_200": "2x2x2 fine-tune run"}
SUBSET_COLUMNS = ("H3", "H2", "T22", "own_valid", "All27")
SUBSET_NAMES = {"H3": "H3", "H2": "H2", "T22": "T22", "own_valid": "Own\nvalidation", "All27": "All27"}
CATEGORIES = ("all", *common.ELEMENTS)


def _load():
    fingerprints = common.read(FINGERPRINTS)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    floor = common.read(FLOOR)
    configs = common.read(CONFIGS)
    return fingerprints, cross, cross_configs, floor, configs


def _trained_on(model: str, subset: str, configs: pd.DataFrame) -> bool:
    """Whether ``subset`` contains configurations the model was trained on."""
    if model == "foundation" or subset == "own_valid":
        return False
    split = configs[f"{model.split('_')[0]}_split"]
    members = configs if subset == "All27" else configs[configs["subset"] == subset]
    return bool((split.loc[members.index] == "train").any())


def _panel_a(ax, label_ax, assigned_ax, fingerprints: pd.DataFrame) -> None:
    rows, positions, y = [], [], 0.0
    offsets = dict(zip(common.MODEL_ORDER, np.linspace(-0.3, 0.3, len(common.MODEL_ORDER))))
    for series in SERIES_ORDER:
        block = fingerprints[fingerprints["series"] == series].sort_values("temperature_K")
        start = y
        for run in block.itertuples():
            for model in common.MODEL_ORDER:
                value = getattr(run, f"rms_{model}_meV_A")
                ax.plot(value, y + offsets[model], **common.marker_kwargs(model, "o", 3.0, edge=0.6), zorder=3)
                rows.append({"series": series, "temperature_K": run.temperature_K, "candidate": model,
                             "rms_residual_meV_A": value, "assigned_model": run.assigned_model})
            label_ax.text(1.0, y, f"{run.temperature_K} K", ha="right", va="center", fontsize=style.SMALL)
            assigned = run.assigned_model
            assigned_ax.text(0.05, y, common.SHORT_LABELS.get(assigned, assigned), ha="left", va="center",
                             fontsize=style.SMALL, color=style.MODEL_COLORS.get(assigned, "black"))
            positions.append(y)
            y += 1.0
        label_ax.text(0.0, (start + y - 1.0) / 2, SERIES_LABELS[series].replace(" run", "\nrun"), ha="left",
                      va="center", fontsize=style.SMALL, linespacing=1.1)
        if series != SERIES_ORDER[-1]:
            ax.axhline(y - 0.5, color=style.LIGHT_GREY, lw=0.5, zorder=1)
    ax.axvline(FINGERPRINT_MATCH_meV_A, color="black", lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.text(FINGERPRINT_MATCH_meV_A * 1.25, -0.75, f"{FINGERPRINT_MATCH_meV_A:g} meV/Å threshold", ha="left",
            va="bottom", fontsize=style.SMALL)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    style.log_ticks(ax, "x")
    ax.set_ylim(y - 0.5, -0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("RMS force residual against the stored MD forces (meV/Å)")
    for axis in (label_ax, assigned_ax):
        axis.set_ylim(y - 0.5, -0.9)
        axis.set_xlim(0, 1)
        axis.axis("off")
    assigned_ax.text(0.05, -0.75, "Assigned", ha="left", va="bottom", fontsize=style.SMALL, fontweight="bold")
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, label_ax, cross: pd.DataFrame, configs: pd.DataFrame, primary: list[str]) -> None:
    table = cross[cross["element"] == "all"]
    cmap = colormaps["Greys"]
    values = table["force_rmse_meV_A"]
    norm = LogNorm(vmin=values.min() * 0.75, vmax=values.max() * 1.6)
    rows, y = [], 0
    for model in common.MODEL_ORDER:
        for labels in ("k222", "gamma"):
            for x, subset in enumerate(SUBSET_COLUMNS):
                match = table[(table["model"] == model) & (table["labels"] == labels) & (table["subset"] == subset)]
                if match.empty:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor=style.LIGHT_GREY,
                                           lw=0.4, hatch="////"))
                    ax.text(x, y, "n/a", ha="center", va="center", fontsize=style.SMALL, color=style.GREY)
                    rows.append({"model": model, "labels": labels, "subset": subset, "force_rmse_meV_A": np.nan,
                                 "n_configs": np.nan, "includes_training_configs": False})
                    continue
                row = match.iloc[0]
                value = row["force_rmse_meV_A"]
                shade = norm(value)
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=cmap(shade), edgecolor="white", lw=0.8))
                trained = _trained_on(model, subset, configs)
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=style.SMALL,
                        color="white" if shade > 0.55 else "black", style="italic" if trained else "normal")
                rows.append({"model": model, "labels": labels, "subset": subset, "force_rmse_meV_A": value,
                             "n_configs": int(row["n_configs"]), "includes_training_configs": trained})
            label_ax.text(0.98, y, f"vs {common.LABEL_SET_NAMES[labels]}", ha="right", va="center",
                          fontsize=style.SMALL, color=style.GREY)
            y += 1
        color = "black" if model in primary else style.GREY
        label_ax.text(0.0, y - 1.5, common.SHORT_LABELS[model] + ("" if model in primary else "\n(not used for MD)"),
                      ha="left", va="center", fontsize=style.SMALL, color=color, linespacing=1.1)
        label_ax.plot(-0.06, y - 1.5, clip_on=False, **common.marker_kwargs(model, "o", 3.0, edge=0.6))
    ax.set_xlim(-0.5, len(SUBSET_COLUMNS) - 0.5)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_yticks([])
    counts = {s: sorted(set(table.loc[table["subset"] == s, "n_configs"].dropna().astype(int))) for s in SUBSET_COLUMNS}
    ax.set_xticks(range(len(SUBSET_COLUMNS)),
                  [f"{SUBSET_NAMES[s]}\n(n = {' or '.join(str(c) for c in counts[s])})" for s in SUBSET_COLUMNS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    common.no_frame(ax)
    label_ax.set_ylim(y - 0.5, -0.5)
    label_ax.set_xlim(0, 1)
    label_ax.axis("off")
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(ax, cross: pd.DataFrame) -> None:
    offsets = dict(zip(common.MODEL_ORDER, np.linspace(-0.3, 0.3, len(common.MODEL_ORDER))))
    rows = []
    ax.axhline(1.0, color=style.GREY, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for index, element in enumerate(CATEGORIES):
        for model in common.MODEL_ORDER:
            row = cross[(cross["model"] == model) & (cross["labels"] == "k222") & (cross["subset"] == "H3")
                        & (cross["element"] == element)].iloc[0]
            ax.plot(index + offsets[model], row["softening_slope"], **common.marker_kwargs(model, "o", 3.2, edge=0.6),
                    zorder=3)
            rows.append({"element": element, "model": model, "softening_slope": row["softening_slope"],
                         "n_components": int(row["n_components"])})
    ax.set_xticks(range(len(CATEGORIES)), ["All", *common.ELEMENTS])
    ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Softening slope on H3\n(model vs 2x2x2 force)")
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, cross: pd.DataFrame, cross_configs: pd.DataFrame, floor: pd.DataFrame) -> None:
    pressure_floor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"), "pressure_rmse_kbar"].iloc[0]
    ax.axhspan(-pressure_floor, pressure_floor, color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    rows = []
    for index, model in enumerate(common.MODEL_ORDER):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")].sort_values(
            "config_id")
        offsets = common.jitter(len(per), 0.45, seed=20 + index)
        held = (per["subset"] == "H3").to_numpy()
        values = per["pressure_error_kbar"].to_numpy()
        ax.scatter(index + offsets[~held], values[~held], zorder=2,
                   **common.scatter_kwargs(model, 3.0, 0.4, alpha=0.35))
        ax.scatter(index + offsets[held], values[held], zorder=3, **common.scatter_kwargs(model, 11.0, 0.9))
        value = cross[(cross["model"] == model) & (cross["labels"] == "k222") & (cross["subset"] == "All27")
                      & (cross["element"] == "all")]["pressure_rmse_kbar"].iloc[0]
        ax.text(index, 3.1, f"{value:.2f}", ha="center", va="bottom", fontsize=style.SMALL)
        rows.extend({"model": model, "config_id": c, "subset": s, "x": index + o, "pressure_error_kbar": v,
                     "all27_rmse_kbar": value}
                    for c, s, o, v in zip(per["config_id"], per["subset"], offsets, values))
    ax.text(-0.4, 4.3, "All27 RMSE (kbar)", ha="left", va="bottom", fontsize=style.SMALL)
    ticks = [common.SHORT_LABELS[m].replace(", ", ",\n").replace("MACE-MP-0", "MACE-\nMP-0") for m in common.MODEL_ORDER]
    ax.set_xticks(range(len(common.MODEL_ORDER)), ticks, fontsize=style.SMALL)
    ax.set_xlim(-0.6, len(common.MODEL_ORDER) - 0.4)
    ax.set_ylim(-8.5, 5.5)
    ax.set_yticks([-8, -6, -4, -2, 0, 2])
    ax.set_ylabel("Pressure error vs 2x2x2 (kbar)")
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", markersize=3.3, markerfacecolor=style.GREY,
                              markeredgecolor=style.GREY, label="H3"),
                       Line2D([0], [0], marker="o", linestyle="none", markersize=1.8, markerfacecolor=style.GREY,
                              markeredgecolor=style.GREY, alpha=0.35, label="T22, H2"),
                       Rectangle((0, 0), 1, 1, facecolor=style.LIGHT_GREY, edgecolor="none",
                                 label=f"label floor, +/-{pressure_floor:.2f}")],
              loc="lower left", fontsize=style.SMALL, handletextpad=0.4, borderaxespad=0.2, ncol=3,
              columnspacing=0.8)
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _legend(fig, layout: common.Layout, top: float) -> None:
    handles = [Line2D([0], [0], label=common.SHORT_LABELS[m], **common.marker_kwargs(m, "o", 3.4, edge=0.7))
               for m in common.MODEL_ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(layout.x(14.0), layout.y(top)),
               ncol=len(handles), fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0,
               frameon=False)


def make() -> list[Path]:
    fingerprints, cross, cross_configs, floor, configs = _load()
    primary = common.primary_models(cross)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    _legend(fig, L, 3.0)

    top1, height1 = 16.0, 44.0
    ax_a_labels = L.axes(4.0, top1, 22.0, height1)
    ax_a = L.axes(27.0, top1, 70.0, height1)
    ax_a_assigned = L.axes(98.0, top1, 22.0, height1)
    _panel_a(ax_a, ax_a_labels, ax_a_assigned, fingerprints)
    ax_c = L.axes(140.0, top1, 41.0, height1)
    _panel_c(ax_c, cross)

    top2, height2 = 82.0, 46.0
    ax_b_labels = L.axes(4.0, top2, 30.0, height2)
    ax_b = L.axes(36.0, top2, 84.0, height2)
    _panel_b(ax_b, ax_b_labels, cross, configs, primary)
    ax_d = L.axes(140.0, top2 - 2.0, 41.0, height2 + 2.0)
    _panel_d(ax_d, cross, cross_configs, floor)

    L.letter(ax_a, "a", 3.0, 13.0)
    L.letter(ax_c, "c", 127.0, 13.0)
    L.letter(ax_b, "b", 3.0, 74.0)
    L.letter(ax_d, "d", 127.0, 74.0)
    return style.save(fig, "ed07_models", "extended")


def _caption() -> str:
    fingerprints, cross, cross_configs, floor, configs = _load()
    models = list(common.MODEL_ORDER)
    residual = fingerprints[[f"rms_{m}_meV_A" for m in models]].to_numpy()
    best = residual.min(axis=1)
    others = np.sort(residual, axis=1)[:, 1]
    assigned = fingerprints.groupby("series")["assigned_model"].agg(lambda s: ", ".join(sorted(set(s))))
    run_text = "; ".join(f"{SERIES_LABELS[s]} -> {common.SHORT_LABELS.get(assigned[s], assigned[s])}"
                         for s in SERIES_ORDER)
    table = cross[cross["element"] == "all"].set_index(["model", "labels", "subset"])
    h3 = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'H3'), 'force_rmse_meV_A']:.1f}" for m in models)
    slopes = cross[(cross["labels"] == "k222") & (cross["subset"] == "H3")].set_index(["model", "element"])
    found = ", ".join(f"{e} {slopes.loc[('foundation', e), 'softening_slope']:.2f}" for e in CATEGORIES)
    tuned = slopes.loc[[m for m in models if m != "foundation"], "softening_slope"]
    pressure = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'All27'), 'pressure_rmse_kbar']:.2f}"
                         for m in models)
    pk = cross_configs[cross_configs["labels"] == "k222"].groupby("model")["pressure_error_kbar"].mean()
    tuned_mean = pk.drop("foundation")
    pfloor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"), "pressure_rmse_kbar"].iloc[0]
    return (
        "Extended Data Fig. 7 | Model identity and the full cross-evaluation. "
        f"**a**, Force fingerprints of the 416-atom, 200-ps MD runs at 600, 800 and 1000 K: RMS difference between "
        f"the forces stored in four frames of each run (first, one third, two thirds, last) and the forces of each "
        f"candidate model recomputed in float32 on a CPU (logarithmic axis). A run is assigned to a model if the "
        f"residual is below {FINGERPRINT_MATCH_meV_A:g} meV/A (dashed line) and every other candidate is at least "
        f"{FINGERPRINT_RATIO:g} times worse. The matching model gives {best.min():.4f}-{best.max():.4f} meV/A, the "
        f"next best {others.min():.1f}-{others.max():.1f} meV/A. Assignments: {run_text}. These three models are "
        f"the primary rows of Fig. 3; the other two fine-tuned models were not used for MD. "
        f"**b**, Force RMSE (meV/A) of all five models against both label sets on each subset (own validation: "
        f"V5 = H3 + H2 for the Gamma fine-tune, H3 for the 2x2x2 fine-tune; not applicable to the foundation). "
        f"Shading is logarithmic in the RMSE. Italic values include configurations the model was trained on; the "
        f"primary estimand is the H3 column against 2x2x2 ({h3}). "
        f"**c**, Softening slope (least-squares slope through the origin of model against 2x2x2 force components) "
        f"on H3, overall and per element. MACE-MP-0: {found}; fine-tuned models {tuned.min():.2f}-{tuned.max():.2f}. "
        f"Dashed line, slope 1. "
        f"**d**, Pressure error against the 2x2x2 labels for each configuration (H3 large and opaque), a "
        f"diagnostic only: no model was trained on stress. All27 RMSE (kbar): {pressure}. The fine-tuned models "
        f"underestimate the pressure by {-tuned_mean.max():.1f}-{-tuned_mean.min():.1f} kbar on average (MACE-MP-0: "
        f"{pk['foundation']:+.2f} kbar), well outside the label floor of {pfloor:.2f} kbar (grey band). "
        f"Model names as in Fig. 3; stage 2 filled, stage 1 open markers. "
        + common.sources([FINGERPRINTS, CROSS, CROSS_CONFIGS, FLOOR, CONFIGS])
    )


CAPTION = _caption()
