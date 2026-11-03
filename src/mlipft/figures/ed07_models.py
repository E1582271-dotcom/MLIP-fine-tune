"""Extended Data figure "models": model identity and the full cross-evaluation.

a, force fingerprints: RMS residual between the forces stored in each MD run and the forces of
every candidate model, and the model each run is assigned to; b, force RMSE of all five models
against both label sets on every subset; c, softening slope per element; d, pressure error per
configuration (a diagnostic: no model was trained on stress); e, the cross term of the label-floor
decomposition, which places each model between the two label sets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from ..ingest.models import FINGERPRINT_RATIO, FINGERPRINT_MATCH_meV_A
from . import common, registry, style

KEY = "models"
NAME = registry.number(KEY)
PANELS = "abcde"
WIDTH, HEIGHT = 183.0, 142.0
FINGERPRINTS = paths.MODELS / "fingerprints.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
DECOMPOSITION = paths.RESULTS / "floor_decomposition.csv"
CONFIGS = paths.DFT / "configs.csv"
SERIES_ORDER = ("mace_416_200", "ftG_416_200", "ftK_416_200")
SERIES_LABELS = {"mace_416_200": "MACE-MP-0 run", "ftG_416_200": "Gamma fine-tune run",
                 "ftK_416_200": "2x2x2 fine-tune run"}
SERIES_TEXT = {"mace_416_200": "MACE-MP-0\nrun", "ftG_416_200": "Gamma\nfine-tune run",
               "ftK_416_200": "2x2x2\nfine-tune run"}
SUBSET_COLUMNS = ("H3", "H2", "T22", "own_valid", "All27")
SUBSET_NAMES = {"H3": "H3", "H2": "H2", "T22": "T22", "own_valid": "Own\nvalidation", "All27": "All27"}
CATEGORIES = ("all", *common.ELEMENTS)
SIZE = 2.5  # marker diameter (pt) of the model markers
EDGE = 0.5  # marker edge width (pt)
SPREAD = 0.38  # the five models sit within +/-SPREAD of a row (a) or a category (c)
HEAT = LinearSegmentedColormap.from_list("ed07_rmse", [style.WHITE, style.BLACK])
H3_OFFSETS = np.array([0.1, 0.27, 0.44])  # d: the three H3 configurations, in id order
DECOMPOSITION_SUBSETS = {"H3": {"dx": -0.15, "size": 3.4}, "All27": {"dx": 0.15, "size": 2.2}}


def _load():
    fingerprints = common.read(FINGERPRINTS)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    floor = common.read(FLOOR)
    decomposition = common.read(DECOMPOSITION)
    configs = common.read(CONFIGS)
    return fingerprints, cross, cross_configs, floor, decomposition, configs


def _trained_on(model: str, subset: str, configs: pd.DataFrame) -> bool:
    """Whether ``subset`` contains configurations the model was trained on."""
    if model == "foundation" or subset == "own_valid":
        return False
    split = configs[f"{model.split('_')[0]}_split"]
    members = configs if subset == "All27" else configs[configs["subset"] == subset]
    return bool((split.loc[members.index] == "train").any())


def _model_ticks(ax) -> None:
    ticks = [common.SHORT_LABELS[m].replace(", ", ",\n") for m in common.MODEL_ORDER]
    ax.set_xticks(range(len(common.MODEL_ORDER)), ticks, fontsize=style.SMALL)
    ax.set_xlim(-0.6, len(common.MODEL_ORDER) - 0.4)


def _grey_handle(size: float, label: str, alpha: float = 1.0) -> Line2D:
    return Line2D([0], [0], marker="o", linestyle="none", markersize=size, markerfacecolor=style.GRAY,
                  markeredgecolor=style.GRAY, alpha=alpha, label=label)


# ------------------------------------------------------------------------------------------- panels


def _panel_a(ax, series_ax, assigned_ax, fingerprints: pd.DataFrame) -> None:
    rows, ticks, labels, y = [], [], [], 0.0
    offsets = dict(zip(common.MODEL_ORDER, np.linspace(-SPREAD, SPREAD, len(common.MODEL_ORDER))))
    for series in SERIES_ORDER:
        block = fingerprints[fingerprints["series"] == series].sort_values("temperature_K")
        start = y
        for run in block.itertuples():
            for model in common.MODEL_ORDER:
                value = getattr(run, f"rms_{model}_meV_A")
                ax.plot(value, y + offsets[model], **common.model_style(model, size=SIZE, edge=EDGE, zorder=3))
                rows.append({"series": series, "temperature_K": run.temperature_K, "candidate": model,
                             "rms_residual_meV_A": value, "assigned_model": run.assigned_model,
                             "y": y + offsets[model]})
            ticks.append(y)
            labels.append(f"{run.temperature_K} K")
            assigned = run.assigned_model
            assigned_ax.text(0.06, y, common.SHORT_LABELS.get(assigned, assigned), ha="left", va="center",
                             fontsize=style.SMALL, color=style.MODEL_COLORS.get(assigned, style.BLACK))
            y += 1.0
        series_ax.text(0.0, (start + y - 1.0) / 2, SERIES_TEXT[series], ha="left", va="center",
                       fontsize=style.SMALL, linespacing=1.1)
        if series != SERIES_ORDER[-1]:
            ax.axhline(y - 0.5, color=style.LIGHT_GRAY, linewidth=0.5, zorder=1)
    ax.axvline(FINGERPRINT_MATCH_meV_A, **{**style.IDENTITY_LINE, "color": style.BLACK, "linewidth": 0.6})
    ax.text(FINGERPRINT_MATCH_meV_A / 1.7, 0.0, f"{FINGERPRINT_MATCH_meV_A:g} meV/Å\nthreshold", ha="right",
            va="center", fontsize=style.SMALL, linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    style.log_ticks(ax, "x")
    limits = (y - 0.3, -0.7)
    ax.set_ylim(*limits)
    ax.set_yticks(ticks, labels, fontsize=style.SMALL)
    ax.set_xlabel("RMS residual vs stored MD forces (meV/Å)")
    for axis in (series_ax, assigned_ax):
        axis.set_ylim(*limits)
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
    assigned_ax.text(0.06, -0.85, "Assigned", ha="left", va="bottom", fontsize=style.SMALL, fontweight="bold")
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, label_ax, cross: pd.DataFrame, configs: pd.DataFrame, primary: list[str]) -> None:
    table = cross[cross["element"] == "all"]
    values = table["force_rmse_meV_A"]
    norm = LogNorm(vmin=values.min() * 0.75, vmax=values.max() * 1.6)
    rows, y = [], 0
    for model in common.MODEL_ORDER:
        for labels in ("k222", "gamma"):
            for x, subset in enumerate(SUBSET_COLUMNS):
                match = table[(table["model"] == model) & (table["labels"] == labels) & (table["subset"] == subset)]
                if match.empty:  # own validation: the fine-tunes only
                    continue
                row = match.iloc[0]
                value = row["force_rmse_meV_A"]
                shade = float(norm(value))
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=HEAT(shade), edgecolor=style.WHITE,
                                       linewidth=0.8))
                trained = _trained_on(model, subset, configs)
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=style.SMALL,
                        color=style.WHITE if shade > 0.5 else style.BLACK, style="italic" if trained else "normal")
                rows.append({"model": model, "labels": labels, "subset": subset, "force_rmse_meV_A": value,
                             "n_configs": int(row["n_configs"]), "includes_training_configs": trained,
                             "column": x, "row": y})
            label_ax.text(0.99, y, f"vs {common.LABEL_SET_NAMES[labels]}", ha="right", va="center",
                          fontsize=style.SMALL, color=style.GRAY)
            y += 1
        primary_model = model in primary
        label_ax.text(0.1, y - 1.5, common.SHORT_LABELS[model] + ("" if primary_model else "\n(no MD run)"),
                      ha="left", va="center", fontsize=style.SMALL, linespacing=1.1,
                      color=style.BLACK if primary_model else style.GRAY)
        label_ax.plot(0.04, y - 1.5, **common.model_style(model, size=3.0, edge=0.6))
    for axis in (ax, label_ax):
        axis.set_ylim(y - 0.5, -0.5)
    ax.set_xlim(-0.5, len(SUBSET_COLUMNS) - 0.5)
    ax.set_yticks([])
    counts = {s: sorted(set(table.loc[table["subset"] == s, "n_configs"].dropna().astype(int))) for s in SUBSET_COLUMNS}
    ax.set_xticks(range(len(SUBSET_COLUMNS)),
                  [f"{SUBSET_NAMES[s]}\n(n = {' or '.join(str(c) for c in counts[s])})" for s in SUBSET_COLUMNS],
                  fontsize=style.SMALL)
    ax.tick_params(axis="x", pad=2)
    label_ax.set_xlim(0, 1)
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _panel_c(ax, cross: pd.DataFrame) -> None:
    offsets = dict(zip(common.MODEL_ORDER, np.linspace(-SPREAD, SPREAD, len(common.MODEL_ORDER))))
    rows = []
    ax.axhline(1.0, **style.UNITY_LINE)
    for index, element in enumerate(CATEGORIES):
        for model in common.MODEL_ORDER:
            row = cross[(cross["model"] == model) & (cross["labels"] == "k222") & (cross["subset"] == "H3")
                        & (cross["element"] == element)].iloc[0]
            ax.plot(index + offsets[model], row["softening_slope"],
                    **common.model_style(model, size=SIZE, edge=EDGE, zorder=3))
            rows.append({"element": element, "model": model, "x": index + offsets[model],
                         "softening_slope": row["softening_slope"], "n_components": int(row["n_components"])})
    ax.set_xticks(range(len(CATEGORIES)), ["All", *common.ELEMENTS])
    ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
    ax.set_ylim(0.75, 1.03)
    ax.set_yticks([0.8, 0.9, 1.0])
    ax.set_ylabel("Softening slope on H3\n(model vs 2x2x2 force)")
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, cross: pd.DataFrame, cross_configs: pd.DataFrame, floor: pd.DataFrame) -> None:
    pressure_floor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"), "pressure_rmse_kbar"].iloc[0]
    ax.axhspan(-pressure_floor, pressure_floor, **style.BAND_FILL)
    ax.axhline(0, **style.UNITY_LINE)
    rows = []
    for index, model in enumerate(common.MODEL_ORDER):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")].sort_values(
            "config_id")
        held = (per["subset"] == "H3").to_numpy()
        values = per["pressure_error_kbar"].to_numpy()
        # T22 and H2 as a jittered strip on the left of the slot, the three H3 side by side on the right
        offsets = np.empty(len(per))
        offsets[~held] = -0.22 + common.jitter(int((~held).sum()), 0.34, seed=20 + index)
        offsets[held] = H3_OFFSETS[:int(held.sum())]
        ax.scatter(index + offsets[~held], values[~held],
                   **common.model_scatter(model, size=1.7, edge=0.4, alpha=0.4, zorder=2))
        ax.scatter(index + offsets[held], values[held], **common.model_scatter(model, size=2.8, edge=0.8, zorder=3))
        value = cross[(cross["model"] == model) & (cross["labels"] == "k222") & (cross["subset"] == "All27")
                      & (cross["element"] == "all")]["pressure_rmse_kbar"].iloc[0]
        ax.text(index, 3.0, f"{value:.2f}", ha="center", va="bottom", fontsize=style.SMALL)
        rows.extend({"model": model, "config_id": c, "subset": s, "x": index + o, "pressure_error_kbar": v,
                     "all27_rmse_kbar": value, "label_floor_kbar": pressure_floor}
                    for c, s, o, v in zip(per["config_id"], per["subset"], offsets, values))
    ax.text(-0.45, 4.35, "All27 RMSE (kbar)", ha="left", va="bottom", fontsize=style.SMALL)
    _model_ticks(ax)
    ax.set_ylim(-10.3, 5.9)
    ax.set_yticks([-8, -6, -4, -2, 0, 2])
    ax.set_ylabel("Pressure error vs 2x2x2 (kbar)")
    ax.legend(handles=[_grey_handle(2.8, "H3"), _grey_handle(1.7, "T22, H2", alpha=0.4),
                       Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none",
                                 label=f"label floor, +/-{pressure_floor:.2f}")],
              loc="lower left", fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.7, ncol=3,
              columnspacing=0.7, handlelength=1.0)
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _panel_e(ax, decomposition: pd.DataFrame) -> None:
    ax.axhline(0.0, **style.UNITY_LINE)
    ax.axhline(-2.0, **style.UNITY_LINE)
    rows = []
    for index, model in enumerate(common.MODEL_ORDER):
        for subset, look in DECOMPOSITION_SUBSETS.items():
            row = decomposition[(decomposition["model"] == model) & (decomposition["subset"] == subset)].iloc[0]
            ax.plot(index + look["dx"], row["cross_fraction"],
                    **common.model_style(model, size=look["size"], edge=EDGE, zorder=3))
            rows.append({"model": model, "subset": subset, "x": index + look["dx"],
                         "rmse_vs_gamma_meV_A": row["rmse_vs_gamma_meV_A"],
                         "rmse_vs_k222_meV_A": row["rmse_vs_k222_meV_A"], "floor_meV_A": row["floor_meV_A"],
                         "cross_term_meV2_A2": row["cross_term_meV2_A2"], "cross_fraction": row["cross_fraction"]})
    ax.text(-0.45, 0.4, "tracks the Gamma labels", ha="left", va="bottom", fontsize=style.SMALL,
            color=style.CONTEXT_TEXT)
    ax.text(-0.45, -2.4, "tracks the 2x2x2 labels", ha="left", va="top", fontsize=style.SMALL,
            color=style.CONTEXT_TEXT)
    _model_ticks(ax)
    ax.set_ylim(-7.2, 1.4)
    ax.set_yticks([-6, -4, -2, 0])
    ax.set_ylabel("Cross term / floor²")
    ax.legend(handles=[_grey_handle(3.4, "H3"), _grey_handle(2.2, "All27")], loc="lower right",
              fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.7, handlelength=1.0)
    style.save_source(pd.DataFrame(rows), NAME, "e")


def _legend(fig, layout: common.Layout, top: float) -> None:
    handles = [common.handle(common.model_style(m, size=3.4), common.SHORT_LABELS[m]) for m in common.MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, layout.y(top)), ncol=len(handles),
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0, frameon=False)


def make() -> list[Path]:
    fingerprints, cross, cross_configs, floor, decomposition, configs = _load()
    primary = common.primary_models(cross)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    _legend(fig, L, 2.5)

    # row 1: a (fingerprints, the hero), b (cross-evaluation table)
    top1, height1 = 15.0, 60.0
    ax_a_series = L.axes(2.0, top1, 14.0, height1, frame="none")
    ax_a = L.axes(25.0, top1, 47.0, height1, row="1", col="left")
    ax_a_assigned = L.axes(72.5, top1, 17.0, height1, frame="none")
    _panel_a(ax_a, ax_a_series, ax_a_assigned, fingerprints)
    ax_b_labels = L.axes(99.0, top1, 27.0, height1, frame="none")
    ax_b = L.axes(127.0, top1, 54.0, height1, frame="none")
    _panel_b(ax_b, ax_b_labels, cross, configs, primary)

    # row 2: c (softening), d (pressure), e (floor decomposition)
    top2, height2 = 95.0, 36.0
    ax_c = L.axes(25.0, top2, 35.0, height2, row="2", col="left")
    _panel_c(ax_c, cross)
    ax_d = L.axes(72.0, top2, 49.0, height2, row="2")
    _panel_d(ax_d, cross, cross_configs, floor)
    ax_e = L.axes(134.0, top2, 47.0, height2, row="2")
    _panel_e(ax_e, decomposition)

    L.letter(ax_a, "a", 3.0, 11.0)
    L.letter(ax_b, "b", 97.0, 11.0)
    L.letter(ax_c, "c", 3.0, top2 - 4.0)
    L.letter(ax_d, "d", 64.0, top2 - 4.0)
    L.letter(ax_e, "e", 126.0, top2 - 4.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    fingerprints, cross, cross_configs, floor, decomposition, configs = _load()
    fingerprints = fingerprints[fingerprints["series"].isin(SERIES_ORDER)]  # the runs panel a draws
    models = list(common.MODEL_ORDER)
    md = common.md_models(cross)
    residual =fingerprints[[f"rms_{m}_meV_A" for m in models]].to_numpy()
    best = residual.min(axis=1)
    others = np.sort(residual, axis=1)[:, 1]
    assigned = fingerprints.groupby("series")["assigned_model"].agg(lambda s: ", ".join(sorted(set(s))))
    run_text = "; ".join(f"{SERIES_LABELS[s]} -> {common.SHORT_LABELS.get(assigned[s], assigned[s])}"
                         for s in SERIES_ORDER)
    temperatures = sorted(fingerprints["temperature_K"].unique())
    temperatures = ", ".join(str(t) for t in temperatures[:-1]) + f" and {temperatures[-1]}"
    table = cross[cross["element"] == "all"].set_index(["model", "labels", "subset"])
    h3 = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'H3'), 'force_rmse_meV_A']:.1f}" for m in models)
    slopes = cross[(cross["labels"] == "k222") & (cross["subset"] == "H3")].set_index(["model", "element"])
    components = ", ".join(f"{e} {int(slopes.loc[('foundation', e), 'n_components'])}" for e in common.ELEMENTS)
    found = ", ".join(f"{e} {slopes.loc[('foundation', e), 'softening_slope']:.2f}" for e in CATEGORIES)
    tuned = slopes.loc[[m for m in models if m != "foundation"], "softening_slope"]
    pressure = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'All27'), 'pressure_rmse_kbar']:.2f}"
                         for m in models)
    pk = cross_configs[cross_configs["labels"] == "k222"].groupby("model")["pressure_error_kbar"].mean()
    n_pressure = cross_configs[(cross_configs["labels"] == "k222") & (cross_configs["model"] == "foundation")]
    tuned_mean = pk.loc[[m for m in models if m != "foundation"]]
    pfloor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"), "pressure_rmse_kbar"].iloc[0]
    dec = decomposition.set_index(["model", "subset"])
    floors = {s: dec.xs(s, level="subset")["floor_meV_A"].iloc[0] for s in DECOMPOSITION_SUBSETS}
    fractions = "; ".join(f"{common.SHORT_LABELS[m]} {dec.loc[(m, 'H3'), 'cross_fraction']:.2f} "
                          f"({dec.loc[(m, 'All27'), 'cross_fraction']:.2f})" for m in models)
    return (
        registry.heading(KEY)
        + "Markers (legend at top): MACE-MP-0 red; the fine-tuned models navy (Gamma labels) or blue (2x2x2 "
        "labels), filled for stage 2 and open for stage 1. "
        + common.naming_sentence(configs) + " "
        f"**a**, Force fingerprints of the 416-atom, 200-ps MD runs at {temperatures} K (n = {len(fingerprints)} "
        f"runs): RMS difference between the forces stored in four frames of each run (first, one third, two "
        f"thirds, last) and the forces of each candidate model recomputed in float32 on a CPU (logarithmic axis). "
        f"A run is assigned to a model if the residual is below {FINGERPRINT_MATCH_meV_A:g} meV/A (dashed line) "
        f"and every other candidate is at least {FINGERPRINT_RATIO:g} times worse. The matching model gives "
        f"{best.min():.4f}-{best.max():.4f} meV/A, the next best {others.min():.1f}-{others.max():.1f} meV/A. "
        f"Assignments: {run_text}. These are the models of the MD runs in {registry.label('diffusion')} and "
        f"{registry.label('finetune')} ({'; '.join(common.SHORT_LABELS[md[f]] for f in common.RUN_FAMILIES)}); "
        "the other two fine-tuned models have no MD run. Single values, no error bars. "
        f"**b**, Force RMSE (meV/A, all force components of the n configurations under each column) of all five "
        f"models against both label sets on each subset. Own validation is defined for the fine-tunes only: "
        f"V5 = H3 + H2 for the Gamma fine-tune, H3 for the 2x2x2 fine-tune. Shading is logarithmic in the RMSE. "
        f"Italic values include configurations the model was trained on; the primary estimand is the H3 column "
        f"against 2x2x2 ({h3}), as in {registry.label('forces')}. Single values, no error bars. "
        f"**c**, Softening slope (least-squares slope through the origin of model against 2x2x2 force components) "
        f"on H3, overall and per element (n = {int(slopes.loc[('foundation', 'all'), 'n_components'])} "
        f"components; {components}). MACE-MP-0: {found}; fine-tuned models {tuned.min():.2f}-{tuned.max():.2f}. "
        f"Line, slope 1. Single values, no error bars. "
        f"**d**, Pressure error against the 2x2x2 labels for each configuration (n = {len(n_pressure)} per model; "
        f"T22 and H2 small and pale on the left of each model, the three H3 configurations large on the right), a "
        f"diagnostic only: no model was trained on stress. Numbers above, All27 RMSE "
        f"(kbar): {pressure}. The fine-tuned models underestimate the pressure by "
        f"{-tuned_mean.max():.1f}-{-tuned_mean.min():.1f} kbar on average (MACE-MP-0: {pk['foundation']:+.2f} "
        f"kbar), well outside the label floor of +/-{pfloor:.2f} kbar (grey band). Single values per "
        f"configuration, no error bars. "
        f"**e**, Decomposition of the force error against the two label sets, RMSE(m vs 2x2x2)^2 = "
        f"RMSE(m vs Gamma)^2 + floor^2 + cross term, over all force components of H3 (large markers; floor "
        f"{floors['H3']:.1f} meV/A) and All27 (small markers; floor {floors['All27']:.1f} meV/A). The cross term "
        f"divided by floor^2 is 0 for a model whose error against the Gamma labels is uncorrelated with the "
        f"Gamma minus 2x2x2 difference (it tracks the Gamma labels) and -2 for a model whose error against the "
        f"2x2x2 labels is uncorrelated with it (it tracks the 2x2x2 labels). H3 (All27): {fractions}. The Gamma "
        f"fine-tunes lie between the two references, closer to 0 on All27, which contains their training "
        f"configurations; the 2x2x2 fine-tunes lie near -2. MACE-MP-0, trained on neither label set, lies below "
        f"-2: its error against the 2x2x2 labels is anticorrelated with the label difference. Single values, no "
        f"error bars. "
        + common.sources([FINGERPRINTS, CROSS, CROSS_CONFIGS, FLOOR, DECOMPOSITION, CONFIGS])
    )


CAPTION = _caption()
