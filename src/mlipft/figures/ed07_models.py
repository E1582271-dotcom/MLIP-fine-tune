"""Extended Data figure "models": model identity and the full cross-evaluation.

a, force fingerprints: RMS residual between the forces stored in each MACE MD run and the forces of
every candidate MACE model, and the model each run is assigned to (both 2x2x2 runs: the July run and
the run of amendment A5); b, force RMSE of the two universal potentials and the four fine-tunes against
both label sets on every subset; c, softening slope per element; d, pressure error per configuration (a
diagnostic: no model was trained on stress); e, the cross term of the label-floor decomposition, which
places each model between the two label sets. d and e share their model rows. MatterSim (amendment A4)
is evaluated in b-e; it is not a candidate in a, its runs were checked against MatterSim alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from ..ingest.models import FINGERPRINT_RATIO, FINGERPRINT_MATCH_meV_A
from . import common, registry, style

KEY = "models"
NAME = registry.number(KEY)
PANELS = "abcde"
WIDTH, HEIGHT = 183.0, 164.0
FINGERPRINTS = paths.MODELS / "fingerprints.csv"
MS_IDENTITY = paths.MODELS / "mattersim_identity.csv"  # amendment A4
EVALUATION = paths.MODELS / "evaluation.json"
MS_EVALUATION = paths.MODELS / "mattersim_evaluation.json"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
DECOMPOSITION = paths.RESULTS / "floor_decomposition.csv"
CONFIGS = paths.DFT / "configs.csv"
# b-e: the universal potentials, then the fine-tunes in MODEL_ORDER (a: the MACE candidates, MODEL_ORDER)
MODELS = (*style.UNIVERSAL_MODELS, *(m for m in common.MODEL_ORDER if m not in style.UNIVERSAL_MODELS))
SERIES_ORDER = ("mace_416_200", "ftG_416_200", "ftK_416_200", "ftK2_416_200")
SERIES_LABELS = {"mace_416_200": "MACE-MP-0 run", "ftG_416_200": "Gamma fine-tune run",
                 "ftK_416_200": "July 2x2x2 fine-tune run", "ftK2_416_200": "2x2x2 fine-tune run of amendment A5"}
SERIES_TEXT = {"mace_416_200": "MACE-MP-0\nrun", "ftG_416_200": "Gamma\nfine-tune run",
               "ftK_416_200": "2x2x2\nfine-tune run\n(July)", "ftK2_416_200": "2x2x2\nfine-tune run\n(A5)"}
SUBSET_COLUMNS = ("H3", "H2", "T22", "own_valid", "All27")
SUBSET_NAMES = {"H3": "H3", "H2": "H2", "T22": "T22", "own_valid": "Own\nvalidation", "All27": "All27"}
CATEGORIES = ("all", *common.ELEMENTS)
SIZE = 2.4  # marker diameter (pt) of the model markers
EDGE = 0.5  # marker edge width (pt)
SPREAD = 0.42  # the candidate models sit within +/-SPREAD of a row (a), the six models of a category (c)
HEAT = LinearSegmentedColormap.from_list("ed07_rmse", [style.WHITE, style.BLACK])
H3_OFFSETS = np.array([-0.07, -0.23, -0.39])  # d: the three H3 configurations, in id order, above the row centre
STRIP = (0.24, 0.24)  # d: centre and width of the T22 and H2 strip, below the row centre
RMSE_X = 5.9  # d: x of the All27 RMSE column (kbar)
DECOMPOSITION_SUBSETS = {"H3": {"dy": -0.15, "size": 3.4}, "All27": {"dy": 0.15, "size": 2.2}}


def _load():
    fingerprints = common.read(FINGERPRINTS)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    floor = common.read(FLOOR)
    decomposition = common.read(DECOMPOSITION)
    configs = common.read(CONFIGS)
    return fingerprints, cross, cross_configs, floor, decomposition, configs


def _drove_md(cross: pd.DataFrame) -> set[str]:
    """Models that drove an MD run (non-empty ``drove_md``), MatterSim included."""
    return set(cross.loc[cross["drove_md"].fillna("").astype(str) != "", "model"])


def _trained_on(model: str, subset: str, configs: pd.DataFrame) -> bool:
    """Whether ``subset`` contains configurations the model was trained on."""
    if model in style.UNIVERSAL_MODELS or subset == "own_valid":
        return False
    split = configs[f"{model.split('_')[0]}_split"]
    members = configs if subset == "All27" else configs[configs["subset"] == subset]
    return bool((split.loc[members.index] == "train").any())


def _model_rows(ax, labels: bool) -> None:
    """d, e: one row per model, top to bottom in ``MODELS``."""
    ax.set_yticks(range(len(MODELS)), [common.SHORT_LABELS[m] for m in MODELS], fontsize=style.SMALL)
    ax.tick_params(axis="y", labelleft=labels)
    ax.set_ylim(len(MODELS) - 0.45, -0.62)


def _above(ax, x: float, text: str, **kwargs):
    """Text centred on data ``x``, 1.5 pt above the top frame of ``ax``."""
    above = transforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=1.5, units="points")
    return ax.text(x, 1.0, text, transform=above, ha="center", va="bottom", fontsize=style.SMALL, linespacing=1.1,
                   **kwargs)


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
            if assigned in style.MODEL_COLORS:
                assigned_ax.plot(0.07, y, **common.model_style(assigned, size=SIZE + 0.5, edge=0.6))
            assigned_ax.text(0.15, y, common.SHORT_LABELS.get(assigned, assigned), ha="left", va="center",
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
    assigned_ax.text(0.03, -0.85, "Assigned", ha="left", va="bottom", fontsize=style.SMALL, fontweight="bold")
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, label_ax, cross: pd.DataFrame, configs: pd.DataFrame) -> None:
    table = cross[cross["element"] == "all"]
    values = table["force_rmse_meV_A"]
    norm = LogNorm(vmin=values.min() * 0.75, vmax=values.max() * 1.6)
    drove = _drove_md(cross)
    rows, y = [], 0
    for model in MODELS:
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
                             "drove_md": model in drove, "column": x, "row": y})
            label_ax.text(0.99, y, f"vs {common.LABEL_SET_NAMES[labels]}", ha="right", va="center",
                          fontsize=style.SMALL, color=style.GRAY)
            y += 1
        md_model = model in drove
        label_ax.text(0.1, y - 1.5, common.SHORT_LABELS[model] + ("" if md_model else "\n(no MD run)"),
                      ha="left", va="center", fontsize=style.SMALL, linespacing=1.1,
                      color=style.BLACK if md_model else style.GRAY)
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
    offsets = dict(zip(MODELS, np.linspace(-SPREAD, SPREAD, len(MODELS))))
    rows = []
    ax.axhline(1.0, **style.UNITY_LINE)
    for index, element in enumerate(CATEGORIES):
        for model in MODELS:
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
    ax.axvspan(-pressure_floor, pressure_floor, **style.BAND_FILL)
    ax.axvline(0, **style.UNITY_LINE)
    rows = []
    for index, model in enumerate(MODELS):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")].sort_values(
            "config_id")
        held = (per["subset"] == "H3").to_numpy()
        values = per["pressure_error_kbar"].to_numpy()
        # T22 and H2 as a jittered strip below the row centre, the three H3 staggered above it
        offsets = np.empty(len(per))
        offsets[~held] = STRIP[0] + common.jitter(int((~held).sum()), STRIP[1], seed=20 + index)
        offsets[held] = H3_OFFSETS[:int(held.sum())]
        ax.scatter(values[~held], index + offsets[~held],
                   **common.model_scatter(model, size=1.7, edge=0.4, alpha=0.4, zorder=2))
        ax.scatter(values[held], index + offsets[held], **common.model_scatter(model, size=2.7, edge=0.7, zorder=3))
        value = cross[(cross["model"] == model) & (cross["labels"] == "k222") & (cross["subset"] == "All27")
                      & (cross["element"] == "all")]["pressure_rmse_kbar"].iloc[0]
        ax.text(RMSE_X, index, f"{value:.2f}", ha="center", va="center", fontsize=style.SMALL)
        rows.extend({"model": model, "config_id": c, "subset": s, "y": index + o, "pressure_error_kbar": v,
                     "all27_rmse_kbar": value, "label_floor_kbar": pressure_floor}
                    for c, s, o, v in zip(per["config_id"], per["subset"], offsets, values))
    _above(ax, RMSE_X, "All27\nRMSE")
    _model_rows(ax, labels=True)
    ax.set_xlim(-9.2, 7.6)
    ax.set_xticks([-8, -6, -4, -2, 0, 2, 4])
    ax.set_xlabel("Pressure error vs 2x2x2 (kbar)")
    ax.legend(handles=[_grey_handle(2.7, "H3"), _grey_handle(1.7, "T22, H2", alpha=0.4),
                       Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none",
                                 label=f"label floor,\n+/-{pressure_floor:.2f}")],
              loc="upper left", fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.4, handlelength=1.0,
              labelspacing=0.3)
    style.save_source(pd.DataFrame(rows), NAME, "d")


def _panel_e(ax, decomposition: pd.DataFrame) -> None:
    rows = []
    for index, model in enumerate(MODELS):
        for subset, look in DECOMPOSITION_SUBSETS.items():
            row = decomposition[(decomposition["model"] == model) & (decomposition["subset"] == subset)].iloc[0]
            ax.plot(row["cross_fraction"], index + look["dy"],
                    **common.model_style(model, size=look["size"], edge=EDGE, zorder=3))
            rows.append({"model": model, "subset": subset, "y": index + look["dy"],
                         "rmse_vs_gamma_meV_A": row["rmse_vs_gamma_meV_A"],
                         "rmse_vs_k222_meV_A": row["rmse_vs_k222_meV_A"], "floor_meV_A": row["floor_meV_A"],
                         "cross_term_meV2_A2": row["cross_term_meV2_A2"], "cross_fraction": row["cross_fraction"]})
    _model_rows(ax, labels=False)
    for x, labels in ((0.0, "gamma"), (-2.0, "k222")):  # the two references, named above the frame
        ax.axvline(x, **style.UNITY_LINE)
        _above(ax, x, f"tracks\n{common.LABEL_SET_NAMES[labels]}", color=style.CONTEXT_TEXT)
    ax.set_xlim(-7.2, 1.0)
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_xlabel("Cross term / floor²")
    ax.legend(handles=[_grey_handle(3.4, "H3"), _grey_handle(2.2, "All27")], loc="lower left",
              fontsize=style.SMALL, handletextpad=0.3, borderaxespad=0.5, handlelength=1.0)
    style.save_source(pd.DataFrame(rows), NAME, "e")


def _legend(fig, layout: common.Layout, top: float) -> None:
    handles = [common.handle(common.model_style(m, size=3.4), common.SHORT_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, layout.y(top)), ncol=len(handles),
               fontsize=style.SMALL, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0, frameon=False)


def make() -> list[Path]:
    fingerprints, cross, cross_configs, floor, decomposition, configs = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)
    _legend(fig, L, 2.5)

    # row 1: a (fingerprints, the hero), b (cross-evaluation table); twelve rows each
    top1, height1 = 15.0, 68.0
    ax_a_series = L.axes(2.0, top1, 14.0, height1, frame="none")
    ax_a = L.axes(25.0, top1, 47.0, height1, row="1", col="left")
    ax_a_assigned = L.axes(73.0, top1, 18.0, height1, frame="none")
    _panel_a(ax_a, ax_a_series, ax_a_assigned, fingerprints)
    ax_b_labels = L.axes(99.0, top1, 27.0, height1, frame="none")
    ax_b = L.axes(127.0, top1, 54.0, height1, frame="none")
    _panel_b(ax_b, ax_b_labels, cross, configs)

    # row 2: c (softening), d (pressure) and e (floor decomposition), d and e on shared model rows
    top2, height2 = 104.0, 50.0
    ax_c = L.axes(25.0, top2, 41.0, height2, row="2", col="left")
    _panel_c(ax_c, cross)
    ax_d = L.axes(88.0, top2, 47.0, height2, row="2")
    _panel_d(ax_d, cross, cross_configs, floor)
    ax_e = L.axes(139.0, top2, 42.0, height2, row="2")
    _panel_e(ax_e, decomposition)

    L.letter(ax_a, "a", 3.0, 11.0)
    L.letter(ax_b, "b", 97.0, 11.0)
    L.letter(ax_c, "c", 3.0, top2 - 4.0)
    L.letter(ax_d, "d", 70.0, top2 - 4.0)
    L.letter(ax_e, "e", 138.0, top2 - 4.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    fingerprints, cross, cross_configs, floor, decomposition, configs = _load()
    fingerprints = fingerprints[fingerprints["series"].isin(SERIES_ORDER)]  # the runs panel a draws
    candidates = list(common.MODEL_ORDER)
    models = list(MODELS)
    tuned_models = [m for m in models if m not in style.UNIVERSAL_MODELS]
    md = common.md_models(cross)
    drove = _drove_md(cross)
    no_md = [m for m in candidates if m not in drove]
    residual = fingerprints[[f"rms_{m}_meV_A" for m in candidates]].to_numpy()
    best = residual.min(axis=1)
    others = np.sort(residual, axis=1)[:, 1]
    assigned = fingerprints.groupby("series")["assigned_model"].agg(lambda s: ", ".join(sorted(set(s))))
    run_text = "; ".join(f"{SERIES_LABELS[s]} -> {common.SHORT_LABELS.get(assigned[s], assigned[s])}"
                         for s in SERIES_ORDER)
    temperatures = sorted(fingerprints["temperature_K"].unique())
    temperatures = ", ".join(str(t) for t in temperatures[:-1]) + f" and {temperatures[-1]}"
    identity = common.read(MS_IDENTITY)
    precision = json.loads(MS_EVALUATION.read_text(encoding="utf-8"))["precision"]
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    mace_precision = evaluation["labels_dtype"]
    table = cross[cross["element"] == "all"].set_index(["model", "labels", "subset"])
    h3 = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'H3'), 'force_rmse_meV_A']:.1f}" for m in models)
    slopes = cross[(cross["labels"] == "k222") & (cross["subset"] == "H3")].set_index(["model", "element"])
    components = ", ".join(f"{e} {int(slopes.loc[('foundation', e), 'n_components'])}" for e in common.ELEMENTS)

    def slope_text(model):
        return ", ".join(f"{e} {slopes.loc[(model, e), 'softening_slope']:.2f}" for e in CATEGORIES)

    tuned = slopes.loc[tuned_models, "softening_slope"]
    pressure = "; ".join(f"{common.SHORT_LABELS[m]} {table.loc[(m, 'k222', 'All27'), 'pressure_rmse_kbar']:.2f}"
                         for m in models)
    pk = cross_configs[cross_configs["labels"] == "k222"].groupby("model")["pressure_error_kbar"].mean()
    n_pressure = cross_configs[(cross_configs["labels"] == "k222") & (cross_configs["model"] == "foundation")]
    tuned_mean = pk.loc[tuned_models]
    universal_mean = "; ".join(f"{common.SHORT_LABELS[m]} {pk[m]:+.2f}" for m in style.UNIVERSAL_MODELS)
    pfloor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"), "pressure_rmse_kbar"].iloc[0]
    dec = decomposition.set_index(["model", "subset"])
    floors = {s: dec.xs(s, level="subset")["floor_meV_A"].iloc[0] for s in DECOMPOSITION_SUBSETS}
    position = {m: dec.loc[m, "cross_fraction"] for m in style.UNIVERSAL_MODELS}
    if not ((position["mattersim"].between(-2.0, 0.0)).all() and (position["foundation"] < -2.0).all()):
        raise ValueError("the universal potentials changed sides in the floor decomposition; reword e")
    fractions = "; ".join(f"{common.SHORT_LABELS[m]} {dec.loc[(m, 'H3'), 'cross_fraction']:.2f} "
                          f"({dec.loc[(m, 'All27'), 'cross_fraction']:.2f})" for m in models)
    return (
        registry.heading(KEY)
        + "Markers (legend at top): the universal potentials red (MACE-MP-0 dark, MatterSim light); the fine-tuned "
        "models navy (Gamma labels) or blue (2x2x2 labels), filled for stage 2 and open for stage 1. "
        + common.naming_sentence(configs) + " "
        f"**a**, Force fingerprints of the 416-atom, 200-ps MD runs of MACE-MP-0 and its fine-tunes at "
        f"{temperatures} K (n = {len(fingerprints)} runs; both 2x2x2 fine-tune runs, the July run and the run of "
        f"protocol amendment A5): RMS difference between the forces stored in four frames of each run (first, one "
        f"third, two thirds, last) and the forces of each candidate MACE model recomputed in "
        f"{evaluation['fingerprint_dtype']} on a {evaluation['device'].upper()} (logarithmic axis; the markers of a "
        f"row in the order of the legend, top to bottom). A run is assigned to a model if the residual is below "
        f"{FINGERPRINT_MATCH_meV_A:g} meV/A (dashed line) and every other candidate is at least "
        f"{FINGERPRINT_RATIO:g} times worse. The matching model gives "
        f"{best.min():.4f}-{best.max():.4f} meV/A, the next best {others.min():.1f}-{others.max():.1f} meV/A. "
        f"Assignments: {run_text}. These are the models of the MACE MD runs in {registry.label('diffusion')} and "
        f"{registry.label('finetune')} ({'; '.join(common.SHORT_LABELS[md[f]] for f in common.RUN_FAMILIES)}); "
        f"{' and '.join(common.SHORT_LABELS[m] for m in no_md)} has no MD run. MatterSim is not a candidate: the "
        f"forces stored in its runs, recomputed with MatterSim, match to "
        f"{identity['rms_meV_A'].min():.4f}-{identity['rms_meV_A'].max():.4f} meV/A (n = {len(identity)} runs). Single values, no error bars. "
        f"In b-e, MatterSim was evaluated in {precision} (the precision of its MD runs) and the MACE models in "
        f"{mace_precision} (protocol amendment A4). "
        f"**b**, Force RMSE (meV/A, all force components of the n configurations under each column) of every "
        f"model against both label sets on each subset. Own validation is defined for the fine-tunes only: "
        f"V5 = H3 + H2 for the Gamma fine-tune, H3 for the 2x2x2 fine-tune. Shading is logarithmic in the RMSE. "
        f"Italic values include configurations the model was trained on; the primary estimand is the H3 column "
        f"against 2x2x2 ({h3}), as in {registry.label('forces')}. Single values, no error bars. "
        f"**c**, Softening slope (least-squares slope through the origin of model against 2x2x2 force components) "
        f"on H3, overall and per element (n = {int(slopes.loc[('foundation', 'all'), 'n_components'])} "
        f"components; {components}); the markers of a category in the order of the legend, left to right. "
        f"MACE-MP-0: {slope_text('foundation')}; MatterSim: {slope_text('mattersim')}; fine-tuned models "
        f"{tuned.min():.2f}-{tuned.max():.2f}. Line, slope 1. Single values, no error bars. "
        f"**d**, Pressure error against the 2x2x2 labels for each configuration (n = {len(n_pressure)} per model; "
        f"T22 and H2 small and pale below the centre of each model's row, the three H3 configurations large above "
        f"it), a diagnostic only: no model was trained on stress. Numbers on the right, All27 RMSE "
        f"(kbar): {pressure}. The fine-tuned models underestimate the pressure by "
        f"{-tuned_mean.max():.1f}-{-tuned_mean.min():.1f} kbar on average (mean error of the universal potentials: "
        f"{universal_mean} kbar), well outside the label floor of +/-{pfloor:.2f} kbar (grey band). Single values "
        f"per configuration, no error bars. "
        f"**e**, Decomposition of the force error against the two label sets, RMSE(m vs 2x2x2)^2 = "
        f"RMSE(m vs Gamma)^2 + floor^2 + cross term, over all force components of H3 (large markers; floor "
        f"{floors['H3']:.1f} meV/A) and All27 (small markers; floor {floors['All27']:.1f} meV/A); model rows as in "
        f"d. The cross term divided by floor^2 is 0 for a model whose error against the Gamma labels is "
        f"uncorrelated with the Gamma minus 2x2x2 difference (it tracks the Gamma labels) and -2 for a model whose "
        f"error against the 2x2x2 labels is uncorrelated with it (it tracks the 2x2x2 labels). H3 (All27): "
        f"{fractions}. The Gamma fine-tunes lie between the two references, closer to 0 on All27, which contains "
        f"their training configurations; the 2x2x2 fine-tunes lie near -2. Of the universal potentials, trained on "
        f"neither label set, MatterSim lies between the two references and MACE-MP-0 below -2: the error of "
        f"MACE-MP-0 against the 2x2x2 labels is anticorrelated with the label difference. Single values, no error "
        f"bars. "
        + common.sources([FINGERPRINTS, MS_IDENTITY, EVALUATION, MS_EVALUATION, CROSS, CROSS_CONFIGS, FLOOR,
                          DECOMPOSITION, CONFIGS])
    )


CAPTION = _caption()
