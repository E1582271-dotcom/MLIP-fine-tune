"""Fig. 3: the fine-tuned models against their DFT reference.

a, the 27 labelled configurations and their subsets; b, the label-error floor (Gamma against
2x2x2 labels); c, force RMSE on H3 against both label sets; d, force parity against the 2x2x2
labels for the three models that drove MD; e, per-element force RMSE on H3; f, centred energy
error per configuration. Fine-tuned models are compared with DFT and with the foundation only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from . import common, style

NAME = "fig3"
WIDTH, HEIGHT = 183.0, 150.0
CONFIGS = paths.DFT / "configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
FLOOR_CONFIGS = paths.RESULTS / "label_floor_configs.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FORCES = paths.MODELS / "predictions_forces.csv.gz"
SUBSET_ORDER = ("T22", "H3", "H2")
SUBSET_TEXT = {"T22": "T22: trained on by both fine-tunes", "H3": "H3: held out by both (primary test set)",
               "H2": "H2: held out by the Gamma fine-tune only"}
LABEL_MARKERS = {"k222": "o", "gamma": "s"}


def _load():
    configs = common.ordered_configs(common.read(CONFIGS))
    floor = common.read(FLOOR)
    floor_configs = common.read(FLOOR_CONFIGS)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    forces = common.read(FORCES).merge(configs[["config_id", "subset"]], on="config_id")
    return configs, floor, floor_configs, cross, cross_configs, forces


def _row(cross: pd.DataFrame, model: str, labels: str, subset: str, element: str = "all") -> pd.Series:
    rows = cross[(cross["model"] == model) & (cross["labels"] == labels) & (cross["subset"] == subset)
                 & (cross["element"] == element)]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {model}, {labels}, {subset}, {element}")
    return rows.iloc[0]


def _components(forces: pd.DataFrame, prefix: str) -> np.ndarray:
    return forces[[f"{prefix}_f{axis}" for axis in "xyz"]].to_numpy()


def _floor_per_config_element(forces: pd.DataFrame) -> pd.DataFrame:
    """RMSE of Gamma minus 2x2x2 force components per configuration and element (meV/A)."""
    diff = (_components(forces, "dft_gamma") - _components(forces, "dft_k222")) * 1000.0
    frame = forces[["config_id", "subset", "species"]].assign(sq=(diff ** 2).mean(axis=1))
    out = frame.groupby(["config_id", "subset", "species"], sort=True)["sq"].mean().pow(0.5)
    return out.rename("force_rmse_meV_A").reset_index().rename(columns={"species": "element"})


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(ax, configs: pd.DataFrame, layout: common.Layout) -> None:
    temperatures = sorted(configs["temperature_K"].unique())
    frames = sorted(configs["frame"].unique())
    for row in configs.itertuples():
        x, y = frames.index(row.frame), temperatures.index(row.temperature_K)
        ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor=style.SUBSET_COLORS[row.subset],
                               edgecolor="none"))
        if row.subset != "T22":
            ax.text(x, y, row.subset, ha="center", va="center", fontsize=style.SMALL,
                    color="white" if row.subset == "H3" else "black")
    ax.set_xlim(-0.55, len(frames) - 0.45)
    ax.set_ylim(len(temperatures) - 0.45, -0.55)
    ax.set_xticks(range(len(frames)), [str(f) for f in frames], fontsize=style.SMALL)
    ax.set_yticks(range(len(temperatures)), [f"{t} K" for t in temperatures])
    ax.tick_params(length=0, pad=1.5)
    common.no_frame(ax)
    ax.set_xlabel("Frame of the 50-ps MACE-MP-0 run", labelpad=2)
    counts = configs["subset"].value_counts()
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=4.5, markeredgecolor="none",
                      markerfacecolor=style.SUBSET_COLORS[s], label=f"{SUBSET_TEXT[s]} ({counts[s]})")
               for s in SUBSET_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.32, -0.62), handletextpad=0.4,
              labelspacing=0.45, borderaxespad=0.0, fontsize=style.SMALL)
    style.save_source(configs[["config_id", "temperature_K", "frame", "subset", "ftG_split", "ftK_split"]], NAME,
                      "a")


def _panel_b(ax, floor: pd.DataFrame, floor_configs: pd.DataFrame, per_element: pd.DataFrame,
             layout: common.Layout, header: tuple[float, float]) -> None:
    categories = ["all", *common.ELEMENTS]
    rows = []
    for index, element in enumerate(categories):
        if element == "all":
            values = floor_configs[["config_id", "subset", "force_rmse_meV_A"]]
            color = style.GREY
        else:
            values = per_element.loc[per_element["element"] == element, ["config_id", "subset", "force_rmse_meV_A"]]
            color = style.ELEMENT_COLORS[element]
        values = values.sort_values("config_id")
        offsets = common.jitter(len(values), 0.5, seed=index)
        ax.scatter(index + offsets, values["force_rmse_meV_A"], s=4, facecolors=color, edgecolors="none",
                   alpha=0.75, zorder=2)
        total = floor.loc[(floor["subset"] == "All27") & (floor["element"] == element), "force_rmse_meV_A"].iloc[0]
        ax.plot([index - 0.33, index + 0.33], [total, total], color="black", lw=1.0, solid_capstyle="butt", zorder=3)
        rows.extend({"element": element, "config_id": c, "subset": s, "force_rmse_meV_A": v, "x": index + o,
                     "all27_rmse_meV_A": total}
                    for c, s, v, o in zip(values["config_id"], values["subset"], values["force_rmse_meV_A"], offsets))
    ax.set_xticks(range(len(categories)), ["All", *common.ELEMENTS])
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Force RMSE (meV/Å)\nGamma vs 2x2x2 labels")
    whole = floor[(floor["subset"] == "All27") & (floor["element"] == "all")].iloc[0]
    layout.text(header[0], header[1], f"Centred energy floor {whole['energy_rmse_centred_meV_atom']:.2f} meV/atom\n"
                                      f"Pressure floor {whole['pressure_rmse_kbar']:.2f} kbar",
                ha="left", va="bottom", fontsize=style.SMALL, linespacing=1.35)
    ax.legend(handles=[Line2D([0], [0], color="black", lw=1.0, label="All27"),
                       Line2D([0], [0], marker="o", color=style.GREY, linestyle="none", markersize=2.2,
                              markeredgecolor="none", label="per configuration")],
              loc="upper right", fontsize=style.SMALL, handlelength=1.0, borderaxespad=0.1, labelspacing=0.35,
              handletextpad=0.4)
    source = pd.DataFrame(rows)
    source["energy_rmse_centred_all27_meV_atom"] = whole["energy_rmse_centred_meV_atom"]
    source["pressure_rmse_all27_kbar"] = whole["pressure_rmse_kbar"]
    style.save_source(source, NAME, "b")


def _panel_c(ax, label_ax, value_ax, cross: pd.DataFrame, cross_configs: pd.DataFrame, floor: pd.DataFrame,
             primary: list[str]) -> None:
    models = [m for m in common.MODEL_ORDER if m in primary] + [m for m in common.MODEL_ORDER if m not in primary]
    h3_floor = floor.loc[(floor["subset"] == "H3") & (floor["element"] == "all"), "force_rmse_meV_A"].iloc[0]
    lo, hi = 10.0, 400.0
    ax.axvspan(lo, h3_floor, color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.text(np.sqrt(lo * h3_floor), 0.5, "label floor", rotation=90, ha="center", va="center",
            fontsize=style.SMALL, color="#555555", transform=ax.get_xaxis_transform())
    positions, rows = {}, []
    gap = 0.6
    for index, model in enumerate(models):
        base = index + (gap if model not in primary else 0.0)
        positions[model] = base
        faded = model not in primary
        color = common.model_color(model, faded)
        for labels, dy in (("k222", -0.18), ("gamma", 0.18)):
            y = base + dy
            row = _row(cross, model, labels, "H3")
            per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == labels)
                                & (cross_configs["subset"] == "H3")].sort_values("config_id")
            ax.scatter(per["force_rmse_meV_A"], np.full(len(per), y), s=2.5,
                       facecolors=common.lighten(color, 0.35), edgecolors="none", zorder=2)
            ax.plot(row["force_rmse_meV_A"], y, **common.marker_kwargs(model, LABEL_MARKERS[labels], 3.6,
                                                                         faded=faded), zorder=3)
            rows.append({"model": model, "labels": labels, "config_id": "H3 (all 3)",
                         "force_rmse_meV_A": row["force_rmse_meV_A"], "drove_md": model in primary})
            rows.extend({"model": model, "labels": labels, "config_id": c, "force_rmse_meV_A": v,
                         "drove_md": model in primary} for c, v in zip(per["config_id"], per["force_rmse_meV_A"]))
    top, bottom = -0.6, positions[models[-1]] + 0.6
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    style.log_ticks(ax, "x")
    ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Force RMSE on H3 (meV/Å)")
    separator = (positions[models[len(primary) - 1]] + positions[models[len(primary)]]) / 2
    ax.axhline(separator, color=style.LIGHT_GREY, lw=0.5, zorder=1)
    for axis in (label_ax, value_ax):
        axis.set_ylim(bottom, top)
        axis.set_xlim(0, 1)
        axis.axis("off")
    for model in models:
        color = "black" if model in primary else style.GREY
        label_ax.text(1.0, positions[model], style.MODEL_LABELS[model].replace(", ", ",\n"), ha="right",
                      va="center", fontsize=style.SMALL, color=color, linespacing=1.1)
        for labels, x in (("k222", 0.25), ("gamma", 0.70)):
            value = _row(cross, model, labels, "H3")["force_rmse_meV_A"]
            value_ax.text(x, positions[model], f"{value:.1f}", ha="center", va="center", fontsize=style.SMALL,
                          color=color)
    value_ax.text(0.25, top - 0.05, "2x2x2", ha="center", va="bottom", fontsize=style.SMALL)
    value_ax.text(0.70, top - 0.05, "Gamma", ha="center", va="bottom", fontsize=style.SMALL)
    label_ax.text(0.0, separator, "not used for MD", ha="left", va="center", fontsize=style.SMALL,
                  color=style.GREY, style="italic")
    handles = [Line2D([0], [0], marker=LABEL_MARKERS[k], linestyle="none", markersize=3.6, markerfacecolor="black",
                      markeredgecolor="black", label=f"vs {common.LABEL_SET_NAMES[k]} labels")
               for k in ("k222", "gamma")]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=1.8, markerfacecolor=style.GREY,
                          markeredgecolor="none", label="one H3 configuration"))
    label_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=style.SMALL,
                    handletextpad=0.2, columnspacing=1.0, borderaxespad=0.3, labelspacing=0.3)
    source = pd.DataFrame(rows)
    source["h3_label_floor_meV_A"] = h3_floor
    style.save_source(source, NAME, "c")


def _panel_d(axes, forces: pd.DataFrame, cross: pd.DataFrame, primary: list[str]) -> None:
    reference = _components(forces, "dft_k222").ravel()
    species = np.repeat(forces["species"].to_numpy(), 3)
    held_out = np.repeat((forces["subset"] == "H3").to_numpy(), 3)
    limit = 5.0
    source = pd.DataFrame({"config_id": np.repeat(forces["config_id"].to_numpy(), 3),
                           "atom": np.repeat(forces["atom"].to_numpy(), 3),
                           "species": species, "subset": np.repeat(forces["subset"].to_numpy(), 3),
                           "component": np.tile(list("xyz"), len(forces)), "dft_k222_eV_A": reference})
    for index, (ax, model) in enumerate(zip(axes, primary)):
        predicted = _components(forces, model).ravel()
        source[f"{model}_eV_A"] = predicted
        ax.plot([-limit, limit], [-limit, limit], color=style.GREY, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.scatter(reference[~held_out], predicted[~held_out], s=0.6, facecolors=style.LIGHT_GREY,
                   edgecolors="none", zorder=2)
        for element in common.ELEMENTS:
            keep = held_out & (species == element)
            ax.scatter(reference[keep], predicted[keep], s=2.2, facecolors=style.ELEMENT_COLORS[element],
                       edgecolors="none", zorder=3)
        row = _row(cross, model, "k222", "H3")
        slope = row["softening_slope"]
        ax.plot([-limit, limit], [-limit * slope, limit * slope], color=style.MODEL_COLORS[model], lw=0.7, zorder=4)
        ax.text(0.04, 0.96, common.SHORT_LABELS[model], transform=ax.transAxes, ha="left", va="top",
                fontsize=style.FONT_SIZE, color=style.MODEL_COLORS[model], fontweight="bold")
        ax.text(0.04, 0.86, f"H3 RMSE {row['force_rmse_meV_A']:.1f} meV/Å\nslope {slope:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=style.SMALL, linespacing=1.3)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Model force component (eV/Å)")
        if index == 1:
            ax.set_xlabel("2x2x2 DFT force component (eV/Å)")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=2.6, markeredgecolor="none",
                      markerfacecolor=style.ELEMENT_COLORS[e], label=e) for e in common.ELEMENTS]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=2.0, markeredgecolor="none",
                          markerfacecolor=style.LIGHT_GREY, label="T22, H2"))
    axes[-1].legend(handles=handles, loc="lower right", fontsize=style.SMALL, handletextpad=0.1,
                    borderaxespad=0.3, labelspacing=0.3, title="H3 atoms", title_fontsize=style.SMALL)
    style.save_source(source, NAME, "d")


def _panel_e(ax, cross: pd.DataFrame, floor: pd.DataFrame, primary: list[str]) -> None:
    offsets = np.linspace(-0.24, 0.24, len(primary))
    rows = []
    for index, element in enumerate(common.ELEMENTS):
        value = floor.loc[(floor["subset"] == "H3") & (floor["element"] == element), "force_rmse_meV_A"].iloc[0]
        ax.plot([index - 0.36, index + 0.36], [value, value], color=style.GREY, lw=1.2, solid_capstyle="butt",
                zorder=1)
        rows.append({"element": element, "series": "label floor (Gamma vs 2x2x2)", "force_rmse_meV_A": value})
        for model, dx in zip(primary, offsets):
            value = _row(cross, model, "k222", "H3", element)["force_rmse_meV_A"]
            ax.plot(index + dx, value, **common.marker_kwargs(model, "o", 3.6), zorder=3)
            rows.append({"element": element, "series": model, "force_rmse_meV_A": value})
    ax.set_yscale("log")
    ax.set_ylim(5, 1000)
    style.log_ticks(ax, "y")
    ax.set_yticks([10, 100, 1000], ["10", "100", "1000"])
    ax.set_xticks(range(len(common.ELEMENTS)), common.ELEMENTS)
    ax.set_xlim(-0.6, len(common.ELEMENTS) - 0.4)
    ax.set_ylabel("Force RMSE on H3 vs 2x2x2 (meV/Å)")
    handles = [Line2D([0], [0], label=common.SHORT_LABELS[m], **common.marker_kwargs(m, "o", 3.6)) for m in primary]
    handles.append(Line2D([0], [0], color=style.GREY, lw=1.2, label="label floor"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=style.SMALL,
              handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3, ncol=2, columnspacing=0.8)
    style.save_source(pd.DataFrame(rows), NAME, "e")


def _panel_f(ax, configs: pd.DataFrame, cross: pd.DataFrame, cross_configs: pd.DataFrame, floor: pd.DataFrame,
             primary: list[str]) -> None:
    ids = list(configs["config_id"])
    x = common.config_positions(configs)
    position = dict(zip(ids, x))
    centred_floor = floor.loc[(floor["subset"] == "All27") & (floor["element"] == "all"),
                              "energy_rmse_centred_meV_atom"].iloc[0]
    ax.axhspan(-centred_floor, centred_floor, color=style.LIGHT_GREY, lw=0, zorder=0)
    ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    common.subset_bands(ax, configs, x)
    offsets = np.linspace(-0.22, 0.22, len(primary))
    rows = []
    for model, dx in zip(primary, offsets):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")]
        per = per.set_index("config_id").loc[ids]
        for subset, alpha, size in (("T22", 0.4, 2.6), ("H2", 0.7, 3.0), ("H3", 1.0, 3.4)):
            keep = (per["subset"] == subset).to_numpy()
            ax.plot(x[keep] + dx, per["energy_centred_meV_atom"].to_numpy()[keep],
                    **common.marker_kwargs(model, "o", size, alpha=alpha), zorder=3)
        rows.extend({"model": model, "config_id": c, "subset": s, "x": position[c] + dx,
                     "energy_centred_meV_atom": v}
                    for c, s, v in zip(ids, per["subset"], per["energy_centred_meV_atom"]))
    common.config_axis(ax, configs, x)
    ax.set_ylabel("Centred energy error\nvs 2x2x2 (meV/atom)")
    handles = []
    for model in primary:
        value = _row(cross, model, "k222", "H3")["energy_rmse_centred_meV_atom"]
        handles.append(Line2D([0], [0], label=f"{common.SHORT_LABELS[model]} (H3 RMSE {value:.2f})",
                              **common.marker_kwargs(model, "o", 3.4)))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=style.LIGHT_GREY, edgecolor="none",
                             label=f"label floor, +/-{centred_floor:.2f}"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=len(handles), fontsize=style.SMALL,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
    source = pd.DataFrame(rows)
    source["label_floor_centred_rmse_meV_atom"] = centred_floor
    style.save_source(source, NAME, "f")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    configs, floor, floor_configs, cross, cross_configs, forces = _load()
    primary = common.primary_models(cross)
    per_element = _floor_per_config_element(forces)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    # row 1: a, b, c
    ax_a = L.axes(12.0, 13.0, 36.0, 12.0)
    _panel_a(ax_a, configs, L)
    ax_b = L.axes(68.0, 13.0, 36.0, 31.0)
    _panel_b(ax_b, floor, floor_configs, per_element, L, (69.0, 11.5))
    ax_c_labels = L.axes(110.0, 13.0, 19.0, 31.0)
    ax_c = L.axes(130.0, 13.0, 37.0, 31.0)
    ax_c_values = L.axes(168.0, 13.0, 14.0, 31.0)
    _panel_c(ax_c, ax_c_labels, ax_c_values, cross, cross_configs, floor, primary)

    # row 2: d (three parity panels), e
    top2, size = 61.0, 36.0
    ax_d = [L.axes(12.0 + i * (size + 3.0), top2, size, size) for i in range(3)]
    _panel_d(ax_d, forces, cross, primary)
    ax_e = L.axes(140.0, top2, 41.0, size)
    _panel_e(ax_e, cross, floor, primary)

    # row 3: f
    ax_f = L.axes(14.0, 117.0, 167.0, 24.0)
    _panel_f(ax_f, configs, cross, cross_configs, floor, primary)

    L.letter(ax_a, "a", 3.0, 6.0)
    L.letter(ax_b, "b", 56.0, 6.0)
    L.letter(ax_c, "c", 106.0, 6.0)
    L.letter(ax_d[0], "d", 3.0, 57.0)
    L.letter(ax_e, "e", 128.0, 57.0)
    L.letter(ax_f, "f", 3.0, 109.0)
    return style.save(fig, "fig3_dft", "main")


def _caption() -> str:
    configs, floor, floor_configs, cross, cross_configs, _ = _load()
    primary = common.primary_models(cross)
    counts = configs["subset"].value_counts()
    h3_ids = ", ".join(configs.loc[configs["subset"] == "H3", "config_id"])
    held_1000 = int(((configs["temperature_K"] == 1000) & (configs["subset"] != "T22")).sum())
    f27 = floor[(floor["subset"] == "All27")].set_index("element")
    fh3 = floor[(floor["subset"] == "H3")].set_index("element")
    elements = ", ".join(f"{e} {f27.loc[e, 'force_rmse_meV_A']:.1f}" for e in common.ELEMENTS)

    def force(model, labels="k222", subset="H3", element="all"):
        return _row(cross, model, labels, subset, element)

    c_text = "; ".join(f"{common.SHORT_LABELS[m] if m != 'foundation' else 'MACE-MP-0'} "
                       f"{force(m)['force_rmse_meV_A']:.1f} (2x2x2) and {force(m, 'gamma')['force_rmse_meV_A']:.1f} "
                       f"(Gamma)" for m in common.MODEL_ORDER)
    slopes = "; ".join(f"{common.SHORT_LABELS[m]} {force(m)['softening_slope']:.2f}" for m in primary)
    p_text = "; ".join(f"{common.SHORT_LABELS[m]} {force(m, element='P')['force_rmse_meV_A']:.0f}" for m in primary)
    e_text = "; ".join(f"{common.SHORT_LABELS[m]} {force(m)['energy_rmse_centred_meV_atom']:.2f}" for m in primary)
    e_all = ", ".join(f"{force(m, subset='All27')['energy_rmse_centred_meV_atom']:.2f}" for m in primary)
    fk = cross_configs[(cross_configs["model"] == "foundation") & (cross_configs["labels"] == "k222")].copy()
    fk["temperature_K"] = fk["config_id"].map(lambda c: common.config_key(c)[0])
    fmeans = fk.groupby("temperature_K")["energy_centred_meV_atom"].mean()
    trend = ", ".join(f"{fmeans[t]:+.1f} at {t} K" for t in sorted(fmeans.index))
    return (
        "Fig. 3 | Fine-tuned models against their DFT reference. "
        f"**a**, The 27 DFT-labelled configurations (frames 100-900 of the 52-atom MACE-MP-0 runs at 600, 800 and "
        f"1000 K) by subset: T22, used in training by both fine-tunes ({counts['T22']}); H3, held out by both "
        f"({counts['H3']}: {h3_ids}), the primary test set; H2, held out by the Gamma fine-tune only "
        f"({counts['H2']}). {'No' if held_1000 == 0 else held_1000} 1000 K configuration is held out. "
        f"**b**, Label-error floor: RMSE of Gamma minus 2x2x2 force components over All27 "
        f"({f27.loc['all', 'force_rmse_meV_A']:.1f} meV/A; {elements} meV/A; black bars) and per configuration "
        f"(dots). Centred energy floor {f27.loc['all', 'energy_rmse_centred_meV_atom']:.2f} meV/atom (the Gamma "
        f"energies lie on average {f27.loc['all', 'energy_mean_raw_meV_atom']:.2f} meV/atom above the 2x2x2 ones "
        f"before centring); pressure floor {f27.loc['all', 'pressure_rmse_kbar']:.2f} kbar. "
        f"**c**, Force RMSE on H3 (meV/A): {c_text}. Large markers, all three H3 configurations; small dots, "
        f"single configurations; grey band, the H3 label floor ({fh3.loc['all', 'force_rmse_meV_A']:.1f} meV/A). "
        f"Rows above the line are the models that drove the MD runs (identified by force fingerprints, Extended "
        f"Data Fig. 7a); the two rows below were not used for MD. Logarithmic axis. "
        f"**d**, Force components predicted by the three MD models against the 2x2x2 labels; H3 atoms coloured by "
        f"element, the T22 and H2 configurations in grey (T22 is training data of both fine-tunes, H2 of the "
        f"2x2x2 fine-tune). Dashed line, identity; "
        f"solid line, least-squares slope through the origin on H3 ({slopes}). "
        f"**e**, Force RMSE on H3 against the 2x2x2 labels per element, with the per-element H3 label floor (grey "
        f"bars). P carries the largest error ({p_text} meV/A). Logarithmic axis. "
        f"**f**, Energy error per configuration against the 2x2x2 labels after removing each model's constant offset "
        f"(mean over T22; all configurations have the composition Li24P4S20Cl4), ordered by source temperature. "
        f"H3 (dark band) and H2 (light band) configurations in full colour, T22 faint. RMSE on H3: {e_text} "
        f"meV/atom (All27: {e_all} in the same order). The foundation's centred error drifts with source temperature (mean "
        f"{trend} meV/atom). Grey band, the centred label floor. "
        f"{common.naming_sentence(configs)} Fine-tuned models are compared with DFT and with the foundation model "
        f"only. "
        f"With three held-out configurations no confidence intervals are given. "
        + common.sources([CONFIGS, FLOOR, FLOOR_CONFIGS, CROSS, CROSS_CONFIGS, FORCES])
    )


CAPTION = _caption()
