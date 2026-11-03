"""Forces (and centred energies) of the four potentials of the main figures against their DFT labels.

a, force parity against the 2x2x2 labels of MACE-MP-0, MatterSim and the two stage-two fine-tunes (hero);
b, force RMSE on H3 of the four models against both label sets; c, per-element force RMSE on H3 with the
per-element label floor; d, centred energy error per configuration. All four models drove MD runs; the
stage-one fine-tunes are in Extended Data. Every model is judged against DFT, never against experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from . import common, registry, style

KEY = "forces"
NAME = registry.number(KEY)
PANELS = "abcd"
WIDTH, HEIGHT = 183.0, 153.0
CONFIGS = paths.DFT / "configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FORCES = paths.MODELS / "predictions_forces.csv.gz"
MS_FORCES = paths.MODELS / "predictions_mattersim_forces.csv.gz"  # amendment A4
MS_IDENTITY = paths.MODELS / "mattersim_identity.csv"
MS_EVALUATION = paths.MODELS / "mattersim_evaluation.json"
EVALUATION = paths.MODELS / "evaluation.json"
PARITY_LIMIT = 5.0  # eV/A, both axes of the parity panels
RMSE_LIMITS = (6.0, 700.0)  # meV/A, the shared log axis of b and c
RMSE_TICKS = (10, 20, 50, 100, 200, 500)
# Atom groups of the parity panels (a): Li against the framework; per-element errors are in c.
ATOM_GROUPS = (("P, S, Cl", ("P", "S", "Cl"), style.DARK_GRAY), ("Li", ("Li",), style.ELEMENT_COLORS["Li"]))
B_DOT_X, B_DOT_SPREAD = -0.42, 0.06  # b: the per-configuration dots left of each model, one column per configuration
C_OFFSET = 0.30  # horizontal offset (element units) of the outer models of c
D_OFFSET = 0.36  # horizontal offset (configuration units) of the outer models of d
D_MARKERS = (("T22", 0.4, 2.2), ("H2", 0.7, 2.6), ("H3", 1.0, 3.0))  # subset, opacity, marker size (pt) in d
D_EDGE = 0.5  # marker edge width (pt) in d: four models share each configuration


def _load():
    configs = common.ordered_configs(common.read(CONFIGS))
    floor = common.read(FLOOR)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    # MatterSim (amendment A4) comes from its own file, merged as in dft.agreement.load
    forces = common.read(FORCES).merge(common.read(MS_FORCES), on=["config_id", "atom", "species"], how="left",
                                       validate="one_to_one")
    if forces["mattersim_fx"].isna().any():
        raise ValueError("MatterSim predictions do not cover every atom")
    forces = forces.merge(configs[["config_id", "subset"]], on="config_id")
    return configs, floor, cross, cross_configs, forces


def _row(cross: pd.DataFrame, model: str, labels: str, subset: str, element: str = "all") -> pd.Series:
    rows = cross[(cross["model"] == model) & (cross["labels"] == labels) & (cross["subset"] == subset)
                 & (cross["element"] == element)]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {model}, {labels}, {subset}, {element}")
    return rows.iloc[0]


def _floor(floor: pd.DataFrame, subset: str, element: str = "all", column: str = "force_rmse_meV_A") -> float:
    return float(floor.loc[(floor["subset"] == subset) & (floor["element"] == element), column].iloc[0])


def _components(forces: pd.DataFrame, prefix: str) -> np.ndarray:
    return forces[[f"{prefix}_f{axis}" for axis in "xyz"]].to_numpy()


def _key_axes(layout: common.Layout, left: float, top: float, width: float, height: float):
    """A frameless axes without ticks that only holds a legend."""
    ax = layout.axes(left, top, width, height, frame="none")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _rmse_axis(ax, label: str) -> None:
    """The shared logarithmic force-RMSE axis of panels b and c."""
    ax.set_yscale("log")
    ax.set_ylim(*RMSE_LIMITS)
    style.log_ticks(ax, "y")
    ax.set_yticks(RMSE_TICKS, [str(t) for t in RMSE_TICKS])
    ax.set_ylabel(label)


# ----------------------------------------------------------------------------------------------- panels


def _panel_a(axes, key_ax, forces: pd.DataFrame, cross: pd.DataFrame, models: dict[str, str]) -> None:
    """Force parity against the 2x2x2 labels, one square axes per model; the key in ``key_ax`` above them."""
    reference = _components(forces, "dft_k222").ravel()
    species = np.repeat(forces["species"].to_numpy(), 3)
    held_out = np.repeat((forces["subset"] == "H3").to_numpy(), 3)
    limit = PARITY_LIMIT
    group = {element: label for label, elements, _ in ATOM_GROUPS for element in elements}
    source = pd.DataFrame({"config_id": np.repeat(forces["config_id"].to_numpy(), 3),
                           "atom": np.repeat(forces["atom"].to_numpy(), 3),
                           "species": species, "atom_group": [group[e] for e in species],
                           "subset": np.repeat(forces["subset"].to_numpy(), 3),
                           "component": np.tile(list("xyz"), len(forces)), "dft_k222_eV_A": reference})
    for index, (ax, (family, model)) in enumerate(zip(axes, models.items())):
        predicted = _components(forces, model).ravel()
        source[f"{model}_eV_A"] = predicted
        ax.plot([-limit, limit], [-limit, limit], **style.IDENTITY_LINE)
        # One scatter artist per parity cloud (a cloud of force components overlaps by nature), drawn in a fixed
        # order: T22 and H2 atoms behind, then the H3 framework atoms, then H3 Li on top. The cloud is rasterized
        # at the export resolution in PDF and SVG (about 14 000 points per axes); text and lines stay vector.
        layers = [(~held_out, style.BAND, 1.0)]
        layers += [(held_out & np.isin(species, elements), color, 2.2) for _, elements, color in ATOM_GROUPS]
        order = np.concatenate([np.flatnonzero(keep) for keep, _, _ in layers])
        colors = np.concatenate([np.full(int(keep.sum()), color, dtype=object) for keep, color, _ in layers])
        sizes = np.concatenate([np.full(int(keep.sum()), size) for keep, _, size in layers])
        ax.scatter(reference[order], predicted[order], s=sizes, facecolors=list(colors), edgecolors="none", zorder=3,
                   rasterized=True)
        row = _row(cross, model, "k222", "H3")
        slope = float(row["softening_slope"])
        source[f"{model}_slope_H3"] = slope
        ax.plot([-limit, limit], [-limit * slope, limit * slope], color=style.BLACK, linewidth=0.6, zorder=4)
        ax.text(0.05, 0.95, common.MAIN_SHORT_LABELS[family], transform=ax.transAxes, ha="left", va="top",
                fontsize=style.FONT_SIZE, color=style.MODEL_COLORS[model], fontweight="bold")
        ax.text(0.05, 0.85, f"H3 RMSE {common.sig(row['force_rmse_meV_A'])} meV/Å\nH3 slope {slope:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=style.SMALL, linespacing=1.35)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Model force component (eV/Å)")
    # one axis label, centred under the row: on the second axes, shifted to the middle of the gap after it
    middle = axes[len(axes) // 2 - 1]
    middle.set_xlabel("2x2x2 DFT force component (eV/Å)")
    if len(axes) % 2 == 0:
        right = axes[len(axes) // 2].get_position().x0
        box = middle.get_position()
        middle.xaxis.label.set_x((right - box.x0) / box.width * 0.5 + 0.5)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markeredgecolor="none",
                      markerfacecolor=color, label=f"H3 atoms, {label}") for label, _, color in reversed(ATOM_GROUPS)]
    handles += [Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markeredgecolor=style.LIGHT_GRAY,
                       markeredgewidth=0.3, markerfacecolor=style.BAND, label="T22, H2 atoms"),
                Line2D([0], [0], **{**style.IDENTITY_LINE, "zorder": 2}, label="identity"),
                Line2D([0], [0], color=style.BLACK, linewidth=0.6, label="least squares through origin, H3")]
    key_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles), handletextpad=0.4,
                  columnspacing=1.6, borderaxespad=0.0, borderpad=0.0, handlelength=1.4)
    style.save_source(source, NAME, "a")


def _panel_b(ax, legend_ax, configs: pd.DataFrame, cross: pd.DataFrame, cross_configs: pd.DataFrame,
             floor: pd.DataFrame, models: dict[str, str]) -> None:
    """Force RMSE on H3 of every model: marker against 2x2x2, grey bar against Gamma, dots per configuration."""
    positions = {family: float(i) for i, family in enumerate(models)}
    h3 = list(configs.loc[configs["subset"] == "H3", "config_id"])  # the order of d
    # one H3 configuration per dot column, left to right in the order of d, so that close values stay visible
    dot_x = dict(zip(h3, B_DOT_X + np.linspace(-B_DOT_SPREAD, B_DOT_SPREAD, len(h3))))
    h3_floor = _floor(floor, "H3")
    lo = RMSE_LIMITS[0]
    ax.axhspan(lo, h3_floor, **style.BAND_FILL)
    rows = []
    for family, model in models.items():
        x = positions[family]
        k222 = _row(cross, model, "k222", "H3")["force_rmse_meV_A"]
        gamma = _row(cross, model, "gamma", "H3")["force_rmse_meV_A"]
        ax.plot([x - 0.24, x + 0.24], [gamma, gamma], color=style.GRAY, linewidth=1.0, solid_capstyle="butt",
                zorder=2)
        ax.plot(x, k222, **common.model_style(model, zorder=4))
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")]
        per = per.set_index("config_id").loc[h3]
        dots = x + np.array([dot_x[c] for c in h3])
        ax.plot(dots, per["force_rmse_meV_A"], **common.model_style(model, size=1.8, edge=0.4, zorder=3))
        common_fields = {"model": family, "md_model": model}
        rows.append({**common_fields, "labels": "k222", "config_id": f"H3 (all {len(h3)})", "x": x,
                     "force_rmse_meV_A": k222})
        rows.append({**common_fields, "labels": "gamma", "config_id": f"H3 (all {len(h3)})", "x": x,
                     "force_rmse_meV_A": gamma})
        rows.extend({**common_fields, "labels": "k222", "config_id": c, "x": d, "force_rmse_meV_A": v}
                    for c, d, v in zip(h3, dots, per["force_rmse_meV_A"]))
    last = max(positions.values())
    ax.set_xlim(-0.7, last + 0.5)
    _rmse_axis(ax, "Force RMSE on H3 (meV/Å)")
    ax.set_xticks(list(positions.values()), [common.MAIN_SHORT_LABELS[f].replace(", ", ",\n") for f in models])
    ax.tick_params(axis="x", labelsize=style.SMALL)
    ax.text(last + 0.3, np.sqrt(lo * h3_floor), f"H3 label floor {h3_floor:.1f} meV/Å", ha="right", va="center",
            fontsize=style.SMALL, color=style.GRAY)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=common.MARKER_SIZE,
                      markerfacecolor=style.BLACK, markeredgecolor=style.BLACK, label="vs 2x2x2 labels"),
               Line2D([0], [0], color=style.GRAY, linewidth=1.0, label="vs Gamma labels"),
               Line2D([0], [0], marker="o", linestyle="none", markersize=1.8, markerfacecolor=style.BLACK,
                      markeredgecolor=style.BLACK, label="one H3 configuration, vs 2x2x2")]
    legend_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3, handletextpad=0.5,
                     columnspacing=1.0, borderaxespad=0.0, borderpad=0.0, handlelength=1.2)
    source = pd.DataFrame(rows)
    source["h3_label_floor_meV_A"] = h3_floor
    style.save_source(source, NAME, "b")


def _panel_c(ax, legend_ax, cross: pd.DataFrame, floor: pd.DataFrame, models: dict[str, str]) -> None:
    """Per-element force RMSE on H3 against the 2x2x2 labels, with the per-element H3 label floor."""
    offsets = np.linspace(-C_OFFSET, C_OFFSET, len(models))
    rows = []
    for index, element in enumerate(common.ELEMENTS):
        value = _floor(floor, "H3", element)
        lo = RMSE_LIMITS[0]
        ax.add_patch(Rectangle((index - 0.4, lo), 0.8, value - lo, facecolor=style.BAND, edgecolor="none",
                               zorder=0))
        rows.append({"element": element, "series": "label floor (Gamma vs 2x2x2)", "md_model": "", "x": index,
                     "force_rmse_meV_A": value})
        for (family, model), dx in zip(models.items(), offsets):
            value = _row(cross, model, "k222", "H3", element)["force_rmse_meV_A"]
            ax.plot(index + dx, value, **common.model_style(model, zorder=3))
            rows.append({"element": element, "series": family, "md_model": model, "x": index + dx,
                         "force_rmse_meV_A": value})
    _rmse_axis(ax, "Force RMSE on H3 (meV/Å)")
    ax.set_xticks(range(len(common.ELEMENTS)), common.ELEMENTS)
    ax.set_xlim(-0.6, len(common.ELEMENTS) - 0.4)
    handles = [common.handle(common.model_style(m), common.MAIN_SHORT_LABELS[f]) for f, m in models.items()]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none", label="H3 label floor"))
    legend_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles),
                     handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0, handlelength=1.0)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, legend_ax, configs: pd.DataFrame, cross: pd.DataFrame, cross_configs: pd.DataFrame,
             models: dict[str, str]) -> None:
    """Centred energy error per configuration against the 2x2x2 labels."""
    ids = list(configs["config_id"])
    x = common.config_positions(configs)
    position = dict(zip(ids, x))
    ax.axhline(0, **style.UNITY_LINE)
    common.subset_bands(ax, configs, x, labels=False)
    above = transforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=2.0, units="points")
    for position_x, subset in zip(x, configs["subset"]):
        if subset != "T22":  # name the held-out configurations 0.7 mm above the frame
            ax.text(position_x, 1.0, subset, transform=above, ha="center", va="bottom", fontsize=style.SMALL,
                    color=style.BLACK if subset == "H3" else style.GRAY)
    offsets = np.linspace(-D_OFFSET, D_OFFSET, len(models))
    rows = []
    for (family, model), dx in zip(models.items(), offsets):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")]
        per = per.set_index("config_id").loc[ids]
        for subset, alpha, size in D_MARKERS:
            keep = (per["subset"] == subset).to_numpy()
            ax.plot(x[keep] + dx, per["energy_centred_meV_atom"].to_numpy()[keep],
                    **common.model_style(model, size=size, edge=D_EDGE, alpha=alpha, zorder=3))
        rows.extend({"model": family, "md_model": model, "config_id": c, "subset": s, "x": position[c] + dx,
                     "energy_centred_meV_atom": v}
                    for c, s, v in zip(ids, per["subset"], per["energy_centred_meV_atom"]))
    common.config_axis(ax, configs, x)
    values = [row["energy_centred_meV_atom"] for row in rows]
    ax.set_ylim(min(values) - 4.0, max(values) + 4.0)  # room for the markers at both limits
    ax.set_yticks([-20, -10, 0, 10])
    ax.set_ylabel("Centred energy error\nvs 2x2x2 (meV/atom)")
    # row header of the frame tick labels (config_axis: tick length 1.5 pt, pad 1.5 pt)
    header = transforms.offset_copy(ax.transAxes, fig=ax.figure, x=-2.0, y=-3.0, units="points")
    ax.text(0.0, 0.0, "Frame", transform=header, ha="right", va="top", fontsize=style.SMALL)
    handles = []
    for family, model in models.items():
        value = _row(cross, model, "k222", "H3")["energy_rmse_centred_meV_atom"]
        handles.append(common.handle(common.model_style(model),
                                     f"{common.MAIN_SHORT_LABELS[family]} (H3 RMSE {value:.2f} meV/atom)"))
    legend_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles),
                     handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    style.save_source(pd.DataFrame(rows), NAME, "d")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    configs, floor, cross, cross_configs, forces = _load()
    models = common.main_models(cross)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    left, right = 14.0, 181.0

    # row 1: a, four square parity axes under their key
    top_a, gap = 9.0, 3.0
    size = (right - left - (len(models) - 1) * gap) / len(models)
    ax_a = [L.axes(left + i * (size + gap), top_a, size, size, row="a", col="left" if i == 0 else None)
            for i in range(len(models))]
    key_a = _key_axes(L, left, top_a - 5.0, right - left, 3.0)
    _panel_a(ax_a, key_a, forces, cross, models)

    # row 2: b, c (legend strips above them)
    top_bc, height_bc = 65.0, 38.0
    ax_b = L.axes(left, top_bc, 74.0, height_bc, row="bc", col="left")
    key_b = _key_axes(L, left, top_bc - 5.0, 74.0, 3.0)
    _panel_b(ax_b, key_b, configs, cross, cross_configs, floor, models)
    ax_c = L.axes(102.0, top_bc, right - 102.0, height_bc, row="bc")
    key_c = _key_axes(L, 102.0, top_bc - 5.0, right - 102.0, 3.0)
    _panel_c(ax_c, key_c, cross, floor, models)

    # row 3: d
    top_d = 122.0
    ax_d = L.axes(left, top_d, right - left, 23.0, row="d", col="left")
    key_d = _key_axes(L, left, top_d - 8.0, right - left, 3.0)
    _panel_d(ax_d, key_d, configs, cross, cross_configs, models)

    L.letter(ax_a[0], "a", 4.0, top_a - 3.0)
    L.letter(ax_b, "b", 4.0, top_bc - 3.0)
    L.letter(ax_c, "c", 90.0, top_bc - 3.0)
    L.letter(ax_d, "d", 4.0, top_d - 6.0)
    return L.save(*registry.target(KEY))


def _where(configs: pd.DataFrame, subset: str) -> str:
    """Plain-language location of a subset's configurations: 'frame 400 at 600 K and frames 200 and 500 at 800 K'."""
    parts = []
    for temperature, group in configs[configs["subset"] == subset].groupby("temperature_K", sort=True):
        frames = [str(f) for f in sorted(group["frame"])]
        listed = frames[0] if len(frames) == 1 else ", ".join(frames[:-1]) + " and " + frames[-1]
        parts.append(f"frame{'s' if len(frames) > 1 else ''} {listed} at {temperature} K")
    return " and ".join(parts)


def _word(n: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five"}.get(int(n), str(n))


def _naming(configs: pd.DataFrame, models: dict[str, str]) -> str:
    """Caption sentence naming the fine-tunes of the main figures (``common.naming_sentence`` without stages)."""
    g, k = configs["ftG_split"].value_counts(), configs["ftK_split"].value_counts()
    stages = {models[family].partition("_")[2] for family in ("ftG", "ftK2")}
    if stages != {"s2"}:
        raise ValueError(f"the main figures show the stage-two fine-tunes, not {sorted(stages)}")
    return (f"'{common.MAIN_SHORT_LABELS['ftG']}' and '{common.MAIN_SHORT_LABELS['ftK2']}' denote MACE-MP-0 small "
            f"fine-tuned on the Gamma-point labels ({g['train']} training and {g['valid']} validation "
            f"configurations) or on the 2x2x2 labels ({k['train']} and {k['valid']}), each the model after "
            f"stochastic weight averaging (stage 2); the stage-1 models are compared in {registry.label('models')}.")


def _caption() -> str:
    configs, floor, cross, cross_configs, forces = _load()
    models = common.main_models(cross)
    counts = configs["subset"].value_counts()
    n_h3_word, n_word = _word(counts["H3"]), _word(len(models))
    identity = common.read(MS_IDENTITY)
    checked = {len(str(frames).split(";")) for frames in identity["frames"]}
    if len(checked) != 1:
        raise ValueError("the MatterSim identity check used a different number of frames per run")
    n_checked = _word(checked.pop())
    precision = json.loads(MS_EVALUATION.read_text(encoding="utf-8"))["precision"]
    mace_precision = json.loads(EVALUATION.read_text(encoding="utf-8"))["labels_dtype"]

    def force(model, labels="k222", subset="H3", element="all"):
        return _row(cross, model, labels, subset, element)

    def name(family):
        return common.MAIN_SHORT_LABELS[family]

    first = models["foundation"]
    n_h3 = int(force(first)["n_components"])
    n_config = n_h3 // counts["H3"]
    n_element = ", ".join(f"{e} {int(force(first, element=e)['n_components'])}" for e in common.ELEMENTS)
    a_text = "; ".join(f"{name(f)} {common.sig(force(m)['force_rmse_meV_A'])} meV/A and "
                       f"{force(m)['softening_slope']:.2f}" for f, m in models.items())
    b_text = "; ".join(f"{name(f)} {common.sig(force(m)['force_rmse_meV_A'])} (2x2x2) and "
                       f"{common.sig(force(m, 'gamma')['force_rmse_meV_A'])} (Gamma)" for f, m in models.items())
    worst = {max(common.ELEMENTS, key=lambda e: force(m, element=e)["force_rmse_meV_A"]) for m in models.values()}
    if len(worst) != 1:
        raise ValueError(f"the element with the largest H3 force error differs between the models: {sorted(worst)}")
    worst = worst.pop()
    p_text = "; ".join(f"{name(f)} {force(m, element=worst)['force_rmse_meV_A']:.0f}" for f, m in models.items())
    floors = ", ".join(f"{e} {_floor(floor, 'H3', e):.1f}" for e in common.ELEMENTS)
    e_text = "; ".join(f"{name(f)} {force(m)['energy_rmse_centred_meV_atom']:.2f}" for f, m in models.items())
    e_all = ", ".join(f"{force(m, subset='All27')['energy_rmse_centred_meV_atom']:.2f}" for m in models.values())
    centred_floor = _floor(floor, "All27", column="energy_rmse_centred_meV_atom")
    temperatures = sorted(configs["temperature_K"].unique())
    runs = ", ".join(str(t) for t in temperatures[:-1]) + f" or {temperatures[-1]} K"

    def drift(family):
        per = cross_configs[(cross_configs["model"] == models[family]) & (cross_configs["labels"] == "k222")]
        means = per.groupby(per["config_id"].map(lambda c: common.config_key(c)[0]))["energy_centred_meV_atom"].mean()
        return f"{name(family)} " + ", ".join(f"{means[t]:+.1f}" for t in temperatures)

    sizes = {subset: (alpha, size) for subset, alpha, size in D_MARKERS}
    return (
        registry.heading(KEY)
        + f"Shown are the {n_word} potentials of the main figures, MACE-MP-0 and MatterSim (universal potentials) and "
        f"two fine-tunes of MACE-MP-0 small, each of which drove the MD runs of {registry.label('diffusion')}. "
        f"H3 denotes the {n_h3_word} configurations that neither fine-tune was trained on "
        f"({_where(configs, 'H3')}); they are the validation configurations of both fine-tunes and so were used to "
        f"select the epoch. The subsets are mapped in {registry.label('labels', 'a')}. The reference is the 2x2x2 "
        f"k-point label set unless stated. MatterSim was evaluated in {precision}, the precision of its MD runs, "
        f"and the MACE models in {mace_precision} (protocol amendment A4). "
        f"**a**, Force components predicted by the {n_word} models against the 2x2x2 labels; each point is one "
        f"force component (single values, no error bars). The MACE models are those that drove the MD runs "
        f"(identified by force fingerprints, {registry.label('models', 'a')}); MatterSim reproduces the forces "
        f"stored in its runs ({n_checked} frames of each of its {_word(len(identity))} production runs recomputed, RMS "
        f"difference at most {common.sig(identity['rms_meV_A'].max(), 2)} meV/A). H3 atoms are drawn in navy for "
        f"Li and in "
        f"dark grey for the framework atoms P, S and Cl (n = {n_h3} force components per model; per-element errors "
        f"in c); the T22 and H2 configurations are drawn in light grey behind them (T22 is training data of both "
        f"fine-tunes, H2 of the 2x2x2 fine-tune). Dashed line, identity; solid line, least-squares fit through the "
        f"origin on H3, whose slope measures force softening. RMSE and slope on H3: {a_text}. "
        f"**b**, Force RMSE on H3 (meV/A) of the {n_word} models: large markers against the 2x2x2 labels and grey "
        f"bars against the Gamma-point labels, each over all n = {n_h3} H3 force components; small dots, the n = "
        f"{counts['H3']} H3 configurations individually against the 2x2x2 labels ({n_config} components each), left "
        f"to right in the order of d. "
        f"Single values, no error bars; with {n_h3_word} configurations no confidence intervals are given. "
        f"{b_text}. Grey band, the H3 label floor ({_floor(floor, 'H3'):.1f} meV/A): the RMSE of the Gamma minus "
        f"2x2x2 force components over H3, the force error that separates the two label sets themselves. "
        f"**c**, Force RMSE on H3 against the 2x2x2 labels per element for the {n_word} models (n = {n_element} "
        f"force components per model); single values, no error bars. Grey bands, the per-element H3 label floor "
        f"(RMSE of the Gamma minus 2x2x2 force components of that element over H3: {floors} meV/A). {worst} carries "
        f"the largest error of every model ({p_text} meV/A). Panels b and c share a logarithmic axis. "
        f"**d**, Energy error per configuration (n = {len(configs)}) against the 2x2x2 labels after removing each "
        f"model's constant offset, the mean error over the {counts['T22']} T22 configurations (all configurations "
        f"have the composition Li24P4S20Cl4); single values, no error bars. Configurations are ordered by the "
        f"temperature ({runs}) and frame of the 52-atom MACE-MP-0 run they were taken from; the {n_word} models are "
        f"offset horizontally within each configuration. H3 configurations (dark bands) are drawn as opaque "
        f"{sizes['H3'][1]:.1f}-pt markers, H2 configurations (light bands) as {sizes['H2'][1]:.1f}-pt markers at "
        f"{sizes['H2'][0]:.0%} opacity and T22 configurations as {sizes['T22'][1]:.1f}-pt markers at "
        f"{sizes['T22'][0]:.0%} opacity. RMSE on H3: {e_text} meV/atom (All27: {e_all} in the same order). The "
        f"centred errors of the universal potentials drift with source temperature (mean {drift('foundation')}; "
        f"{drift('mattersim')} meV/atom at {', '.join(str(t) for t in temperatures)} K). The centred energy label "
        f"floor, +/-{centred_floor:.2f} meV/atom (the RMSE over all {len(configs)} configurations of the Gamma minus "
        f"2x2x2 energies, centred in the same way), is narrower than a marker and is not drawn. "
        f"{_naming(configs, models)} Every model is compared with DFT, not with experiment. "
        + common.sources([CONFIGS, FLOOR, CROSS, CROSS_CONFIGS, FORCES, MS_FORCES, MS_IDENTITY, MS_EVALUATION,
                          EVALUATION])
    )


CAPTION = _caption()
