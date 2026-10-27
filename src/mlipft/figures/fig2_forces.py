"""Forces (and centred energies) of the models against their DFT labels.

a, force parity against the 2x2x2 labels for the three models that drove MD (hero); b, force RMSE
on H3 of all five models against both label sets; c, per-element force RMSE on H3 with the
per-element label floor; d, centred energy error per configuration. Every model is judged against
DFT (and the fine-tunes against the foundation model), never against experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from . import common, registry, style

KEY = "forces"
NAME = registry.number(KEY)
PANELS = "abcd"
WIDTH, HEIGHT = 183.0, 160.0
CONFIGS = paths.DFT / "configs.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"
CROSS_CONFIGS = paths.RESULTS / "cross_evaluation_configs.csv"
FORCES = paths.MODELS / "predictions_forces.csv.gz"
PARITY_LIMIT = 5.0  # eV/A, both axes of the parity panels
RMSE_LIMITS = (6.0, 700.0)  # meV/A, the shared log axis of b and c
RMSE_TICKS = (10, 20, 50, 100, 200, 500)
MD_GAP = 0.7  # extra space in b between the models that drove MD and the two that did not


def _load():
    configs = common.ordered_configs(common.read(CONFIGS))
    floor = common.read(FLOOR)
    cross = common.read(CROSS)
    cross_configs = common.read(CROSS_CONFIGS)
    forces = common.read(FORCES).merge(configs[["config_id", "subset"]], on="config_id")
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


def _panel_a(axes, key_ax, forces: pd.DataFrame, cross: pd.DataFrame, models: list[str]) -> None:
    """Force parity against the 2x2x2 labels, one square axes per MD model; element key in ``key_ax``."""
    reference = _components(forces, "dft_k222").ravel()
    species = np.repeat(forces["species"].to_numpy(), 3)
    held_out = np.repeat((forces["subset"] == "H3").to_numpy(), 3)
    limit = PARITY_LIMIT
    source = pd.DataFrame({"config_id": np.repeat(forces["config_id"].to_numpy(), 3),
                           "atom": np.repeat(forces["atom"].to_numpy(), 3),
                           "species": species, "subset": np.repeat(forces["subset"].to_numpy(), 3),
                           "component": np.tile(list("xyz"), len(forces)), "dft_k222_eV_A": reference})
    for index, (ax, model) in enumerate(zip(axes, models)):
        predicted = _components(forces, model).ravel()
        source[f"{model}_eV_A"] = predicted
        ax.plot([-limit, limit], [-limit, limit], **style.IDENTITY_LINE)
        ax.scatter(reference[~held_out], predicted[~held_out], s=1.0, facecolors=style.BAND, edgecolors="none",
                   zorder=2)
        for element in common.ELEMENTS:
            keep = held_out & (species == element)
            ax.scatter(reference[keep], predicted[keep], s=2.2, facecolors=style.ELEMENT_COLORS[element],
                       edgecolors="none", zorder=3)
        row = _row(cross, model, "k222", "H3")
        slope = float(row["softening_slope"])
        source[f"{model}_slope_H3"] = slope
        ax.plot([-limit, limit], [-limit * slope, limit * slope], color=style.BLACK, linewidth=0.6, zorder=4)
        ax.text(0.05, 0.95, common.SHORT_LABELS[model], transform=ax.transAxes, ha="left", va="top",
                fontsize=style.FONT_SIZE, color=style.MODEL_COLORS[model], fontweight="bold")
        ax.text(0.05, 0.86, f"H3 RMSE {common.sig(row['force_rmse_meV_A'])} meV/Å\nH3 slope {slope:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=style.SMALL, linespacing=1.35)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Model force component (eV/Å)")
        if index == 1:
            ax.set_xlabel("2x2x2 DFT force component (eV/Å)")
    elements = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markeredgecolor="none",
                       markerfacecolor=style.ELEMENT_COLORS[e], label=e) for e in common.ELEMENTS]
    first = key_ax.legend(handles=elements, loc="upper left", bbox_to_anchor=(0.0, 1.0), title="H3 atoms",
                          title_fontsize=style.SMALL, alignment="left", handletextpad=0.2, borderaxespad=0.0,
                          labelspacing=0.35, borderpad=0.0)
    key_ax.add_artist(first)
    others = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.0, markeredgecolor=style.LIGHT_GRAY,
                     markeredgewidth=0.3, markerfacecolor=style.BAND, label="T22, H2 atoms"),
              Line2D([0], [0], **{**style.IDENTITY_LINE, "zorder": 2}, label="identity"),
              Line2D([0], [0], color=style.BLACK, linewidth=0.6, label="least squares\nthrough origin")]
    key_ax.legend(handles=others, loc="upper left", bbox_to_anchor=(0.0, 0.66), handletextpad=0.4,
                  borderaxespad=0.0, labelspacing=0.5, borderpad=0.0, handlelength=1.4)
    style.save_source(source, NAME, "a")


def _panel_b(ax, legend_ax, cross: pd.DataFrame, cross_configs: pd.DataFrame, floor: pd.DataFrame,
             primary: list[str]) -> None:
    """Force RMSE on H3 of every model: marker against 2x2x2, grey bar against Gamma, dots per configuration."""
    models = [m for m in common.MODEL_ORDER if m in primary] + [m for m in common.MODEL_ORDER if m not in primary]
    positions = {m: i + (MD_GAP if m not in primary else 0.0) for i, m in enumerate(models)}
    h3_floor = _floor(floor, "H3")
    lo = RMSE_LIMITS[0]
    ax.axhspan(lo, h3_floor, **style.BAND_FILL)
    rows = []
    for model in models:
        x = positions[model]
        k222 = _row(cross, model, "k222", "H3")["force_rmse_meV_A"]
        gamma = _row(cross, model, "gamma", "H3")["force_rmse_meV_A"]
        ax.plot([x - 0.24, x + 0.24], [gamma, gamma], color=style.GRAY, linewidth=1.0, solid_capstyle="butt",
                zorder=2)
        ax.plot(x, k222, **common.model_style(model, zorder=4))
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")
                            & (cross_configs["subset"] == "H3")].sort_values("config_id")
        dots = np.full(len(per), x - 0.42)
        ax.plot(dots, per["force_rmse_meV_A"], **common.model_style(model, size=1.8, edge=0.4, zorder=3))
        common_fields = {"model": model, "drove_md": model in primary}
        rows.append({**common_fields, "labels": "k222", "config_id": "H3 (all 3)", "x": x,
                     "force_rmse_meV_A": k222})
        rows.append({**common_fields, "labels": "gamma", "config_id": "H3 (all 3)", "x": x,
                     "force_rmse_meV_A": gamma})
        rows.extend({**common_fields, "labels": "k222", "config_id": c, "x": x - 0.42, "force_rmse_meV_A": v}
                    for c, v in zip(per["config_id"], per["force_rmse_meV_A"]))
    md = [positions[m] for m in models if m in primary]
    other = [positions[m] for m in models if m not in primary]
    ax.set_xlim(-0.7, other[-1] + 0.5)
    _rmse_axis(ax, "Force RMSE on H3 (meV/Å)")
    ax.set_xticks([positions[m] for m in models],
                  [common.SHORT_LABELS[m].replace(", ", ",\n") for m in models])
    ax.tick_params(axis="x", labelsize=style.SMALL)
    for group, text in ((md, "used for MD"), (other, "not used for MD")):
        ax.text((group[0] + group[-1]) / 2, 0.96, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=style.SMALL, color=style.GRAY)
    ax.text(other[-1] + 0.3, np.sqrt(lo * h3_floor), f"H3 label floor {h3_floor:.1f}", ha="right", va="center",
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


def _panel_c(ax, legend_ax, cross: pd.DataFrame, floor: pd.DataFrame, primary: list[str]) -> None:
    """Per-element force RMSE on H3 against the 2x2x2 labels, with the per-element H3 label floor."""
    offsets = np.linspace(-0.24, 0.24, len(primary))
    rows = []
    for index, element in enumerate(common.ELEMENTS):
        value = _floor(floor, "H3", element)
        ax.plot([index - 0.36, index + 0.36], [value, value], color=style.GRAY, linewidth=1.2,
                solid_capstyle="butt", zorder=1)
        rows.append({"element": element, "series": "label floor (Gamma vs 2x2x2)", "x": index,
                     "force_rmse_meV_A": value})
        for model, dx in zip(primary, offsets):
            value = _row(cross, model, "k222", "H3", element)["force_rmse_meV_A"]
            ax.plot(index + dx, value, **common.model_style(model, zorder=3))
            rows.append({"element": element, "series": model, "x": index + dx, "force_rmse_meV_A": value})
    _rmse_axis(ax, "Force RMSE on H3 (meV/Å)")
    ax.set_xticks(range(len(common.ELEMENTS)), common.ELEMENTS)
    ax.set_xlim(-0.6, len(common.ELEMENTS) - 0.4)
    handles = [common.handle(common.model_style(m), common.SHORT_LABELS[m]) for m in primary]
    handles.append(Line2D([0], [0], color=style.GRAY, linewidth=1.2, label="label floor"))
    legend_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles),
                     handletextpad=0.5, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0, handlelength=1.2)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def _panel_d(ax, legend_ax, configs: pd.DataFrame, cross: pd.DataFrame, cross_configs: pd.DataFrame,
             floor: pd.DataFrame, primary: list[str]) -> None:
    """Centred energy error per configuration against the 2x2x2 labels."""
    ids = list(configs["config_id"])
    x = common.config_positions(configs)
    position = dict(zip(ids, x))
    centred_floor = _floor(floor, "All27", column="energy_rmse_centred_meV_atom")
    ax.axhspan(-centred_floor, centred_floor, **style.BAND_FILL)
    ax.axhline(0, **style.UNITY_LINE)
    common.subset_bands(ax, configs, x)
    offsets = np.linspace(-0.22, 0.22, len(primary))
    rows = []
    for model, dx in zip(primary, offsets):
        per = cross_configs[(cross_configs["model"] == model) & (cross_configs["labels"] == "k222")]
        per = per.set_index("config_id").loc[ids]
        for subset, alpha, size in (("T22", 0.4, 2.6), ("H2", 0.7, 3.0), ("H3", 1.0, 3.4)):
            keep = (per["subset"] == subset).to_numpy()
            ax.plot(x[keep] + dx, per["energy_centred_meV_atom"].to_numpy()[keep],
                    **common.model_style(model, size=size, alpha=alpha, zorder=3))
        rows.extend({"model": model, "config_id": c, "subset": s, "x": position[c] + dx,
                     "energy_centred_meV_atom": v}
                    for c, s, v in zip(ids, per["subset"], per["energy_centred_meV_atom"]))
    common.config_axis(ax, configs, x)
    values = [row["energy_centred_meV_atom"] for row in rows]
    ax.set_ylim(min(values) - 4.0, max(values) + 4.0)  # room for the markers at both limits
    ax.set_yticks([-20, -10, 0, 10])
    ax.set_ylabel("Centred energy error\nvs 2x2x2 (meV/atom)")
    handles = []
    for model in primary:
        value = _row(cross, model, "k222", "H3")["energy_rmse_centred_meV_atom"]
        handles.append(common.handle(common.model_style(model),
                                     f"{common.SHORT_LABELS[model]} (H3 RMSE {value:.2f} meV/atom)"))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=style.BAND, edgecolor="none",
                             label=f"label floor, ±{centred_floor:.2f} meV/atom"))
    legend_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=len(handles),
                     handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    source = pd.DataFrame(rows)
    source["label_floor_centred_rmse_meV_atom"] = centred_floor
    style.save_source(source, NAME, "d")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    configs, floor, cross, cross_configs, forces = _load()
    primary = common.primary_models(cross)
    fig, L = common.new_figure(WIDTH, HEIGHT)
    left = 14.0

    # row 1: a, three square parity axes and the element key
    top_a, size, gap = 7.0, 46.0, 3.0
    ax_a = [L.axes(left + i * (size + gap), top_a, size, size, row="a", col="left" if i == 0 else None)
            for i in range(3)]
    key_a = _key_axes(L, left + 3 * size + 2 * gap + 4.0, top_a, 21.0, size)
    _panel_a(ax_a, key_a, forces, cross, primary)

    # row 2: b, c (legend strips above them)
    top_bc, height_bc = 72.0, 38.0
    ax_b = L.axes(left, top_bc, 74.0, height_bc, row="bc", col="left")
    key_b = _key_axes(L, left, top_bc - 5.0, 74.0, 3.0)
    _panel_b(ax_b, key_b, cross, cross_configs, floor, primary)
    ax_c = L.axes(102.0, top_bc, 79.0, height_bc, row="bc")
    key_c = _key_axes(L, 102.0, top_bc - 5.0, 79.0, 3.0)
    _panel_c(ax_c, key_c, cross, floor, primary)

    # row 3: d
    top_d = 128.0
    ax_d = L.axes(left, top_d, 167.0, 23.0, row="d", col="left")
    key_d = _key_axes(L, left, top_d - 8.0, 167.0, 3.0)
    _panel_d(ax_d, key_d, configs, cross, cross_configs, floor, primary)

    L.letter(ax_a[0], "a", 4.0, 6.0)
    L.letter(ax_b, "b", 4.0, 69.0)
    L.letter(ax_c, "c", 90.0, 69.0)
    L.letter(ax_d, "d", 4.0, 122.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    configs, floor, cross, cross_configs, forces = _load()
    primary = common.primary_models(cross)
    counts = configs["subset"].value_counts()
    h3_ids = ", ".join(configs.loc[configs["subset"] == "H3", "config_id"])

    def force(model, labels="k222", subset="H3", element="all"):
        return _row(cross, model, labels, subset, element)

    def name(model):
        return "MACE-MP-0" if model == "foundation" else common.SHORT_LABELS[model]

    n_h3 = int(force(primary[0])["n_components"])
    n_config = n_h3 // counts["H3"]
    n_element = ", ".join(f"{e} {int(force(primary[0], element=e)['n_components'])}" for e in common.ELEMENTS)
    a_text = "; ".join(f"{name(m)} {common.sig(force(m)['force_rmse_meV_A'])} meV/A and "
                       f"{force(m)['softening_slope']:.2f}" for m in primary)
    others = [m for m in common.MODEL_ORDER if m not in primary]
    b_text = "; ".join(f"{name(m)} {common.sig(force(m)['force_rmse_meV_A'])} (2x2x2) and "
                       f"{common.sig(force(m, 'gamma')['force_rmse_meV_A'])} (Gamma)" for m in [*primary, *others])
    p_text = "; ".join(f"{name(m)} {force(m, element='P')['force_rmse_meV_A']:.0f}" for m in primary)
    floors = ", ".join(f"{e} {_floor(floor, 'H3', e):.1f}" for e in common.ELEMENTS)
    e_text = "; ".join(f"{name(m)} {force(m)['energy_rmse_centred_meV_atom']:.2f}" for m in primary)
    e_all = ", ".join(f"{force(m, subset='All27')['energy_rmse_centred_meV_atom']:.2f}" for m in primary)
    fk = cross_configs[(cross_configs["model"] == "foundation") & (cross_configs["labels"] == "k222")].copy()
    fk["temperature_K"] = fk["config_id"].map(lambda c: common.config_key(c)[0])
    fmeans = fk.groupby("temperature_K")["energy_centred_meV_atom"].mean()
    trend = ", ".join(f"{fmeans[t]:+.1f} at {t} K" for t in sorted(fmeans.index))
    return (
        registry.heading(KEY)
        + f"The test set H3 consists of the {counts['H3']} configurations held out by both fine-tunes ({h3_ids}; "
        f"the subsets are mapped in {registry.label('labels', 'a')}); the reference is the 2x2x2 k-point label "
        f"set unless stated. "
        f"**a**, Force components predicted by the three models that drove the MD runs (identified by force "
        f"fingerprints, {registry.label('models', 'a')}) against the 2x2x2 labels. H3 atoms are coloured by "
        f"element (n = {n_h3} force components per model); the T22 and H2 configurations are drawn in light grey "
        f"behind them (T22 is training data of both fine-tunes, H2 of the 2x2x2 fine-tune). Dashed line, identity; "
        f"solid line, least-squares fit through the origin on H3, whose slope measures force softening. RMSE and "
        f"slope on H3: {a_text}. "
        f"**b**, Force RMSE on H3 (meV/A) of all five models: large markers against the 2x2x2 labels and grey bars "
        f"against the Gamma-point labels, each over all n = {n_h3} H3 force components; small dots, the n = "
        f"{counts['H3']} H3 configurations individually against the 2x2x2 labels ({n_config} components each). "
        f"{b_text}. The three models on the left drove the MD runs; the two on the right were not used for MD. "
        f"Grey band, the H3 label floor ({_floor(floor, 'H3'):.1f} meV/A): the RMSE of the Gamma minus 2x2x2 force "
        f"components over H3, the force error that separates the two label sets themselves. With "
        f"{counts['H3']} held-out configurations no confidence intervals are given. "
        f"**c**, Force RMSE on H3 against the 2x2x2 labels per element for the three MD models (n = {n_element} "
        f"force components per model), with the per-element H3 label floor (grey bars; RMSE of the Gamma minus "
        f"2x2x2 force components of that element over H3: {floors} meV/A). P carries the largest error "
        f"({p_text} meV/A). Panels b and c share a logarithmic axis. "
        f"**d**, Energy error per configuration (n = {len(configs)}) against the 2x2x2 labels after removing each "
        f"model's constant offset, the mean error over the {counts['T22']} T22 configurations (all configurations "
        f"have the composition Li24P4S20Cl4), ordered by source temperature and frame. H3 (dark band) and H2 "
        f"(light band) configurations are in full colour, T22 faint. RMSE on H3: {e_text} meV/atom (All27: {e_all} "
        f"in the same order). The foundation model's centred error drifts with source temperature (mean {trend} "
        f"meV/atom). Grey band, the centred energy label floor: the RMSE over all {len(configs)} configurations of the "
        f"Gamma minus 2x2x2 energies, centred in the same way (offset over T22). "
        f"{common.naming_sentence(configs)} Fine-tuned models are compared with DFT and with the foundation model "
        f"only. "
        + common.sources([CONFIGS, FLOOR, CROSS, CROSS_CONFIGS, FORCES])
    )


CAPTION = _caption()
