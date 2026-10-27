"""Extended Data figure "labels": the two DFT label sets (Gamma point and 2x2x2 k-point mesh).

a, the 27 labelled configurations by source run, frame and subset; b, Gamma minus 2x2x2 per
configuration (energy, force RMSE, pressure); c, parity of the force components by element;
d, cost of the SCF calculations; e, distribution of the DFT force magnitudes per element.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import paths
from . import common, registry, style

KEY = "labels"
NAME = registry.number(KEY)
PANELS = "abcde"
WIDTH, HEIGHT = 183.0, 164.0
CONFIGS = paths.DFT / "configs.csv"
QE_RUNS = paths.DFT / "qe_runs.csv"
PSEUDOPOTENTIALS = paths.DFT / "pseudopotentials.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
FLOOR_CONFIGS = paths.RESULTS / "label_floor_configs.csv"
FORCES = paths.MODELS / "predictions_forces.csv.gz"
LABEL_SETS = ("gamma", "k222")
SUBSET_ORDER = ("T22", "H3", "H2")
SUBSET_TEXT = {"T22": "trained on by both\nfine-tunes", "H3": "trained on by neither;\nvalidation set of both",
               "H2": "trained on by the 2x2x2\nfine-tune only"}
# Subset markers in b: the colours of the grid in a; the light T22 face gets a grey edge to stay visible.
SUBSET_EDGES = {"T22": style.GRAY, "H2": style.GRAY, "H3": style.SUBSET_COLORS["H3"]}
MARKER = 3.0
DASHED = {**style.IDENTITY_LINE, "linewidth": 0.6}  # dashed guide lines of b


def _load():
    configs = common.ordered_configs(common.read(CONFIGS))
    runs = common.read(QE_RUNS)
    floor = common.read(FLOOR)
    floor_configs = common.read(FLOOR_CONFIGS).set_index("config_id").loc[configs["config_id"]].reset_index()
    forces = common.read(FORCES)
    return configs, runs, floor, floor_configs, forces


def _components(forces: pd.DataFrame, labels: str) -> np.ndarray:
    return forces[[f"dft_{labels}_f{axis}" for axis in "xyz"]].to_numpy()


def _and(values) -> str:
    """'600, 800 and 1000' from (600, 800, 1000)."""
    items = [str(v) for v in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _energy_offset(floor_configs: pd.DataFrame) -> float:
    """The T22 mean of Gamma minus 2x2x2 energy per atom (raw minus centred is that constant)."""
    return float((floor_configs["energy_raw_meV_atom"] - floor_configs["energy_centred_meV_atom"]).mean())


# ------------------------------------------------------------------------------------------- panels


def _panel_a(ax, legend_ax, configs: pd.DataFrame) -> None:
    temperatures = sorted(configs["temperature_K"].unique())
    frames = sorted(configs["frame"].unique())
    rows = []
    for row in configs.itertuples():
        x, y = temperatures.index(row.temperature_K), frames.index(row.frame)
        ax.add_patch(Rectangle((x - 0.46, y - 0.45), 0.92, 0.9, facecolor=style.SUBSET_COLORS[row.subset],
                               edgecolor="none"))
        if row.subset != "T22":
            ax.text(x, y, row.subset, ha="center", va="center", fontsize=style.SMALL,
                    color=style.WHITE if row.subset == "H3" else style.BLACK)
        rows.append({"config_id": row.config_id, "temperature_K": row.temperature_K, "frame": row.frame,
                     "subset": row.subset, "ftG_split": row.ftG_split, "ftK_split": row.ftK_split,
                     "grid_column": x, "grid_row": y})
    ax.set_xlim(-0.5, len(temperatures) - 0.5)
    ax.set_ylim(len(frames) - 0.5, -0.5)
    ax.set_xticks(range(len(temperatures)), [f"{t} K" for t in temperatures], fontsize=style.SMALL)
    ax.set_yticks(range(len(frames)), [str(f) for f in frames], fontsize=style.SMALL)
    ax.tick_params(pad=1.5)
    ax.set_xlabel("Source temperature", labelpad=2)
    ax.set_ylabel("Frame", labelpad=2)
    counts = configs["subset"].value_counts()
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=5.0, markeredgecolor="none",
                      markerfacecolor=style.SUBSET_COLORS[s], label=f"{s} ({counts[s]}): {SUBSET_TEXT[s]}")
               for s in SUBSET_ORDER]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5), handletextpad=0.5,
                     labelspacing=1.4, borderaxespad=0.0, fontsize=style.SMALL, handlelength=1.0)
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _strip(ax, x: np.ndarray, values: np.ndarray, subsets) -> None:
    subsets = np.asarray(subsets)
    for subset in SUBSET_ORDER:
        keep = subsets == subset
        ax.plot(x[keep], values[keep], marker="o", linestyle="none", markersize=MARKER, markeredgewidth=0.5,
                markerfacecolor=style.SUBSET_COLORS[subset], markeredgecolor=SUBSET_EDGES[subset], zorder=3)


def _subset_names(ax, configs: pd.DataFrame, x: np.ndarray) -> None:
    """Name the shaded H3 and H2 columns above the frame, lifted clear of the frame line."""
    lifted = transforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=1.2, units="points")
    for position, subset in zip(x, configs["subset"]):
        if subset != "T22":
            ax.text(position, 1.0, subset, transform=lifted, ha="center", va="bottom", fontsize=style.SMALL,
                    color=style.BLACK if subset == "H3" else style.GRAY)


def _panel_b(axes, configs: pd.DataFrame, floor: pd.DataFrame, floor_configs: pd.DataFrame) -> None:
    x = common.config_positions(configs)
    whole = floor[(floor["subset"] == "All27") & (floor["element"] == "all")].iloc[0]
    offset = _energy_offset(floor_configs)
    energy_ax, force_ax, pressure_ax = axes
    for ax in axes:
        common.subset_bands(ax, configs, x, labels=False)
        common.config_axis(ax, configs, x, labels=ax is pressure_ax)
        ax.yaxis.set_label_coords(-0.075, 0.5)
    _subset_names(energy_ax, configs, x)
    # every other frame is labelled (100, 300, ..., 900): 27 labels side by side read as one run of digits
    pressure_ax.set_xticks(x, [str(f) if f % 200 == 100 else "" for f in configs["frame"]])
    right = x[-1] + 0.4  # annotations sit above the 1000 K configurations, where no column is shaded

    energy_ax.axhline(offset, **DASHED)
    _strip(energy_ax, x, floor_configs["energy_raw_meV_atom"].to_numpy(), floor_configs["subset"])
    energy_ax.set_ylim(3.5, offset + 2.3)
    energy_ax.set_yticks([4, 5, 6])
    energy_ax.set_ylabel("ΔE\n(meV/atom)")
    energy_ax.text(right, offset + 2.15, f"Dashed: T22 mean offset, {offset:.2f} meV/atom", ha="right", va="top",
                   fontsize=style.SMALL)
    energy_ax.set_title("Gamma minus 2x2x2 labels, per configuration", loc="left", fontsize=style.FONT_SIZE,
                        pad=9)

    force_ax.axhline(whole["force_rmse_meV_A"], **DASHED)
    _strip(force_ax, x, floor_configs["force_rmse_meV_A"].to_numpy(), floor_configs["subset"])
    force_ax.set_ylim(0, 30)
    force_ax.set_yticks([0, 10, 20])
    force_ax.set_ylabel("ΔF RMSE\n(meV/Å)")
    force_ax.text(right, 1.5, f"Dashed: All27 RMSE, {whole['force_rmse_meV_A']:.1f} meV/Å", ha="right",
                  va="bottom", fontsize=style.SMALL)

    _strip(pressure_ax, x, floor_configs["pressure_kbar"].to_numpy(), floor_configs["subset"])
    pressure_ax.set_ylim(0, 0.8)
    pressure_ax.set_yticks([0, 0.2, 0.4, 0.6])
    pressure_ax.set_ylabel("ΔP\n(kbar)")
    pressure_ax.text(right, 0.04, f"All27 RMSE {whole['pressure_rmse_kbar']:.2f} kbar", ha="right", va="bottom",
                     fontsize=style.SMALL)

    source = floor_configs.assign(temperature_K=configs["temperature_K"].to_numpy(),
                                  frame=configs["frame"].to_numpy(), x=x, energy_offset_T22_meV_atom=offset,
                                  force_rmse_all27_meV_A=whole["force_rmse_meV_A"],
                                  pressure_rmse_all27_kbar=whole["pressure_rmse_kbar"])
    style.save_source(source, NAME, "b")


def _panel_c(axes, forces: pd.DataFrame, floor: pd.DataFrame, layout: common.Layout, xlabel_at) -> None:
    gamma, k222 = _components(forces, "gamma"), _components(forces, "k222")
    species = forces["species"].to_numpy()
    rows = []
    for index, (ax, element) in enumerate(zip(axes, common.ELEMENTS)):
        keep = species == element
        ref, val = k222[keep].ravel(), gamma[keep].ravel()
        limit = {"Li": 2.75, "P": 5.5, "S": 4.5, "Cl": 2.5}[element]
        ax.plot([-limit, limit], [-limit, limit], **style.IDENTITY_LINE)
        ax.scatter(ref, val, s=1.2, facecolors=style.ELEMENT_COLORS[element], edgecolors="none", zorder=2)
        row = floor[(floor["subset"] == "All27") & (floor["element"] == element)].iloc[0]
        ax.legend(handles=[Line2D([0], [0], marker="s", linestyle="none", markersize=4.0, markeredgecolor="none",
                                  markerfacecolor=style.ELEMENT_COLORS[element], label=element)],
                  loc="upper left", fontsize=style.FONT_SIZE, prop={"weight": "bold", "size": style.FONT_SIZE},
                  handlelength=0.8, handletextpad=0.4, borderaxespad=0.5)
        ax.text(0.95, 0.05, f"RMSE {row['force_rmse_meV_A']:.1f} meV/Å\nslope {row['softening_slope']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=style.SMALL, linespacing=1.3)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ticks = [-4, -2, 0, 2, 4] if limit > 4 else [-2, -1, 0, 1, 2]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if index == 0:
            ax.set_ylabel("Gamma force component (eV/Å)")
        rows.append(pd.DataFrame({"element": element, "dft_k222_eV_A": ref, "dft_gamma_eV_A": val,
                                  "rmse_meV_A": row["force_rmse_meV_A"], "slope": row["softening_slope"]}))
    layout.text(*xlabel_at, "2x2x2 force component (eV/Å)", ha="center", va="top")
    style.save_source(pd.concat(rows, ignore_index=True), NAME, "c")


def _panel_d(time_ax, iter_ax, runs: pd.DataFrame) -> None:
    rows = []
    for index, labels in enumerate(LABEL_SETS):
        frame = runs[runs["label_set"] == labels].sort_values("config_id")
        hours = frame["wall_s"].to_numpy() / 3600.0
        offsets = common.jitter(len(frame), 0.4, seed=10 + index)
        color = style.LABEL_COLORS[labels]
        time_ax.scatter(index + offsets, hours, s=4, facecolors=color, edgecolors="none", zorder=3)
        median = float(np.median(hours))
        time_ax.plot([index - 0.3, index + 0.3], [median, median], color=color, lw=0.8, zorder=2)
        text = f"{median * 60:.0f} min" if median < 1 else f"{median:.1f} h"
        time_ax.text(index + 0.34, median, text, ha="left", va="center", fontsize=style.SMALL)
        rows.append(frame.assign(wall_h=hours, x=index + offsets, median_wall_h=median)[
            ["label_set", "config_id", "k_points", "mpi_processes", "scf_iterations", "wall_s", "cpu_s", "wall_h",
             "x", "median_wall_h"]])
    ticklabels = []
    for labels in LABEL_SETS:
        frame = runs[runs["label_set"] == labels]
        k, mpi = int(frame["k_points"].iloc[0]), int(frame["mpi_processes"].iloc[0])
        ticklabels.append(f"{common.LABEL_SET_NAMES[labels]}\n{k} k-point{'s' if k > 1 else ''}\n{mpi} MPI ranks")
    time_ax.set_xticks(range(len(LABEL_SETS)), ticklabels, fontsize=style.SMALL)
    time_ax.set_xlim(-0.5, 1.85)
    time_ax.set_ylim(0, 6.5)
    time_ax.set_ylabel("Wall time per\nconfiguration (h)")

    iterations = np.arange(runs["scf_iterations"].min(), runs["scf_iterations"].max() + 1)
    width = 0.38
    counts_rows = []
    for index, labels in enumerate(LABEL_SETS):
        counts = runs[runs["label_set"] == labels]["scf_iterations"].value_counts().reindex(iterations, fill_value=0)
        iter_ax.bar(iterations + (index - 0.5) * width, counts.to_numpy(), width=width,
                    color=style.LABEL_COLORS[labels], label=common.LABEL_SET_NAMES[labels], lw=0)
        counts_rows.append(pd.DataFrame({"label_set": labels, "scf_iterations": iterations,
                                         "configurations": counts.to_numpy()}))
    iter_ax.set_xticks(iterations)
    iter_ax.set_xlim(iterations[0] - 0.6, iterations[-1] + 0.6)
    iter_ax.set_xlabel("SCF iterations")
    iter_ax.set_ylabel("Configurations")
    iter_ax.set_ylim(0, 36)
    iter_ax.set_yticks([0, 10, 20, 30])
    iter_ax.legend(loc="upper left", fontsize=style.SMALL, handlelength=0.8, handletextpad=0.3, borderaxespad=0.3,
                   ncol=2, columnspacing=0.8)
    histogram = pd.concat(counts_rows, ignore_index=True)
    source = pd.concat([pd.concat(rows, ignore_index=True).assign(quantity="wall_time"),
                        histogram.assign(quantity="scf_iteration_count")], ignore_index=True)
    style.save_source(source, NAME, "d")


def _panel_e(ax, forces: pd.DataFrame) -> None:
    magnitude = np.linalg.norm(_components(forces, "k222"), axis=1)
    species = forces["species"].to_numpy()
    edges = np.arange(0.0, 5.4 + 1e-9, 0.2)
    rows = []
    for element in common.ELEMENTS:
        density, _ = np.histogram(magnitude[species == element], bins=edges, density=True)
        ax.stairs(density, edges, color=style.ELEMENT_COLORS[element], lw=0.9,
                  label=f"{element} ({int((species == element).sum())})")
        rows.append(pd.DataFrame({"element": element, "bin_low_eV_A": edges[:-1], "bin_high_eV_A": edges[1:],
                                  "density_per_eV_A": density}))
    ax.set_xlim(-0.15, 5.55)  # outlines (zero-density bins, the edges at 0 and 5.4) clear of the spines
    ax.set_ylim(-0.06, 1.75)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel("2x2x2 force magnitude (eV/Å)")
    ax.set_ylabel("Probability density (Å/eV)")
    ax.legend(loc="upper right", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.4, borderaxespad=0.4,
              title="Atoms of All27", title_fontsize=style.SMALL)
    style.save_source(pd.concat(rows, ignore_index=True), NAME, "e")


def make() -> list[Path]:
    configs, runs, floor, floor_configs, forces = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)

    # row 1: a (grid and its legend), b (three strips, the hero)
    top1, strip, gap = 13.0, 14.0, 3.5
    bottom1 = top1 + 3 * strip + 2 * gap
    ax_a = L.axes(13.0, top1, 21.0, bottom1 - top1, frame="none")
    ax_a_legend = L.axes(37.5, top1, 32.0, bottom1 - top1, frame="none")
    _panel_a(ax_a, ax_a_legend, configs)
    axes_b = [L.axes(84.0, top1 + i * (strip + gap), 97.0, strip, row=f"b{i}", col="b") for i in range(3)]
    _panel_b(axes_b, configs, floor, floor_configs)

    # row 2: c (four parity panels)
    top2, size, step = 80.0, 30.0, 45.0
    axes_c = [L.axes(16.0 + i * step, top2, size, size, row="c", col="left" if i == 0 else None) for i in range(4)]
    _panel_c(axes_c, forces, floor, L, (16.0 + (3 * step + size) / 2, top2 + size + 5.0))

    # row 3: d (wall time, SCF iterations), e (force magnitudes)
    top3, height3 = 130.0, 22.0
    ax_time = L.axes(16.0, top3, 34.0, height3, row="de", col="left")
    ax_iter = L.axes(66.0, top3, 32.0, height3, row="de")
    _panel_d(ax_time, ax_iter, runs)
    ax_e = L.axes(126.0, top3, 55.0, height3, row="de")
    _panel_e(ax_e, forces)

    L.letter(ax_a, "a", 3.0, 7.0)
    L.letter(axes_b[0], "b", 71.0, 7.0)
    L.letter(axes_c[0], "c", 3.0, top2 - 4.0)
    L.letter(ax_time, "d", 3.0, top3 - 4.0)
    L.letter(ax_e, "e", 111.0, top3 - 4.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    configs, runs, floor, floor_configs, forces = _load()
    counts = configs["subset"].value_counts()
    h3_ids = ", ".join(configs.loc[configs["subset"] == "H3", "config_id"])
    held_1000 = int(((configs["temperature_K"] == 1000) & (configs["subset"] != "T22")).sum())
    f27 = floor[floor["subset"] == "All27"].set_index("element")
    offset = _energy_offset(floor_configs)
    raw = floor_configs["energy_raw_meV_atom"]
    worst = floor_configs.loc[floor_configs["force_rmse_meV_A"].idxmax()]
    higher = bool((floor_configs["pressure_kbar"] > 0).all())
    elements = ", ".join(f"{e} {f27.loc[e, 'force_rmse_meV_A']:.1f} (slope {f27.loc[e, 'softening_slope']:.3f})"
                         for e in common.ELEMENTS)
    cost = {}
    for labels in LABEL_SETS:
        frame = runs[runs["label_set"] == labels]
        cost[labels] = {"k": int(frame["k_points"].iloc[0]), "mpi": int(frame["mpi_processes"].iloc[0]),
                        "wall": float(frame["wall_s"].median()), "lo": int(frame["scf_iterations"].min()),
                        "hi": int(frame["scf_iterations"].max()), "n": len(frame),
                        "core_h": float((frame["wall_s"] * frame["mpi_processes"]).sum() / 3600.0)}
    g, k = cost["gamma"], cost["k222"]
    minute = bool(((runs.loc[runs["label_set"] == "k222", "wall_s"] % 60) == 0).all())
    magnitude = {labels: np.linalg.norm(_components(forces, labels), axis=1) for labels in LABEL_SETS}
    species = forces["species"].to_numpy()
    atoms = {e: int((species == e).sum()) for e in common.ELEMENTS}
    means = {e: magnitude["k222"][species == e].mean() for e in common.ELEMENTS}
    mean_diff = max(abs(magnitude["gamma"][species == e].mean() - means[e]) for e in common.ELEMENTS) * 1000.0
    settings = {}
    for column in ("qe_version", "ecutwfc_Ry", "ecutrho_Ry", "smearing_Ry", "conv_thr_Ry"):
        values = runs[column].unique()
        if len(values) != 1:
            raise ValueError(f"{column} differs between the label sets")
        settings[column] = values[0]
    threshold = f"{settings['conv_thr_Ry']:.0e}".replace("e-0", "e-")
    libraries = common.read(PSEUDOPOTENTIALS)["library"].str.split(";").str[0].unique()
    n_configs = len(configs)
    return (
        registry.heading(KEY) + f"Both sets label the same {n_configs} configurations with Quantum "
        f"ESPRESSO {str(settings['qe_version']).removeprefix('v.')} ({' and '.join(libraries)} pseudopotentials, "
        f"{settings['ecutwfc_Ry']:g}/{settings['ecutrho_Ry']:g} Ry, Gaussian smearing {settings['smearing_Ry']:g} "
        f"Ry, convergence threshold {threshold} Ry) and differ only in the k-point sampling: "
        f"Gamma point only, or a 2x2x2 Monkhorst-Pack mesh (taken as the reference). "
        f"**a**, The {n_configs} labelled configurations, frames 100-900 of the 52-atom, 50-ps MACE-MP-0 runs at "
        f"{_and(sorted(configs['temperature_K'].unique()))} K, by subset: T22, trained on by "
        f"both fine-tunes (n = {counts['T22']}); H3, trained on by neither (n = {counts['H3']}: {h3_ids}), the "
        f"validation configurations of both fine-tunes, which therefore took part in selecting their epochs, and "
        f"the subset of the primary comparison; H2, validation configurations of the Gamma fine-tune and "
        f"training configurations of the 2x2x2 fine-tune (n = {counts['H2']}). "
        f"{'No' if held_1000 == 0 else held_1000} 1000 K configuration is outside the training data. "
        f"**b**, Gamma minus 2x2x2 for each of the {n_configs} configurations, ordered by source temperature and "
        f"frame (every other frame labelled) and coloured by subset as in a (H3 and H2 columns shaded): energy per atom "
        f"({raw.min():.2f} to {raw.max():.2f} meV/atom; dashed line, the mean offset over T22, "
        f"{offset:.2f} meV/atom, which the energy comparisons subtract; centred RMSE over All27 "
        f"{f27.loc['all', 'energy_rmse_centred_meV_atom']:.2f} meV/atom), force RMSE over all components of a "
        f"configuration (dashed line, the All27 value {f27.loc['all', 'force_rmse_meV_A']:.1f} meV/A, the label "
        f"floor used in {registry.label('forces')}; largest {worst['force_rmse_meV_A']:.1f} meV/A for "
        f"{worst['config_id']}) and pressure (RMSE {f27.loc['all', 'pressure_rmse_kbar']:.2f} kbar"
        + ("; the Gamma pressure is the higher one for every configuration" if higher else "")
        + "). Single values per configuration, no error bars. "
        f"**c**, Gamma against 2x2x2 force components of all atoms of the {n_configs} configurations, by element "
        f"(n = " + ", ".join(f"{atoms[e]} {e}" for e in common.ELEMENTS) + " atoms, three components each); "
        "RMSE (meV/A) and "
        f"least-squares slope through the origin: {elements}. Dashed line, identity; no error bars. "
        f"**d**, Cost of the SCF single points: wall time of each configuration (n = {g['n']} per label set; "
        f"horizontal line, median: {g['wall'] / 60:.0f} min with {g['k']} k-point on {g['mpi']} MPI processes; "
        f"{k['wall'] / 3600:.1f} h with {k['k']} k-points on {k['mpi']} processes; {g['core_h']:.0f} and "
        f"{k['core_h']:.0f} core-hours for the {n_configs} configurations) and the number of SCF iterations "
        f"({g['lo']}-{g['hi']} and {k['lo']}-{k['hi']}); single values, no error bars. "
        + ("The 2x2x2 wall and CPU times are printed by Quantum ESPRESSO in hours and minutes, so they are "
           "resolved to 1 min. " if minute else "")
        + f"**e**, Distribution of the 2x2x2 force magnitudes per element over the {n_configs} configurations "
        "(bins of 0.2 eV/A; atoms per element in the legend; mean "
        + ", ".join(f"{e} {means[e]:.2f}" for e in common.ELEMENTS)
        + f" eV/A); the Gamma distributions are indistinguishable at this scale (mean magnitudes differ by at most "
        f"{mean_diff:.1f} meV/A); no error bars. "
        + common.sources([CONFIGS, QE_RUNS, PSEUDOPOTENTIALS, FLOOR, FLOOR_CONFIGS, FORCES])
    )


CAPTION = _caption()
