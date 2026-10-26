"""Extended Data Fig. 5: the two DFT label sets (Gamma point and 2x2x2 k-point mesh).

a, Gamma minus 2x2x2 differences per configuration (energy raw and centred, force RMSE, pressure);
b, parity of the force components by element; c, cost of the SCF calculations; d, distribution
of the DFT force magnitudes per element.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import paths
from . import common, style

NAME = "ed05"
WIDTH, HEIGHT = 183.0, 122.0
CONFIGS = paths.DFT / "configs.csv"
QE_RUNS = paths.DFT / "qe_runs.csv"
PSEUDOPOTENTIALS = paths.DFT / "pseudopotentials.csv"
FLOOR = paths.RESULTS / "label_floor.csv"
FLOOR_CONFIGS = paths.RESULTS / "label_floor_configs.csv"
FORCES = paths.MODELS / "predictions_forces.csv.gz"
LABEL_SETS = ("gamma", "k222")
SUBSET_MARKERS = {"T22": {"markerfacecolor": style.LIGHT_GREY, "markeredgecolor": style.GREY, "markersize": 3.0},
                  "H2": {"markerfacecolor": style.GREY, "markeredgecolor": style.GREY, "markersize": 3.2},
                  "H3": {"markerfacecolor": "black", "markeredgecolor": "black", "markersize": 3.2}}


def _load():
    configs = common.ordered_configs(common.read(CONFIGS))
    runs = common.read(QE_RUNS)
    floor = common.read(FLOOR)
    floor_configs = common.read(FLOOR_CONFIGS).set_index("config_id").loc[configs["config_id"]].reset_index()
    forces = common.read(FORCES)
    return configs, runs, floor, floor_configs, forces


def _components(forces: pd.DataFrame, labels: str) -> np.ndarray:
    return forces[[f"dft_{labels}_f{axis}" for axis in "xyz"]].to_numpy()


def _strip(ax, x: np.ndarray, values: np.ndarray, subsets) -> None:
    subsets = np.asarray(subsets)
    for subset, kwargs in SUBSET_MARKERS.items():
        keep = subsets == subset
        ax.plot(x[keep], values[keep], marker="o", linestyle="none", markeredgewidth=0.5, zorder=3, **kwargs)


def _panel_a(axes, configs: pd.DataFrame, floor: pd.DataFrame, floor_configs: pd.DataFrame) -> None:
    x = common.config_positions(configs)
    whole = floor[(floor["subset"] == "All27") & (floor["element"] == "all")].iloc[0]
    offset = float((floor_configs["energy_raw_meV_atom"] - floor_configs["energy_centred_meV_atom"]).mean())
    energy_ax, force_ax, pressure_ax = axes
    for ax in axes:
        common.subset_bands(ax, configs, x, labels=ax is energy_ax)
        common.config_axis(ax, configs, x, labels=ax is pressure_ax)

    energy_ax.axhline(offset, color=style.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    _strip(energy_ax, x, floor_configs["energy_raw_meV_atom"].to_numpy(), floor_configs["subset"])
    energy_ax.set_ylabel("ΔE\n(meV/atom)")
    energy_ax.set_title("Gamma minus 2x2x2 labels, per configuration", loc="left", fontsize=style.SMALL, pad=10)
    energy_ax.set_ylim(offset - 1.6, offset + 1.6)
    right = energy_ax.secondary_yaxis("right", functions=(lambda raw: raw - offset, lambda centred: centred + offset))
    right.set_ylabel("Centred (meV/atom)", fontsize=style.SMALL)
    right.tick_params(labelsize=style.SMALL)

    force_ax.axhline(whole["force_rmse_meV_A"], color=style.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    _strip(force_ax, x, floor_configs["force_rmse_meV_A"].to_numpy(), floor_configs["subset"])
    force_ax.set_ylim(0, 30)
    force_ax.set_ylabel("ΔF RMSE\n(meV/Å)")

    pressure_ax.axhline(0, color=style.GREY, lw=0.5, zorder=1)
    _strip(pressure_ax, x, floor_configs["pressure_kbar"].to_numpy(), floor_configs["subset"])
    pressure_ax.set_ylim(0, 0.8)
    pressure_ax.set_ylabel("ΔP\n(kbar)")

    handles = [Line2D([0], [0], marker="o", linestyle="none", markeredgewidth=0.5, label=subset, **kwargs)
               for subset, kwargs in SUBSET_MARKERS.items()]
    handles.append(Line2D([0], [0], color=style.GREY, lw=0.6, ls=(0, (3, 2)),
                          label="T22 mean (energy), All27 RMSE (force)"))
    energy_ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=4, fontsize=style.SMALL,
                     handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    source = floor_configs.assign(temperature_K=configs["temperature_K"].to_numpy(), frame=configs["frame"].to_numpy(),
                                  x=x, energy_offset_T22_meV_atom=offset)
    style.save_source(source, NAME, "a")


def _panel_b(axes, forces: pd.DataFrame, floor: pd.DataFrame, layout: common.Layout, xlabel_at) -> None:
    gamma, k222 = _components(forces, "gamma"), _components(forces, "k222")
    species = forces["species"].to_numpy()
    rows = []
    for index, (ax, element) in enumerate(zip(axes, common.ELEMENTS)):
        keep = species == element
        ref, val = k222[keep].ravel(), gamma[keep].ravel()
        limit = 5.5 if element == "P" else (4.5 if element == "S" else 2.5)
        ax.plot([-limit, limit], [-limit, limit], color=style.GREY, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.scatter(ref, val, s=1.2, facecolors=style.ELEMENT_COLORS[element], edgecolors="none", zorder=2)
        row = floor[(floor["subset"] == "All27") & (floor["element"] == element)].iloc[0]
        ax.text(0.05, 0.96, element, transform=ax.transAxes, ha="left", va="top", fontsize=style.FONT_SIZE,
                fontweight="bold", color=style.ELEMENT_COLORS[element])
        ax.text(0.95, 0.04, f"RMSE {row['force_rmse_meV_A']:.1f} meV/Å\nslope {row['softening_slope']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=style.SMALL, linespacing=1.3)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ticks = [-4, -2, 0, 2, 4] if limit > 4 else [-2, -1, 0, 1, 2]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if index == 0:
            ax.set_ylabel("Gamma force component (eV/Å)")
        rows.append(pd.DataFrame({"element": element, "dft_k222_eV_A": ref, "dft_gamma_eV_A": val}))
    layout.text(*xlabel_at, "2x2x2 force component (eV/Å)", ha="center", va="top")
    style.save_source(pd.concat(rows, ignore_index=True), NAME, "b")


def _panel_c(time_ax, iter_ax, runs: pd.DataFrame) -> None:
    rows = []
    for index, labels in enumerate(LABEL_SETS):
        frame = runs[runs["label_set"] == labels].sort_values("config_id")
        hours = frame["wall_s"].to_numpy() / 3600.0
        offsets = common.jitter(len(frame), 0.4, seed=10 + index)
        time_ax.scatter(index + offsets, hours, s=4, facecolors=style.DFT_COLORS[labels], edgecolors="none", zorder=3)
        median = float(np.median(hours))
        time_ax.plot([index - 0.3, index + 0.3], [median, median], color=style.DFT_COLORS[labels], lw=0.8, zorder=2)
        text = f"{median * 60:.0f} min" if median < 1 else f"{median:.1f} h"
        time_ax.text(index + 0.32, median, text, ha="left", va="center", fontsize=style.SMALL)
        rows.append(frame.assign(wall_h=hours, x=index + offsets)[["label_set", "config_id", "k_points",
                                                                   "mpi_processes", "scf_iterations", "wall_s",
                                                                   "cpu_s", "wall_h", "x"]])
    ticklabels = []
    for labels in LABEL_SETS:
        frame = runs[runs["label_set"] == labels]
        k, mpi = int(frame["k_points"].iloc[0]), int(frame["mpi_processes"].iloc[0])
        ticklabels.append(f"{common.LABEL_SET_NAMES[labels]}\n{k} k-point{'s' if k > 1 else ''}\n{mpi} MPI ranks")
    time_ax.set_xticks(range(len(LABEL_SETS)), ticklabels, fontsize=style.SMALL)
    time_ax.set_xlim(-0.55, 1.75)
    time_ax.set_ylim(0, 6.5)
    time_ax.set_ylabel("Wall time per\nconfiguration (h)")

    iterations = np.arange(runs["scf_iterations"].min(), runs["scf_iterations"].max() + 1)
    width = 0.38
    for index, labels in enumerate(LABEL_SETS):
        counts = runs[runs["label_set"] == labels]["scf_iterations"].value_counts().reindex(iterations, fill_value=0)
        iter_ax.bar(iterations + (index - 0.5) * width, counts.to_numpy(), width=width, color=style.DFT_COLORS[labels],
                    label=common.LABEL_SET_NAMES[labels], lw=0)
    iter_ax.set_xticks(iterations)
    iter_ax.set_xlabel("SCF iterations")
    iter_ax.set_ylabel("Configurations")
    iter_ax.set_ylim(0, 30)
    iter_ax.legend(loc="upper left", fontsize=style.SMALL, handlelength=0.8, handletextpad=0.3, borderaxespad=0.1)
    style.save_source(pd.concat(rows, ignore_index=True), NAME, "c")


def _panel_d(ax, forces: pd.DataFrame) -> None:
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
    ax.set_xlim(0, 5.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("2x2x2 force magnitude (eV/Å)")
    ax.set_ylabel("Probability density (Å/eV)")
    ax.legend(loc="upper right", fontsize=style.SMALL, handlelength=1.0, handletextpad=0.4, borderaxespad=0.1,
              title="atoms of All27", title_fontsize=style.SMALL)
    style.save_source(pd.concat(rows, ignore_index=True), NAME, "d")


def make() -> list[Path]:
    configs, runs, floor, floor_configs, forces = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)
    left, width = 16.0, 106.0
    tops = (11.0, 31.0, 51.0)
    axes_a = [L.axes(left, top, width, 16.0) for top in tops]
    _panel_a(axes_a, configs, floor, floor_configs)
    ax_time = L.axes(147.0, 11.0, 34.0, 27.0)
    ax_iter = L.axes(147.0, 53.0, 34.0, 14.0)
    _panel_c(ax_time, ax_iter, runs)
    top2, size, gap = 86.0, 24.0, 7.5
    axes_b = [L.axes(left + i * (size + gap), top2, size, size) for i in range(4)]
    _panel_b(axes_b, forces, floor, L, (left + (4 * size + 3 * gap) / 2, top2 + size + 5.0))
    ax_d = L.axes(150.0, top2, 31.0, size)
    _panel_d(ax_d, forces)
    L.letter(axes_a[0], "a", 3.0, 7.0)
    L.letter(ax_time, "c", 134.0, 7.0)
    L.letter(axes_b[0], "b", 3.0, 82.0)
    L.letter(ax_d, "d", 134.0, 82.0)
    return style.save(fig, "ed05_labels", "extended")


def _caption() -> str:
    configs, runs, floor, floor_configs, forces = _load()
    f27 = floor[floor["subset"] == "All27"].set_index("element")
    offset = float((floor_configs["energy_raw_meV_atom"] - floor_configs["energy_centred_meV_atom"]).mean())
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
                        "hi": int(frame["scf_iterations"].max()),
                        "core_h": float((frame["wall_s"] * frame["mpi_processes"]).sum() / 3600.0)}
    g, k = cost["gamma"], cost["k222"]
    minute = bool(((runs.loc[runs["label_set"] == "k222", "wall_s"] % 60) == 0).all())
    magnitude = {labels: np.linalg.norm(_components(forces, labels), axis=1) for labels in LABEL_SETS}
    species = forces["species"].to_numpy()
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
    return (
        f"Extended Data Fig. 5 | The two DFT label sets. Both sets label the same 27 configurations with Quantum "
        f"ESPRESSO {str(settings['qe_version']).removeprefix('v.')} ({' and '.join(libraries)} pseudopotentials, "
        f"{settings['ecutwfc_Ry']:g}/{settings['ecutrho_Ry']:g} Ry, Gaussian smearing {settings['smearing_Ry']:g} "
        f"Ry, convergence threshold {threshold} Ry) and differ only in the k-point sampling: "
        f"Gamma point only, or a 2x2x2 Monkhorst-Pack mesh (taken as the better reference). "
        "**a**, Gamma minus 2x2x2 per configuration, ordered by source temperature and frame: energy per atom "
        f"(left axis raw, {raw.min():.2f} to {raw.max():.2f} meV/atom; right axis after subtracting the T22 mean "
        f"of {offset:.2f} meV/atom, dashed line), force RMSE over all components (dashed line, All27 value "
        f"{f27.loc['all', 'force_rmse_meV_A']:.1f} meV/A; largest {worst['force_rmse_meV_A']:.1f} meV/A for "
        f"{worst['config_id']}) and pressure (RMSE {f27.loc['all', 'pressure_rmse_kbar']:.2f} kbar"
        + ("; the Gamma pressure is the higher one for every configuration" if higher else "")
        + "). Shaded columns, H3 (dark) and H2 (light). "
        f"**b**, Gamma against 2x2x2 force components for all atoms of the 27 configurations, by element: RMSE "
        f"(meV/A) {elements}. Dashed line, identity. "
        f"**c**, Cost of the SCF single points: wall time per configuration (median {g['wall'] / 60:.0f} min with "
        f"{g['k']} k-point on {g['mpi']} MPI processes; {k['wall'] / 3600:.1f} h with {k['k']} k-points on "
        f"{k['mpi']} processes; {g['core_h']:.0f} and {k['core_h']:.0f} core-hours for the 27 configurations) and "
        f"the number of SCF iterations ({g['lo']}-{g['hi']} and {k['lo']}-{k['hi']}). "
        + ("The 2x2x2 wall and CPU times are printed by Quantum ESPRESSO in hours and minutes, so they are "
           "resolved to 1 min. " if minute else "")
        + "**d**, Distribution of the 2x2x2 force magnitudes per element over all 27 configurations (mean "
        + ", ".join(f"{e} {means[e]:.2f}" for e in common.ELEMENTS)
        + f" eV/A); the Gamma distributions are indistinguishable at this scale (mean magnitudes differ by at most "
        f"{mean_diff:.1f} meV/A). "
        + common.sources([CONFIGS, QE_RUNS, PSEUDOPOTENTIALS, FLOOR, FLOOR_CONFIGS, FORCES])
    )


CAPTION = _caption()
