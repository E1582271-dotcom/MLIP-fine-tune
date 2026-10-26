"""Helpers shared by the figure modules: reading tables, millimetre layout, model and configuration styling.

Layout is in millimetres from the top-left corner of the figure, so panels in the same row or column
share their frame edges exactly. Model order and styling follow ``style``: colour by fine-tune
(foundation blue, Gamma labels vermillion, 2x2x2 labels green), stage two filled and stage one open.
The primary models (those that drove the MD runs) come from the ``drove_md`` column of the
cross-evaluation table, not from this file.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import to_hex, to_rgb

from .. import paths
from . import style

MODEL_ORDER = ("foundation", "ftG_s2", "ftK_s1", "ftG_s1", "ftK_s2")
SHORT_LABELS = {"foundation": "MACE-MP-0", "ftG_s1": "Gamma, stage 1", "ftG_s2": "Gamma, stage 2",
                "ftK_s1": "2x2x2, stage 1", "ftK_s2": "2x2x2, stage 2"}
LABEL_SET_NAMES = {"gamma": "Gamma", "k222": "2x2x2"}
ELEMENTS = ("Li", "P", "S", "Cl")
SECONDARY_FADE = 0.55  # secondary models are drawn in their colour mixed with this much white
_CONFIG = re.compile(r"T(\d+)_f(\d+)")


def read(path: Path | str, **kwargs) -> pd.DataFrame:
    """A committed table with exact floats; empty cells are NaN, the string 'NA' stays a string."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip", **kwargs)


def sources(files: Iterable[Path | str]) -> str:
    """The closing 'Sources: ...' sentence of a caption, with repository-relative paths."""
    return "Sources: " + ", ".join(paths.relpath(path) for path in files) + "."


def config_key(config_id: str) -> tuple[int, int]:
    """(source temperature in K, frame) of an id such as ``T600_f400``."""
    match = _CONFIG.fullmatch(config_id)
    if match is None:
        raise ValueError(f"unexpected configuration id {config_id!r}")
    return int(match.group(1)), int(match.group(2))


def ordered_configs(configs: pd.DataFrame) -> pd.DataFrame:
    """Configurations ordered by source temperature, then frame, with ``temperature_K`` and ``frame``."""
    keys = configs["config_id"].map(config_key)
    out = configs.assign(temperature_K=[k[0] for k in keys], frame=[k[1] for k in keys])
    return out.sort_values(["temperature_K", "frame"]).reset_index(drop=True)


def naming_sentence(configs: pd.DataFrame) -> str:
    """Caption sentence explaining the short model names, with the split sizes from ``configs.csv``."""
    g, k = configs["ftG_split"].value_counts(), configs["ftK_split"].value_counts()
    return ("'Gamma' and '2x2x2' denote MACE-MP-0 small fine-tuned on the Gamma-point labels "
            f"({g['train']} training and {g['valid']} validation configurations) or on the 2x2x2 labels "
            f"({k['train']} and {k['valid']}); stage 1 is the best epoch before stochastic weight averaging, "
            "stage 2 the model after it.")


def primary_models(cross: pd.DataFrame) -> list[str]:
    """Models that drove an MD run (non-empty ``drove_md``), in ``MODEL_ORDER``."""
    drove = set(cross.loc[cross["drove_md"].fillna("").astype(str) != "", "model"])
    return [model for model in MODEL_ORDER if model in drove]


def filled(model: str) -> bool:
    """Stage two filled, stage one open; the foundation is filled."""
    if model == "foundation":
        return True
    return style.STAGE_FILLED[model.split("_")[1]]


def lighten(color: str, amount: float) -> str:
    """``color`` mixed with ``amount`` (0-1) of white, as an opaque hex colour."""
    rgb = np.array(to_rgb(color))
    return to_hex(rgb + (1.0 - rgb) * amount)


def model_color(model: str, faded: bool = False) -> str:
    color = style.MODEL_COLORS[model]
    return lighten(color, SECONDARY_FADE) if faded else color


def marker_kwargs(model: str, marker: str = "o", size: float = 3.5, edge: float = 0.7, alpha: float = 1.0,
                  faded: bool = False) -> dict:
    """Keyword arguments for ``ax.plot`` drawing one model's marker (``faded``: secondary model)."""
    color = model_color(model, faded)
    return {"marker": marker, "markersize": size, "markeredgewidth": edge, "markeredgecolor": color,
            "markerfacecolor": color if filled(model) else "white", "linestyle": "none", "alpha": alpha}


def scatter_kwargs(model: str, size: float = 9.0, edge: float = 0.6, alpha: float = 1.0, faded: bool = False) -> dict:
    """Keyword arguments for ``ax.scatter`` drawing one model's marker (``size`` in pt^2)."""
    color = model_color(model, faded)
    return {"s": size, "linewidths": edge, "edgecolors": color, "facecolors": color if filled(model) else "white",
            "alpha": alpha}


def jitter(n: int, width: float, seed: int) -> np.ndarray:
    """Deterministic horizontal offsets in [-width/2, width/2] for a strip of ``n`` points."""
    return (np.random.default_rng(seed).random(n) - 0.5) * width


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, float) ** 2)))


def config_positions(configs: pd.DataFrame) -> np.ndarray:
    """x position of each configuration of ``ordered_configs``: one unit apart, a gap between temperatures."""
    temperatures = configs["temperature_K"].to_numpy()
    order = sorted(set(temperatures))
    return np.arange(len(configs)) + np.array([order.index(t) for t in temperatures]) * 0.8


def config_axis(ax, configs: pd.DataFrame, x: np.ndarray, labels: bool = True) -> None:
    """Frame numbers as x tick labels and a bracket per source temperature below them."""
    ax.set_xticks(x, [str(f) for f in configs["frame"]] if labels else [])
    ax.tick_params(axis="x", length=1.5, pad=1.5, labelsize=style.SMALL)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    if not labels:
        return
    blend = ax.get_xaxis_transform()
    line = transforms.offset_copy(blend, fig=ax.figure, y=-9.0, units="points")
    text = transforms.offset_copy(blend, fig=ax.figure, y=-10.5, units="points")
    temperatures = configs["temperature_K"].to_numpy()
    for temperature in sorted(set(temperatures)):
        group = x[temperatures == temperature]
        ax.plot([group[0] - 0.3, group[-1] + 0.3], [0, 0], color="black", lw=0.5, transform=line, clip_on=False)
        ax.text(group.mean(), 0, f"Source temperature {temperature} K", transform=text, ha="center", va="top",
                fontsize=style.FONT_SIZE)


def subset_bands(ax, configs: pd.DataFrame, x: np.ndarray, labels: bool = True) -> None:
    """Shade the held-out configurations (H3 darker, H2 lighter); optionally name them above the axes."""
    for position, subset in zip(x, configs["subset"]):
        if subset == "T22":
            continue
        ax.axvspan(position - 0.45, position + 0.45, color=style.SUBSET_COLORS[subset],
                   alpha=0.18 if subset == "H3" else 0.12, lw=0, zorder=0)
        if labels:
            ax.text(position, 1.0, subset, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=style.SMALL, color="black" if subset == "H3" else style.GREY)


class Layout:
    """Axes and panel letters placed in millimetres from the top-left corner of a figure."""

    def __init__(self, fig, width_mm: float, height_mm: float):
        self.fig, self.width, self.height = fig, width_mm, height_mm

    def axes(self, left: float, top: float, width: float, height: float, **kwargs):
        return self.fig.add_axes((left / self.width, 1.0 - (top + height) / self.height, width / self.width,
                                  height / self.height), **kwargs)

    def x(self, mm: float) -> float:
        return mm / self.width

    def y(self, mm: float) -> float:
        """Figure fraction of a distance ``mm`` from the top edge."""
        return 1.0 - mm / self.height

    def letter(self, ax, letter: str, left: float, top: float) -> None:
        """Panel letter whose right edge is at ``left`` mm and whose bottom is at ``top`` mm."""
        style.panel_label(ax, letter, x=self.x(left), y=self.y(top), transform=self.fig.transFigure)

    def text(self, left: float, top: float, text: str, **kwargs):
        kwargs.setdefault("fontsize", style.FONT_SIZE)
        return self.fig.text(self.x(left), self.y(top), text, **kwargs)


def new_figure(width_mm: float, height_mm: float):
    """Figure of ``width_mm`` x ``height_mm`` and its millimetre layout."""
    fig = style.figure(width_mm * style.MM, height_mm * style.MM)
    return fig, Layout(fig, width_mm, height_mm)


def no_frame(ax) -> None:
    """Hide all four spines (heatmaps, tables and grids drawn on their own)."""
    for spine in ax.spines.values():
        spine.set_visible(False)
