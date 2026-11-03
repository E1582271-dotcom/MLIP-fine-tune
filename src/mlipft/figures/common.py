"""Helpers shared by the figure modules: tables, millimetre layout, the encoding helpers, number formats.

Layout is in millimetres from the top-left corner of the figure. ``Layout.axes(..., row=, col=)``
tags every framed panel so that ``qa.alignment`` can verify that panels in the same row share their
top and bottom frame edges and panels in the same column their left edge; ``Layout.save`` runs the
QA checks and writes the figure.

Every marker of a model, of AIMD or of a measured value is drawn through ``model_style`` /
``model_scatter`` / ``aimd_style`` / ``context_style``, which apply the vocabulary of ``style``:
colour by model, fill by training stage, shape by run type. The models that drove the MD runs come
from the ``drove_md`` column of the cross-evaluation table (``primary_models``, ``md_models``,
``main_models``), not from this file.

Run families are the ``model`` field of the series catalogue: foundation and mattersim (the universal
potentials), ftG (Gamma labels), ftK (2x2x2 labels, the July run, stage one) and ftK2 (2x2x2 labels, the
run of amendment A5, stage two). The main figures show ``MAIN_FAMILIES``.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D

from .. import paths
from ..constants import SERIES_BY_ID, TEMPERATURES
from . import qa, style

MODEL_ORDER = ("foundation", "ftG_s2", "ftK_s2", "ftG_s1", "ftK_s1")  # MACE-MP-0 and its fine-tunes
SHORT_LABELS = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG_s1": "Gamma, stage 1",
                "ftG_s2": "Gamma, stage 2", "ftK_s1": "2x2x2, stage 1", "ftK_s2": "2x2x2, stage 2"}
LABEL_SET_NAMES = {"gamma": "Gamma", "k222": "2x2x2"}
ELEMENTS = ("Li", "P", "S", "Cl")
RUN_FAMILIES = ("foundation", "ftG", "ftK", "ftK2")  # MD runs of MACE-MP-0 and of its fine-tunes
# The main figures: the universal potentials and the stage-two fine-tunes, named without a stage.
MAIN_FAMILIES = ("foundation", "mattersim", "ftG", "ftK2")
MAIN_SHORT_LABELS = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG": "Fine-tuned, Gamma",
                     "ftK2": "Fine-tuned, 2x2x2"}
CAGED_SLOPE = 0.25  # AIMD Li MSD log-log slope in the fit window below which the Li ions are caged
MARKER_SIZE = 3.4  # default marker diameter (pt) of the style helpers
_CONFIG = re.compile(r"T(\d+)_f(\d+)")


# --------------------------------------------------------------------------------------- tables


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
    """MACE models (``MODEL_ORDER``) that drove an MD run (non-empty ``drove_md``), in that order; MatterSim,
    which also drove runs, is not among them."""
    drove = set(cross.loc[cross["drove_md"].fillna("").astype(str) != "", "model"])
    return [model for model in MODEL_ORDER if model in drove]


def md_models(cross: pd.DataFrame) -> dict[str, str]:
    """Run family -> id of the model that drove its MD run, from the ``drove_md`` column (the family is the
    catalogue ``model`` of the series it drove).

    For the committed tables: ``{"foundation": "foundation", "ftG": "ftG_s2", "ftK": "ftK_s1", "ftK2": "ftK_s2"}``.
    MatterSim also drove runs but is not a MACE run family: the figures draw it as itself.
    """
    drove = cross.loc[cross["drove_md"].fillna("").astype(str) != "", ["model", "drove_md"]].drop_duplicates()
    out: dict[str, str] = {}
    for model, series in drove.itertuples(index=False):
        family = SERIES_BY_ID[series]["model"]
        if family not in RUN_FAMILIES and family in style.UNIVERSAL_MODELS:
            continue
        if out.get(family, model) != model:
            raise ValueError(f"two models drove the {family} runs: {out[family]} and {model}")
        out[family] = str(model)
    if set(out) != set(RUN_FAMILIES):
        raise ValueError(f"expected one MD model per run family {RUN_FAMILIES}, got {sorted(out)}")
    return {family: out[family] for family in RUN_FAMILIES}


def main_models(cross: pd.DataFrame) -> dict[str, str]:
    """``MAIN_FAMILIES`` -> model id: MatterSim as itself, the MACE families from ``md_models``.

    For the committed tables: ``{"foundation": "foundation", "mattersim": "mattersim", "ftG": "ftG_s2",
    "ftK2": "ftK_s2"}``.
    """
    md = md_models(cross)
    return {family: md.get(family, family) for family in MAIN_FAMILIES}


def caged_temperatures(aimd: pd.DataFrame, threshold: float = CAGED_SLOPE) -> tuple[int, ...]:
    """AIMD temperatures (ascending) at which the Li MSD is almost flat in the fit window (log-log slope below
    ``threshold``): there the AIMD D measures caged motion, not diffusion."""
    caged = aimd.loc[aimd["msd_loglog_slope_fit_window"] < threshold, "temperature_K"].astype(int)
    return tuple(sorted(set(caged)))


# ------------------------------------------------------------------------------ encoding helpers


def filled(model: str) -> bool:
    """Whether the model's markers are filled: universal potentials and stage two filled, stage one open.

    Accepts model ids (``ftG_s1``) and run families (``ftG``, ``ftK``, ``ftK2``: the stage in ``style.MD_STAGE``).
    """
    if model in style.UNIVERSAL_MODELS:
        return True
    family, _, stage = model.partition("_")
    return style.STAGE_FILLED[stage or style.MD_STAGE[family]]


def model_style(model: str, kind: str = "production", size: float = MARKER_SIZE, edge: float = 0.7,
                **overrides) -> dict:
    """``ax.plot`` keyword arguments for one model's markers.

    ``model``: a key of ``style.MODEL_COLORS`` (foundation, mattersim, ftG, ftK, ftG_s1, ...). ``kind``: the
    run type, a key of ``style.RUN_MARKERS`` (production circle, cell_52 triangle, clexcess inverted
    triangle). ``size``: marker diameter in pt; ``edge``: edge width in pt. Filled or white face by stage
    (``filled``). ``overrides`` (zorder, label, linestyle, ...) are applied last.
    """
    color = style.MODEL_COLORS[model]
    kwargs = {"color": color, "marker": style.RUN_MARKERS[kind], "markersize": size, "markeredgewidth": edge,
              "markeredgecolor": color, "markerfacecolor": color if filled(model) else style.WHITE,
              "linestyle": "none"}
    kwargs.update(overrides)
    return kwargs


def aimd_style(size: float = MARKER_SIZE, edge: float = 0.7, **overrides) -> dict:
    """``ax.plot`` keyword arguments for AIMD: a filled black square, the same in every figure."""
    kwargs = {"color": style.AIMD_COLOR, "marker": style.AIMD_MARKER, "markersize": size, "markeredgewidth": edge,
              "markeredgecolor": style.AIMD_COLOR, "markerfacecolor": style.AIMD_COLOR, "linestyle": "none"}
    kwargs.update(overrides)
    return kwargs


def context_style(size: float = MARKER_SIZE, edge: float = 0.5, **overrides) -> dict:
    """``ax.plot`` keyword arguments for a measured value (context only, Extended Data only): light-grey diamond.

    Name such marks in ``style.CONTEXT_TEXT``; never draw them in a panel that shows a fine-tuned model.
    """
    kwargs = {"color": style.CONTEXT_COLOR, "marker": style.MEASURED_MARKER, "markersize": size,
              "markeredgewidth": edge, "markeredgecolor": style.CONTEXT_COLOR, "markerfacecolor": style.CONTEXT_COLOR,
              "linestyle": "none"}
    kwargs.update(overrides)
    return kwargs


def to_scatter(plot_kwargs: dict) -> dict:
    """Convert keyword arguments of the ``*_style`` helpers from ``ax.plot`` to ``ax.scatter`` (same look).

    The marker diameter ``markersize`` (pt) becomes the area ``s`` (pt^2); other keys such as ``zorder``,
    ``alpha``, ``label`` or ``clip_on`` pass through.
    """
    kwargs = dict(plot_kwargs)
    out = {"marker": kwargs.pop("marker", style.MLIP_MARKER), "s": kwargs.pop("markersize", MARKER_SIZE) ** 2,
           "facecolors": kwargs.pop("markerfacecolor", None), "edgecolors": kwargs.pop("markeredgecolor", None),
           "linewidths": kwargs.pop("markeredgewidth", 0.7)}
    for key in ("color", "linestyle", "ls"):
        kwargs.pop(key, None)
    out.update(kwargs)
    return out


def model_scatter(model: str, kind: str = "production", size: float = MARKER_SIZE, edge: float = 0.7,
                  **overrides) -> dict:
    """``ax.scatter`` keyword arguments for one model's markers; the same look as ``model_style``."""
    return to_scatter(model_style(model, kind, size, edge, **overrides))


def handle(plot_kwargs: dict, label: str) -> Line2D:
    """A legend handle drawn with keyword arguments of the ``*_style`` helpers."""
    return Line2D([0], [0], label=label, **plot_kwargs)


def lighten(color: str, amount: float) -> str:
    """``color`` mixed with ``amount`` (0-1) of white, as an opaque hex colour."""
    rgb = np.array(to_rgb(color))
    return to_hex(rgb + (1.0 - rgb) * amount)


# ------------------------------------------------------------------------- annotations, numbers


def temperature_labels(ax, temperatures: Iterable[int] = TEMPERATURES, y: float = 0.97, scale: float = 1000.0,
                       **kwargs) -> list:
    """Write "600 K", "800 K", ... inside the frame above the data columns of a ``scale``/T axis.

    Replaces the forbidden secondary top axis of Arrhenius plots. ``y`` is in axes coordinates (top of the
    text); call after the x limits are final: temperatures outside them are skipped. Returns the texts.
    """
    kwargs = {"ha": "center", "va": "top", "fontsize": style.SMALL, "color": style.CONTEXT_TEXT, **kwargs}
    low, high = sorted(ax.get_xlim())
    texts = []
    for temperature in temperatures:
        x = scale / float(temperature)
        if low <= x <= high:
            texts.append(ax.text(x, y, f"{temperature:g} K", transform=ax.get_xaxis_transform(), **kwargs))
    return texts


def sig(value: float, digits: int = 3) -> str:
    """``value`` to ``digits`` significant figures, keeping trailing zeros: 0.260, 2.44, 213, 5.55, 10.0."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    rounded = float(f"{value:.{digits}g}")
    decimals = max(digits - 1 - math.floor(math.log10(abs(rounded))), 0)
    return f"{rounded:.{decimals}f}"


def sci(value: float, digits: int = 2) -> str:
    """``value`` in ASCII scientific notation with ``digits`` significant figures: 1.7e-5, 3.0e8."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def allow_overlap(*artists):
    """Mark artists (or containers, lists of artists) as deliberate overlaps that QA must not report.

    An exempt text is skipped by every overlap check; an exempt line, collection or bar is skipped as data
    (text-over-data and clipping checks); an exempt Axes has no text-over-data or clipping checks. Use
    sparingly, for example for a label written on purpose on top of the line it names. Returns the first
    argument, so ``allow_overlap(ax.text(...))`` chains.
    """
    for artist in artists:
        if artist is None:
            continue
        if isinstance(artist, (list, tuple)):  # lists, tuples and matplotlib containers (errorbar, bar)
            allow_overlap(*artist)
        else:
            qa.EXEMPT.add(artist)
    return artists[0] if artists else None


# --------------------------------------------------------------------------- configuration axes


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
        ax.plot([group[0] - 0.3, group[-1] + 0.3], [0, 0], color=style.BLACK, lw=0.5, transform=line, clip_on=False)
        ax.text(group.mean(), 0, f"Source temperature {temperature} K", transform=text, ha="center", va="top",
                fontsize=style.FONT_SIZE)


def subset_bands(ax, configs: pd.DataFrame, x: np.ndarray, labels: bool = True) -> None:
    """Shade the held-out configurations (H3 darker, H2 lighter); optionally name them above the axes."""
    for position, subset in zip(x, configs["subset"]):
        if subset == "T22":
            continue
        ax.axvspan(position - 0.45, position + 0.45, color=style.SUBSET_COLORS[subset],
                   alpha=0.18 if subset == "H3" else 0.12, lw=0, zorder=0)
        if labels:  # 2 pt above the frame, so the label never sits on the frame line
            above = transforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=2.0, units="points")
            ax.text(position, 1.0, subset, transform=above, ha="center", va="bottom",
                    fontsize=style.SMALL, color=style.BLACK if subset == "H3" else style.GRAY)


# --------------------------------------------------------------------------------------- layout


class AxesRecord(NamedTuple):
    ax: object
    row: str | None
    col: str | None
    frame: str  # "closed" or "none"
    box_mm: tuple[float, float, float, float]  # requested (left, top, width, height)


class Layout:
    """Axes and panel letters placed in millimetres from the top-left corner of a figure.

    ``records`` lists every axes made by ``axes`` with its row/column tags; after ``save`` (or
    ``qa.gate``), ``name`` and ``findings`` hold the figure name and its QA findings.
    """

    def __init__(self, fig, width_mm: float, height_mm: float):
        self.fig, self.width, self.height = fig, width_mm, height_mm
        self.records: list[AxesRecord] = []
        self.name: str | None = None
        self.findings: list | None = None
        fig._mlipft_layout = self
        qa.track(self)

    def axes(self, left: float, top: float, width: float, height: float, *, row: str | None = None,
             col: str | None = None, frame: str = "closed", **kwargs):
        """Axes whose frame spans ``width`` x ``height`` mm with its top-left corner at (``left``, ``top``) mm.

        ``row``: tag shared by panels whose top and bottom frame edges must align (required for every framed
        axes); ``col``: tag shared by panels whose left frame edges must align. ``frame="none"`` makes a
        frameless axes (schematic, heatmap, table; see ``frameless``). Other keyword arguments go to
        ``fig.add_axes``.
        """
        if frame not in ("closed", "none"):
            raise ValueError(f"frame must be 'closed' or 'none', not {frame!r}")
        ax = self.fig.add_axes((left / self.width, 1.0 - (top + height) / self.height, width / self.width,
                                height / self.height), **kwargs)
        if frame == "none":
            frameless(ax)
        self.records.append(AxesRecord(ax, None if row is None else str(row), None if col is None else str(col),
                                       frame, (left, top, width, height)))
        return ax

    def x(self, mm: float) -> float:
        """Figure fraction of a distance ``mm`` from the left edge."""
        return mm / self.width

    def y(self, mm: float) -> float:
        """Figure fraction of a distance ``mm`` from the top edge."""
        return 1.0 - mm / self.height

    def letter(self, ax, letter: str, left: float, top: float):
        """Panel letter whose right edge is at ``left`` mm and whose bottom is at ``top`` mm; returns the text."""
        return style.panel_label(ax, letter, x=self.x(left), y=self.y(top), transform=self.fig.transFigure)

    def text(self, left: float, top: float, text: str, **kwargs):
        kwargs.setdefault("fontsize", style.FONT_SIZE)
        return self.fig.text(self.x(left), self.y(top), text, **kwargs)

    def save(self, name: str, folder: str = "main", strict: bool | None = None) -> list[Path]:
        """Check the figure (``qa.check``), write it (``style.save``), then report through ``qa.gate``.

        ``name, folder = registry.target(KEY)``. The checks run on the live figure before it is written;
        the strict-mode error is raised after writing, so that a failing figure can still be inspected
        (and crop-audited when the build asks for it). Returns the written paths.
        """
        findings = qa.check(self, name) if qa.mode(strict) != "off" else []
        written = style.save(self.fig, name, folder)
        png = next((path for path in written if path.suffix == ".png"), None)
        qa.gate(self, strict, name=name, png=png, findings=findings)
        return written


def new_figure(width_mm: float, height_mm: float):
    """Figure of ``width_mm`` x ``height_mm`` and its millimetre layout."""
    fig = style.figure(width_mm * style.MM, height_mm * style.MM)
    return fig, Layout(fig, width_mm, height_mm)


def frameless(ax):
    """Draw ``ax`` without a frame (schematics, heatmaps, tables): hide all spines and tick marks.

    Tick labels stay available as row and column names (``ax.set_xticks([])`` drops them). The axes is
    registered so that QA skips its frame and alignment checks and does not check its text against its
    own drawing. Returns ``ax``.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", length=0)
    qa.FRAMELESS.add(ax)
    return ax
