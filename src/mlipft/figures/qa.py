"""Computed quality checks of a drawn figure, and the zoomed crop audit.

Every check runs on the figure drawn at 600 dpi (the resolution of the PNG export; nothing is
rasterised for the checks) and reports ``Finding`` records whose boxes are in millimetres from the
top-left corner of the figure:

* ``alignment``: framed axes with the same ``row`` tag share their drawn top and bottom frame edges,
  axes with the same ``col`` tag their drawn left edge (``Layout.axes(..., row=, col=)``); every
  framed axes must carry a row tag.
* ``frames``: every axes that is not frameless has four visible frame lines, no tick marks or tick
  labels on the top or right, and no secondary axis.
* ``overlaps``: text against text (panel letters, figure texts, axes texts and annotations, tick
  labels, axis labels, titles, legend boxes); text touching any frame line, its own or another
  axes' (no tolerance: any contact with the drawn spine counts); framed axes against each other;
  text or legend boxes against the data drawn in the same axes (markers, line vertices, scatter
  points, error-bar segments, bars) and against the reference lines that span it (axhline,
  axvline, identity and unity lines; no tolerance); markers of different artists in the same axes
  closer than 0.1 mm (``marker-marker``); error bars and caps of one series closer than 0.1 mm to
  the marker of another (``errorbar-marker``); tick marks within 0.3 mm of another axes' frame
  (``tick-frame``).
* ``clipped``: markers that cross the frame, markers within 0.1 mm of a spine, data points outside
  the axis limits.
* ``fonts``: text between 5 and 7 pt, panel letters 8 pt bold; a warning when the font resolves to
  DejaVu (the matplotlib fallback).
* ``size``: width 89, 120 or 183 mm, height at most 247 mm.

Exempt from ``overlaps`` and ``clipped``: artists registered with ``allow_overlap`` (via
``common.allow_overlap``), filled bands and other patches that are not bars, images and meshes,
reference lines as data (they are checked against text only, and may meet the spines), the markers
of one artist against each other, an error bar against its own data point (the marker of the bar's
colour whose centre lies on the bar), and the text and data of frameless axes (schematics,
heatmaps, tables; ``common.frameless``) against each other. Text in a frameless axes is still
checked against all other text and against framed axes. A legend without a frame is measured by
its entries, not by its invisible padding.

Mode: ``strict`` raises ``FigureQAError`` listing every error, ``warn`` prints a compact summary
to stderr, ``off`` skips the checks. The mode comes from the ``strict`` argument, else from an
active ``session`` (``mlipft figures --strict``), else from the environment variable MLIPFT_QA,
else ``strict``. CI sets MLIPFT_QA=warn because font metrics differ between Linux and macOS; the
gate is enforced where the figures are made, together with ``crop_audit`` (``--audit DIR``).
"""
from __future__ import annotations

import json
import math
import os
import sys
import weakref
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib import font_manager
from matplotlib import lines as mlines
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex, to_rgba_array
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetBox
from matplotlib.text import Text

from .. import paths
from . import style

QA_VERSION = 2  # 2: strict text-frame contact, text against reference lines, marker and error-bar spacing,
#                    tick marks near other frames, markers near spines
DPI = style.DPI
MM_PER_INCH = 25.4
ALIGN_TOL_MM = 0.05
OVERLAP_TOL_MM = 0.2  # text against text and text against data: overlaps smaller than this pass
FRAME_TOL_MM = 0.02  # a marker crosses the frame when it reaches this far past the frame line
MARKER_GAP_MM = 0.1  # least gap between markers of different artists, and between a marker and a foreign error bar
SPINE_GAP_MM = 0.1  # least gap between a marker and the drawn spine
TICK_GAP_MM = 0.3  # least gap between a tick mark and the frame of another axes
ON_BAR_MM = 0.02  # a marker centre this close to an error bar of its colour is the bar's own data point
SHAPE_STEP_MM = 0.05  # dashes, error bars and caps are sampled at this spacing
REFERENCE_SPAN = 0.98  # a straight marker-less line covering this fraction of the axes is a reference line
SAMPLE_MM = 0.25  # line segments are sampled at this spacing for the text-against-data check
FONT_RANGE = (style.SMALL, style.LABEL)
MODES = ("strict", "warn", "off")
SUMMARY_LINES = 12  # findings listed in a warn-mode summary

FRAMELESS: weakref.WeakSet = weakref.WeakSet()  # axes drawn without a frame on purpose (common.frameless)
EXEMPT: weakref.WeakSet = weakref.WeakSet()  # artists allowed to overlap (common.allow_overlap)

_SESSIONS: list[dict] = []
_ACTIVE: dict[int, Drawn] = {}


class Finding(NamedTuple):
    """One problem found by a check."""

    figure: str  # figure name, e.g. "fig2_forces" (empty when unknown)
    check: str  # alignment | frames | overlaps | clipped | fonts | size
    kind: str  # short code within the check, e.g. "text-text", "row-top", "top-ticks"
    subject: str  # the artist's text or a description of the axes
    box: tuple[float, float, float, float] | None  # (left, top, right, bottom), mm from the figure's top-left
    detail: str = ""
    severity: str = "error"  # "warning" never fails a strict build

    def __str__(self) -> str:
        where = "" if self.box is None else (f" at x {self.box[0]:.2f}-{self.box[2]:.2f} mm, "
                                             f"y {self.box[1]:.2f}-{self.box[3]:.2f} mm")
        detail = f" ({self.detail})" if self.detail else ""
        flag = "" if self.severity == "error" else f" [{self.severity}]"
        return f"{self.figure or '<figure>'} {self.check}/{self.kind}: {self.subject}{where}{detail}{flag}"

    def as_dict(self) -> dict:
        return {**self._asdict(), "box": None if self.box is None else [round(v, 3) for v in self.box]}


class FigureQAError(AssertionError):
    """Raised in strict mode; ``findings`` holds every finding of the failing figure(s)."""

    def __init__(self, findings: list[Finding]):
        self.findings = list(findings)
        errors = [f for f in self.findings if f.severity == "error"]
        names = sorted({f.figure or "<figure>" for f in errors})
        super().__init__(f"figure QA failed for {', '.join(names)}: {len(errors)} errors\n"
                         + "\n".join(f"  {f}" for f in self.findings))


# ------------------------------------------------------------------------------------- drawing


class Drawn:
    """A figure drawn at QA resolution: its renderer and the conversion of display units to millimetres."""

    def __init__(self, fig, renderer):
        self.fig, self.renderer, self.dpi = fig, renderer, fig.dpi
        self.height = fig.bbox.height

    def px(self, mm: float) -> float:
        """Millimetres -> display pixels."""
        return mm / MM_PER_INCH * self.dpi

    def to_mm(self, value: float) -> float:
        return value / self.dpi * MM_PER_INCH

    def mm(self, bbox) -> tuple[float, float, float, float]:
        """Display bbox -> (left, top, right, bottom) in mm from the figure's top-left corner."""
        return (self.to_mm(bbox.x0), self.to_mm(self.height - bbox.y1), self.to_mm(bbox.x1),
                self.to_mm(self.height - bbox.y0))

    def point_mm(self, x: float, y: float) -> tuple[float, float]:
        return self.to_mm(x), self.to_mm(self.height - y)


@contextmanager
def drawn(fig) -> Iterator[Drawn]:
    """Lay ``fig`` out at 600 dpi without rasterising and yield a ``Drawn``; nested calls reuse it.

    The figure's dpi is restored afterwards, so drawing for QA does not change the exported files.
    """
    key = id(fig)
    if key in _ACTIVE:
        yield _ACTIVE[key]
        return
    old = fig.dpi
    fig.set_dpi(DPI)
    try:
        canvas = fig.canvas
        renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else fig._get_renderer()
        with renderer._draw_disabled():
            fig.draw(renderer)
        state = Drawn(fig, renderer)
        _ACTIVE[key] = state
        yield state
    finally:
        _ACTIVE.pop(key, None)
        fig.set_dpi(old)


# ------------------------------------------------------------------------------------- helpers


def is_frameless(ax) -> bool:
    """Registered with ``common.frameless`` or switched off with ``ax.axis("off")``."""
    return ax in FRAMELESS or not ax.axison


def is_exempt(artist) -> bool:
    return artist in EXEMPT


def _split(target):
    """(figure, layout or None) of a Layout or a Figure."""
    if hasattr(target, "fig") and hasattr(target, "records"):
        return target.fig, target
    return target, getattr(target, "_mlipft_layout", None)


def _name(target, name: str | None) -> str:
    if name:
        return name
    _, layout = _split(target)
    return getattr(layout, "name", None) or ""


def _tree(fig) -> list[tuple[object, object | None]]:
    """(axes, parent axes or None) for every visible axes, child axes (insets, secondary axes) included."""
    out = []

    def visit(ax, parent):
        if not ax.get_visible():
            return
        out.append((ax, parent))
        for child in getattr(ax, "child_axes", []):
            visit(child, ax)

    for ax in fig.axes:
        visit(ax, None)
    return out


def _ancestors(ax, parents: dict) -> set[int]:
    out = set()
    while parents.get(id(ax)) is not None:
        ax = parents[id(ax)]
        out.add(id(ax))
    return out


def _is_colorbar(ax) -> bool:
    return getattr(ax, "_colorbar", None) is not None


def _framed(fig) -> list:
    """Visible axes drawn with a frame (secondary axes excluded)."""
    return [ax for ax, _ in _tree(fig) if not is_frameless(ax) and not isinstance(ax, SecondaryAxis)]


def describe(ax, layout=None) -> str:
    """Short name of an axes: its index, label and row/column tags."""
    record = _record(ax, layout)
    fig = ax.figure
    index = fig.axes.index(ax) if ax in fig.axes else -1
    parts = [f"axes {index}" if index >= 0 else "child axes"]
    label = ax.get_label()
    if label and not label.startswith("_"):
        parts.append(repr(label))
    if record is not None:
        parts.append(f"row={record.row!r} col={record.col!r}")
    return " ".join(parts)


def _record(ax, layout):
    if layout is None:
        return None
    for record in layout.records:
        if record.ax is ax:
            return record
    return None


def _drawn_ticks(axis) -> list:
    try:
        return list(axis._update_ticks())
    except Exception:  # pragma: no cover - private API changed
        return axis.get_major_ticks() + axis.get_minor_ticks()


def _box_array(boxes) -> np.ndarray:
    return np.array([[b.x0, b.y0, b.x1, b.y1] for b in boxes], float).reshape(-1, 4)


def _intersection(a, b) -> tuple[float, float]:
    """Overlap width and height of two display bboxes (negative when apart)."""
    return min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)


class _Item(NamedTuple):
    artist: object
    owner: object | None  # axes the text belongs to; None for figure texts and panel letters
    role: str  # letter | figtext | text | tick | label | title | legend
    box: object  # display bbox
    name: str


def _text_box(artist, renderer):
    if isinstance(artist, Text):
        return Text.get_window_extent(artist, renderer)  # the text only, not an annotation's arrow
    if isinstance(artist, Legend) and not artist.get_frame_on() and getattr(artist, "_legend_box", None) is not None:
        return artist._legend_box.get_window_extent(renderer)  # the drawn entries, not the invisible padding
    return artist.get_window_extent(renderer)


def text_items(fig, d: Drawn) -> list[_Item]:
    """Every drawn text and legend box of ``fig`` with its owner axes and role."""
    items: list[_Item] = []

    def add(artist, owner, role):
        if artist is None or not artist.get_visible():
            return
        if isinstance(artist, Text):
            text = artist.get_text()
            if not text or not text.strip():
                return
            if getattr(artist, "_mlipft_role", "") == "letter":
                owner, role = None, "letter"
            name = text.replace("\n", " ")
        else:
            name = "legend" if isinstance(artist, Legend) else type(artist).__name__
        box = _text_box(artist, d.renderer)
        if not np.all(np.isfinite([box.x0, box.y0, box.x1, box.y1])) or box.width <= 0 or box.height <= 0:
            return
        items.append(_Item(artist, owner, role, box, name))

    for text in fig.texts:
        add(text, None, "figtext")
    for legend in fig.legends:
        add(legend, None, "legend")
    for ax, _ in _tree(fig):
        for text in ax.texts:
            add(text, ax, "text")
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if not axis.get_visible():
                    continue
                for tick in _drawn_ticks(axis):
                    add(tick.label1, ax, "tick")
                    add(tick.label2, ax, "tick")
                add(axis.label, ax, "label")
                add(axis.offsetText, ax, "label")
        for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            add(title, ax, "title")
        legends = [ax.get_legend()] + [a for a in ax.artists if isinstance(a, (Legend, OffsetBox))]
        for legend in dict.fromkeys(a for a in legends if a is not None):
            add(legend, ax, "legend")
    return items


def _all_texts(fig) -> list[tuple[Text, str]]:
    """Every drawn Text (legend entries included) with its role, for the font check."""
    out = []
    for ax, _ in _tree(fig):
        legends = [ax.get_legend()] + [a for a in ax.artists if isinstance(a, Legend)]
        for legend in dict.fromkeys(a for a in legends if a is not None):
            if legend.get_visible():
                out += [(t, "legend") for t in legend.get_texts()] + [(legend.get_title(), "legend")]
    for legend in fig.legends:
        if legend.get_visible():
            out += [(t, "legend") for t in legend.get_texts()] + [(legend.get_title(), "legend")]
    return [(t, role) for t, role in out if t.get_visible() and t.get_text().strip()]


def _marker_radius_px(size_pt: float, edge_pt: float, d: Drawn) -> float:
    return (size_pt / 2.0 + edge_pt / 2.0) * d.dpi / 72.0


def _no_marker(marker) -> bool:
    return marker is None or (isinstance(marker, str) and marker in ("", " ", "None", "none"))


def _no_line(line: Line2D) -> bool:
    return line.get_linestyle() in ("None", "", " ", "none") or line.get_linewidth() <= 0


def _is_reference_line(line: Line2D, ax) -> bool:
    """Straight marker-less lines spanning the axes: axhline, axvline, identity and other guide lines."""
    if not _no_marker(line.get_marker()):
        return False
    xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return False
    rel = ax.transAxes.inverted().transform(xy)
    span_x, span_y = np.ptp(rel[:, 0]), np.ptp(rel[:, 1])
    if len(xy) == 2:
        return span_x >= REFERENCE_SPAN or span_y >= REFERENCE_SPAN
    flat_y, flat_x = np.ptp(xy[:, 1]) < 0.5, np.ptp(xy[:, 0]) < 0.5
    return (flat_y and span_x >= REFERENCE_SPAN) or (flat_x and span_y >= REFERENCE_SPAN)


def _sample(xy: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline (display units) at most ``step`` apart; NaN vertices break it."""
    if len(xy) < 2:
        return xy[np.isfinite(xy).all(axis=1)]
    p0, p1 = xy[:-1], xy[1:]
    ok = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
    p0, p1 = p0[ok], p1[ok]
    if not len(p0):
        return np.empty((0, 2))
    n = np.maximum(1, np.ceil(np.hypot(*(p1 - p0).T) / step)).astype(int)
    index = np.repeat(np.arange(len(p0)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    points = p0[index] + (p1 - p0)[index] * t[:, None]
    return np.vstack([points, p1[-1:]])


class _Data(NamedTuple):
    artist: object
    xy: np.ndarray  # (n, 2) display units
    radius: np.ndarray  # (n,) display units
    clip_on: bool
    marker: bool  # True for markers and scatter points, False for sampled line segments
    name: str


def _artist_name(artist) -> str:
    label = artist.get_label() if hasattr(artist, "get_label") else ""
    if label and not str(label).startswith("_"):
        return f"{type(artist).__name__} {label!r}"
    return type(artist).__name__


def data_points(ax, d: Drawn, markers_only: bool = False) -> list[_Data]:
    """The data drawn in ``ax`` as points with a radius: markers, scatter points, sampled line segments."""
    out: list[_Data] = []
    step = d.px(SAMPLE_MM)
    for line in ax.lines:
        if not line.get_visible() or is_exempt(line):
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float).reshape(-1, 2))
        if not _no_marker(line.get_marker()) and line.get_markersize() > 0:
            radius = _marker_radius_px(line.get_markersize(), line.get_markeredgewidth(), d)
            keep = np.isfinite(xy).all(axis=1)
            out.append(_Data(line, xy[keep], np.full(int(keep.sum()), radius), line.get_clip_on(), True,
                             _artist_name(line)))
        if markers_only or _no_line(line) or _is_reference_line(line, ax):
            continue
        points = _sample(xy, step)
        out.append(_Data(line, points, np.full(len(points), line.get_linewidth() / 2 * d.dpi / 72.0),
                         line.get_clip_on(), False, _artist_name(line)))
    for collection in ax.collections:
        if not collection.get_visible() or is_exempt(collection):
            continue
        if isinstance(collection, PathCollection):
            offsets = np.ma.filled(np.ma.asarray(collection.get_offsets(), float), np.nan).reshape(-1, 2)
            if not len(offsets):
                continue
            xy = collection.get_offset_transform().transform(offsets)
            sizes = np.asarray(collection.get_sizes(), float)
            widths = np.asarray(collection.get_linewidths(), float)
            if not len(sizes):
                continue
            sizes = np.resize(sizes, len(xy))
            widths = np.resize(widths if len(widths) else np.zeros(1), len(xy))
            radius = (np.sqrt(sizes) / 2.0 + widths / 2.0) * d.dpi / 72.0
            keep = np.isfinite(xy).all(axis=1)
            out.append(_Data(collection, xy[keep], radius[keep], collection.get_clip_on(), True,
                             _artist_name(collection)))
        elif isinstance(collection, LineCollection) and not markers_only:
            transform = collection.get_transform()
            widths = np.asarray(collection.get_linewidths(), float)
            width = float(widths.max()) if len(widths) else 0.0
            if width <= 0:
                continue
            chunks = [_sample(transform.transform(np.asarray(segment, float).reshape(-1, 2)), step)
                      for segment in collection.get_segments() if len(segment)]
            if not chunks:
                continue
            points = np.vstack(chunks)
            out.append(_Data(collection, points, np.full(len(points), width / 2 * d.dpi / 72.0),
                             collection.get_clip_on(), False, _artist_name(collection)))
    return out


def _bars(ax, d: Drawn) -> list[tuple[object, object]]:
    """(patch, display bbox) of every visible bar."""
    out = []
    for container in ax.containers:
        if isinstance(container, BarContainer):
            for patch in container.patches:
                if patch.get_visible() and not is_exempt(patch):
                    out.append((patch, patch.get_window_extent(d.renderer)))
    return out


def _clip_visible(data: _Data, frame) -> np.ndarray:
    """Mask of the points that are drawn: inside the frame (by their radius) when the artist is clipped."""
    if not data.clip_on:
        return np.ones(len(data.xy), bool)
    x, y, r = data.xy[:, 0], data.xy[:, 1], data.radius
    return (x > frame.x0 - r) & (x < frame.x1 + r) & (y > frame.y0 - r) & (y < frame.y1 + r)


def _spine_half_width(ax, d: Drawn) -> float:
    """Half the width of the widest visible spine of ``ax`` (display units)."""
    widths = [spine.get_linewidth() for spine in ax.spines.values() if spine.get_visible()]
    return (max(widths) if widths else 0.0) / 2 * d.dpi / 72.0


def _grown(box, by: float) -> tuple[float, float, float, float]:
    return box.x0 - by, box.y0 - by, box.x1 + by, box.y1 + by


def _box_meets(box, other: tuple[float, float, float, float]) -> bool:
    """Whether a display bbox and an (x0, y0, x1, y1) rectangle share an area (no tolerance)."""
    return min(box.x1, other[2]) - max(box.x0, other[0]) > 0 and min(box.y1, other[3]) - max(box.y0, other[1]) > 0


def _frame_contact(box, frame, half_width: float) -> str:
    """Name of the drawn frame line (a band ``half_width`` either side of the frame edge) that ``box`` touches."""
    x0, y0, x1, y1, w = frame.x0, frame.y0, frame.x1, frame.y1, half_width
    for side, band in (("left", (x0 - w, y0 - w, x0 + w, y1 + w)), ("right", (x1 - w, y0 - w, x1 + w, y1 + w)),
                       ("bottom", (x0 - w, y0 - w, x1 + w, y0 + w)), ("top", (x0 - w, y1 - w, x1 + w, y1 + w))):
        if _box_meets(box, band):
            return side
    return ""


def _clip_segment(p0, p1, box: tuple[float, float, float, float]):
    """The part of segment ``p0``-``p1`` strictly inside ``box`` (Liang-Barsky), or None."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - box[0]), (dx, box[2] - p0[0]), (-dy, p0[1] - box[1]), (dy, box[3] - p0[1])):
        if p == 0:
            if q <= 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return None
    return (p0[0] + t0 * dx, p0[1] + t0 * dy), (p0[0] + t1 * dx, p0[1] + t1 * dy)


def _reference_segments(ax, d: Drawn) -> list[tuple[object, list, float]]:
    """(line, drawn segments, half width) of every visible reference line of ``ax``, clipped to the frame."""
    out = []
    frame = (ax.bbox.x0, ax.bbox.y0, ax.bbox.x1, ax.bbox.y1)
    for line in ax.lines:
        if not line.get_visible() or is_exempt(line) or _no_line(line) or not _is_reference_line(line, ax):
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float).reshape(-1, 2))
        segments = []
        for p0, p1 in zip(xy[:-1], xy[1:]):
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            part = _clip_segment(p0, p1, frame) if line.get_clip_on() else (tuple(p0), tuple(p1))
            if part is not None:
                segments.append(part)
        if segments:
            out.append((line, segments, line.get_linewidth() / 2 * d.dpi / 72.0))
    return out


class _Marks(NamedTuple):
    """The drawn markers of one artist as discs: dashes ("_", "|") are sampled along their length."""

    artist: object
    centres: np.ndarray  # (m, 2) marker centres, display units
    xy: np.ndarray  # (n, 2) disc centres
    radius: np.ndarray  # (n,)
    index: np.ndarray  # (n,) marker each disc belongs to
    colors: np.ndarray  # (k, 3) RGB of the marker edges and faces
    name: str


def _rgb(colors) -> np.ndarray:
    """Opaque RGB rows of a colour or colour array (fully transparent entries dropped)."""
    try:
        rgba = to_rgba_array(colors)
    except (TypeError, ValueError):
        return np.empty((0, 3))
    return rgba[rgba[:, 3] > 0, :3]


def _same_colour(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(len(a) and len(b) and (np.abs(a[:, None, :] - b[None, :, :]).max(axis=2) < 1e-3).any())


def _discs(centres: np.ndarray, marker, size_px: np.ndarray, edge_px: np.ndarray, step: float):
    """(disc centres, radii, marker index) of markers: a disc of radius size/2 + edge/2, a dash as a sampled
    segment of half-width edge/2."""
    if marker in ("_", "|", mlines.TICKLEFT, mlines.TICKRIGHT, mlines.TICKUP, mlines.TICKDOWN):
        horizontal = marker in ("_", mlines.TICKLEFT, mlines.TICKRIGHT)
        xy, radius, index = [], [], []
        for i, (centre, size, edge) in enumerate(zip(centres, size_px, edge_px)):
            offsets = np.linspace(-size / 2, size / 2, max(2, int(np.ceil(size / step)) + 1))
            if marker in (mlines.TICKLEFT, mlines.TICKDOWN):
                offsets = offsets - size / 2
            elif marker in (mlines.TICKRIGHT, mlines.TICKUP):
                offsets = offsets + size / 2
            points = np.repeat(centre[None, :], len(offsets), axis=0)
            points[:, 0 if horizontal else 1] += offsets
            xy.append(points)
            radius.append(np.full(len(offsets), edge / 2))
            index.append(np.full(len(offsets), i))
        if not xy:
            return np.empty((0, 2)), np.empty(0), np.empty(0, int)
        return np.vstack(xy), np.concatenate(radius), np.concatenate(index)
    return centres, size_px / 2 + edge_px / 2, np.arange(len(centres))


def _errorbar_artists(ax) -> tuple[set[int], dict[int, int]]:
    """ids of the caps and bar collections of every error bar, and data line id -> container id."""
    parts, data_lines = set(), {}
    for container in ax.containers:
        if isinstance(container, ErrorbarContainer):
            data_line, caplines, barlinecols = container.lines
            parts |= {id(artist) for artist in (*caplines, *barlinecols)}
            if data_line is not None:
                data_lines[id(data_line)] = id(container)
    return parts, data_lines


def marker_sets(ax, d: Drawn) -> list[_Marks]:
    """The drawn markers of every Line2D and scatter collection of ``ax`` (error-bar caps excluded)."""
    parts, _ = _errorbar_artists(ax)
    step, px = d.px(SHAPE_STEP_MM), d.dpi / 72.0
    out = []
    for line in ax.lines:
        if not line.get_visible() or is_exempt(line) or id(line) in parts or _no_marker(line.get_marker()) \
                or line.get_markersize() <= 0:
            continue
        centres = line.get_transform().transform(np.asarray(line.get_xydata(), float).reshape(-1, 2))
        centres = centres[np.isfinite(centres).all(axis=1)]
        n = len(centres)
        xy, radius, index = _discs(centres, line.get_marker(), np.full(n, line.get_markersize() * px),
                                   np.full(n, line.get_markeredgewidth() * px), step)
        colors = np.vstack([_rgb(line.get_markeredgecolor()), _rgb(line.get_markerfacecolor())])
        out.append(_Marks(line, centres, xy, radius, index, colors, _artist_name(line)))
    for collection in ax.collections:
        if not isinstance(collection, PathCollection) or not collection.get_visible() or is_exempt(collection) \
                or id(collection) in parts:
            continue
        offsets = np.ma.filled(np.ma.asarray(collection.get_offsets(), float), np.nan).reshape(-1, 2)
        sizes = np.asarray(collection.get_sizes(), float)
        if not len(offsets) or not len(sizes):
            continue
        centres = collection.get_offset_transform().transform(offsets)
        widths = np.asarray(collection.get_linewidths(), float)
        sizes = np.resize(np.sqrt(sizes), len(centres)) * px
        widths = np.resize(widths if len(widths) else np.zeros(1), len(centres)) * px
        keep = np.isfinite(centres).all(axis=1)
        centres = centres[keep]
        colors = np.vstack([_rgb(collection.get_edgecolors()), _rgb(collection.get_facecolors())])
        out.append(_Marks(collection, centres, centres, sizes[keep] / 2 + widths[keep] / 2,
                          np.arange(len(centres)), colors, _artist_name(collection)))
    frame = ax.bbox
    shown = []
    for marks in out:  # only what is drawn: clipped artists lose the markers outside the frame
        if marks.artist.get_clip_on() and len(marks.xy):
            x, y, r = marks.xy[:, 0], marks.xy[:, 1], marks.radius
            keep = (x > frame.x0 - r) & (x < frame.x1 + r) & (y > frame.y0 - r) & (y < frame.y1 + r)
            marks = marks._replace(xy=marks.xy[keep], radius=marks.radius[keep], index=marks.index[keep])
        if len(marks.xy):
            shown.append(marks)
    return shown


class _Bar(NamedTuple):
    """One error bar of one data point, with its caps, as discs."""

    container: int
    segment: tuple | None  # (p0, p1) of the bar, display units, for the ownership test
    xy: np.ndarray
    radius: np.ndarray
    colors: np.ndarray
    name: str


def errorbar_bars(ax, d: Drawn) -> list[_Bar]:
    """Every drawn error bar of ``ax`` with the caps at its ends."""
    step, px, join = d.px(SHAPE_STEP_MM), d.dpi / 72.0, d.px(0.05)
    out = []
    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        _, caplines, barlinecols = container.lines
        bars = []  # [p0, p1, xy list, radius list, colors]
        for collection in barlinecols:
            if not collection.get_visible() or is_exempt(collection):
                continue
            widths = np.asarray(collection.get_linewidths(), float)
            half = (float(widths.max()) if len(widths) else 0.0) / 2 * px
            colors = _rgb(collection.get_colors())
            transform = collection.get_transform()
            for segment in collection.get_segments():
                points = transform.transform(np.asarray(segment, float).reshape(-1, 2))
                if len(points) < 2 or not np.isfinite(points).all():
                    continue
                sampled = _sample(points, step)
                bars.append([points[0], points[-1], [sampled], [np.full(len(sampled), half)], colors])
        loose = []
        for capline in caplines:
            if not capline.get_visible() or is_exempt(capline) or capline.get_markersize() <= 0:
                continue
            centres = capline.get_transform().transform(np.asarray(capline.get_xydata(), float).reshape(-1, 2))
            centres = centres[np.isfinite(centres).all(axis=1)]
            n = len(centres)
            xy, radius, index = _discs(centres, capline.get_marker(), np.full(n, capline.get_markersize() * px),
                                       np.full(n, capline.get_markeredgewidth() * px), step)
            for i, centre in enumerate(centres):
                ends = [min(np.hypot(*(centre - bar[0])), np.hypot(*(centre - bar[1]))) for bar in bars]
                target = bars[int(np.argmin(ends))] if ends and min(ends) <= join else None
                part = (xy[index == i], radius[index == i])
                if target is None:
                    loose.append(part + (_rgb(capline.get_markeredgecolor()),))
                else:
                    target[2].append(part[0])
                    target[3].append(part[1])
        label = container.get_label()
        for p0, p1, xy, radius, colors in bars:
            name = f"error bar {_hex(colors)}" + (f" {label!r}" if label and not label.startswith("_") else "")
            out.append(_Bar(id(container), (p0, p1), np.vstack(xy), np.concatenate(radius), colors, name))
        for xy, radius, colors in loose:
            out.append(_Bar(id(container), None, xy, radius, colors, f"error-bar cap {_hex(colors)}"))
    return out


def _hex(colors: np.ndarray) -> str:
    return to_hex(colors[0]) if len(colors) else "(no colour)"


def _on_segment(centres: np.ndarray, segment, tolerance: float) -> np.ndarray:
    """Mask of the centres that lie on the segment (within ``tolerance``)."""
    p0, p1 = np.asarray(segment[0], float), np.asarray(segment[1], float)
    direction = p1 - p0
    length2 = float(direction @ direction)
    if length2 == 0:
        return np.hypot(*(centres - p0).T) <= tolerance
    t = ((centres - p0) @ direction) / length2
    nearest = p0 + np.clip(t, 0.0, 1.0)[:, None] * direction
    return np.hypot(*(centres - nearest).T) <= tolerance


def _touching(a_xy: np.ndarray, a_r: np.ndarray, b_xy: np.ndarray, b_r: np.ndarray, gap: float,
              a_centre: np.ndarray | None = None, b_centre: np.ndarray | None = None, concentric: float = 0.0):
    """(mask over the a discs that come closer than ``gap`` to a b disc, first such (a, b) disc pair).

    With ``a_centre``/``b_centre`` (the marker centre of every disc), pairs of markers whose centres lie within
    ``concentric`` of each other are one composite marker (a halo or outline drawn around a marker) and pass.
    """
    hit = np.zeros(len(a_xy), bool)
    if not len(a_xy) or not len(b_xy):
        return hit, None
    reach = float(a_r.max() + b_r.max() + gap)
    low, high = a_xy.min(axis=0) - reach, a_xy.max(axis=0) + reach
    keep = ((b_xy >= low) & (b_xy <= high)).all(axis=1)
    b_xy, b_r = b_xy[keep], b_r[keep]
    if b_centre is not None:
        b_centre = b_centre[keep]
    if not len(b_xy):
        return hit, None
    first = None
    chunk = max(1, 2_000_000 // len(b_xy))
    for start in range(0, len(a_xy), chunk):
        pa, ra = a_xy[start:start + chunk], a_r[start:start + chunk]
        distance = np.hypot(pa[:, None, 0] - b_xy[None, :, 0], pa[:, None, 1] - b_xy[None, :, 1])
        close = distance < ra[:, None] + b_r[None, :] + gap
        if a_centre is not None and b_centre is not None:
            ca = a_centre[start:start + chunk]
            close &= np.hypot(ca[:, None, 0] - b_centre[None, :, 0], ca[:, None, 1] - b_centre[None, :, 1]) \
                >= concentric
        rows = close.any(axis=1)
        hit[start:start + chunk] = rows
        if first is None and rows.any():
            i = int(np.argmax(rows))
            first = (pa[i], b_xy[int(np.argmax(close[i]))])
    return hit, first


def tick_marks(ax, d: Drawn) -> np.ndarray:
    """(n, 4) display boxes (x0, y0, x1, y1) of the drawn tick marks of ``ax``."""
    boxes = []
    if not ax.axison:
        return np.empty((0, 4))
    px = d.dpi / 72.0
    for axis in (ax.xaxis, ax.yaxis):
        if not axis.get_visible():
            continue
        for tick in _drawn_ticks(axis):
            for line in (tick.tick1line, tick.tick2line):
                if not line.get_visible() or line.get_markersize() <= 0:
                    continue
                xy = line.get_transform().transform(np.asarray(line.get_xydata(), float).reshape(-1, 2))
                length, half = line.get_markersize() * px, line.get_markeredgewidth() / 2 * px
                marker = line.get_marker()
                for x, y in xy[np.isfinite(xy).all(axis=1)]:
                    if marker == mlines.TICKDOWN:
                        boxes.append((x - half, y - length, x + half, y))
                    elif marker == mlines.TICKUP:
                        boxes.append((x - half, y, x + half, y + length))
                    elif marker == mlines.TICKLEFT:
                        boxes.append((x - length, y - half, x, y + half))
                    elif marker == mlines.TICKRIGHT:
                        boxes.append((x, y - half, x + length, y + half))
                    elif marker == "|":
                        boxes.append((x - half, y - length / 2, x + half, y + length / 2))
                    else:
                        boxes.append((x - length / 2, y - half, x + length / 2, y + half))
    return np.array(boxes, float).reshape(-1, 4)


# -------------------------------------------------------------------------------------- checks


def alignment(layout, tol_mm: float = ALIGN_TOL_MM, name: str | None = None) -> list[Finding]:
    """Same ``row`` tag: equal drawn top and bottom frame edges; same ``col`` tag: equal left edges.

    Every framed axes of the figure must carry a row tag (``missing-row``).
    """
    fig, layout = _split(layout)
    name = _name(layout or fig, name)
    findings = []
    with drawn(fig) as d:
        rows, cols = defaultdict(list), defaultdict(list)
        for ax in fig.axes:
            if not ax.get_visible() or is_frameless(ax):
                continue
            record = _record(ax, layout)
            box = d.mm(ax.bbox)
            if record is None or record.row is None:
                findings.append(Finding(name, "alignment", "missing-row", describe(ax, layout), box,
                                        "framed axes without a row tag (Layout.axes(..., row=...))"))
            else:
                rows[record.row].append((ax, box))
            if record is not None and record.col is not None:
                cols[record.col].append((ax, box))
        for tag, members in rows.items():
            for kind, index in (("row-top", 1), ("row-bottom", 3)):
                values = [box[index] for _, box in members]
                if max(values) - min(values) > tol_mm:
                    findings.append(Finding(name, "alignment", kind, f"row {tag!r}", _union(b for _, b in members),
                                            "; ".join(f"{describe(ax, layout)}: {box[index]:.3f} mm"
                                                      for ax, box in members)))
        for tag, members in cols.items():
            values = [box[0] for _, box in members]
            if max(values) - min(values) > tol_mm:
                findings.append(Finding(name, "alignment", "col-left", f"col {tag!r}", _union(b for _, b in members),
                                        "; ".join(f"{describe(ax, layout)}: {box[0]:.3f} mm" for ax, box in members)))
    return findings


def _union(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def alignment_table(layout) -> list[dict]:
    """Drawn frame edges (mm from the top-left) of every axes of a layout, with its tags: the numeric alignment record."""
    fig, layout = _split(layout)
    rows = []
    with drawn(fig) as d:
        for index, ax in enumerate(fig.axes):
            record = _record(ax, layout)
            left, top, right, bottom = d.mm(ax.bbox)
            rows.append({"axes": index, "label": ax.get_label(), "row": getattr(record, "row", None),
                         "col": getattr(record, "col", None),
                         "frame": "none" if is_frameless(ax) else "closed",
                         "left_mm": round(left, 3), "top_mm": round(top, 3), "right_mm": round(right, 3),
                         "bottom_mm": round(bottom, 3)})
    return rows


def frames(fig, name: str | None = None) -> list[Finding]:
    """Closed frame on every axes that is not frameless; no top or right ticks or tick labels; no secondary axes."""
    fig, layout = _split(fig)
    name = _name(fig, name)
    findings = []
    with drawn(fig) as d:
        for ax, _ in _tree(fig):
            box = d.mm(ax.bbox)
            if isinstance(ax, SecondaryAxis):
                findings.append(Finding(name, "frames", "secondary-axis", describe(ax, layout), box,
                                        "secondary axes are not allowed; label the data inside the frame"))
                continue
            if is_frameless(ax):
                continue
            if _is_colorbar(ax):
                outline = ax.spines.get("outline")
                if outline is None or not outline.get_visible():
                    findings.append(Finding(name, "frames", "open-frame", describe(ax, layout), box,
                                            "colorbar without outline"))
                continue
            for side in ("left", "bottom", "right", "top"):
                spine = ax.spines.get(side)
                if spine is None or not spine.get_visible() or spine.get_linewidth() <= 0 \
                        or spine.get_edgecolor()[3] == 0:
                    findings.append(Finding(name, "frames", "open-frame", describe(ax, layout), box,
                                            f"{side} spine hidden"))
                elif spine.get_position() not in (("outward", 0.0), ("outward", 0)):
                    findings.append(Finding(name, "frames", "offset-spine", describe(ax, layout), box,
                                            f"{side} spine at {spine.get_position()}"))
            if not ax.axison:
                continue
            for axis, side in ((ax.xaxis, "top"), (ax.yaxis, "right")):
                ticks = _drawn_ticks(axis) if axis.get_visible() else []
                if any(t.tick2line.get_visible() and t.tick2line.get_markersize() > 0 for t in ticks):
                    findings.append(Finding(name, "frames", f"{side}-ticks", describe(ax, layout), box,
                                            f"tick marks on the {side}"))
                if any(t.label2.get_visible() and t.label2.get_text().strip() for t in ticks):
                    findings.append(Finding(name, "frames", f"{side}-labels", describe(ax, layout), box,
                                            f"tick labels on the {side}"))
    return findings


def overlaps(fig, tol_mm: float = OVERLAP_TOL_MM, name: str | None = None) -> list[Finding]:
    """Text against text, text against frames, axes against axes, text against data in the same axes."""
    fig, layout = _split(fig)
    name = _name(fig, name)
    findings = []
    with drawn(fig) as d:
        tol = d.px(tol_mm)
        tree = _tree(fig)
        parents = {id(ax): parent for ax, parent in tree}
        items = [item for item in text_items(fig, d) if not is_exempt(item.artist)]

        # text against text
        boxes = _box_array(item.box for item in items)
        if len(items) > 1:
            dx = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
            dy = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
            hit = np.triu((dx > tol) & (dy > tol), k=1)
            for i, j in zip(*np.nonzero(hit)):
                a, b = items[i], items[j]
                box = (max(a.box.x0, b.box.x0), max(a.box.y0, b.box.y0), min(a.box.x1, b.box.x1),
                       min(a.box.y1, b.box.y1))
                kind = "legend-text" if "legend" in (a.role, b.role) else "text-text"
                findings.append(Finding(name, "overlaps", kind, f"{a.name!r} / {b.name!r}",
                                        _mm_tuple(d, box), f"{a.role} / {b.role}"))

        # text against frames: any contact with a drawn frame line, or any overlap with another axes' frame
        framed = [ax for ax, _ in tree if not is_frameless(ax) and not isinstance(ax, SecondaryAxis)]
        half_widths = {id(ax): _spine_half_width(ax, d) for ax in framed}
        for item in items:
            owner_line = _ancestors(item.owner, parents) | {id(item.owner)} if item.owner is not None else set()
            for ax in framed:
                frame, half = ax.bbox, half_widths[id(ax)]
                if item.owner is ax:
                    side = _frame_contact(item.box, frame, half)
                    if side:
                        findings.append(Finding(name, "overlaps", "text-frame", repr(item.name), d.mm(item.box),
                                                f"{item.role} touches the {side} frame line of {describe(ax, layout)}"))
                    continue
                if id(ax) in owner_line:
                    continue
                if _box_meets(item.box, _grown(frame, half)):
                    findings.append(Finding(name, "overlaps", "text-axes", repr(item.name), d.mm(item.box),
                                            f"{item.role} touches or enters the frame of {describe(ax, layout)}"))

        # tick marks against the frame of other axes
        for a in framed:
            ticks = tick_marks(a, d)
            if not len(ticks):
                continue
            for b in framed:
                if b is a or id(a) in _ancestors(b, parents) or id(b) in _ancestors(a, parents):
                    continue
                band = _grown(b.bbox, half_widths[id(b)])
                gx = np.maximum.reduce([band[0] - ticks[:, 2], ticks[:, 0] - band[2], np.zeros(len(ticks))])
                gy = np.maximum.reduce([band[1] - ticks[:, 3], ticks[:, 1] - band[3], np.zeros(len(ticks))])
                near = np.hypot(gx, gy) < d.px(TICK_GAP_MM)
                if near.any():
                    first = ticks[int(np.argmax(near))]
                    findings.append(Finding(name, "overlaps", "tick-frame", describe(a, layout),
                                            _mm_tuple(d, tuple(first)),
                                            f"{int(near.sum())} tick marks within {TICK_GAP_MM:g} mm of the frame of "
                                            f"{describe(b, layout)}"))

        # framed axes against each other
        for i, a in enumerate(framed):
            for b in framed[i + 1:]:
                if id(a) in _ancestors(b, parents) or id(b) in _ancestors(a, parents):
                    continue
                w, h = _intersection(a.bbox, b.bbox)
                if w > tol and h > tol:
                    findings.append(Finding(name, "overlaps", "axes-axes",
                                            f"{describe(a, layout)} / {describe(b, layout)}",
                                            _mm_tuple(d, (max(a.bbox.x0, b.bbox.x0), max(a.bbox.y0, b.bbox.y0),
                                                          min(a.bbox.x1, b.bbox.x1), min(a.bbox.y1, b.bbox.y1)))))

        # inside each axes (not in frameless schematics)
        for ax, _ in tree:
            if is_frameless(ax) or isinstance(ax, SecondaryAxis) or is_exempt(ax):
                continue
            findings += _marker_spacing(ax, d, name, layout)
            own = [item for item in items if item.owner is ax and item.role in ("text", "legend")]
            if not own:
                continue
            # text and legends against reference lines: no contact at all
            for line, segments, half in _reference_segments(ax, d):
                for item in own:
                    box = _grown(item.box, half)
                    if any(_clip_segment(p0, p1, box) is not None for p0, p1 in segments):
                        findings.append(Finding(name, "overlaps", "text-reference", repr(item.name), d.mm(item.box),
                                                f"{item.role} crossed by the reference line {_artist_name(line)} in "
                                                f"{describe(ax, layout)}"))
            # text and legends against the data
            series = data_points(ax, d)
            bars = _bars(ax, d)
            for item in own:
                b = item.box
                for data in series:
                    if not len(data.xy):
                        continue
                    shown = _clip_visible(data, ax.bbox)
                    x, y, r = data.xy[:, 0], data.xy[:, 1], data.radius
                    inside = shown & (x > b.x0 - r + tol) & (x < b.x1 + r - tol) & (y > b.y0 - r + tol) \
                        & (y < b.y1 + r - tol)
                    if inside.any():
                        first = int(np.argmax(inside))
                        px, py = d.point_mm(x[first], y[first])
                        what = "markers" if data.marker else "line"
                        findings.append(Finding(name, "overlaps", f"{item.role}-data", repr(item.name), d.mm(b),
                                                f"over {int(inside.sum())} {what} points of {data.name} in "
                                                f"{describe(ax, layout)}, first at ({px:.2f}, {py:.2f}) mm"))
                for _patch, box in bars:
                    w, h = _intersection(b, box)
                    if w > tol and h > tol:
                        findings.append(Finding(name, "overlaps", f"{item.role}-data", repr(item.name), d.mm(b),
                                                f"over a bar of {describe(ax, layout)}"))
                        break
    return findings


def _mm_tuple(d: Drawn, box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (d.to_mm(x0), d.to_mm(d.height - y1), d.to_mm(x1), d.to_mm(d.height - y0))


def _marker_spacing(ax, d: Drawn, name: str, layout) -> list[Finding]:
    """Markers of different artists, and error bars against foreign markers, closer than MARKER_GAP_MM."""
    findings = []
    gap, on_bar = d.px(MARKER_GAP_MM), d.px(ON_BAR_MM)
    marks = marker_sets(ax, d)
    _, data_lines = _errorbar_artists(ax)
    for i, a in enumerate(marks):
        for b in marks[i + 1:]:
            hit, first = _touching(a.xy, a.radius, b.xy, b.radius, gap, a.centres[a.index], b.centres[b.index],
                                   on_bar)
            if first is None:
                continue
            (ax_, ay_), (bx_, by_) = first
            box = _mm_tuple(d, (min(ax_, bx_), min(ay_, by_), max(ax_, bx_), max(ay_, by_)))
            findings.append(Finding(name, "overlaps", "marker-marker", f"{a.name} / {b.name}", box,
                                    f"{len(np.unique(a.index[hit]))} markers closer than {MARKER_GAP_MM:g} mm to "
                                    f"markers of another artist in {describe(ax, layout)}"))
    for bar in errorbar_bars(ax, d):
        # the bar's own data point: a marker of the bar's colour centred on the bar, and markers concentric with it
        owners = np.empty((0, 2))
        if bar.segment is not None:
            owners = np.vstack([np.empty((0, 2))] + [m.centres[_on_segment(m.centres, bar.segment, on_bar)]
                                                     for m in marks if _same_colour(m.colors, bar.colors)])
        for m in marks:
            if data_lines.get(id(m.artist)) == bar.container:
                continue  # the error bar's own data line
            owned = np.zeros(len(m.centres), bool)
            if len(owners) and len(m.centres):
                owned = np.hypot(m.centres[:, None, 0] - owners[None, :, 0],
                                 m.centres[:, None, 1] - owners[None, :, 1]).min(axis=1) < on_bar
            keep = ~owned[m.index]
            hit, first = _touching(m.xy[keep], m.radius[keep], bar.xy, bar.radius, gap)
            if first is None:
                continue
            (mx, my), (bx, by) = first
            box = _mm_tuple(d, (min(mx, bx), min(my, by), max(mx, bx), max(my, by)))
            findings.append(Finding(name, "overlaps", "errorbar-marker", f"{bar.name} / {m.name}", box,
                                    f"{len(np.unique(m.index[keep][hit]))} markers closer than {MARKER_GAP_MM:g} mm "
                                    f"to an error bar or cap of another series in {describe(ax, layout)}"))
    return findings


def clipped(fig, name: str | None = None) -> list[Finding]:
    """Markers that cross the frame of framed axes or come within 0.1 mm of a spine, and clipped data points
    outside the axis limits."""
    fig, layout = _split(fig)
    name = _name(fig, name)
    findings = []
    with drawn(fig) as d:
        tol, spine_gap = d.px(FRAME_TOL_MM), d.px(SPINE_GAP_MM)
        for ax, _ in _tree(fig):
            if is_frameless(ax) or isinstance(ax, SecondaryAxis) or is_exempt(ax):
                continue
            f, half = ax.bbox, _spine_half_width(ax, d)
            for data in data_points(ax, d, markers_only=True):
                if not len(data.xy):
                    continue
                x, y, r = data.xy[:, 0], data.xy[:, 1], data.radius
                outside = (x < f.x0 - tol) | (x > f.x1 + tol) | (y < f.y0 - tol) | (y > f.y1 + tol)
                inner = np.minimum.reduce([x - f.x0, f.x1 - x, y - f.y0, f.y1 - y])
                # distance from an outside centre to the frame rectangle
                ox = np.maximum.reduce([f.x0 - x, np.zeros_like(x), x - f.x1])
                oy = np.maximum.reduce([f.y0 - y, np.zeros_like(y), y - f.y1])
                outer = np.hypot(ox, oy)
                crossing = np.where(outside, outer < r - tol, inner < r - tol)
                if data.clip_on:
                    crossing &= ~outside
                    if outside.any():
                        first = int(np.argmax(outside))
                        px, py = d.point_mm(x[first], y[first])
                        findings.append(Finding(name, "clipped", "outside-limits", data.name, d.mm(f),
                                                f"{int(outside.sum())} points outside the limits of "
                                                f"{describe(ax, layout)}, first at ({px:.2f}, {py:.2f}) mm"))
                if crossing.any():
                    first = int(np.argmax(crossing))
                    px, py = d.point_mm(x[first], y[first])
                    findings.append(Finding(name, "clipped", "marker-on-frame", data.name, d.mm(f),
                                            f"{int(crossing.sum())} markers cross the frame of "
                                            f"{describe(ax, layout)}, first at ({px:.2f}, {py:.2f}) mm"))
                # gap between the marker edge and the drawn spine
                near = ~crossing & (np.where(outside, outer, inner) - r - half < spine_gap)
                if data.clip_on:
                    near &= ~outside
                if near.any():
                    first = int(np.argmax(near))
                    px, py = d.point_mm(x[first], y[first])
                    findings.append(Finding(name, "clipped", "marker-near-frame", data.name, d.mm(f),
                                            f"{int(near.sum())} markers within {SPINE_GAP_MM:g} mm of a spine of "
                                            f"{describe(ax, layout)}, first at ({px:.2f}, {py:.2f}) mm"))
    return findings


def fonts(fig, name: str | None = None) -> list[Finding]:
    """Text 5-7 pt, panel letters 8 pt bold; a warning when a font resolves to DejaVu."""
    fig, _ = _split(fig)
    name = _name(fig, name)
    findings = []
    with drawn(fig) as d:
        texts = [(item.artist, item.role) for item in text_items(fig, d) if isinstance(item.artist, Text)]
        texts += _all_texts(fig)
        fallback = set()
        for artist, role in texts:
            size, box = float(artist.get_fontsize()), d.mm(_text_box(artist, d.renderer))
            subject = repr(artist.get_text().replace("\n", " "))
            if role == "letter":
                if abs(size - style.PANEL) > 0.01 or artist.get_fontweight() not in ("bold", 700):
                    findings.append(Finding(name, "fonts", "letter", subject, box,
                                            f"{size:g} pt, weight {artist.get_fontweight()}; panel letters are "
                                            f"{style.PANEL:g} pt bold"))
            elif not FONT_RANGE[0] - 0.01 <= size <= FONT_RANGE[1] + 0.01:
                findings.append(Finding(name, "fonts", "size", subject, box,
                                        f"{size:g} pt; text is {FONT_RANGE[0]:g}-{FONT_RANGE[1]:g} pt"))
            try:
                path = font_manager.findfont(artist.get_fontproperties(), fallback_to_default=True)
            except Exception:  # pragma: no cover - font lookup failure
                continue
            if "dejavu" in Path(path).name.lower():
                fallback.add(Path(path).name)
        for font in sorted(fallback):
            findings.append(Finding(name, "fonts", "fallback-font", font, None,
                                    "text resolves to the DejaVu fallback (Arial, Helvetica or Liberation Sans "
                                    "is not installed)", "warning"))
    return findings


def size(fig, name: str | None = None) -> list[Finding]:
    """Width 89, 120 or 183 mm; height at most 247 mm."""
    fig, _ = _split(fig)
    name = _name(fig, name)
    width, height = fig.get_figwidth() * MM_PER_INCH, fig.get_figheight() * MM_PER_INCH
    findings = []
    if not any(abs(width - target) <= 0.05 for target in style.WIDTHS_MM):
        findings.append(Finding(name, "size", "width", f"{width:.2f} mm", None, "allowed: 89, 120 or 183 mm"))
    if height > style.MAX_HEIGHT_MM + 0.05:
        findings.append(Finding(name, "size", "height", f"{height:.2f} mm", None, "at most 247 mm"))
    return findings


def check(target, name: str | None = None) -> list[Finding]:
    """Every check on a Layout (alignment included) or a Figure, from one drawing at 600 dpi."""
    fig, layout = _split(target)
    name = _name(target, name)
    with drawn(fig):
        findings = alignment(layout, name=name) if layout is not None else []
        findings += frames(fig, name) + overlaps(fig, name=name) + clipped(fig, name) + fonts(fig, name)
        findings += size(fig, name)
    return findings


# --------------------------------------------------------------------------- mode, gate, session


def mode(strict: bool | None = None) -> str:
    """'strict', 'warn' or 'off': the argument, else the innermost session, else MLIPFT_QA, else 'strict'.

    CI sets MLIPFT_QA=warn: text metrics differ between the Linux fonts there and the fonts the figures are
    checked with, so the gate is enforced where the figures are made.
    """
    if strict is not None:
        return "strict" if strict else "warn"
    for state in reversed(_SESSIONS):
        if state["strict"] is not None:
            return "strict" if state["strict"] else "warn"
    value = os.environ.get("MLIPFT_QA", "strict").strip().lower()
    return value if value in MODES else "strict"


@contextmanager
def session(strict: bool | None = None, audit: Path | str | None = None) -> Iterator[dict]:
    """Build-wide QA settings (used by ``registry.build``): mode, crop-audit directory, layouts created meanwhile."""
    state = {"strict": strict, "audit": outside_repository(audit) if audit else None, "layouts": []}
    _SESSIONS.append(state)
    try:
        yield state
    finally:
        _SESSIONS.remove(state)


def track(layout) -> None:
    """Called by ``common.Layout``: remember the layout in the active session (if any)."""
    if _SESSIONS:
        _SESSIONS[-1]["layouts"].append(layout)


def summary(findings: list[Finding], name: str, current: str) -> str:
    if not findings:
        return f"QA {name}: pass"
    counts = defaultdict(int)
    for finding in findings:
        counts[finding.check] += 1
    head = (f"QA {name}: {len(findings)} findings ("
            + ", ".join(f"{check} {count}" for check, count in sorted(counts.items())) + ")")
    if current != "strict":
        head += " [warn; --strict or MLIPFT_QA=strict to fail]"
    lines = [head] + [f"  {finding}" for finding in findings[:SUMMARY_LINES]]
    if len(findings) > SUMMARY_LINES:
        lines.append(f"  ... and {len(findings) - SUMMARY_LINES} more")
    return "\n".join(lines)


def gate(layout, strict: bool | None = None, *, name: str | None = None, png: Path | str | None = None,
         findings: list[Finding] | None = None) -> list[Finding]:
    """Check a figure, write its crop audit when a session asks for one, then report.

    ``findings`` (from ``check``) skips the checks; ``png`` is the exported PNG for the crop audit.
    Strict mode raises ``FigureQAError`` when any finding is an error; otherwise a summary goes to
    stderr. The layout's ``name`` and ``findings`` attributes are set, which marks it as checked.
    """
    current = mode(strict)
    name = _name(layout, name)
    if current == "off":
        findings = []
    elif findings is None:
        findings = check(layout, name)
    if hasattr(layout, "records"):
        layout.name, layout.findings = name, findings
    if current == "off":
        return findings
    state = _SESSIONS[-1] if _SESSIONS else None
    if state is not None and state["audit"] is not None and png is not None:
        crop_audit(layout, png, state["audit"], findings=findings)
    errors = [f for f in findings if f.severity == "error"]
    if current == "strict" and errors:
        raise FigureQAError(findings)
    print(summary(findings, name or "<figure>", current), file=sys.stderr)
    return findings


# ---------------------------------------------------------------------------------- crop audit


def outside_repository(path: Path | str) -> Path:
    """``path`` resolved; refuses any location inside the repository (crops are never committed)."""
    path = Path(path).expanduser().resolve()
    repo = paths.REPO.resolve()
    if path == repo or repo in path.parents:
        raise ValueError(f"the audit directory must be outside the repository, got {paths.relpath(path)}")
    return path


def _near_line(box, vertical: bool, at: float, low: float, high: float, near: float) -> bool:
    """Whether ``box`` comes within ``near`` of the segment at x (vertical) or y = ``at`` from ``low`` to ``high``."""
    if vertical:
        gap = 0.0 if box.x0 <= at <= box.x1 else min(abs(box.x0 - at), abs(box.x1 - at))
        return gap <= near and box.y1 >= low - near and box.y0 <= high + near
    gap = 0.0 if box.y0 <= at <= box.y1 else min(abs(box.y0 - at), abs(box.y1 - at))
    return gap <= near and box.x1 >= low - near and box.x0 <= high + near


def crop_windows(target) -> list[dict]:
    """Where to look, as display boxes at 600 dpi: 10-mm windows at the corners and edge midpoints of every
    framed axes ("frame-*") and of every frameless axes ("panel-*"), every legend (+1.5 mm), every text
    within 3 mm of a right or top frame line (the text and that line, +1.5 mm) and every marker within
    1 mm of the axis limits (6-mm windows, one per 5-mm cell)."""
    fig, layout = _split(target)
    windows: list[dict] = []
    with drawn(fig) as d:
        half, margin, near, close = d.px(5.0), d.px(1.5), d.px(3.0), d.px(1.0)
        framed = _framed(fig)
        for ax, _ in _tree(fig):
            if isinstance(ax, SecondaryAxis):
                continue
            b = ax.bbox
            prefix = "panel" if is_frameless(ax) else "frame"
            label = describe(ax, layout)
            mid_x, mid_y = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            for where, (x, y) in (("top-left", (b.x0, b.y1)), ("top", (mid_x, b.y1)), ("top-right", (b.x1, b.y1)),
                                  ("right", (b.x1, mid_y)), ("bottom-right", (b.x1, b.y0)),
                                  ("bottom", (mid_x, b.y0)), ("bottom-left", (b.x0, b.y0)),
                                  ("left", (b.x0, mid_y))):
                windows.append({"kind": f"{prefix}-{where}", "subject": label,
                                "box": (x - half, y - half, x + half, y + half)})
        items = text_items(fig, d)
        for item in items:
            b = item.box
            grown = [b.x0 - margin, b.y0 - margin, b.x1 + margin, b.y1 + margin]
            if item.role == "legend":
                windows.append({"kind": "legend", "subject": item.name, "box": tuple(grown)})
                continue
            for ax in framed:
                f = ax.bbox
                right = _near_line(b, True, f.x1, f.y0, f.y1, near)
                if not right and not _near_line(b, False, f.y1, f.x0, f.x1, near):
                    continue
                if right:  # include the stretch of the frame line next to the text
                    grown[0], grown[2] = min(grown[0], f.x1 - margin), max(grown[2], f.x1 + margin)
                else:
                    grown[1], grown[3] = min(grown[1], f.y1 - margin), max(grown[3], f.y1 + margin)
                if not any(_contains(w["box"], grown) for w in windows):
                    windows.append({"kind": "text-near-frame", "subject": item.name, "box": tuple(grown)})
                break
        cell, size = d.px(5.0), d.px(3.0)
        for ax in framed:
            f = ax.bbox
            cells: dict[tuple[int, int], list] = defaultdict(list)
            for data in data_points(ax, d, markers_only=True):
                if not len(data.xy):
                    continue
                x, y, r = data.xy[:, 0], data.xy[:, 1], data.radius
                inside = (x >= f.x0) & (x <= f.x1) & (y >= f.y0) & (y <= f.y1)
                edge = np.minimum.reduce([x - f.x0, f.x1 - x, y - f.y0, f.y1 - y])
                for px, py in data.xy[inside & (edge < close + r)]:
                    cells[(int(px // cell), int(py // cell))].append((px, py))
            for key in sorted(cells):
                points = np.array(cells[key])
                cx, cy = points.mean(axis=0)
                windows.append({"kind": "marker-near-limit", "subject": describe(ax, layout),
                                "box": (cx - size, cy - size, cx + size, cy + size)})
    return windows


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def crop_audit(target, png_path: Path | str, out_dir: Path | str, *,
               findings: list[Finding] | None = None) -> list[Path]:
    """Write zoomed 600-dpi crops of a saved figure for visual review, outside the repository.

    ``out_dir/<png stem>/`` receives one PNG per window of ``crop_windows`` (corners and edge
    midpoints of every axes, legends, text near a right or top frame, markers near the limits),
    ``contact_sheet_NN.png`` pages of labelled thumbnails, and ``index.json`` (crop boxes in
    mm, the QA findings and the drawn alignment table). Old crops in that folder are removed first.
    Returns the written paths.
    """
    from PIL import Image, ImageDraw, ImageFont

    out_dir = outside_repository(out_dir)
    fig, layout = _split(target)
    png_path = Path(png_path)
    folder = out_dir / png_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    for old in list(folder.glob("*.png")) + [folder / "index.json"]:
        old.unlink(missing_ok=True)
    image = Image.open(png_path)
    image.load()
    image = image.convert("RGB")
    with drawn(fig) as d:
        sx, sy = image.width / d.fig.bbox.width, image.height / d.fig.bbox.height
        windows = crop_windows(target)
        for window in windows:
            x0, y0, x1, y1 = window["box"]
            window["box_mm"] = [round(v, 3) for v in _mm_tuple(d, (x0, y0, x1, y1))]
            window["pixels"] = (int(math.floor(x0 * sx)), int(math.floor(image.height - y1 * sy)),
                                int(math.ceil(x1 * sx)), int(math.ceil(image.height - y0 * sy)))
        table = alignment_table(layout) if layout is not None else []
    written, entries, tiles = [], [], []
    for index, window in enumerate(windows):
        left, top, right, bottom = window["pixels"]
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right - left < 2 or bottom - top < 2:
            continue
        crop = image.crop((left, top, right, bottom))
        path = folder / f"{index:03d}_{window['kind']}.png"
        crop.save(path, dpi=(DPI, DPI))
        written.append(path)
        tiles.append((crop, f"{index:03d} {window['kind']}"))
        entries.append({"file": path.name, "kind": window["kind"], "subject": window["subject"],
                        "box_mm": window["box_mm"]})
    sheets = _contact_sheets(tiles, folder, Image, ImageDraw, ImageFont)
    written += sheets
    index_path = folder / "index.json"
    index_path.write_text(json.dumps({
        "figure": png_path.stem, "png": str(png_path), "dpi": DPI,
        "size_mm": [round(fig.get_figwidth() * MM_PER_INCH, 3), round(fig.get_figheight() * MM_PER_INCH, 3)],
        "contact_sheets": [p.name for p in sheets], "crops": entries,
        "findings": [f.as_dict() for f in (findings or [])], "alignment": table}, indent=1) + "\n",
        encoding="utf-8")
    written.append(index_path)
    return written


def _contact_sheets(tiles, folder: Path, Image, ImageDraw, ImageFont, columns: int = 6, per_page: int = 36,
                    tile: int = 300, caption: int = 22) -> list[Path]:
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow without FreeType sizes
        font = ImageFont.load_default()
    pages = []
    for page, start in enumerate(range(0, len(tiles), per_page), start=1):
        chunk = tiles[start:start + per_page]
        rows = math.ceil(len(chunk) / columns)
        sheet = Image.new("RGB", (columns * (tile + 8) + 8, rows * (tile + caption + 8) + 8), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (crop, label) in enumerate(chunk):
            thumb = crop.copy()
            thumb.thumbnail((tile, tile))
            x = 8 + (i % columns) * (tile + 8)
            y = 8 + (i // columns) * (tile + caption + 8)
            draw.rectangle((x - 1, y - 1, x + tile, y + tile), outline=(200, 200, 200))
            sheet.paste(thumb, (x + (tile - thumb.width) // 2, y + (tile - thumb.height) // 2))
            draw.text((x, y + tile + 3), label[:44], fill=(0, 0, 0), font=font)
        path = folder / f"contact_sheet_{page:02d}.png"
        sheet.save(path)
        pages.append(path)
    return pages
