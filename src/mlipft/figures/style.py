"""Figure style (Nature family), the single encoding vocabulary, and deterministic export.

Sizes follow Nature's column widths (89 mm single, 120 mm one-and-a-half, 183 mm double, at most
247 mm tall); text is 5-7 pt sans-serif, panel letters 8 pt bold lower case. Data panels have a
closed frame with ticks on the bottom and left only.

Every colour and marker used by a figure comes from this module (``tests/test_figures.py`` checks
that no figure module writes a hex colour of its own):

* Fine-tuned models are blue (NAVY for the Gamma-label model that drove MD, BLUE for the 2x2x2-label
  one, BLUE_MID and BLUE_LIGHT for the two stages that did not drive MD); universal potentials are
  red (MACE-MP-0 DEEP_RED, MatterSim LIGHT_RED); reference data are black and grey.
* Fill means training stage and nothing else: stage two (and every universal potential) filled,
  stage one open (``STAGE_FILLED``, ``MD_STAGE``).
* Marker shape means run type and nothing else: production run circle, 52-atom run triangle,
  Cl-excess x = 0 run inverted triangle (``RUN_MARKERS``), AIMD a filled black square
  (``AIMD_MARKER``), a measured value a light-grey diamond (``MEASURED_MARKER``, Extended Data only).
* Do not combine in one panel: temperature reds with model reds, element navy with fine-tune navy,
  Cl-content blues with fine-tune blues.

Every figure is saved as PDF and SVG (editable text) and as a 600 dpi PNG, without timestamps, so
a rebuild gives identical bytes. Each panel's plotted values are written to
``figures/source_data/<figure>_<panel>.csv``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..paths import FIGURES  # noqa: E402

MM = 1 / 25.4
SINGLE = 89 * MM
ONE_HALF = 120 * MM
DOUBLE = 183 * MM
MAX_HEIGHT = 247 * MM
WIDTHS_MM = (89.0, 120.0, 183.0)
MAX_HEIGHT_MM = 247.0
DPI = 600

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE = 6.0  # default text, tick labels and axis labels
SMALL = 5.0  # legends, dense annotations
LABEL = 7.0  # the largest text allowed apart from panel letters
PANEL = 8.0  # panel letters, bold
LINE = 0.6
HASHSALT = "mlip-fine-tune"

# ------------------------------------------------------------------------------------------- palette
NAVY = "#0F4D92"
BLUE = "#3775BA"
BLUE_MID = "#6E9BCF"
BLUE_LIGHT = "#A8C5E5"
DEEP_RED = "#B64342"
LIGHT_RED = "#E9A6A1"
BLACK = "#272727"
DARK_GRAY = "#4A4A4A"
GRAY = "#767676"
MID_GRAY = "#A3A3A3"
LIGHT_GRAY = "#CFCECE"
BAND = "#EDEDED"  # windows, label floors and other filled bands
SCHEMATIC_FILL = "#F0F0F0"  # box fill in schematics
WHITE = "#FFFFFF"

# ------------------------------------------------------------------------------------------ models
MODEL_COLORS = {"foundation": DEEP_RED, "mattersim": LIGHT_RED,
                "ftG": NAVY, "ftG_s2": NAVY, "ftK": BLUE, "ftK_s1": BLUE,
                "ftG_s1": BLUE_MID, "ftK_s2": BLUE_LIGHT}
MODEL_LABELS = {"foundation": "MACE-MP-0", "mattersim": "MatterSim", "ftG": "Fine-tuned, Gamma labels",
                "ftK": "Fine-tuned, 2x2x2 labels", "ftG_s1": "Gamma labels, stage one",
                "ftG_s2": "Gamma labels, stage two", "ftK_s1": "2x2x2 labels, stage one",
                "ftK_s2": "2x2x2 labels, stage two"}
UNIVERSAL_MODELS = ("foundation", "mattersim")
STAGE_FILLED = {"s1": False, "s2": True}  # stage two filled, stage one open; universal models filled
# Stage of the model that drove each fine-tuned MD run (ftG: stage two, ftK: stage one). The cross-evaluation
# table (drove_md column) is the authority; tests/test_figures.py checks that the two agree.
MD_STAGE = {"ftG": "s2", "ftK": "s1"}

# ----------------------------------------------------------------------------------------- markers
MLIP_MARKER = "o"
AIMD_MARKER = "s"  # always filled AIMD_COLOR
MEASURED_MARKER = "D"  # LIGHT_GRAY face; Extended Data only, never next to a fine-tuned model
RUN_MARKERS = {"production": "o", "cell_52": "^", "clexcess": "v"}  # 416-atom production, 52-atom, Cl-excess x = 0

# --------------------------------------------------------------------------------------- references
AIMD_COLOR = BLACK
LABEL_COLORS = {"k222": BLACK, "gamma": GRAY}  # the two DFT label sets
CONTEXT_COLOR = LIGHT_GRAY  # measured values and other context marks
CONTEXT_TEXT = GRAY  # text that names context marks

# ------------------------------------------------------------------------------------ other series
TEMPERATURE_COLORS = {600: "#EDB3AE", 800: "#CC6660", 1000: "#8C2A29"}  # red lightness tiers
ELEMENT_COLORS = {"Li": NAVY, "P": DARK_GRAY, "S": GRAY, "Cl": MID_GRAY}
SUBSET_COLORS = {"H3": NAVY, "H2": MID_GRAY, "T22": LIGHT_GRAY}
CL_COLORS = {1.0: BLUE_LIGHT, 1.25: BLUE_MID, 1.5: BLUE, 1.75: NAVY}

# ---------------------------------------------------------------------------------- reference lines
# Keyword arguments (full matplotlib names, so ``{**IDENTITY_LINE, "linewidth": 0.8}`` overrides cleanly).
IDENTITY_LINE = {"color": GRAY, "linewidth": 0.5, "linestyle": (0, (3, 2)), "zorder": 1}  # ax.plot of y = x
UNITY_LINE = {"color": GRAY, "linewidth": 0.5, "zorder": 1}  # ax.axhline / axvline at 0 or 1
BAND_FILL = {"color": BAND, "linewidth": 0, "zorder": 0}  # ax.axvspan / axhspan / fill_between: windows, floors

# --------------------------------------------------------------- legacy, removed in phase 2
# Old names mapped to the new values so that modules not yet migrated still draw.
OKABE_ITO = {"black": BLACK, "orange": LIGHT_RED, "sky": BLUE_LIGHT, "green": BLUE, "yellow": LIGHT_GRAY,
             "blue": NAVY, "vermillion": DEEP_RED, "purple": DARK_GRAY}  # legacy, removed in phase 2
GREY = GRAY  # legacy, removed in phase 2
LIGHT_GREY = LIGHT_GRAY  # legacy, removed in phase 2
DFT_COLORS = LABEL_COLORS  # legacy, removed in phase 2
EXPERIMENT_COLOR = CONTEXT_COLOR  # legacy, removed in phase 2
REFERENCE_COLOR = GRAY  # legacy, removed in phase 2


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": SMALL,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": LINE,
        "axes.edgecolor": BLACK,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.labeltop": False,
        "ytick.labelright": False,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.major.width": LINE,
        "ytick.major.width": LINE,
        "xtick.minor.width": LINE * 0.7,
        "ytick.minor.width": LINE * 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "errorbar.capsize": 1.5,
        "patch.linewidth": LINE,
        "svg.fonttype": "none",
        "svg.hashsalt": HASHSALT,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "standard",
        "path.simplify": False,
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
    })


def figure(width: float, height: float):
    """A new figure of the given size (inches); width must be a Nature column width."""
    if round(width / MM) not in (89, 120, 183):
        raise ValueError("Figure width must be 89, 120 or 183 mm")
    if height > MAX_HEIGHT + 1e-9:
        raise ValueError("Figure taller than 247 mm")
    apply_style()
    return plt.figure(figsize=(width, height))


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.0, transform=None):
    """Bold lower-case panel letter at the top-left of ``ax`` (figure coordinates by default); returns the text."""
    fig = ax.figure
    bbox = ax.get_position()
    if transform is None:
        text = fig.text(bbox.x0 + x, bbox.y1 + 0.004 if y == 1.0 else y, letter, fontsize=PANEL, fontweight="bold",
                        ha="right", va="bottom")
    else:
        text = ax.text(x, y, letter, transform=transform, fontsize=PANEL, fontweight="bold", ha="right",
                       va="bottom")
    text._mlipft_role = "letter"  # read by figures.qa
    return text


def save(fig, name: str, folder: str = "main") -> list[Path]:
    """Write ``figures/<folder>/<name>.{pdf,svg,png}`` deterministically."""
    target = FIGURES / folder
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for extension, metadata in (("pdf", {"CreationDate": None, "ModDate": None, "Producer": None,
                                         "Creator": None}),
                                ("svg", {"Date": None, "Creator": None}),
                                ("png", {"Software": None})):
        path = target / f"{name}.{extension}"
        fig.savefig(path, metadata=metadata, dpi=DPI if extension == "png" else None)
        written.append(path)
    plt.close(fig)
    return written


def save_source(frame: pd.DataFrame, figure_name: str, panel: str) -> Path:
    """Plotted values of one panel: ``figures/source_data/<figure>_<panel>.csv``."""
    target = FIGURES / "source_data"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{figure_name}_{panel}.csv"
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")
    return path


def log_ticks(ax, axis: str = "y") -> None:
    from matplotlib.ticker import LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10, numticks=12))
    target.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
    target.set_minor_formatter(NullFormatter())


def legend_handles(labels_colors: Iterable[tuple[str, str]], marker: str = "s", size: float = 4.0):
    from matplotlib.lines import Line2D

    return [Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=color, markeredgecolor="none",
                   markersize=size, label=label) for label, color in labels_colors]
