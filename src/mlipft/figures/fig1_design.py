"""Study design and model system (registry key "design").

a, the 52-atom Li6PS5Cl cell (config0) in an oblique projection, with the P-S bonds of its four PS4
tetrahedra and the cell edges; its ordered S/Cl arrangement (Cl on 4a, S on 4c) is the one simulated by
the AIMD reference. b, the study design (protocol amendment A3): every model is compared with DFT. Grey
boxes are inputs and simulations; the three outlined boxes are the comparisons, each marked with the
figure that shows it: forces and energies against the DFT labels; Li D of all four models against AIMD of
the same ordered cell (Deng et al. 2017); fine-tuned MD against foundation MD with the reading fixed in
advance. c, the 42 MD runs: published in July 2026 and reproduced here, or analysed here for the first
time, and whether the July run log survives.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .. import io, paths
from ..constants import SERIES, TEMPERATURES, run_id
from . import common, registry, style

KEY = "design"
NAME = registry.number(KEY)
PANELS = "abc"
WIDTH, HEIGHT = 183.0, 143.0
STRUCTURE = paths.STRUCTURES / "config0.cif"
CONFIGS = paths.DFT / "configs.csv"
QE_RUNS = paths.DFT / "qe_runs.csv"
REPRODUCTION = paths.RESULTS / "reproduction_runs.csv"
THERMOSTAT = paths.RESULTS / "thermostat.csv"
AIMD = paths.REFERENCES / "aimd_deng2017" / "transport.csv"

BOND_CUTOFF_A = 2.6
ORIGIN_SHIFT = 0.18  # fractional shift of the drawn origin along (1, 1, 1): every PS4 unit lies inside the cell
AZIMUTH_DEG, ELEVATION_DEG = 24.0, 17.0
DIAMETER_A = {"Li": 0.72, "P": 0.78, "S": 1.15, "Cl": 1.25}
ELEMENT_ORDER = ("Li", "P", "S", "Cl")
DEPTH_FADE = 0.40  # the farthest atoms are mixed with this much white
LINE_MM = 2.35  # line pitch of 5-pt text in the schematic
TITLE_MM = 2.65  # line pitch of 6-pt bold titles in the schematic
BASELINE_MM = 1.28  # baseline of a 5-pt line below the top of its letters (Arial)
BADGE_MM = (7.6, 3.3)  # figure badge of a comparison box (width, height)
SWATCH_PT = 2.8  # marker diameter of the schematic and table swatches
_FORMULA = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)")


def _load():
    cell = io.read_cif(STRUCTURE)
    configs = common.ordered_configs(common.read(CONFIGS))
    qe = common.read(QE_RUNS)
    reproduction = common.read(REPRODUCTION)
    thermostat = common.read(THERMOSTAT)
    aimd = common.read(AIMD)
    return cell, configs, qe, reproduction, thermostat, aimd


def _formula(text: str) -> str:
    """Chemical formula with mathtext subscripts (Li5.75PS4.75Cl1.25 -> Li$_{5.75}$PS$_{4.75}$Cl$_{1.25}$)."""
    return _FORMULA.sub(r"\1$_{\2}$", text)


def _plain(text: str) -> str:
    """Mathtext subscripts removed (source-data text)."""
    return re.sub(r"\$_\{([^}]*)\}\$", r"\1", text)


# ----------------------------------------------------------------------------------------------- a: structure


def _structure(cell: dict) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Atoms of the drawn cell (origin shifted along the body diagonal) and the P-S bonds."""
    a = cell["a"]
    species = np.array(cell["species"])
    fractional = np.mod(cell["fractional"] + ORIGIN_SHIFT, 1.0)
    cartesian = fractional * a
    bonds = []
    for i in np.flatnonzero(species == "P"):
        for j in np.flatnonzero(species == "S"):
            d = cell["fractional"][j] - cell["fractional"][i]
            d -= np.round(d)
            if np.linalg.norm(d * a) < BOND_CUTOFF_A:
                if np.linalg.norm(cartesian[j] - cartesian[i]) > BOND_CUTOFF_A:
                    raise ValueError("a PS4 tetrahedron is cut by the drawn cell; change ORIGIN_SHIFT")
                bonds.append((int(i), int(j)))
    atoms = pd.DataFrame({"atom": np.arange(len(species)), "element": species,
                          "frac_x": fractional[:, 0], "frac_y": fractional[:, 1], "frac_z": fractional[:, 2],
                          "x_A": cartesian[:, 0], "y_A": cartesian[:, 1], "z_A": cartesian[:, 2]})
    return atoms, bonds


def _wyckoff(fractional: np.ndarray) -> str:
    """Site of F-43m (origin on 4a) occupied by a fractional position: 4a, 4b, 4c, 4d or 'other'."""
    quarters = np.asarray(fractional, float) * 4.0
    if np.abs(quarters - np.round(quarters)).max() > 1e-4:
        return "other"
    q = np.mod(np.round(quarters).astype(int), 4)
    if np.all(q % 2 == 0):
        return "4a" if int(np.sum(q == 2)) % 2 == 0 else "4b"
    if np.all(q % 2 == 1):
        return "4c" if int(np.sum(q == 3)) % 2 == 0 else "4d"
    return "other"


def _anion_sites(cell: dict) -> dict:
    """Sites of Cl and of the free S (not bonded to P): one site each, or an error."""
    species = np.array(cell["species"])
    fractional = cell["fractional"]
    bonded = {j for _, j in _structure(cell)[1]}
    groups = {"Cl": np.flatnonzero(species == "Cl"),
              "free S": [j for j in np.flatnonzero(species == "S") if j not in bonded]}
    sites = {}
    for name, members in groups.items():
        found = {_wyckoff(fractional[j]) for j in members}
        if len(found) != 1 or "other" in found:
            raise ValueError(f"{name} does not occupy a single anion site: {sorted(found)}")
        sites[name] = found.pop()
    return sites


def _project(points: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic oblique view: rotate about the c axis, then tilt the viewer up; depth grows away."""
    centred = np.asarray(points, float) - a / 2.0
    th, el = np.radians(AZIMUTH_DEG), np.radians(ELEVATION_DEG)
    x = centred[:, 0] * np.cos(th) - centred[:, 1] * np.sin(th)
    y = centred[:, 0] * np.sin(th) + centred[:, 1] * np.cos(th)
    z = centred[:, 2]
    return x, z * np.cos(el) + y * np.sin(el), y * np.cos(el) - z * np.sin(el)


def _panel_a(ax, cell: dict, layout: common.Layout, box: tuple[float, float, float, float]) -> Path:
    a = cell["a"]
    atoms, bonds = _structure(cell)
    xyz = atoms[["x_A", "y_A", "z_A"]].to_numpy()
    sx, sy, depth = _project(xyz, a)
    atoms["screen_x_A"], atoms["screen_y_A"], atoms["depth_A"] = sx, sy, depth
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * a
    cx, cy, cd = _project(corners, a)
    edges = [(p, q) for p in range(8) for q in range(p + 1, 8) if np.sum(np.abs(corners[p] - corners[q]) > 0) == 1]
    hidden = int(np.argmax(cd))  # the three edges at the farthest corner are hidden edges

    # axes limits with the axes aspect, so one Angstrom is the same length in x and y
    left, top, width, height = box
    half_x = max(np.abs(cx).max(), np.abs(sx).max()) + 0.7
    half_y = half_x * height / width
    centre_y = (cy.max() + cy.min()) / 2.0
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(centre_y - half_y, centre_y + half_y)
    ax.set_axis_off()
    points_per_A = width / 25.4 * 72.0 / (2.0 * half_x)
    lo, hi = depth.min(), depth.max()

    def shade(d: float) -> float:
        return (d - lo) / (hi - lo)

    # painter's order: every atom, bond and cell edge gets a z-order from its depth (farthest first)
    items = [(d, "atom", i) for i, d in enumerate(depth)]
    items += [((depth[i] + depth[j]) / 2.0, "bond", (i, j)) for i, j in bonds]
    items += [((cd[p] + cd[q]) / 2.0, "edge", (p, q)) for p, q in edges]
    items.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    for rank, (d, kind, key) in enumerate(items):
        z = 2.0 + rank * 0.01
        if kind == "atom":
            element = atoms.at[key, "element"]
            ax.scatter([sx[key]], [sy[key]], s=(DIAMETER_A[element] * points_per_A) ** 2,
                       facecolors=common.lighten(style.ELEMENT_COLORS[element], DEPTH_FADE * shade(d)),
                       edgecolors=common.lighten(style.BLACK, 0.45 + 0.35 * shade(d)), linewidths=0.3, zorder=z)
        elif kind == "bond":
            i, j = key
            ax.plot([sx[i], sx[j]], [sy[i], sy[j]], color=common.lighten(style.DARK_GRAY, DEPTH_FADE * shade(d)),
                    lw=1.4, solid_capstyle="round", zorder=z)
        else:
            p, q = key
            front = hidden not in key
            ax.plot([cx[p], cx[q]], [cy[p], cy[q]], color=style.GRAY if front else style.MID_GRAY,
                    lw=0.6 if front else 0.45, zorder=z)

    counts = atoms["element"].value_counts()
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=DIAMETER_A[e] * 3.6,
                      markerfacecolor=style.ELEMENT_COLORS[e], markeredgecolor=style.GRAY, markeredgewidth=0.3,
                      label=f"{e} ({counts[e]})") for e in ELEMENT_ORDER]
    handles.append(Line2D([0], [0], color=style.DARK_GRAY, lw=1.4, label=f"P-S < {BOND_CUTOFF_A:g} Å"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=style.SMALL,
              handletextpad=0.5, columnspacing=0.9, borderaxespad=0.0, handlelength=1.1)
    formula = "".join(f"{e}{counts[e]}" for e in ELEMENT_ORDER)
    layout.text(left + width / 2.0, top - 0.5, f"{_formula(formula)} ({_formula('Li6PS5Cl')}), a = {a:.2f} Å",
                ha="center", va="bottom", fontsize=style.FONT_SIZE)
    sites = _anion_sites(cell)
    layout.text(left + width / 2.0, top + height + 6.0,
                f"Ordered S/Cl: Cl on {sites['Cl']}, free S on {sites['free S']},\nthe arrangement of the AIMD "
                "reference (Deng et al. 2017)", ha="center", va="top", fontsize=style.SMALL, linespacing=1.2)

    bond_rows = pd.DataFrame([{"item": "P-S bond", "atom": i, "atom_2": j,
                               "length_A": float(np.linalg.norm(xyz[i] - xyz[j]))} for i, j in bonds])
    edge_rows = pd.DataFrame([{"item": "cell edge", "screen_x_A": cx[p], "screen_y_A": cy[p], "screen_x2_A": cx[q],
                               "screen_y2_A": cy[q]} for p, q in edges])
    source = pd.concat([atoms.assign(item="atom"), bond_rows, edge_rows], ignore_index=True)
    source["a_A"] = a
    source["origin_shift_frac"] = ORIGIN_SHIFT
    return style.save_source(source, NAME, "a")


# ----------------------------------------------------------------------------------------------- b: design


class _Schematic:
    """Boxes and arrows in millimetres (x right, y down) inside one frameless axes.

    A line of a box is a text, or (text, swatch): a swatch is a dict of ``ax.plot`` marker keywords (the
    ``common.*_style`` helpers) or a colour, drawn as a short bar (a DFT label set).
    """

    def __init__(self, ax, width: float, height: float):
        self.ax = ax
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
        self.rows: list[dict] = []

    def _lines(self, x: float, y: float, lines) -> None:
        for line in lines:
            text, swatch = (line, None) if isinstance(line, str) else line
            tx = x
            if isinstance(swatch, dict):
                self.ax.plot(x + 0.8, y + 0.95, **{**swatch, "zorder": 4})
                tx += 2.2
            elif swatch is not None:
                self.ax.plot([x + 0.05, x + 1.55], [y + 0.95, y + 0.95], color=swatch, lw=1.6,
                             solid_capstyle="butt", zorder=4)
                tx += 2.2
            # on the baseline, so that a tall glyph (the ring of Å) does not push its line down
            self.ax.text(tx, y + BASELINE_MM, text, ha="left", va="baseline", fontsize=style.SMALL, zorder=4)
            y += LINE_MM

    def box(self, x0: float, x1: float, y0: float, y1: float, title: str, lines=(), figure: str | None = None,
            centre: bool = False, side=(), side_at: float | None = None) -> None:
        """A step (grey fill) or, with ``figure``, a comparison (outlined, with a badge naming that figure).

        ``centre`` places the text block in the vertical middle of the box (tall boxes); ``side`` is a second
        column of lines starting ``side_at`` mm from the left edge, level with ``lines``."""
        comparison = figure is not None
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                         facecolor=style.WHITE if comparison else style.SCHEMATIC_FILL,
                                         edgecolor=style.BLACK if comparison else "none",
                                         linewidth=0.8 if comparison else 0.0, zorder=2))
        n_title = title.count("\n") + 1
        gap = 1.5 if comparison else 0.9
        n_lines = max(len(lines), len(side))
        top = y0 + 1.3
        if centre:
            block = TITLE_MM * n_title + gap + LINE_MM * n_lines - 0.6
            top = (y0 + y1) / 2.0 - block / 2.0
        tx = x0 + 1.4
        if comparison:
            badge_w, badge_h = BADGE_MM
            self.ax.add_patch(FancyBboxPatch((x0 + 1.2, top - 0.1), badge_w, badge_h,
                                             boxstyle="round,pad=0,rounding_size=0.6", facecolor=style.BLACK,
                                             edgecolor="none", zorder=3))
            self.ax.text(x0 + 1.2 + badge_w / 2.0, top + badge_h / 2.0 - 0.1, figure, ha="center", va="center",
                         fontsize=style.FONT_SIZE, fontweight="bold", color=style.WHITE, zorder=4)
            tx = x0 + 1.2 + badge_w + 1.1
        self.ax.text(tx, top, title, ha="left", va="top", fontsize=style.FONT_SIZE, fontweight="bold",
                     zorder=4, linespacing=1.1)
        y = top + TITLE_MM * n_title + gap
        self._lines(x0 + 1.4, y, lines)
        if side:
            self._lines(x0 + side_at, y, side)
        texts = [line if isinstance(line, str) else line[0] for line in (*lines, *side)]
        self.rows.append({"kind": "comparison" if comparison else "step", "figure": figure or "",
                          "title": title.replace("\n", " "), "text": _plain("; ".join(texts)),
                          "x0_mm": x0, "y0_mm": y0, "x1_mm": x1, "y1_mm": y1})

    def arrow(self, points: list[tuple[float, float]], label: str = "", label_at: tuple[float, float] | None = None,
              ha: str = "left") -> None:
        for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1]):
            self.ax.plot([xa, xb], [ya, yb], color=style.BLACK, lw=0.7, solid_capstyle="projecting", zorder=1)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>,head_length=2.0,head_width=1.1",
                                          mutation_scale=1.0, shrinkA=0, shrinkB=0, color=style.BLACK, lw=0.7,
                                          zorder=1))
        if label:
            x, y = label_at if label_at is not None else points[0]
            self.ax.text(x, y, label, ha=ha, va="center", fontsize=style.SMALL, color=style.DARK_GRAY, zorder=4,
                         linespacing=1.15)
        self.rows.append({"kind": "arrow", "figure": "", "title": label.replace("\n", " "),
                          "text": " -> ".join(f"({x:g}, {y:g})" for x, y in points),
                          "x0_mm": points[0][0], "y0_mm": points[0][1], "x1_mm": points[-1][0],
                          "y1_mm": points[-1][1]})

    def rule(self, y0: float, y1: float, text: str) -> None:
        """The study's rule, set apart in a full-width outlined box."""
        x1 = self.ax.get_xlim()[1]
        self.ax.add_patch(FancyBboxPatch((0.0, y0), x1, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                         facecolor=style.WHITE, edgecolor=style.BLACK, linewidth=0.8, zorder=2))
        self.ax.text(x1 / 2.0, (y0 + y1) / 2.0, text, ha="center", va="center", fontsize=style.FONT_SIZE, zorder=4,
                     linespacing=1.25)
        self.rows.append({"kind": "note", "figure": "", "title": text.replace("\n", " "), "text": "",
                          "x0_mm": 0.0, "y0_mm": y0, "x1_mm": x1, "y1_mm": y1})


def _aimd_facts(aimd: pd.DataFrame) -> dict:
    """Cell, length and temperatures of the AIMD reference, and the temperatures shared with the MLIP runs."""
    for column in ("n_atoms", "a_A"):
        if aimd[column].nunique() != 1:
            raise ValueError(f"the AIMD runs differ in {column}")
    temperatures = sorted(int(t) for t in aimd["temperature_K"])
    shared = [t for t in TEMPERATURES if t in temperatures]
    return {"n_atoms": int(aimd["n_atoms"].iloc[0]), "a": float(aimd["a_A"].iloc[0]), "temperatures": temperatures,
            "shared": shared, "ps": float(aimd["length_ps"].round(0).max()),
            "time_step_fs": float(aimd["time_step_fs"].iloc[0])}


def _design_facts() -> dict:
    """Sizes and lengths of the MD series of the design (from the series catalogue)."""
    fine_tuned = {(sr["n_atoms"], sr["production_ps"]) for sr in SERIES if sr["model"] in ("ftG", "ftK")}
    if len(fine_tuned) != 1:
        raise ValueError("the fine-tuned series differ in size or length")
    universal = [sr for sr in SERIES if sr["model"] in style.UNIVERSAL_MODELS and sr["cl_content"] == 1.0
                 and not sr["id"].startswith("clx")]
    production = sorted({(sr["n_atoms"], sr["production_ps"]) for sr in universal if sr["n_atoms"] > 52})
    if production != sorted(fine_tuned):
        raise ValueError("the fine-tuned runs do not follow the protocol of the universal production runs")
    cl = [sr for sr in SERIES if sr["id"].startswith("clx")]
    return {"n_prod": production[0][0], "ps_prod": production[0][1],
            "tiers": sorted({sr["production_ps"] for sr in universal if sr["n_atoms"] == 52}),
            "cl_atoms": sorted(sr["n_atoms"] for sr in cl), "cl_ps": sorted({sr["production_ps"] for sr in cl}),
            "cl_x": sorted(sr["cl_content"] - 1.0 for sr in cl)}


def _panel_b(ax, configs: pd.DataFrame, qe: pd.DataFrame, aimd: pd.DataFrame, width: float, height: float) -> Path:
    s = _Schematic(ax, width, height)
    frames = sorted(configs["frame"].unique())
    temperatures = sorted(configs["temperature_K"].unique())
    qe_version = str(qe["qe_version"].iloc[0]).removeprefix("v.")
    facts = _design_facts()
    ref = _aimd_facts(aimd)
    shared = f"{ref['shared'][0]}-{ref['shared'][-1]} K"
    protocol = f"{facts['n_prod']} atoms, {facts['ps_prod']} ps"

    # columns (mm): inputs | universal MD over configurations, DFT labels, fine-tunes | AIMD, against AIMD
    col_in = (0.0, 19.0)
    col_ref = (width - 22.0, width)
    x_right = col_ref[0] - 2.5  # connector: universal MD -> against AIMD
    col_ft = (col_ref[0] - 28.0, col_ref[0] - 5.0)
    col_dft = (col_ft[0] - 22.0, col_ft[0] - 4.0)
    col_cfg = (col_dft[0] - 22.5, col_dft[0] - 4.0)
    md = (col_in[1] + 5.0, col_ft[1])
    x_left = md[0] + 1.8  # connector: universal MD -> fine-tuned against foundation
    col_ftmd = (col_ft[0] + 3.6, col_ft[1])
    x_elbow = col_ft[0] + 1.6  # connector: fine-tunes -> against DFT labels

    r1, r2, r3, r4 = (1.0, 14.5), (21.0, 35.5), (42.0, 57.0), (63.5, 76.0)
    m1, m2, m3, m4 = [(lo + hi) / 2.0 for lo, hi in (r1, r2, r3, r4)]
    swatch = {"size": SWATCH_PT, "edge": 0.6}

    # inputs and simulations
    s.box(*col_in, *r1, "Structure", [_formula("Li6PS5Cl"), "52 atoms (a)", f"{facts['n_prod']} atoms (2x2x2)"])
    s.box(*md, *r1, "MD, universal potentials",
          [("MACE-MP-0 small (foundation)", common.model_style("foundation", **swatch)),
           ("MatterSim", common.model_style("mattersim", **swatch))],
          side=[f"52 atoms: {', '.join(str(t) for t in facts['tiers'])} ps", f"{protocol}",
                f"{', '.join(str(t) for t in temperatures)} K"], side_at=38.0)
    s.box(*col_ref, r1[0], r2[1], "AIMD of the\nsame ordered\ncell",
          [("Deng et al. 2017", common.aimd_style(**swatch)), f"{ref['n_atoms']} atoms", f"a = {ref['a']:.2f} Å",
           f"{ref['temperatures'][0]}-{ref['temperatures'][-1]} K", f"{ref['ps']:.0f} ps per T",
           registry.label("aimd", short=True)], centre=True)
    s.box(*col_in, r2[0], r3[1], "Cl excess",
          ["Li$_{6-x}$PS$_{5-x}$Cl$_{1+x}$", f"x = {facts['cl_x'][0]:g}-"
           f"{facts['cl_x'][-1]:g}", "MACE-MP-0", f"{facts['cl_atoms'][0]}-{facts['cl_atoms'][-1]} atoms",
           f"{', '.join(str(v) for v in facts['cl_ps'])} ps", registry.label("clexcess", short=True)], centre=True)
    s.box(*col_cfg, *r2, f"{len(configs)}\nconfigurations", [f"frames {frames[0]}-{frames[-1]}",
                                                            f"{len(frames)} per temperature"])
    s.box(*col_dft, *r2, "DFT labels", [f"QE {qe_version}, PBE", ("Gamma point", style.LABEL_COLORS["gamma"]),
                                        ("2x2x2 k-points", style.LABEL_COLORS["k222"])])
    s.box(*col_ft, *r2, "Fine-tunes of\nMACE-MP-0", [("Gamma labels", common.model_style("ftG", **swatch)),
                                                    ("2x2x2 labels", common.model_style("ftK", **swatch))])
    s.box(*col_ftmd, *r3, "Fine-tuned MD", [protocol, f"{', '.join(str(t) for t in TEMPERATURES)} K",
                                            "Gamma: stage 2", "2x2x2: stage 1"])

    # the three comparisons, each marked with its figure
    s.box(col_cfg[0], col_dft[1], *r3, "Against DFT labels",
          ["forces and energies", "MACE-MP-0 and both fine-tunes", "held-out configurations"],
          figure=registry.label("forces"))
    s.box(*col_ref, r3[0], r4[1], "Against\nAIMD", [f"Li D at {shared}", "all four models", "no extrapolation"],
          figure=registry.label("diffusion"), centre=True)
    s.box(col_cfg[0], col_ft[1], *r4, "Fine-tuned against foundation",
          ["fine-tuned MD against MACE-MP-0 MD", "pre-registered reading"],
          side=["same MD protocol", f"{protocol}, {shared}"], side_at=36.0, figure=registry.label("finetune"))

    # flow
    x_in = (col_in[0] + col_in[1]) / 2.0
    s.arrow([(col_in[1], m1), (md[0], m1)])
    s.arrow([(x_in, r1[1]), (x_in, r2[0])])
    x_cfg = (col_cfg[0] + col_cfg[1]) / 2.0
    s.arrow([(x_cfg, r1[1]), (x_cfg, r2[0])], label="MACE-MP-0, 52 atoms, 50 ps",
            label_at=(x_cfg + 1.3, (r1[1] + r2[0]) / 2.0))
    s.arrow([(col_cfg[1], m2), (col_dft[0], m2)])
    s.arrow([(col_dft[1], m2), (col_ft[0], m2)])
    x_dft = (col_dft[0] + col_dft[1]) / 2.0
    s.arrow([(x_dft, r2[1]), (x_dft, r3[0])])
    s.arrow([(x_elbow, r2[1]), (x_elbow, m3), (col_dft[1], m3)])
    x_ftmd = (col_ftmd[0] + col_ftmd[1]) / 2.0
    s.arrow([(x_ftmd, r2[1]), (x_ftmd, r3[0])])
    x_ref = (col_ref[0] + col_ref[1]) / 2.0
    s.arrow([(x_ref, r2[1]), (x_ref, r3[0])])  # AIMD -> against AIMD
    y_universal, y_fine = r3[0] + 3.0, r3[0] + 9.0
    s.arrow([(md[1], m1), (x_right, m1), (x_right, y_universal), (col_ref[0], y_universal)])
    s.arrow([(col_ftmd[1], y_fine), (col_ref[0], y_fine)])
    s.arrow([(x_ftmd, r3[1]), (x_ftmd, r4[0])])
    s.arrow([(x_left, r1[1]), (x_left, m4), (col_cfg[0], m4)])

    s.rule(height - 6.5, height, "Every model is compared with DFT: forces with the DFT labels, Li diffusion with "
                                 "AIMD of the same ordered cell.")
    return style.save_source(pd.DataFrame(s.rows), NAME, "b")


# ----------------------------------------------------------------------------------------------- c: runs

GROUPS = {"A": "Universal\npotentials", "C": "Fine-tuned", "D": "Cl excess"}  # analysis-plan part -> table group


def _runs(reproduction: pd.DataFrame, thermostat: pd.DataFrame) -> pd.DataFrame:
    """One row per MD run: published (and its reproduction classes) or new, and the July run log if any."""
    classes = reproduction.set_index("run_id")
    logs = thermostat.set_index("run_id")["run_log"].fillna("").astype(str)
    rows = []
    for series in SERIES:
        for temperature in TEMPERATURES:
            rid = run_id(series["id"], temperature)
            published = rid in classes.index
            if published != (series["published"] is not None):
                raise ValueError(f"{rid}: reproduction table and series catalogue disagree")
            log = logs.get(rid, "")
            rows.append({"run_id": rid, "series": series["id"], "label": series["label"], "part": series["part"],
                         "group": GROUPS[series["part"]].replace("\n", " "), "model": series["model"],
                         "run_type": _run_type(series), "n_atoms": series["n_atoms"],
                         "production_ps": series["production_ps"], "temperature_K": temperature,
                         "published_july_2026": published,
                         "pymatgen_class": classes.at[rid, "pymatgen_class"] if published else "",
                         "kinisi_D_class": classes.at[rid, "kinisi_D_class"] if published else "",
                         "kinisi_std_class": classes.at[rid, "kinisi_std_class"] if published else "",
                         "run_log": log, "run_log_survives": log != ""})
    return pd.DataFrame(rows)


def _run_type(series: dict) -> str:
    """Key of ``style.RUN_MARKERS``: Cl-excess series, 52-atom cell or production run."""
    if series["id"].startswith("clx"):
        return "clexcess"
    return "cell_52" if series["n_atoms"] == 52 else "production"


def _row_label(label: str) -> str:
    name = label.split(", ")[0].replace("Fine-tuned on ", "Fine-tuned, ")
    return _formula(name) if name.startswith("Li") else name


def _panel_c(ax, runs: pd.DataFrame, width: float, height: float) -> Path:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    pitch, gap, header = 2.85, 1.6, 6.0
    cell_w, cell_h = 11.0, 2.5
    x_group, x_bracket, x_dot, x_label = 1.0, 11.6, 13.4, 15.2
    x_cells = [43.0 + i * (cell_w + 1.2) for i in range(len(TEMPERATURES))]
    x_atoms, x_ps = x_cells[-1] + cell_w + 6.0, x_cells[-1] + cell_w + 16.0
    face = {True: style.DARK_GRAY, False: style.WHITE}
    series_ids = list(dict.fromkeys(runs["series"]))
    parts = [runs.loc[runs["series"] == sid, "part"].iloc[0] for sid in series_ids]
    ys, y = [], header + pitch / 2.0
    for index, part in enumerate(parts):
        if index and part != parts[index - 1]:
            y += gap
        ys.append(y)
        y += pitch
    position = dict(zip(series_ids, ys))

    head = header - 1.0
    for x, temperature in zip(x_cells, TEMPERATURES):
        ax.text(x + cell_w / 2.0, head, f"{temperature} K", ha="center", va="bottom", fontsize=style.SMALL)
    ax.text(x_atoms, head, "Atoms", ha="center", va="bottom", fontsize=style.SMALL)
    ax.text(x_ps, head, "Production\n(ps)", ha="center", va="bottom", fontsize=style.SMALL, linespacing=1.0)
    ax.text(x_label, head, "Potential or composition", ha="left", va="bottom", fontsize=style.SMALL)

    for part in dict.fromkeys(parts):
        members = [position[sid] for sid, p in zip(series_ids, parts) if p == part]
        top, bottom = members[0] - pitch / 2.0 + 0.3, members[-1] + pitch / 2.0 - 0.3
        ax.plot([x_bracket, x_bracket], [top, bottom], color=style.BLACK, lw=0.5)
        ax.text(x_group, (top + bottom) / 2.0, GROUPS[part], ha="left", va="center", fontsize=style.SMALL,
                linespacing=1.1)

    rows = []
    for sid in series_ids:
        y = position[sid]
        group = runs[runs["series"] == sid]
        first = group.iloc[0]
        ax.plot(x_dot, y, **common.model_style(first["model"], first["run_type"], size=SWATCH_PT, edge=0.6))
        ax.text(x_label, y, _row_label(first["label"]), ha="left", va="center", fontsize=style.SMALL)
        ax.text(x_atoms, y, str(first["n_atoms"]), ha="center", va="center", fontsize=style.SMALL)
        ax.text(x_ps, y, str(first["production_ps"]), ha="center", va="center", fontsize=style.SMALL)
        for x, temperature in zip(x_cells, TEMPERATURES):
            run = group[group["temperature_K"] == temperature].iloc[0]
            published = bool(run["published_july_2026"])
            ax.add_patch(Rectangle((x, y - cell_h / 2.0), cell_w, cell_h, facecolor=face[published],
                                   edgecolor=style.DARK_GRAY, linewidth=0.6))
            if run["run_log_survives"]:
                ax.text(x + cell_w / 2.0, y, "log", ha="center", va="center", fontsize=style.SMALL,
                        color=style.WHITE if published else style.BLACK)
            rows.append({**run.to_dict(), "row_y_mm": y, "cell_x_mm": x})

    # key to the right of the matrix: cells, then markers (colour by potential, shape by run type)
    lx = x_ps + 10.0
    published = runs["published_july_2026"]
    n_pub, n_log = int(published.sum()), int(runs["run_log_survives"].sum())
    classes = {column: "/".join(sorted(set(runs.loc[published, column])))
               for column in ("pymatgen_class", "kinisi_D_class", "kinisi_std_class")}
    entries = [
        (face[True], "", f"Published in July 2026 and reproduced here ({n_pub} runs): pymatgen\n"
                         f"class {classes['pymatgen_class']}, kinisi D and its uncertainty class "
                         f"{classes['kinisi_D_class']} ({registry.label('reproduction', short=True)})"),
        (face[False], "", f"Analysed here for the first time ({len(runs) - n_pub} runs)"),
        (None, "log", f"July run log survives ({n_log} runs): thermostat statistics recomputed\n"
                      f"from the trajectory are checked against it; the other {len(runs) - n_log} runs\n"
                      "have statistics from the trajectory only"),
    ]
    ly = header + pitch / 2.0
    for color, text, label in entries:
        if color is not None:
            ax.add_patch(Rectangle((lx, ly - cell_h / 2.0), 7.0, cell_h, facecolor=color, edgecolor=style.DARK_GRAY,
                                   linewidth=0.6))
        if text:
            ax.text(lx + 3.5, ly, text, ha="center", va="center", fontsize=style.SMALL)
        ax.text(lx + 9.0, ly - cell_h / 2.0 + 0.05, label, ha="left", va="top", fontsize=style.SMALL,
                linespacing=1.15)
        ly += 3.6 + 2.3 * label.count("\n")
    ly += 1.4
    shapes = [("production", f"{runs.loc[runs['run_type'] == 'production', 'n_atoms'].iloc[0]} atoms, production"),
              ("cell_52", "52 atoms"), ("clexcess", "Cl-excess series")]
    for index, model in enumerate(("foundation", "mattersim", "ftG", "ftK")):
        ax.plot(lx + 3.5, ly + index * 2.8, **common.model_style(model, size=SWATCH_PT, edge=0.6))
        ax.text(lx + 9.0, ly + index * 2.8, style.MODEL_LABELS[model], ha="left", va="center", fontsize=style.SMALL)
    sx = lx + 36.0
    for index, (kind, label) in enumerate(shapes):
        ax.plot(sx, ly + index * 2.8, marker=style.RUN_MARKERS[kind], markersize=SWATCH_PT, markeredgewidth=0.6,
                markerfacecolor=style.DARK_GRAY, markeredgecolor=style.DARK_GRAY, linestyle="none")
        ax.text(sx + 2.0, ly + index * 2.8, label, ha="left", va="center", fontsize=style.SMALL)
    return style.save_source(pd.DataFrame(rows), NAME, "c")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    cell, configs, qe, reproduction, thermostat, aimd = _load()
    runs = _runs(reproduction, thermostat)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    box_a = (3.0, 11.0, 50.0, 54.0)
    ax_a = L.axes(*box_a, frame="none")
    sources = [_panel_a(ax_a, cell, L, box_a)]
    b_left, b_top, b_width, b_height = 59.0, 4.0, 123.0, 84.0
    ax_b = L.axes(b_left, b_top, b_width, b_height, frame="none")
    sources.append(_panel_b(ax_b, configs, qe, aimd, b_width, b_height))
    c_top, c_height = 93.0, 49.0
    ax_c = L.axes(3.0, c_top, 179.0, c_height, frame="none")
    sources.append(_panel_c(ax_c, runs, 179.0, c_height))

    L.letter(ax_a, "a", 3.0, 6.0)
    L.letter(ax_b, "b", 56.0, 6.0)
    L.letter(ax_c, "c", 3.0, c_top - 1.0)
    return L.save(*registry.target(KEY)) + sources


def _caption() -> str:
    cell, configs, qe, reproduction, thermostat, aimd = _load()
    runs = _runs(reproduction, thermostat)
    atoms, bonds = _structure(cell)
    sites = _anion_sites(cell)
    ref = _aimd_facts(aimd)
    facts = _design_facts()
    counts = atoms["element"].value_counts()
    published = runs["published_july_2026"]
    n_pub, n_log = int(published.sum()), int(runs["run_log_survives"].sum())
    new = []
    for sid in runs.loc[~published, "series"].unique():
        series = runs[runs["series"] == sid].iloc[0]
        name = series["label"].split(", ")[0].replace("Fine-tuned on ", "fine-tuned on ")
        new.append(f"{name}, {series['n_atoms']} atoms, {series['production_ps']} ps")
    classes = {column: ", ".join(sorted(set(runs.loc[published, column])))
               for column in ("pymatgen_class", "kinisi_D_class", "kinisi_std_class")}
    frames = sorted(configs["frame"].unique())
    k_points = qe.groupby("label_set")["k_points"].first()
    qe_version = str(qe["qe_version"].iloc[0]).removeprefix("v.")

    def listed(values, unit: str = "K") -> str:
        values = [str(v) for v in values]
        return ", ".join(values[:-1]) + f" and {values[-1]} {unit}"

    volume_ratio = (ref["a"] / cell["a"]) ** 3
    return (
        registry.heading(KEY)
        + f"**a**, The 52-atom cell of every Li6PS5Cl run (Li{counts['Li']}P{counts['P']}S{counts['S']}Cl{counts['Cl']}, "
        f"cubic, a = {cell['a']:.4f} A; the lowest-Ewald-energy S/Cl ordering of the Materials Project cell), in an "
        f"oblique orthographic projection, farther atoms lighter. The S/Cl arrangement is fully ordered, Cl on "
        f"{sites['Cl']} and the free S on {sites['free S']}; the AIMD reference simulates the same arrangement. "
        f"The origin is shifted by {ORIGIN_SHIFT:g} along each axis so that the {counts['P']} PS4 tetrahedra "
        f"(P-S < {BOND_CUTOFF_A:g} A, minimum image; {len(bonds)} bonds) lie inside the drawn cell. The "
        f"{facts['n_prod']}-atom cell is its 2x2x2 supercell. "
        f"**b**, Study design (protocol amendment A3): every model is compared with DFT. Grey boxes, inputs and "
        f"simulations; outlined boxes, the three comparisons, each marked with the figure that shows it. "
        f"Against DFT labels ({registry.label('forces')}): {len(configs)} configurations (frames "
        f"{frames[0]}-{frames[-1]}, {len(frames)} per source temperature) of the 52-atom, 50-ps MACE-MP-0 runs were "
        f"labelled with Quantum ESPRESSO {qe_version} (PBE) at the Gamma point ({k_points['gamma']} k-point) and on a "
        f"2x2x2 mesh ({k_points['k222']} k-points); MACE-MP-0 small was fine-tuned once on each label set, and the "
        f"forces and energies of MACE-MP-0 and of both fine-tuned models are compared with these labels on held-out "
        f"configurations. Against AIMD ({registry.label('diffusion')}): MD with the two universal potentials "
        f"(MACE-MP-0 small, the foundation model, and MatterSim; 52 atoms for {listed(facts['tiers'], 'ps')} and "
        f"{facts['n_prod']} atoms for {facts['ps_prod']} ps) and with the two fine-tuned models (the Gamma fine-tune "
        f"at stage two, the 2x2x2 fine-tune at stage one; {facts['n_prod']} atoms, {facts['ps_prod']} ps), all at "
        f"{listed(TEMPERATURES)}, is compared with AIMD of the same ordered cell through the Li diffusion "
        f"coefficient D at {listed(ref['shared'])}, without extrapolation. The AIMD reference (Deng et al., Chem. "
        f"Mater. 29, 281 (2017); {registry.label('aimd')}) has {ref['n_atoms']} atoms, a = {ref['a']:.2f} A (a "
        f"{100 * (1 - volume_ratio):.0f} % smaller volume than here), a {ref['time_step_fs']:g}-fs time step and "
        f"{ref['ps']:.0f} ps at each of {listed(ref['temperatures'])}. Fine-tuned against foundation "
        f"({registry.label('finetune')}): the fine-tuned MD runs follow the protocol of the {facts['n_prod']}-atom, "
        f"{facts['ps_prod']}-ps MACE-MP-0 runs and are compared with them through the reading fixed in advance "
        f"(protocol/analysis_plan.md). The Cl-excess series Li6-xPS5-xCl1+x (x = {facts['cl_x'][0]:g}-"
        f"{facts['cl_x'][-1]:g}; {facts['cl_atoms'][0]}-{facts['cl_atoms'][-1]} atoms, {facts['cl_ps'][0]} ps) runs "
        f"with MACE-MP-0 ({registry.label('clexcess')}). Markers: red, the universal potentials; blue, the "
        f"fine-tuned models, filled for the stage-two model and open for the stage-one model; black square, AIMD. "
        f"Black and grey bars, the 2x2x2 and Gamma label sets. "
        f"**c**, The {len(runs)} MD runs (n = {len(runs)}: {n_pub} published, {len(runs) - n_pub} new; {n_log} with "
        f"a surviving run log), one per series and temperature, grouped into the universal potentials, the "
        f"fine-tuned models and the Cl-excess series. Dark cells: the {n_pub} runs published in July 2026, "
        f"re-analysed here from the same trajectories (pymatgen class {classes['pymatgen_class']} for every run, "
        f"kinisi class {classes['kinisi_D_class']}; {registry.label('reproduction')}). White cells: the "
        f"{len(runs) - n_pub} runs analysed here for the first time ({'; '.join(new)}). 'log': the July run log "
        f"survives ({n_log} runs) and the thermostat statistics recomputed from the trajectory were checked against "
        f"it; for the other {len(runs) - n_log} runs they come from the trajectory only. Markers: colour and fill by "
        f"potential as in b; shape by run type (circle, {facts['n_prod']}-atom production run; triangle, 52-atom "
        f"run; inverted triangle, Cl-excess series). "
        + common.sources([STRUCTURE, CONFIGS, QE_RUNS, AIMD, REPRODUCTION, THERMOSTAT])
    )


CAPTION = _caption()
