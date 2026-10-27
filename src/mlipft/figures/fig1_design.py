"""Fig. 1: study design, model system and data.

a, the 52-atom Li6PS5Cl cell (config0) in an oblique projection, with the P-S bonds of its four PS4
tetrahedra and the cell edges; its ordered S/Cl arrangement (Cl on 4a, S on 4c) is the one simulated by
the AIMD reference. b, the study design (protocol amendment A3): every model is compared with DFT. MD
with two universal potentials is compared with AIMD of the same ordered cell (Deng et al. 2017) through
Li D (part A); 27 configurations of the 52-atom MACE-MP-0 runs labelled by DFT at the Gamma point and on
a 2x2x2 k-point mesh, and two fine-tunes of MACE-MP-0 compared with those labels (part B); MD with the
fine-tuned models compared with the foundation model and with the AIMD (part C); the Cl-excess series
with MACE-MP-0 (part D). Measured conductivities are context only and never shown next to a fine-tuned
model. c, the 42 MD runs: published in July 2026 and reproduced here, or analysed here for the first
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
from . import common, style

NAME = "fig1"
WIDTH, HEIGHT = 183.0, 147.0
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
PROCESS_FACE = "#F0F0F0"
PUBLISHED_FACE = "#4D4D4D"
LINE_MM = 2.35  # line pitch of 5-pt text in the schematic
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
    return re.sub(r"\$_\{([^}]*)\}\$", r"\1", text).replace("$\\sigma$", "sigma")


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
                       edgecolors=common.lighten("#000000", 0.45 + 0.35 * shade(d)), linewidths=0.3, zorder=z)
        elif kind == "bond":
            i, j = key
            ax.plot([sx[i], sx[j]], [sy[i], sy[j]], color=common.lighten("#4A4A4A", DEPTH_FADE * shade(d)), lw=1.4,
                    solid_capstyle="round", zorder=z)
        else:
            p, q = key
            front = hidden not in key
            ax.plot([cx[p], cx[q]], [cy[p], cy[q]], color="#6E6E6E" if front else "#B0B0B0",
                    lw=0.6 if front else 0.45, zorder=z)

    counts = atoms["element"].value_counts()
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=DIAMETER_A[e] * 3.6,
                      markerfacecolor=style.ELEMENT_COLORS[e], markeredgecolor="#666666", markeredgewidth=0.3,
                      label=f"{e} ({counts[e]})") for e in ELEMENT_ORDER]
    handles.append(Line2D([0], [0], color="#4A4A4A", lw=1.4, label=f"P-S < {BOND_CUTOFF_A:g} Å"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=style.SMALL,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0.0, handlelength=1.1)
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
    """Boxes and arrows in millimetres (x right, y down) inside one axes."""

    def __init__(self, ax, width: float, height: float):
        self.ax = ax
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
        self.rows: list[dict] = []

    def box(self, x0: float, x1: float, y0: float, y1: float, title: str, lines=(), part: str | None = None,
            centre: bool = False) -> None:
        """A step (grey) or a comparison (outlined, with its part letter); ``lines`` are text or (text, marker).

        ``centre`` places the text block in the vertical middle of the box (tall boxes)."""
        comparison = part is not None
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                         facecolor="white" if comparison else PROCESS_FACE,
                                         edgecolor="black" if comparison else "none",
                                         linewidth=0.8 if comparison else 0.0, zorder=2))
        n_title = title.count("\n") + 1
        gap = 1.5 if comparison else 0.9
        top = y0 + 1.3
        if centre:
            block = 2.65 * n_title + gap + LINE_MM * len(lines) - 0.6
            top = (y0 + y1) / 2.0 - block / 2.0
        tx = x0 + 1.4
        if comparison:
            self.ax.add_patch(FancyBboxPatch((x0 + 1.2, top - 0.1), 3.3, 3.3, boxstyle="round,pad=0,rounding_size=0.6",
                                             facecolor="black", edgecolor="none", zorder=3))
            self.ax.text(x0 + 2.85, top + 1.55, part, ha="center", va="center", fontsize=style.FONT_SIZE,
                         fontweight="bold", color="white", zorder=4)
            tx = x0 + 5.5
        self.ax.text(tx, top, title, ha="left", va="top", fontsize=style.FONT_SIZE, fontweight="bold",
                     zorder=4, linespacing=1.1)
        y = top + 2.65 * n_title + gap
        for line in lines:
            text, marker = (line, None) if isinstance(line, str) else line
            x = x0 + 1.4
            if marker is not None:
                color, shape = marker
                self.ax.plot(x + 0.8, y + 0.95, marker=shape, markersize=2.8 if shape != "D" else 2.4,
                             markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5, linestyle="none",
                             zorder=4)
                x += 2.2
            self.ax.text(x, y, text, ha="left", va="top", fontsize=style.SMALL, zorder=4)
            y += LINE_MM
        self.rows.append({"kind": "comparison" if comparison else "step", "part": part or "",
                          "title": title.replace("\n", " "),
                          "text": _plain("; ".join(line if isinstance(line, str) else line[0] for line in lines)),
                          "x0_mm": x0, "y0_mm": y0, "x1_mm": x1, "y1_mm": y1})

    def arrow(self, points: list[tuple[float, float]], label: str = "", label_at: tuple[float, float] | None = None,
              ha: str = "left") -> None:
        for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1]):
            self.ax.plot([xa, xb], [ya, yb], color="black", lw=0.7, solid_capstyle="projecting", zorder=1)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>,head_length=2.0,head_width=1.1",
                                          mutation_scale=1.0, shrinkA=0, shrinkB=0, color="black", lw=0.7, zorder=1))
        if label:
            x, y = label_at if label_at is not None else points[0]
            self.ax.text(x, y, label, ha=ha, va="center", fontsize=style.SMALL, color="#333333", zorder=4,
                         linespacing=1.15)
        self.rows.append({"kind": "arrow", "part": "", "title": label.replace("\n", " "),
                          "text": " -> ".join(f"({x:g}, {y:g})" for x, y in points),
                          "x0_mm": points[0][0], "y0_mm": points[0][1], "x1_mm": points[-1][0],
                          "y1_mm": points[-1][1]})

    def note(self, x: float, y: float, text: str, ha: str = "center", size: float = style.SMALL,
             color: str = "#666666") -> None:
        self.ax.text(x, y, text, ha=ha, va="top", fontsize=size, color=color, zorder=4, linespacing=1.15)
        self.rows.append({"kind": "note", "part": "", "title": _plain(text.replace("\n", " ")), "text": "",
                          "x0_mm": x, "y0_mm": y, "x1_mm": x, "y1_mm": y})


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


def _panel_b(ax, configs: pd.DataFrame, qe: pd.DataFrame, aimd: pd.DataFrame, width: float, height: float) -> Path:
    s = _Schematic(ax, width, height)
    frames = sorted(configs["frame"].unique())
    temperatures = sorted(configs["temperature_K"].unique())
    qe_version = str(qe["qe_version"].iloc[0]).removeprefix("v.")
    fine_tuned = {(sr["n_atoms"], sr["production_ps"]) for sr in SERIES if sr["part"] == "C"}
    if len(fine_tuned) != 1:
        raise ValueError("the fine-tuned series differ in size or length")
    n_ft, ps_ft = fine_tuned.pop()
    cl_atoms = sorted(sr["n_atoms"] for sr in SERIES if sr["part"] == "D")
    cl_ps = sorted({sr["production_ps"] for sr in SERIES if sr["part"] == "D"})
    tiers = sorted({sr["production_ps"] for sr in SERIES if sr["n_atoms"] == 52})
    production = sorted({(sr["n_atoms"], sr["production_ps"]) for sr in SERIES
                         if sr["part"] == "A" and sr["n_atoms"] > 52})
    if len(production) != 1:
        raise ValueError("more than one production size in part A")
    n_prod, ps_prod = production[0]
    ref = _aimd_facts(aimd)
    shared = f"{ref['shared'][0]}-{ref['shared'][-1]} K"

    # columns (mm): structure, D | universal MD, configurations, B, C | DFT | fine-tunes, A | AIMD
    col_d = (0.0, 20.0)
    col_ref = (width - 20.0, width)
    col_cmp = (col_ref[0] - 25.0, col_ref[0] - 4.0)
    col_dft = (col_cmp[0] - 22.0, col_cmp[0] - 4.0)
    col_cfg = (col_dft[0] - 21.0, col_dft[0] - 4.0)
    md = (col_d[1] + 5.0, col_dft[1] + 1.0)
    x_found = col_cfg[0] - 2.5  # vertical connector from the universal MD to C
    col_ftmd = (col_cmp[0] + 3.6, col_cmp[1])
    x_elbow = col_cmp[0] + 1.6  # connector from the fine-tunes to B

    # row 1: structure -> universal MD -> A <- AIMD
    r1, r2, r3, r4 = (1.0, 18.0), (25.0, 39.0), (45.0, 60.0), (66.0, 77.0)
    m1, m2, m3, m4 = 9.5, 32.0, 52.5, 71.5
    s.box(*col_d, *r1, "Structure", [_formula("Li6PS5Cl"), "52 atoms (a)", f"{n_prod} atoms (2x2x2)"])
    s.box(*md, *r1, "MD, universal potentials",
          [("MACE-MP-0 small (foundation)", (style.MODEL_COLORS["foundation"], "o")),
           ("MatterSim", (style.MODEL_COLORS["mattersim"], "o")),
           f"52 atoms: {', '.join(str(t) for t in tiers)} ps",
           f"{n_prod} atoms: {ps_prod} ps",
           f"{', '.join(str(t) for t in temperatures)} K"])
    a_box = (r1[0], r1[0] + 12.5)
    s.box(*col_cmp, *a_box, "Against AIMD", [f"Li D at {shared}", "Fig. 2"], part="A")
    s.note((col_cmp[0] + col_cmp[1]) / 2.0, a_box[1] + 0.8,
           "measured $\\sigma$ (disordered\npolycrystals): context only")
    s.box(*col_ref, r1[0], r4[1], "AIMD of the\nsame ordered\ncell",
          ["Deng et al. 2017", f"{ref['n_atoms']} atoms", f"a = {ref['a']:.2f} Å",
           f"{ref['temperatures'][0]}-{ref['temperatures'][-1]} K", f"{ref['ps']:.0f} ps per T",
           "Ext. Data Fig. 9"], centre=True)
    s.arrow([(col_d[1], m1), (md[0], m1)])
    s.arrow([(md[1], m1), (col_cmp[0], m1)])
    s.arrow([(col_ref[0], m1), (col_cmp[1], m1)])

    # row 2: configurations -> DFT labels -> fine-tunes; D below the structure
    s.box(*col_cfg, *r2, f"{len(configs)}\nconfigurations", [f"frames {frames[0]}-{frames[-1]}",
                                                            f"{len(frames)} per temperature"])
    s.box(*col_dft, *r2, "DFT labels", [f"QE {qe_version}, PBE", ("Gamma point", (style.DFT_COLORS["gamma"], "s")),
                                        ("2x2x2 k-points", (style.DFT_COLORS["k222"], "s"))])
    s.box(*col_cmp, *r2, "Fine-tunes of\nMACE-MP-0", [("Gamma labels", (style.MODEL_COLORS["ftG"], "o")),
                                                     ("2x2x2 labels", (style.MODEL_COLORS["ftK"], "o"))])
    x_cfg = (col_cfg[0] + col_cfg[1]) / 2.0
    s.arrow([(x_cfg, r1[1]), (x_cfg, r2[0])], label="MACE-MP-0, 52 atoms, 50 ps",
            label_at=(x_cfg + 1.3, (r1[1] + r2[0]) / 2.0))
    s.arrow([(col_cfg[1], m2), (col_dft[0], m2)])
    s.arrow([(col_dft[1], m2), (col_cmp[0], m2)])
    s.box(*col_d, r2[0], 47.0, "Cl excess",
          ["Li$_{6-x}$PS$_{5-x}$Cl$_{1+x}$", "x = 0-0.75", "MACE-MP-0", f"{cl_atoms[0]}-{cl_atoms[-1]} atoms",
           f"{', '.join(str(v) for v in cl_ps)} ps", "Ext. Data Fig. 8"], part="D")
    s.arrow([((col_d[0] + col_d[1]) / 2.0, r1[1]), ((col_d[0] + col_d[1]) / 2.0, r2[0])])

    # row 3: B, fed by the labels and the fine-tunes; fine-tuned MD
    s.box(col_cfg[0], col_dft[1], *r3, "Against DFT labels", ["forces and energies", "held-out configurations; Fig. 3"],
          part="B")
    s.box(*col_ftmd, *r3, "Fine-tuned MD", [f"{n_ft} atoms, {ps_ft} ps", "same protocol", "Gamma: stage 2",
                                            "2x2x2: stage 1"])
    x_dft = (col_dft[0] + col_dft[1]) / 2.0
    s.arrow([(x_dft, r2[1]), (x_dft, r3[0])])
    s.arrow([(x_elbow, r2[1]), (x_elbow, m3), (col_dft[1], m3)])
    x_ftmd = (col_ftmd[0] + col_ftmd[1]) / 2.0
    s.arrow([(x_ftmd, r2[1]), (x_ftmd, r3[0])])

    # row 4: C <- fine-tuned MD, foundation MD (row 1) and AIMD
    s.box(col_cfg[0], col_cmp[1], *r4, "Against the foundation model and AIMD",
          [f"fine-tuned MD against MACE-MP-0 MD ({n_ft} atoms, {ps_ft} ps)",
           f"and against AIMD: Li D at {shared}; Fig. 4"],
          part="C")
    s.arrow([(x_ftmd, r3[1]), (x_ftmd, r4[0])])
    s.arrow([(x_found, r1[1]), (x_found, m4), (col_cfg[0], m4)])
    s.arrow([(col_ref[0], m4), (col_cmp[1], m4)])

    # the rule, set apart at the bottom
    note = ("Every model is compared with DFT: forces with the DFT labels, Li diffusion with AIMD.\n"
            "Measured conductivities are context only, and never shown next to a fine-tuned model.")
    ax.add_patch(FancyBboxPatch((0.0, height - 7.0), width, 7.0, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="white", edgecolor=style.MODEL_COLORS["ftG"], linewidth=0.8, zorder=2))
    ax.text(width / 2.0, height - 3.5, note, ha="center", va="center", fontsize=style.FONT_SIZE, zorder=4,
            linespacing=1.25)
    s.rows.append({"kind": "note", "part": "", "title": note.replace("\n", " "), "text": "", "x0_mm": 0.0,
                   "y0_mm": height - 7.0, "x1_mm": width, "y1_mm": height})
    return style.save_source(pd.DataFrame(s.rows), NAME, "b")


# ----------------------------------------------------------------------------------------------- c: runs


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
                         "model": series["model"], "n_atoms": series["n_atoms"],
                         "production_ps": series["production_ps"], "temperature_K": temperature,
                         "published_july_2026": published,
                         "pymatgen_class": classes.at[rid, "pymatgen_class"] if published else "",
                         "kinisi_D_class": classes.at[rid, "kinisi_D_class"] if published else "",
                         "kinisi_std_class": classes.at[rid, "kinisi_std_class"] if published else "",
                         "run_log": log, "run_log_survives": log != ""})
    return pd.DataFrame(rows)


def _row_label(label: str) -> str:
    name = label.split(", ")[0].replace("Fine-tuned on ", "Fine-tuned, ")
    return _formula(name) if name.startswith("Li") else name


def _panel_c(ax, runs: pd.DataFrame, width: float, height: float) -> Path:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    pitch, gap, header = 2.85, 1.6, 6.0
    cell_w, cell_h = 11.0, 2.5
    x_part, x_dot, x_label = 1.0, 8.0, 9.8
    x_cells = [42.0 + i * (cell_w + 1.2) for i in range(len(TEMPERATURES))]
    x_atoms, x_ps = x_cells[-1] + cell_w + 7.0, x_cells[-1] + cell_w + 18.0
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
    ax.text(x_part, head, "Part", ha="left", va="bottom", fontsize=style.SMALL)
    ax.text(x_label, head, "Potential or composition", ha="left", va="bottom", fontsize=style.SMALL)

    for part in dict.fromkeys(parts):
        members = [position[sid] for sid, p in zip(series_ids, parts) if p == part]
        top, bottom = members[0] - pitch / 2.0 + 0.3, members[-1] + pitch / 2.0 - 0.3
        ax.plot([x_part + 4.0, x_part + 4.0], [top, bottom], color="black", lw=0.5)
        ax.text(x_part + 1.6, (top + bottom) / 2.0, part, ha="center", va="center", fontsize=style.FONT_SIZE,
                fontweight="bold")

    rows = []
    for sid in series_ids:
        y = position[sid]
        group = runs[runs["series"] == sid]
        first = group.iloc[0]
        ax.plot(x_dot, y, marker="o", markersize=2.8, markerfacecolor=style.MODEL_COLORS[first["model"]],
                markeredgecolor="none", linestyle="none")
        ax.text(x_label, y, _row_label(first["label"]), ha="left", va="center", fontsize=style.SMALL)
        ax.text(x_atoms, y, str(first["n_atoms"]), ha="center", va="center", fontsize=style.SMALL)
        ax.text(x_ps, y, str(first["production_ps"]), ha="center", va="center", fontsize=style.SMALL)
        for x, temperature in zip(x_cells, TEMPERATURES):
            run = group[group["temperature_K"] == temperature].iloc[0]
            published = bool(run["published_july_2026"])
            ax.add_patch(Rectangle((x, y - cell_h / 2.0), cell_w, cell_h,
                                   facecolor=PUBLISHED_FACE if published else "white",
                                   edgecolor=PUBLISHED_FACE, linewidth=0.6))
            if run["run_log_survives"]:
                ax.text(x + cell_w / 2.0, y, "log", ha="center", va="center", fontsize=style.SMALL,
                        color="white" if published else "black")
            rows.append({**run.to_dict(), "row_y_mm": y, "cell_x_mm": x})

    # legend to the right of the matrix
    lx = x_ps + 12.0
    published = runs["published_july_2026"]
    n_pub, n_log = int(published.sum()), int(runs["run_log_survives"].sum())
    classes = {column: "/".join(sorted(set(runs.loc[published, column])))
               for column in ("pymatgen_class", "kinisi_D_class", "kinisi_std_class")}
    entries = [
        (PUBLISHED_FACE, "", f"Published in July 2026 and reproduced here ({n_pub} runs): pymatgen\n"
                             f"class {classes['pymatgen_class']}, kinisi D and its uncertainty class "
                             f"{classes['kinisi_D_class']} (Ext. Data Fig. 2)"),
        ("white", "", f"Analysed here for the first time ({len(runs) - n_pub} runs)"),
        (None, "log", f"July run log survives ({n_log} runs): thermostat statistics recomputed\n"
                      f"from the trajectory are checked against it; the other {len(runs) - n_log} runs\n"
                      "have statistics from the trajectory only"),
    ]
    ly = header + pitch / 2.0
    for face, text, label in entries:
        if face is not None:
            ax.add_patch(Rectangle((lx, ly - cell_h / 2.0), 7.0, cell_h, facecolor=face, edgecolor=PUBLISHED_FACE,
                                   linewidth=0.6))
        if text:
            ax.text(lx + 3.5, ly, text, ha="center", va="center", fontsize=style.SMALL)
        ax.text(lx + 9.0, ly - cell_h / 2.0 + 0.05, label, ha="left", va="top", fontsize=style.SMALL,
                linespacing=1.15)
        ly += 3.6 + 2.3 * label.count("\n")
    ly += 1.4
    for model in ("foundation", "mattersim", "ftG", "ftK"):
        ax.plot(lx + 3.5, ly, marker="o", markersize=2.8, markerfacecolor=style.MODEL_COLORS[model],
                markeredgecolor="none", linestyle="none")
        ax.text(lx + 9.0, ly, style.MODEL_LABELS[model], ha="left", va="center", fontsize=style.SMALL)
        ly += 2.8
    return style.save_source(pd.DataFrame(rows), NAME, "c")


# ----------------------------------------------------------------------------------------------- figure


def make() -> list[Path]:
    cell, configs, qe, reproduction, thermostat, aimd = _load()
    runs = _runs(reproduction, thermostat)
    fig, L = common.new_figure(WIDTH, HEIGHT)

    box_a = (3.0, 11.0, 54.0, 56.0)
    ax_a = L.axes(*box_a)
    sources = [_panel_a(ax_a, cell, L, box_a)]
    b_left, b_top, b_width, b_height = 64.0, 4.0, 118.0, 86.0
    ax_b = L.axes(b_left, b_top, b_width, b_height)
    sources.append(_panel_b(ax_b, configs, qe, aimd, b_width, b_height))
    c_top, c_height = 97.0, 49.0
    ax_c = L.axes(3.0, c_top, 179.0, c_height)
    sources.append(_panel_c(ax_c, runs, 179.0, c_height))

    L.letter(ax_a, "a", 3.0, 6.0)
    L.letter(ax_b, "b", 61.0, 6.0)
    L.letter(ax_c, "c", 3.0, c_top - 1.0)
    return style.save(fig, "fig1_design", "main") + sources


def _caption() -> str:
    cell, configs, qe, reproduction, thermostat, aimd = _load()
    runs = _runs(reproduction, thermostat)
    atoms, bonds = _structure(cell)
    sites = _anion_sites(cell)
    ref = _aimd_facts(aimd)
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
    cl = [s for s in SERIES if s["part"] == "D"]
    shared = ", ".join(str(t) for t in ref["shared"][:-1]) + f" and {ref['shared'][-1]} K"
    aimd_t = ", ".join(str(t) for t in ref["temperatures"][:-1]) + f" and {ref['temperatures'][-1]} K"
    volume_ratio = (ref["a"] / cell["a"]) ** 3
    md_t = ", ".join(str(t) for t in TEMPERATURES[:-1]) + f" and {TEMPERATURES[-1]} K"
    return (
        "Fig. 1 | Study design, model system and data. "
        f"**a**, The 52-atom cell of every Li6PS5Cl run (Li{counts['Li']}P{counts['P']}S{counts['S']}Cl{counts['Cl']}, "
        f"cubic, a = {cell['a']:.4f} A; the lowest-Ewald-energy S/Cl ordering of the Materials Project cell), in an "
        f"oblique orthographic projection, farther atoms lighter. The S/Cl arrangement is fully ordered, Cl on "
        f"{sites['Cl']} and the free S on {sites['free S']}; the AIMD reference simulates the same arrangement. "
        f"The origin is shifted by {ORIGIN_SHIFT:g} along each axis so that the {counts['P']} PS4 tetrahedra "
        f"(P-S < {BOND_CUTOFF_A:g} A, minimum image; {len(bonds)} bonds) lie inside the drawn cell. The 416-atom "
        f"cell is its 2x2x2 supercell. "
        f"**b**, Study design (protocol amendment A3). Every model is compared with DFT. Part A (Fig. 2): MD with "
        f"two universal potentials (MACE-MP-0 small, the foundation model, and MatterSim) at {md_t}, compared "
        f"with AIMD of the same ordered cell through the Li diffusion coefficient D at {shared}, without "
        f"extrapolation. The AIMD reference (Deng et al., Chem. Mater. 29, 281 (2017); Extended Data Fig. 9) has "
        f"{ref['n_atoms']} atoms, a = {ref['a']:.2f} A (a {100 * (1 - volume_ratio):.0f} % smaller volume than "
        f"here), a {ref['time_step_fs']:g}-fs time step and {ref['ps']:.0f} ps at each of {aimd_t}. Measured "
        f"conductivities are of S/Cl-disordered polycrystals and are context only. "
        f"Part B: {len(configs)} configurations (frames {frames[0]}-{frames[-1]}) of the 52-atom, 50-ps MACE-MP-0 "
        f"runs were labelled with Quantum ESPRESSO (PBE) at the Gamma point ({k_points['gamma']} k-point) and on a "
        f"2x2x2 mesh ({k_points['k222']} k-points); MACE-MP-0 was fine-tuned once on each label set, and the "
        f"fine-tuned models' forces and energies are compared with these labels (Fig. 3). "
        f"Part C: MD with the fine-tuned models (the Gamma fine-tune at stage two, the 2x2x2 fine-tune at stage "
        f"one) under the protocol of the 416-atom, 200-ps foundation runs, compared with those runs and with the "
        f"AIMD through Li D at {shared} (Fig. 4). "
        f"Part D: the Cl-excess series Li6-xPS5-xCl1+x (x = 0-0.75; {cl[-1]['n_atoms']}-{cl[0]['n_atoms']} atoms, "
        f"{cl[0]['production_ps']} ps) with MACE-MP-0 (Extended Data Fig. 8). Measured conductivities are never "
        f"shown next to a fine-tuned model. "
        f"**c**, The {len(runs)} MD runs, one per series and temperature. Dark cells: the {n_pub} runs published in "
        f"July 2026, re-analysed here from the same trajectories (pymatgen class {classes['pymatgen_class']} for "
        f"every run, kinisi class {classes['kinisi_D_class']}; Extended Data Fig. 2). White cells: the "
        f"{len(runs) - n_pub} runs analysed here for the first time ({'; '.join(new)}). 'log': the July run log "
        f"survives ({n_log} runs) and the thermostat statistics recomputed from the trajectory were checked against "
        f"it; for the other {len(runs) - n_log} runs they come from the trajectory only. Dots, potential (colours as "
        f"in Figs. 2-4). "
        + common.sources([STRUCTURE, CONFIGS, QE_RUNS, AIMD, REPRODUCTION, THERMOSTAT])
    )


CAPTION = _caption()
