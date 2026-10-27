"""Figures: the registry, complete exports at Nature column widths, one source table per panel, captions,
the shared encoding, and a deterministic rebuild."""
from __future__ import annotations

import re
import struct

import pytest

from mlipft import paths
from mlipft.figures import common, registry, style

DPI = 600
WIDTH_MM = 183.0  # every figure spans the double column
FIGURE_CODE = paths.REPO / "src" / "mlipft" / "figures"
EXTENSIONS = ("pdf", "svg", "png")

# Phase-2 switches. While a switch is False, a violation of its check is reported as an expected failure
# (xfail) that lists what is left; phase 2 sets every switch to True.
STRICT_PANELS = True  # every figure module defines PANELS
STRICT_FILES = True  # figures/main, figures/extended and figures/source_data hold exactly the registry's files
STRICT_SOURCE = True  # no hard-coded figure number, hex colour or OKABE_ITO in a figure module


def _png_size(path):
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def _pending(problems: list[str], strict: bool, what: str) -> None:
    if problems and not strict:
        pytest.xfail(f"{what} (switched on in phase 2): " + "; ".join(problems[:8])
                     + (f"; ... {len(problems) - 8} more" if len(problems) > 8 else ""))
    assert not problems, f"{what}: " + "; ".join(problems)


def _figure_modules() -> list:
    return sorted(path for path in FIGURE_CODE.glob("*.py") if re.match(r"(fig\d|ed\d\d)_", path.name))


def test_registry_numbers_labels_and_headings():
    keys = [entry.key for entry in registry.ENTRIES]
    assert len(set(keys)) == len(keys) == 13
    for position, entry in enumerate(registry.MAIN, start=1):
        assert registry.number(entry.key) == f"fig{position}"
    for position, entry in enumerate(registry.EXTENDED, start=1):
        assert registry.number(entry.key) == f"ed{position:02d}"
    for entry in registry.ENTRIES:
        assert entry.module.startswith(registry.number(entry.key) + "_"), entry.module
        assert registry.entry(registry.number(entry.key)) is entry
        assert registry.FIGURES[registry.number(entry.key)] == (entry.module, entry.title)
    assert {path.stem for path in _figure_modules()} == {entry.module for entry in registry.ENTRIES}
    assert registry.label("forces", "b") == "Fig. 2b"
    assert registry.label("aimd") == "Extended Data Fig. 8"
    assert registry.label("aimd", "c", short=True) == "Ext. Data Fig. 8c"
    assert registry.heading("forces") == "Fig. 2 | Forces against DFT. "
    assert registry.heading("clexcess") == "Extended Data Fig. 9 | Cl-excess series under MACE-MP-0. "
    assert registry.target("diffusion") == ("fig3_diffusion", "main")
    assert registry.target("aimd") == ("ed08_aimd", "extended")


@pytest.mark.parametrize("key", [entry.key for entry in registry.ENTRIES])
def test_exports_sizes_source_data_and_caption(key):
    entry, number = registry.entry(key), registry.number(key)
    module = registry.module(key)
    assert module.KEY == key and module.NAME == number
    name, folder = registry.target(key)
    for extension in EXTENSIONS:
        assert (paths.FIGURES / folder / f"{name}.{extension}").is_file(), f"{name}.{extension}"
    width_px, height_px = _png_size(paths.FIGURES / folder / f"{name}.png")
    assert abs(width_px / DPI * 25.4 - WIDTH_MM) < 0.3
    assert height_px / DPI * 25.4 <= 247.3
    letters = sorted(match.group(1) for path in paths.FIGURES.glob(f"source_data/{number}_*.csv")
                     if (match := re.fullmatch(rf"{number}_([a-z])\.csv", path.name)))
    assert letters, f"no source data for {number}"
    caption = module.CAPTION
    assert caption.isascii()
    assert caption.startswith(registry.heading(key)), caption[:80]
    assert "Sources:" in caption
    panels = getattr(module, "PANELS", None)
    _pending([] if panels is not None else [f"{entry.module} defines no PANELS"], STRICT_PANELS, "panel letters")
    assert len(set(panels)) == len(panels) and set(panels) == set(letters), (panels, letters)
    positions = [caption.find(f"**{letter}**") for letter in panels]
    assert -1 not in positions, f"caption of {entry.module} lacks a bold panel letter: {positions}"
    assert positions == sorted(positions), f"panel letters out of order in the caption of {entry.module}"


def test_exported_files_match_the_registry():
    expected = {f"{folder}/{name}.{extension}" for name, folder in map(registry.target, registry.FIGURES)
                for extension in EXTENSIONS}
    actual = {path.relative_to(paths.FIGURES).as_posix() for folder in ("main", "extended")
              for path in (paths.FIGURES / folder).glob("*") if path.is_file()}
    problems = [f"missing {name}" for name in sorted(expected - actual)]
    problems += [f"not in the registry: {name}" for name in sorted(actual - expected)]
    numbers = set(registry.FIGURES)
    problems += [f"source data of no figure: {path.name}" for path in sorted(paths.FIGURES.glob("source_data/*.csv"))
                 if path.name.split("_")[0] not in numbers]
    for key in registry.FIGURES:
        panels = getattr(registry.module(key), "PANELS", None)
        if panels is not None:
            problems += [f"source data of no panel: {path.name}"
                         for path in sorted(paths.FIGURES.glob(f"source_data/{key}_*.csv"))
                         if path.stem.split("_", 1)[1] not in panels]
    _pending(problems, STRICT_FILES, "exported files")


def test_figure_numbers_and_colours_come_from_the_registry_and_style():
    number = re.compile(r"\bFigs?\.\s*\d|Data Fig\. \d")
    colour = re.compile(r"""["']#[0-9A-Fa-f]{3,8}["']""")
    shared, pending = [], []
    for path in sorted(FIGURE_CODE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        target = pending if path in _figure_modules() else shared
        if path.name != "registry.py":
            target += [f"{path.name}:{text.count(chr(10), 0, m.start()) + 1} {m.group(0)!r}"
                       for m in number.finditer(text)]
        if path.name != "style.py":
            target += [f"{path.name}:{text.count(chr(10), 0, m.start()) + 1} {m.group(0)}"
                       for m in colour.finditer(text)]
            target += [f"{path.name}: OKABE_ITO" for _ in range(text.count("OKABE_ITO"))]
    assert not shared, shared
    _pending(pending, STRICT_SOURCE, "hard-coded figure numbers or colours")


def test_md_models_agree_with_the_encoding():
    cross = common.read(paths.RESULTS / "cross_evaluation.csv")
    md = common.md_models(cross)
    assert md["foundation"] == "foundation"
    for family in ("ftG", "ftK"):
        assert md[family] == f"{family}_{style.MD_STAGE[family]}"
        assert style.MODEL_COLORS[family] == style.MODEL_COLORS[md[family]]
        assert common.filled(family) == common.filled(md[family])
    assert common.primary_models(cross) == ["foundation", md["ftG"], md["ftK"]]


def test_encoding_helpers():
    assert common.model_style("ftG")["markerfacecolor"] == style.NAVY
    assert common.model_style("ftK")["markerfacecolor"] == style.WHITE
    assert common.model_style("ftK")["markeredgecolor"] == style.BLUE
    assert common.model_style("foundation", "cell_52")["marker"] == "^"
    assert common.model_style("mattersim", "clexcess", zorder=4)["zorder"] == 4
    assert common.filled("mattersim") and common.filled("ftG_s2") and not common.filled("ftG_s1")
    scatter = common.model_scatter("ftG_s1", size=3.0)
    assert scatter["s"] == 9.0 and scatter["facecolors"] == style.WHITE and "markersize" not in scatter
    assert common.aimd_style()["marker"] == style.AIMD_MARKER == "s"
    assert common.aimd_style()["markerfacecolor"] == style.AIMD_COLOR == style.BLACK
    assert common.context_style()["marker"] == "D" and common.context_style()["markerfacecolor"] == style.LIGHT_GRAY
    assert common.handle(common.aimd_style(), "AIMD").get_label() == "AIMD"


@pytest.mark.parametrize(("value", "digits", "text"), [(0.26, 3, "0.260"), (2.4449, 3, "2.44"), (213.4, 3, "213"),
                                                      (5.554, 3, "5.55"), (9.996, 3, "10.0"), (-0.2219, 2, "-0.22"),
                                                      (21345.0, 3, "21300"), (0.0, 3, "0")])
def test_sig(value, digits, text):
    assert common.sig(value, digits) == text


def test_sci():
    assert common.sci(1.7e-5) == "1.7e-5"
    assert common.sci(1.734e-5, 3) == "1.73e-5"
    assert common.sci(9.96e-6) == "1.0e-5"
    assert common.sci(3.0e8, 1) == "3e8"


def test_full_build_cleans_old_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "FIGURES", tmp_path)
    for relative in ("main/fig9_old.png", "extended/ed99_old.svg", "source_data/fig9_a.csv"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("stale")
    (tmp_path / "source_data" / "README.txt").write_text("kept")
    removed = registry.clean()
    assert len(removed) == 3
    assert not list((tmp_path / "main").iterdir()) and not list((tmp_path / "extended").iterdir())
    assert (tmp_path / "source_data" / "README.txt").exists()


def test_rebuild_is_deterministic(tmp_path, monkeypatch):
    builds = []
    for run in ("first", "second"):
        monkeypatch.setattr(style, "FIGURES", tmp_path / run)
        registry.build(["training"])
        name, folder = registry.target("training")
        builds.append({extension: (tmp_path / run / folder / f"{name}.{extension}").read_bytes()
                       for extension in EXTENSIONS})
    for extension in EXTENSIONS:
        assert builds[0][extension] == builds[1][extension], extension
