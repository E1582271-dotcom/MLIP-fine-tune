"""Every figure, in order; figure numbers follow from the position in ``MAIN`` and ``EXTENDED``.

Figures are referred to by a stable key ("forces", "aimd"), never by a number written into the
code: captions start with ``heading(key)``, cross-references use ``label(key, "b")`` and source
data are written as ``<number(key)>_<panel>.csv``. Each figure module defines

* ``KEY``: its key here; ``NAME = registry.number(KEY)`` ("fig2", "ed08"), used for source data;
* ``PANELS``: its panel letters in caption order ("abcd"), one source-data CSV per letter;
* ``CAPTION``: starting with ``registry.heading(KEY)``, panel letters as ``**a**``, ending with
  ``common.sources(...)``;
* ``make() -> list[Path]``, saving through ``layout.save(*registry.target(KEY))``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import NamedTuple

from . import qa, style


class Entry(NamedTuple):
    key: str
    module: str  # module (and exported file) name, <number>_<word>
    title: str


MAIN = (
    Entry("design", "fig1_design", "Study design and model system"),
    Entry("forces", "fig2_forces", "Forces against DFT"),
    Entry("diffusion", "fig3_diffusion", "Li diffusion against AIMD"),
    Entry("finetune", "fig4_finetune", "What fine-tuning changes"),
)
EXTENDED = (
    Entry("stability", "ed01_stability", "Temperature, energy and pressure of every MD run"),
    Entry("reproduction", "ed02_reproduction", "Re-analysis against the values published in July 2026"),
    Entry("arrhenius", "ed03_arrhenius", "Arrhenius fits of every series"),
    Entry("sampling", "ed04_sampling", "Sampling and estimator checks"),
    Entry("labels", "ed05_labels", "The two DFT label sets"),
    Entry("training", "ed06_training", "Training records of the two fine-tunes"),
    Entry("models", "ed07_models", "Model identity and the full cross-evaluation"),
    Entry("aimd", "ed08_aimd", "The AIMD reference"),
    Entry("clexcess", "ed09_clexcess", "Cl-excess series under MACE-MP-0"),
)
ENTRIES = MAIN + EXTENDED
_BY_KEY = {entry.key: entry for entry in ENTRIES}


def entry(key: str) -> Entry:
    """The entry of a key ("forces") or of a figure number ("fig2")."""
    if key in _BY_KEY:
        return _BY_KEY[key]
    for candidate in ENTRIES:
        if number(candidate.key) == key:
            return candidate
    raise KeyError(f"unknown figure {key!r}; keys: {', '.join(_BY_KEY)}")


def is_main(key: str) -> bool:
    return entry(key) in MAIN


def index(key: str) -> int:
    """1-based figure number within its series."""
    item = entry(key)
    return (MAIN.index(item) if item in MAIN else EXTENDED.index(item)) + 1


def number(key: str) -> str:
    """"fig2" for the second main figure, "ed08" for the eighth Extended Data figure."""
    return f"fig{index(key)}" if is_main(key) else f"ed{index(key):02d}"


def folder(key: str) -> str:
    """Export folder under figures/: "main" or "extended"."""
    return "main" if is_main(key) else "extended"


def target(key: str) -> tuple[str, str]:
    """(file name, folder) for ``Layout.save``: ``layout.save(*registry.target(KEY))``."""
    return entry(key).module, folder(key)


def label(key: str, panels: str = "", short: bool = False) -> str:
    """"Fig. 2b", "Extended Data Fig. 8" or, with ``short``, "Ext. Data Fig. 8c" (main figures stay "Fig. 2")."""
    if is_main(key):
        return f"Fig. {index(key)}{panels}"
    return f"{'Ext. Data' if short else 'Extended Data'} Fig. {index(key)}{panels}"


def heading(key: str) -> str:
    """Caption opening: "Fig. 2 | Forces against DFT. " or "Extended Data Fig. 8 | The AIMD reference. "."""
    return f"{label(key)} | {entry(key).title}. "


# name -> (module, title), for cli, report and tests
FIGURES = {number(item.key): (item.module, item.title) for item in ENTRIES}


def module(name: str):
    """The figure module of a figure number ("fig2") or key ("forces")."""
    return importlib.import_module(f"mlipft.figures.{entry(name).module}")


def clean() -> list[Path]:
    """Delete every exported figure and source-data table (before a full build); returns the deleted paths."""
    removed = []
    for path in sorted((style.FIGURES / "main").glob("*")) + sorted((style.FIGURES / "extended").glob("*")) \
            + sorted((style.FIGURES / "source_data").glob("*.csv")):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def build(only=None, strict: bool | None = None, audit: Path | str | None = None) -> list[Path]:
    """Draw the figures in ``only`` (numbers or keys; every figure when None) and return the written paths.

    A full build first deletes all exports and source data, so files of renamed or removed panels do not
    survive. ``strict`` (default: environment variable MLIPFT_QA, else warn) makes QA errors fail the build
    after every selected figure was drawn; ``audit`` is a directory outside the repository that receives
    the crop audit of each figure (``qa.crop_audit``).
    """
    selected = ENTRIES if only is None else [entry(name) for name in only]
    if only is None:
        clean()
    written: list[Path] = []
    failures: list[qa.Finding] = []
    with qa.session(strict=strict, audit=audit) as state:
        for item in selected:
            state["layouts"].clear()
            try:
                files = module(item.key).make()
            except qa.FigureQAError as error:
                failures += error.findings
                continue
            written.extend(files)
            # A figure saved with style.save instead of Layout.save is checked here, after it was written.
            png = next((path for path in files if path.suffix == ".png" and path.stem == item.module), None)
            for layout in list(state["layouts"]):
                if layout.findings is None:
                    try:
                        qa.gate(layout, strict, name=item.module, png=png)
                    except qa.FigureQAError as error:
                        failures += error.findings
            state["layouts"].clear()
    if failures:
        raise qa.FigureQAError(failures)
    return written
