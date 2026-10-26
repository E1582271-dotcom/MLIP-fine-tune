"""Every figure, in order. Each module exposes ``make() -> list[Path]`` and ``CAPTION``."""
from __future__ import annotations

import importlib
from pathlib import Path

# name -> (module, title)
FIGURES = {
    "fig1": ("fig1_design", "Study design, model system and data"),
    "fig2": ("fig2_universal", "Universal potentials against experiment"),
    "fig3": ("fig3_dft", "Fine-tuned models against their DFT reference"),
    "fig4": ("fig4_transport", "What fine-tuning changes in Li transport"),
    "ed01": ("ed01_stability", "Temperature, energy and pressure of every MD run"),
    "ed02": ("ed02_reproduction", "Re-analysis against the values published in July 2026"),
    "ed03": ("ed03_arrhenius", "Arrhenius fits of every series"),
    "ed04": ("ed04_sampling", "Sampling and estimator checks"),
    "ed05": ("ed05_labels", "The two DFT label sets"),
    "ed06": ("ed06_training", "Training records of the two fine-tunes"),
    "ed07": ("ed07_models", "Model identity and the full cross-evaluation"),
    "ed08": ("ed08_clexcess", "Cl-excess series under MACE-MP-0"),
}


def module(name: str):
    return importlib.import_module(f"mlipft.figures.{FIGURES[name][0]}")


def build(only=None) -> list[Path]:
    written: list[Path] = []
    for name in FIGURES:
        if only and name not in only:
            continue
        written.extend(module(name).make())
    return written
