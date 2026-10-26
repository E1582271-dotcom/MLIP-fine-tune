"""Figures: complete exports at Nature column widths, source data, captions, deterministic rebuild."""
from __future__ import annotations

import struct

import pytest

from mlipft import paths
from mlipft.figures import registry

DPI = 600
WIDTHS_MM = (89, 120, 183)


def _png_size(path):
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


@pytest.mark.parametrize("name", list(registry.FIGURES))
def test_exports_sizes_source_data_and_caption(name):
    module, _ = registry.FIGURES[name]
    folder = paths.FIGURES / ("main" if name.startswith("fig") else "extended")
    for extension in ("pdf", "svg", "png"):
        assert (folder / f"{module}.{extension}").is_file(), f"{module}.{extension}"
    width_px, height_px = _png_size(folder / f"{module}.png")
    assert any(abs(width_px / DPI * 25.4 - target) < 0.3 for target in WIDTHS_MM)
    assert height_px / DPI * 25.4 <= 247.3
    assert list(paths.FIGURES.glob(f"source_data/{name}_*.csv")), f"no source data for {name}"
    caption = registry.module(name).CAPTION
    assert caption.isascii() and "Sources:" in caption


def test_rebuild_is_deterministic(tmp_path, monkeypatch):
    from mlipft.figures import style

    builds = []
    for run in ("first", "second"):
        monkeypatch.setattr(style, "FIGURES", tmp_path / run)
        registry.build(["ed06"])
        builds.append((tmp_path / run / "extended" / "ed06_training.svg").read_bytes())
    assert builds[0] == builds[1]
