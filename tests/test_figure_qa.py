"""The figure QA checks fire on small synthetic figures and pass a clean one."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest

from mlipft import paths
from mlipft.figures import common, qa, registry, style


def _errors(findings, check=None, kind=None):
    return [f for f in findings if f.severity == "error" and (check is None or f.check == check)
            and (kind is None or f.kind == kind)]


def _clean():
    """Two framed panels in one row, aligned, with labels, a legend and data well inside the frames."""
    fig, L = common.new_figure(183.0, 60.0)
    left = L.axes(15.0, 8.0, 70.0, 38.0, row="top", col="left")
    right = L.axes(105.0, 8.0, 70.0, 38.0, row="top")
    for ax in (left, right):
        ax.plot([1, 2, 3], [1, 2, 3], **common.model_style("ftG", linestyle="-", label="Gamma"))
        ax.plot([1, 2, 3], [2.5, 3.0, 3.5], **common.aimd_style(label="AIMD"))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 8)
        ax.set_xlabel("x (unit)")
        ax.set_ylabel("y (unit)")
        ax.legend(loc="upper left")
    L.letter(left, "a", 4.0, 6.0)
    L.letter(right, "b", 95.0, 6.0)
    return fig, L


def test_clean_figure_passes():
    fig, L = _clean()
    findings = qa.check(L, "clean")
    plt.close(fig)
    assert _errors(findings) == [], [str(f) for f in findings]


def test_overlapping_texts():
    fig, L = _clean()
    ax = L.fig.axes[0]
    ax.text(2.5, 6.0, "first label")
    ax.text(2.6, 6.05, "second label")
    findings = qa.overlaps(L, name="texts")
    plt.close(fig)
    hits = _errors(findings, "overlaps", "text-text")
    assert len(hits) == 1 and "first label" in hits[0].subject and "second label" in hits[0].subject
    assert hits[0].box is not None and hits[0].box[0] < hits[0].box[2]


def test_text_over_data_and_its_exemption():
    fig, L = _clean()
    ax = L.fig.axes[0]
    label = ax.text(2.0, 2.0, "on the marker", ha="center", va="center")
    assert _errors(qa.overlaps(L), "overlaps", "text-data")
    common.allow_overlap(label)
    assert not _errors(qa.overlaps(L), "overlaps", "text-data")
    plt.close(fig)


def test_text_across_a_frame_and_into_another_axes():
    fig, L = _clean()
    left = L.fig.axes[0]
    left.text(3.9, 7.0, "beyond the right frame and well into the next panel", ha="left")  # crosses its own frame and reaches the next panel
    findings = qa.overlaps(L)
    plt.close(fig)
    assert _errors(findings, "overlaps", "text-frame")
    assert _errors(findings, "overlaps", "text-axes")


def test_frameless_schematic_is_exempt():
    fig, L = _clean()
    schematic = L.axes(20.0, 50.0, 40.0, 8.0, frame="none")
    schematic.set_xlim(0, 1)
    schematic.set_ylim(0, 1)
    schematic.scatter([0.5], [0.5], s=40)
    schematic.text(0.5, 0.5, "box text", ha="center", va="center")
    findings = qa.check(L)
    plt.close(fig)
    assert _errors(findings) == [], [str(f) for f in findings]


def test_misaligned_row_and_missing_tags():
    fig, L = common.new_figure(183.0, 60.0)
    L.axes(15.0, 8.0, 70.0, 38.0, row="top", col="c")
    L.axes(105.0, 8.5, 70.0, 37.5, row="top")
    L.axes(15.0, 50.0, 70.0, 5.0)  # no row tag
    findings = qa.alignment(L, name="rows")
    plt.close(fig)
    kinds = sorted(f.kind for f in findings)
    assert kinds == ["missing-row", "row-top"], kinds
    assert "8.000" in findings[0].detail or "8.000" in findings[1].detail


def test_column_alignment():
    fig, L = common.new_figure(183.0, 100.0)
    L.axes(15.0, 8.0, 70.0, 30.0, row="1", col="left")
    L.axes(15.3, 50.0, 70.0, 30.0, row="2", col="left")
    findings = qa.alignment(L)
    plt.close(fig)
    assert [f.kind for f in findings] == ["col-left"]
    table = qa.alignment_table(L)
    assert [row["left_mm"] for row in table] == pytest.approx([15.0, 15.3], abs=1e-3)


def test_open_frame_top_ticks_and_secondary_axis():
    fig, L = _clean()
    left, right = L.fig.axes[:2]
    left.spines["top"].set_visible(False)
    right.tick_params(top=True, labeltop=True)
    left.secondary_xaxis("top")
    findings = qa.frames(L)
    plt.close(fig)
    kinds = {f.kind for f in findings}
    assert {"open-frame", "top-ticks", "top-labels", "secondary-axis"} <= kinds, kinds


def test_clipped_marker_and_point_outside_the_limits():
    fig, L = _clean()
    ax = L.fig.axes[0]
    ax.plot([4.0], [4.0], **common.model_style("foundation"))  # on the right frame line
    ax.plot([5.0], [4.0], **common.model_style("mattersim"))  # beyond the x limit
    findings = qa.clipped(L)
    plt.close(fig)
    assert len(_errors(findings, "clipped", "marker-on-frame")) == 1
    assert len(_errors(findings, "clipped", "outside-limits")) == 1


def test_font_sizes_and_figure_size():
    fig, L = _clean()
    L.fig.axes[0].text(1.0, 7.0, "too large", fontsize=9)
    findings = qa.fonts(L)
    plt.close(fig)
    assert [f.subject for f in _errors(findings, "fonts", "size")] == ["'too large'"]
    style.apply_style()
    odd = plt.figure(figsize=(150 * style.MM, 260 * style.MM))
    assert {f.kind for f in qa.size(odd)} == {"width", "height"}
    plt.close(odd)


def test_gate_modes(monkeypatch, capsys):
    fig, L = _clean()
    L.fig.axes[0].text(1.0, 7.0, "too large", fontsize=9)
    monkeypatch.delenv("MLIPFT_QA", raising=False)
    with pytest.raises(qa.FigureQAError, match="too large"):
        qa.gate(L, name="gated")  # strict by default
    monkeypatch.setenv("MLIPFT_QA", "warn")
    findings = qa.gate(L, name="gated")
    assert findings and L.findings == findings and L.name == "gated"
    assert "QA gated:" in capsys.readouterr().err
    with pytest.raises(qa.FigureQAError, match="too large"):
        qa.gate(L, strict=True, name="gated")
    monkeypatch.setenv("MLIPFT_QA", "strict")
    with pytest.raises(qa.FigureQAError):
        qa.gate(L, name="gated")
    with qa.session(strict=False):
        qa.gate(L, name="gated")  # the session overrides the environment
    monkeypatch.setenv("MLIPFT_QA", "off")
    assert qa.gate(L, name="gated") == []
    plt.close(fig)


def test_layout_save_writes_then_raises_in_strict_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "FIGURES", tmp_path / "figures")
    fig, L = _clean()
    L.fig.axes[0].text(1.0, 7.0, "too large", fontsize=9)
    with pytest.raises(qa.FigureQAError):
        L.save("synthetic", "main", strict=True)
    assert (tmp_path / "figures" / "main" / "synthetic.png").is_file()


def test_crop_audit(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="outside the repository"):
        qa.outside_repository(paths.REPO / "figures" / "audit")
    monkeypatch.setattr(style, "FIGURES", tmp_path / "figures")
    fig, L = _clean()
    with qa.session(audit=tmp_path / "audit"):
        written = L.save("synthetic", "main")
    assert (tmp_path / "figures" / "main" / "synthetic.svg") in written
    folder = tmp_path / "audit" / "synthetic"
    index = json.loads((folder / "index.json").read_text())
    kinds = [crop["kind"] for crop in index["crops"]]
    assert kinds.count("frame-top-left") == 2 and kinds.count("legend") == 2
    assert all((folder / crop["file"]).is_file() for crop in index["crops"])
    assert index["contact_sheets"] and (folder / index["contact_sheets"][0]).is_file()
    assert [row["row"] for row in index["alignment"]] == ["top", "top"]
    from PIL import Image

    with Image.open(folder / index["crops"][0]["file"]) as crop:
        assert abs(crop.width - 10 / 25.4 * style.DPI) <= 2  # 10-mm window at 600 dpi


def test_temperature_labels_inside_the_frame():
    fig, L = _clean()
    ax = L.fig.axes[0]
    ax.set_xlim(0.9, 1.8)
    texts = common.temperature_labels(ax, temperatures=(600, 800, 1000, 1200))
    assert [t.get_text() for t in texts] == ["600 K", "800 K", "1000 K"]  # 1000/1200 is outside the limits
    plt.close(fig)


def test_registry_checks_figures_saved_without_layout_save(tmp_path, monkeypatch):
    """A module that still calls style.save is checked after the fact; strict failures are collected per figure."""
    monkeypatch.setattr(style, "FIGURES", tmp_path / "figures")

    class Legacy:
        def __init__(self, key, oversized):
            self.key, self.oversized = key, oversized

        def make(self):
            fig, L = _clean()
            if self.oversized:
                L.fig.axes[0].text(1.0, 7.0, "too large", fontsize=9)
            return style.save(fig, *registry.target(self.key))

    modules = {"forces": Legacy("forces", True), "training": Legacy("training", False),
               "aimd": Legacy("aimd", True)}
    monkeypatch.setattr(registry, "module", lambda name: modules[registry.entry(name).key])
    with pytest.raises(qa.FigureQAError) as error:
        registry.build(["forces", "training", "aimd"], strict=True, audit=tmp_path / "audit")
    assert {f.figure for f in error.value.findings if f.severity == "error"} == {"fig2_forces", "ed08_aimd"}
    assert (tmp_path / "figures" / "extended" / "ed08_aimd.png").is_file()  # later figures were still drawn
    assert (tmp_path / "audit" / "ed06_training" / "index.json").is_file()
    assert registry.build(["training"]) == [tmp_path / "figures" / "extended" / f"ed06_training.{x}"
                                            for x in ("pdf", "svg", "png")]
