"""Repository checks: no CJK text, no private identifiers, size limits."""
from __future__ import annotations

from mlipft import hygiene


def test_repository_is_clean():
    assert hygiene.check() == []


def test_checks_detect_problems():
    owner = "E" + "1582271"
    text = "contact " + "e" + "1234567\n" + chr(0x4E2D) + chr(0x6587) + "\n" + "/n" + "fs/x\n"
    names = {name for _, name in hygiene.scan_text(text)}
    assert {"student number", "CJK text", "cluster path"} <= names
    assert hygiene.scan_text(f"https://github.com/{owner}-dotcom/MLIP-fine-tune") == []
    assert hygiene.scan_text(f"https://{owner.lower()}-dotcom.github.io/x") == []
    assert hygiene.scan_text(f"{owner}-dotcom") != []


def test_sync_conflict_copies_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "fig1_design 2.png").write_bytes(b"x")
    (tmp_path / "figures" / "fig1_design.png").write_bytes(b"x")
    assert [p.name for p in hygiene.sync_copies()] == ["fig1_design 2.png"]
    assert any("file-sync conflict copy" in problem for problem in hygiene.check(files=[]))
