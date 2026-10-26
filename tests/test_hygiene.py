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
