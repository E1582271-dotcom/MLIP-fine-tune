"""The code applies the rule that protocol/reproduction_rule.md states."""
from __future__ import annotations

import re

from mlipft import paths
from mlipft.transport import reproduction as R


def _protocol() -> str:
    return (paths.PROTOCOL / "reproduction_rule.md").read_text(encoding="utf-8")


def test_thresholds_match_the_protocol():
    text = _protocol()
    assert "relative difference <= 1e-9" in text and R.PYMATGEN_EXACT == 1e-9
    assert "<= 1e-6 | > 1e-6" in text and R.PYMATGEN_REPRODUCED == 1e-6
    assert "max(3 s_seed, 2 %)" in text and (R.KINISI_SEED_FACTOR, R.KINISI_D_FLOOR) == (3.0, 0.02)
    assert "max(3 s_seed, 10 %)" in text and R.KINISI_STD_FLOOR == 0.10
    assert "0.1 x published log half-width" in text and R.FIT_REPRODUCED == 0.1
    assert "beyond 1 x the published uncertainty" in text and R.FIT_CONSISTENT == 1.0
    assert "0.2 K for 416 atoms and <= 1 K for 52 atoms" in text
    assert R.THERMOSTAT_T_TOLERANCE_K == {416: 0.2, 52: 1.0}


def test_classifier_boundaries():
    assert R.classify_pymatgen(1.0 + 5e-10, 1.0) == "R1"
    assert R.classify_pymatgen(1.0 + 5e-7, 1.0) == "R2"
    assert R.classify_pymatgen(1.0 + 2e-6, 1.0) == "R3"
    assert R.classify_kinisi(1.019, 1.0, 0.001, R.KINISI_D_FLOOR) == "R2"
    assert R.classify_kinisi(1.03, 1.0, 0.001, R.KINISI_D_FLOOR) == "R3"
    assert R.classify_kinisi(1.03, 1.0, 0.011, R.KINISI_D_FLOOR) == "R2"
    old = {"sigma300_mS_cm": 1.0, "sigma300_min_mS_cm": 0.5, "sigma300_max_mS_cm": 2.0, "Ea_eV": 0.3,
           "Ea_err_eV": 0.01}
    half = 0.6931471805599453  # ln 2
    for scale, expected in ((0.05, "R2"), (0.5, "R2-"), (1.5, "R3")):
        new = {"sigma300_mS_cm": 2.718281828459045 ** (scale * half), "Ea_eV": 0.3}
        assert R.classify_fit(new, old)[0] == expected, scale


def test_amendments_are_dated():
    text = (paths.PROTOCOL / "amendments.md").read_text(encoding="utf-8")
    for heading in re.findall(r"^## (A\d+) \((\d{4}-\d{2}-\d{2})\)", text, re.M):
        assert heading[1] >= "2026-10-26"
    assert "Results seen before the change." in text
