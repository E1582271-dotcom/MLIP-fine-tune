"""The FFT MSD equals the direct average over time origins."""
from __future__ import annotations

import numpy as np

from mlipft.transport import msd


def test_fft_msd_equals_direct_average():
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(size=(60, 5, 3)), axis=0)
    fast = msd.msd_per_atom(positions)
    for lag in (0, 1, 7, 30, 59):
        direct = ((positions[lag:] - positions[:len(positions) - lag]) ** 2).sum(axis=2).mean(axis=0)
        assert np.allclose(fast[lag], direct)


def test_com_correction_removes_a_uniform_drift():
    positions = np.zeros((10, 4, 3))
    positions += np.arange(10)[:, None, None] * np.array([0.1, 0.0, 0.0])
    corrected = msd.com_corrected(positions, np.ones(4))
    assert np.allclose(corrected, 0.0)
