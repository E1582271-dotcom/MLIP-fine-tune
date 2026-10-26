"""Mean-squared displacement over all time origins (FFT algorithm), per species.

Used for the MSD curves and the framework-motion diagnostic; the diffusion coefficients come from
``estimators``. Positions must be unwrapped (the July trajectories store them unwrapped). The
displacement of the mass-weighted centre of the whole cell is removed frame by frame, so that a
slow drift of the cell under the Langevin thermostat does not enter the MSD.
"""
from __future__ import annotations

import numpy as np


def com_corrected(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Positions (frames, atoms, 3) with the centre-of-mass displacement from frame 0 removed."""
    com = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
    return positions - (com - com[0])[:, None, :]


def msd_per_atom(positions: np.ndarray) -> np.ndarray:
    """MSD of each atom at every lag 0..n-1, averaged over all time origins: (lags, atoms)."""
    n = positions.shape[0]
    squared = (positions ** 2).sum(axis=2)
    padded = np.vstack([squared, np.zeros((1, squared.shape[1]))])
    s1 = np.empty_like(squared)
    q = 2.0 * squared.sum(axis=0)
    for m in range(n):
        q = q - padded[m - 1] - padded[n - m]
        s1[m] = q / (n - m)
    spectrum = np.fft.rfft(positions, n=2 * n, axis=0)
    autocorrelation = np.fft.irfft(spectrum * spectrum.conj(), n=2 * n, axis=0)[:n].sum(axis=2)
    s2 = autocorrelation / (n - np.arange(n))[:, None]
    return s1 - 2.0 * s2


def log_lags(n_frames: int, count: int = 150) -> np.ndarray:
    """About ``count`` integer lags from 1 to n_frames - 1, evenly spaced in log."""
    return np.unique(np.round(np.geomspace(1, n_frames - 1, count)).astype(int))


def species_msd(positions: np.ndarray, symbols, masses: np.ndarray, lags: np.ndarray) -> dict[str, np.ndarray]:
    """MSD (A^2) per species at the given lags."""
    corrected = com_corrected(positions, masses)
    per_atom = msd_per_atom(corrected)
    symbols = np.asarray(symbols)
    return {species: per_atom[lags][:, symbols == species].mean(axis=1) for species in ("Li", "P", "S", "Cl")
            if np.any(symbols == species)}


def loglog_slope(time: np.ndarray, msd: np.ndarray, start: float) -> float:
    """Least-squares slope of log MSD against log time for time >= start."""
    keep = (time >= start) & (msd > 0)
    return float(np.polyfit(np.log(time[keep]), np.log(msd[keep]), 1)[0])
