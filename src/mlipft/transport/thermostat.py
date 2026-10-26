"""Temperature, potential energy and pressure of every frame, and their run statistics.

The definitions follow the July MD driver (generation/md/md.py): instantaneous temperature from
the kinetic energy with 3N degrees of freedom; drift is the slope of a straight-line fit of the
potential energy per atom against time. Pressure is -tr(stress)/3 from the stored (virial) stress
plus the kinetic term N k_B T / V.
"""
from __future__ import annotations

import numpy as np

from ..constants import EV_A3_TO_GPA, FRAME_PS, KB_EV_K

BLOCK_FRAMES = 10  # 0.5 ps blocks for the stored time series


def frame_series(frames) -> dict[str, np.ndarray]:
    """Per-frame time (ps), temperature (K), potential energy per atom (eV) and pressure (GPa)."""
    n_atoms = len(frames[0])
    masses = frames[0].get_masses()
    volume = frames[0].get_volume()
    momenta = np.array([frame.get_momenta() for frame in frames])
    kinetic = 0.5 * (momenta ** 2 / masses[None, :, None]).sum(axis=(1, 2))
    temperature = 2.0 * kinetic / (3.0 * n_atoms * KB_EV_K)
    energy = np.array([frame.get_potential_energy() for frame in frames]) / n_atoms
    stress = np.array([frame.get_stress(voigt=True, include_ideal_gas=False) for frame in frames])
    pressure = (-stress[:, :3].mean(axis=1) + n_atoms * KB_EV_K * temperature / volume) * EV_A3_TO_GPA
    time = np.arange(len(frames)) * FRAME_PS
    return {"time_ps": time, "T_K": temperature, "E_eV_atom": energy, "P_GPa": pressure}


def statistics(series: dict[str, np.ndarray]) -> dict[str, float]:
    drift = np.polyfit(series["time_ps"], series["E_eV_atom"], 1)[0] * 1000.0
    return {"mean_T_K": float(series["T_K"].mean()), "std_T_K": float(series["T_K"].std()),
            "mean_E_eV_atom": float(series["E_eV_atom"].mean()), "drift_meV_atom_ps": float(drift),
            "mean_P_GPa": float(series["P_GPa"].mean()), "std_P_GPa": float(series["P_GPa"].std())}


def blocks(series: dict[str, np.ndarray], size: int = BLOCK_FRAMES) -> dict[str, np.ndarray]:
    """Block means of every series (the last incomplete block is dropped)."""
    n = len(series["time_ps"]) // size * size
    return {key: values[:n].reshape(-1, size).mean(axis=1) for key, values in series.items()}
