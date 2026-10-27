"""Li diffusion coefficient and conductivity of one trajectory, with the July 2026 call path.

* pymatgen-analysis-diffusion ``DiffusionAnalyzer.from_structures(..., smoothed="max")`` gives
  D and its Nernst-Einstein conductivity. These are the values that enter the Arrhenius fit.
* kinisi ``DiffusionAnalyzer.from_ase`` with the diffusion fit starting at 30 % of the trajectory
  gives a posterior of D; its mean and standard deviation are kept. In July kinisi ran with
  ``random_state=None``; here the seed is explicit (``protocol/reproduction_rule.md``).
* The kinisi conductivity is the Nernst-Einstein conversion of the kinisi D with the Li number
  density of the first frame.

ase, pymatgen, kinisi and scipp are imported inside the functions.
"""
from __future__ import annotations

import numpy as np

from ..constants import STEP_SKIP, TIME_STEP_FS
from .arrhenius import nernst_einstein_mS_cm


def li_density_per_cm3(atoms) -> float:
    """Li atoms per cm^3 of an ASE Atoms object."""
    n_li = int(np.sum(np.array(atoms.get_chemical_symbols()) == "Li"))
    return n_li / (atoms.get_volume() * 1e-24)


def pymatgen_structures(frames):
    """ASE frames -> pymatgen Structures (the input of the July pymatgen estimator)."""
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(frame) for frame in frames]


def pymatgen_diffusion(structures, temperature: float) -> dict:
    """D (cm^2/s) and conductivity (mS/cm) from pymatgen's DiffusionAnalyzer."""
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

    analyzer = DiffusionAnalyzer.from_structures(structures, specie="Li", temperature=temperature,
                                                 time_step=TIME_STEP_FS, step_skip=STEP_SKIP, smoothed="max")
    return {"D_cm2_s": float(analyzer.diffusivity), "sigma_mS_cm": float(analyzer.conductivity)}


def kinisi_analyzer(frames, time_step_fs: float = TIME_STEP_FS, step_skip: int = STEP_SKIP):
    """kinisi DiffusionAnalyzer of the Li atoms of ``frames`` (a list of ASE Atoms)."""
    import scipp as sc
    from kinisi.analyze import DiffusionAnalyzer

    return DiffusionAnalyzer.from_ase(frames, specie="Li", time_step=sc.scalar(float(time_step_fs), unit="fs"),
                                      step_skip=sc.scalar(step_skip, unit=sc.units.dimensionless),
                                      progress=False)


def kinisi_fit(analyzer, start_fs: float, seed: int) -> tuple[float, float]:
    """Mean and standard deviation (cm^2/s) of the kinisi posterior of D."""
    import scipp as sc

    analyzer.diffusion(sc.scalar(float(start_fs), unit="fs"), progress=False,
                       random_state=np.random.RandomState(seed))
    samples = analyzer.D.to(unit=sc.Unit("cm^2/s")).values
    return float(samples.mean()), float(samples.std())


def kinisi_msd(analyzer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time interval (ps), Li MSD (A^2) and its standard deviation (A^2) as computed by kinisi."""
    dt = analyzer.dt.to(unit="ps").values
    msd = analyzer.msd
    return np.asarray(dt, float), np.asarray(msd.values, float), np.sqrt(np.asarray(msd.variances, float))


def kinisi_sigma(D_cm2_s: float, D_std_cm2_s: float, n_per_cm3: float, temperature: float) -> tuple[float, float]:
    return (nernst_einstein_mS_cm(D_cm2_s, n_per_cm3, temperature),
            nernst_einstein_mS_cm(D_std_cm2_s, n_per_cm3, temperature))
