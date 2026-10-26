"""The reproduction rule of ``protocol/reproduction_rule.md`` as code.

``tests/test_protocol.py`` checks that the thresholds below are the ones written in the protocol.
"""
from __future__ import annotations

import math

PYMATGEN_EXACT = 1e-9
PYMATGEN_REPRODUCED = 1e-6
KINISI_D_FLOOR = 0.02
KINISI_STD_FLOOR = 0.10
KINISI_SEED_FACTOR = 3.0
FIT_EXACT = 1e-6
FIT_REPRODUCED = 0.1  # fraction of the published uncertainty
FIT_CONSISTENT = 1.0
THERMOSTAT_T_TOLERANCE_K = {416: 0.2, 52: 1.0}


def relative(new: float, old: float) -> float:
    return abs(new - old) / abs(old)


def classify_pymatgen(new: float, old: float) -> str:
    difference = relative(new, old)
    if difference <= PYMATGEN_EXACT:
        return "R1"
    return "R2" if difference <= PYMATGEN_REPRODUCED else "R3"


def classify_kinisi(new: float, old: float, scatter: float, floor: float) -> str:
    return "R2" if relative(new, old) <= max(KINISI_SEED_FACTOR * scatter, floor) else "R3"


def classify_fit(new: dict, old: dict) -> tuple[str, dict]:
    """Class of a re-fitted series against the published fit, with the measured differences."""
    d_ln_sigma = math.log(new["sigma300_mS_cm"] / old["sigma300_mS_cm"])
    half_width = (math.log(old["sigma300_max_mS_cm"]) - math.log(old["sigma300_min_mS_cm"])) / 2.0
    d_ea = new["Ea_eV"] - old["Ea_eV"]
    measures = {"d_ln_sigma300": d_ln_sigma, "published_log_half_width": half_width, "d_Ea_eV": d_ea,
                "published_Ea_err_eV": old["Ea_err_eV"],
                "d_ln_sigma300_in_uncertainties": abs(d_ln_sigma) / half_width,
                "d_Ea_in_uncertainties": abs(d_ea) / old["Ea_err_eV"]}
    if (relative(new["sigma300_mS_cm"], old["sigma300_mS_cm"]) <= FIT_EXACT
            and relative(new["Ea_eV"], old["Ea_eV"]) <= FIT_EXACT):
        return "R1", measures
    worst = max(measures["d_ln_sigma300_in_uncertainties"], measures["d_Ea_in_uncertainties"])
    if worst <= FIT_REPRODUCED:
        return "R2", measures
    return ("R2-" if worst <= FIT_CONSISTENT else "R3"), measures


# The MD driver rounds its logged statistics to these numbers of decimals (generation code, run_nvt).
LOGGED_DECIMALS = {"mean_T": 1, "std_T": 1, "mean_E_per_atom": 4, "drift_meV_atom_ps": 3}


def thermostat_check(recomputed: dict, logged: dict, n_atoms: int) -> dict:
    """Item 7: mean T within the tolerance; mean energy and drift within one unit of the logged rounding
    (the logs hold one extra sample at step 0, so the last digit can differ by one)."""
    result = {}
    if logged.get("mean_T") not in (None, ""):
        tolerance = THERMOSTAT_T_TOLERANCE_K[416 if n_atoms > 100 else 52]
        result["d_mean_T_K"] = recomputed["mean_T_K"] - float(logged["mean_T"])
        result["mean_T_ok"] = abs(result["d_mean_T_K"]) <= tolerance
    for key, column in (("mean_E_per_atom", "mean_E_eV_atom"), ("drift_meV_atom_ps", "drift_meV_atom_ps")):
        if logged.get(key) not in (None, ""):
            difference = recomputed[column] - float(logged[key])
            result[f"d_{key}"] = difference
            result[f"{key}_ok"] = abs(difference) <= 10.0 ** -LOGGED_DECIMALS[key] * 1.0000001
    return result
