"""Nernst-Einstein conductivity and the weighted Arrhenius fit used for every reported number.

``arrhenius_fit`` is the function that produced the published values, unchanged apart from
names: ln(sigma T) is fitted against 1/T, weighted by the kinisi uncertainty of each
temperature (Mo group convention), and the unscaled fit covariance is propagated to an
activation-energy error and a 1-sigma interval of sigma(300 K).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..constants import KB_EV_K, KB_J_K, Q_C, T_REF_K


def nernst_einstein_mS_cm(D_cm2_s: float, n_per_cm3: float, T: float, z: int = 1) -> float:
    """Tracer conductivity in mS/cm (no Haven ratio)."""
    return n_per_cm3 * (z * Q_C) ** 2 / (KB_J_K * T) * D_cm2_s * 1e3


def arrhenius_fit(temps: Sequence[float], sigmas_mS_cm: Sequence[float],
                  sigma_errs_mS_cm: Sequence[float] | None = None) -> dict:
    """Fit ln(sigma T) = slope / T + intercept.

    The fit is weighted by ``sigma_errs_mS_cm`` (error of ln(sigma T) = error / sigma) when every
    kept point has a positive error and there are at least three points; otherwise it is an
    unweighted fit without an interval. Points with non-positive sigma are dropped.
    """
    T = np.asarray(temps, float)
    s = np.asarray(sigmas_mS_cm, float)
    ok = np.isfinite(s) & (s > 0) & np.isfinite(T)
    T, s = T[ok], s[ok]
    if len(T) < 2:
        return {"Ea_eV": float("nan"), "sigma300_mS_cm": float("nan"), "R2": float("nan"), "n_points": int(len(T))}
    x = 1.0 / T
    y = np.log(s * T)

    y_err = None
    if sigma_errs_mS_cm is not None:
        se = np.asarray(sigma_errs_mS_cm, float)[ok]
        if np.all(np.isfinite(se)) and np.all(se > 0):
            y_err = se / s

    cov = None
    if y_err is not None and len(T) >= 3:
        (slope, intercept), cov = np.polyfit(x, y, 1, w=1.0 / y_err, cov="unscaled")
    else:
        slope, intercept = np.polyfit(x, y, 1)

    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "Ea_eV": float(-slope * KB_EV_K),
        "sigma300_mS_cm": float(math.exp(slope / T_REF_K + intercept) / T_REF_K),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(len(T)),
        "slope": float(slope),
        "intercept": float(intercept),
        "weighted": cov is not None,
    }
    if cov is not None:
        slope_var, intercept_var, covariance = cov[0, 0], cov[1, 1], cov[0, 1]
        out["slope_err"] = float(np.sqrt(slope_var))
        out["intercept_err"] = float(np.sqrt(intercept_var))
        out["Ea_err_eV"] = float(np.sqrt(slope_var) * KB_EV_K)
        x0 = 1.0 / T_REF_K
        y300_var = x0 * x0 * slope_var + intercept_var + 2.0 * x0 * covariance
        y300, y300_sig = slope * x0 + intercept, float(np.sqrt(max(y300_var, 0.0)))
        out["sigma300_min_mS_cm"] = float(math.exp(y300 - y300_sig) / T_REF_K)
        out["sigma300_max_mS_cm"] = float(math.exp(y300 + y300_sig) / T_REF_K)
    return out


def sigma_model(T, slope: float, intercept: float):
    """sigma(T) in mS/cm of a fitted line."""
    T = np.asarray(T, float)
    return np.exp(slope / T + intercept) / T
