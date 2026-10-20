"""Closed-form references used to validate the FEA."""
from __future__ import annotations

import numpy as np


def kt_infinite_orthotropic(A: np.ndarray) -> float:
    """Lekhnitskii's stress concentration factor at a circular hole in an
    infinite orthotropic plate under uniaxial load along x:

        Kt_inf = 1 + sqrt( 2 (sqrt(Ex/Ey) - nu_xy) + Ex/Gxy )

    Only ratios of moduli are needed, so they come straight from a = A^-1.
    """
    a = np.linalg.inv(A)
    Ex_Ey = a[1, 1] / a[0, 0]
    nuxy = -a[0, 1] / a[0, 0]
    Ex_Gxy = a[2, 2] / a[0, 0]
    return 1.0 + np.sqrt(2.0 * (np.sqrt(Ex_Ey) - nuxy) + Ex_Gxy)


def tan_finite_width_factor(kt_inf: float, two_r_over_w: float) -> float:
    """Tan's (1988) finite-width correction for an orthotropic plate with a
    central circular hole. Returns Kt_finite / Kt_inf (gross-section basis).
    """
    lam = two_r_over_w
    iso = 3 * (1 - lam) / (2 + (1 - lam) ** 3)
    M2 = (np.sqrt(1 - 8 * (iso - 1)) - 1) / (2 * lam**2)
    lM = lam * np.sqrt(M2)
    ratio = iso + 0.5 * lM**6 * (kt_inf - 3) * (1 - lM**2)   # Kt_inf / Kt
    return 1.0 / ratio
