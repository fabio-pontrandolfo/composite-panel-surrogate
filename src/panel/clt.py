"""Classical laminate theory (CLT) and Tsai-Wu ply failure.

Units: MPa, mm, N. Strains are engineering strains (gamma_xy = 2 * eps_xy).

The laminate is assumed symmetric and balanced, loaded in-plane only, so the
B matrix is zero, A16 = A26 = 0 and every ply sees the mid-plane strain.
Under that assumption the stacking order does not matter, only the fraction of
plies at each angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlyMaterial:
    """Unidirectional carbon/epoxy ply (T300/5208-type values, Tsai & Hahn)."""

    E1: float = 181_000.0   # MPa, fibre direction
    E2: float = 10_300.0    # MPa, transverse
    G12: float = 7_170.0    # MPa, in-plane shear
    nu12: float = 0.28
    Xt: float = 1500.0      # MPa, strengths
    Xc: float = 1500.0
    Yt: float = 40.0
    Yc: float = 246.0
    S: float = 68.0

    def Q(self) -> np.ndarray:
        """Reduced stiffness of the ply in its own axes (plane stress)."""
        nu21 = self.nu12 * self.E2 / self.E1
        d = 1.0 - self.nu12 * nu21
        return np.array(
            [
                [self.E1 / d, self.nu12 * self.E2 / d, 0.0],
                [self.nu12 * self.E2 / d, self.E2 / d, 0.0],
                [0.0, 0.0, self.G12],
            ]
        )

    def tsai_wu_coefficients(self) -> dict[str, float]:
        F1 = 1.0 / self.Xt - 1.0 / self.Xc
        F2 = 1.0 / self.Yt - 1.0 / self.Yc
        F11 = 1.0 / (self.Xt * self.Xc)
        F22 = 1.0 / (self.Yt * self.Yc)
        F66 = 1.0 / self.S**2
        F12 = -0.5 * np.sqrt(F11 * F22)  # common default interaction term
        return dict(F1=F1, F2=F2, F11=F11, F22=F22, F66=F66, F12=F12)


DEFAULT_PLY = PlyMaterial()
PLY_ANGLES = (0.0, 45.0, -45.0, 90.0)


def strain_rotation(theta_deg: float) -> np.ndarray:
    """T such that eps_ply = T @ eps_global (engineering shear strain)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c * c, s * s, c * s],
            [s * s, c * c, -c * s],
            [-2 * c * s, 2 * c * s, c * c - s * s],
        ]
    )


def Qbar(theta_deg: float, ply: PlyMaterial = DEFAULT_PLY) -> np.ndarray:
    """Ply stiffness rotated into laminate axes: Qbar = T^T Q T."""
    T = strain_rotation(theta_deg)
    return T.T @ ply.Q() @ T


def A_matrix(p0: float, p45: float, p90: float, t: float,
             ply: PlyMaterial = DEFAULT_PLY) -> np.ndarray:
    """In-plane stiffness A (N/mm) of a symmetric, balanced laminate.

    p0, p45, p90: fraction of plies at 0, +/-45 (split evenly) and 90 deg.
    t: total laminate thickness in mm.
    """
    if not np.isclose(p0 + p45 + p90, 1.0):
        raise ValueError("ply fractions must sum to 1")
    A = t * (p0 * Qbar(0, ply) + 0.5 * p45 * (Qbar(45, ply) + Qbar(-45, ply))
             + p90 * Qbar(90, ply))
    A[0, 2] = A[2, 0] = A[1, 2] = A[2, 1] = 0.0  # balanced: exactly zero
    return A


def engineering_constants(A: np.ndarray, t: float) -> dict[str, float]:
    """Effective membrane moduli of the laminate."""
    a = np.linalg.inv(A)
    return dict(
        Ex=1.0 / (a[0, 0] * t),
        Ey=1.0 / (a[1, 1] * t),
        Gxy=1.0 / (a[2, 2] * t),
        nuxy=-a[0, 1] / a[0, 0],
    )


def ply_stresses(eps_global: np.ndarray, ply: PlyMaterial = DEFAULT_PLY,
                 angles=PLY_ANGLES) -> np.ndarray:
    """Ply-axis stresses for each angle.

    eps_global: (..., 3) array of laminate strains [ex, ey, gxy].
    Returns (..., n_angles, 3) array of [s1, s2, t12].
    """
    Q = ply.Q()
    out = []
    for th in angles:
        T = strain_rotation(th)
        out.append(np.einsum("ij,...j->...i", Q @ T, eps_global))
    return np.stack(out, axis=-2)


def tsai_wu_strength_ratio(stress: np.ndarray,
                           ply: PlyMaterial = DEFAULT_PLY) -> np.ndarray:
    """Load multiplier R at which Tsai-Wu reaches 1 (failure index FI = 1/R).

    Solves a R^2 + b R - 1 = 0 for the positive root, with
    a = F11 s1^2 + F22 s2^2 + F66 t12^2 + 2 F12 s1 s2 and b = F1 s1 + F2 s2.
    """
    c = ply.tsai_wu_coefficients()
    s1, s2, t12 = stress[..., 0], stress[..., 1], stress[..., 2]
    a = c["F11"] * s1**2 + c["F22"] * s2**2 + c["F66"] * t12**2 \
        + 2 * c["F12"] * s1 * s2
    b = c["F1"] * s1 + c["F2"] * s2
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (-b + np.sqrt(b * b + 4 * a)) / (2 * a)
    return np.where(a > 0, R, np.inf)


def unnotched_failure_strain(p0: float, p45: float, p90: float,
                             ply: PlyMaterial = DEFAULT_PLY) -> float:
    """Far-field strain ex at first-ply failure under uniaxial Nx (no hole).

    Independent of thickness: A scales with t, so the strain state does not.
    """
    A = A_matrix(p0, p45, p90, 1.0, ply)
    a = np.linalg.inv(A)
    eps = a[:, 0] / a[0, 0]          # strain state for ex = 1 under Nx only
    present = [th for th, p in zip(PLY_ANGLES, (p0, p45 / 2, p45 / 2, p90)) if p > 0]
    R = tsai_wu_strength_ratio(ply_stresses(eps, ply, present), ply)
    return float(R.min())
