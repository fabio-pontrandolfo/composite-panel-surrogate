"""One complete panel analysis: layup + hole -> stiffness and failure strain."""
from __future__ import annotations

import time

import numpy as np

from .clt import A_matrix, ply_stresses, tsai_wu_strength_ratio, unnotched_failure_strain
from .fea import solve
from .mesh import Hole, make_mesh

# Fixed panel geometry used for the dataset (mm)
PANEL_L = 200.0
PANEL_W = 100.0
D0 = 1.0          # point-stress characteristic distance from the hole edge (mm)
H_EDGE = 0.8      # target element size at the hole edge (mm), from the convergence study
H_BG = 6.0        # far-field element size (mm)


def n_theta_for(r: float, h_edge: float = H_EDGE) -> int:
    """Nodes around the hole for a given edge element size (multiple of 4)."""
    return int(np.clip(4 * np.ceil(2 * np.pi * r / (4 * h_edge)), 48, 160))


def point_stress_failure(sol, d0: float = D0, n_points: int = 180):
    """Point-stress criterion (Whitney-Nuismer style) with Tsai-Wu plies.

    Strains are sampled on a circle at distance d0 from the hole edge. At each
    point and ply angle the Tsai-Wu strength ratio R is found; failure strain
    is the applied far-field strain times the smallest R. Using a distance d0
    instead of the peak at the edge gives a target that does not depend on the
    mesh and that matches how notched composites are sized in practice.
    """
    h = sol.mesh.hole
    th = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rho = h.r + d0
    pts = np.c_[h.cx + rho * np.cos(th), h.cy + rho * np.sin(th)]
    eps = sol.strain_at(pts)                          # (n_points, 3)
    R = tsai_wu_strength_ratio(ply_stresses(eps))     # (n_points, 4 angles)
    i, j = np.unravel_index(np.argmin(R), R.shape)
    return dict(
        fail_strain=float(R[i, j] * sol.applied_strain),
        crit_angle_deg=float(np.degrees(th[i])),
        crit_ply=[0, 45, -45, 90][j],
    )


def run_case(p0, p45, p90, t, r, cx, cy, L=PANEL_L, W=PANEL_W,
             n_theta=None, h_bg=H_BG) -> dict:
    """Full analysis of one design. Returns inputs, outputs and diagnostics."""
    tic = time.perf_counter()
    if n_theta is None:
        n_theta = n_theta_for(r)
    A = A_matrix(p0, p45, p90, t)
    mesh = make_mesh(L, W, Hole(cx, cy, r), n_theta=n_theta, h_bg=h_bg)
    sol = solve(mesh, A)
    out = dict(p0=p0, p45=p45, p90=p90, t=t, r=r, cx=cx, cy=cy,
               stiffness=sol.stiffness)
    out.update(point_stress_failure(sol))
    # physics references (no hole) used later by the hybrid model
    a = np.linalg.inv(A)
    out["stiffness_nohole"] = W / (a[0, 0] * L)
    out["fail_strain_nohole"] = unnotched_failure_strain(p0, p45, p90)
    out["n_dof"] = 2 * len(mesh.nodes)
    out["runtime_s"] = time.perf_counter() - tic
    return out
