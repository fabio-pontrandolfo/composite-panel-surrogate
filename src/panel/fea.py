"""2D membrane FEA with 6-node quadratic triangles (T6).

The laminate is treated as one orthotropic sheet whose constitutive law is the
A matrix: N = A @ eps (N/mm). Element stiffness K_e = integral of B^T A B dA,
integrated with 3-point Gauss quadrature.

Load case: left edge on rollers (u_x = 0, one node also u_y = 0), right edge
pulled by a uniform displacement delta. Edges are otherwise free, so a panel
without a hole carries a uniform N_x with N_y = N_xy = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.tri import Triangulation

from .mesh import Mesh

# 3-point Gauss rule on the reference triangle (exact for quadratics)
GP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
GW = np.full(3, 1 / 6)


def shape_derivs(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """dN/d(xi, eta) for T6 at points (xi, eta). Returns (n, 2, 6)."""
    xi, eta = np.atleast_1d(xi), np.atleast_1d(eta)
    L1 = 1 - xi - eta
    z = np.zeros_like(xi)
    dxi = np.stack([-(4 * L1 - 1), 4 * xi - 1, z,
                    4 * (L1 - xi), 4 * eta, -4 * eta], axis=-1)
    deta = np.stack([-(4 * L1 - 1), z, 4 * eta - 1,
                     -4 * xi, 4 * xi, 4 * (L1 - eta)], axis=-1)
    return np.stack([dxi, deta], axis=1)


def _B(X: np.ndarray, dN: np.ndarray):
    """Strain-displacement matrices.

    X: (ne, 6, 2) element node coordinates; dN: (ne, 2, 6) natural derivatives.
    Returns B (ne, 3, 12) and detJ (ne,).
    """
    J = dN @ X                                   # (ne, 2, 2)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    dNdx = np.linalg.solve(J, dN)                # (ne, 2, 6)
    ne = len(X)
    B = np.zeros((ne, 3, 12))
    B[:, 0, 0::2] = dNdx[:, 0]
    B[:, 1, 1::2] = dNdx[:, 1]
    B[:, 2, 0::2] = dNdx[:, 1]
    B[:, 2, 1::2] = dNdx[:, 0]
    return B, detJ


def _dofs(elems: np.ndarray) -> np.ndarray:
    d = np.empty((len(elems), 12), int)
    d[:, 0::2] = 2 * elems
    d[:, 1::2] = 2 * elems + 1
    return d


@dataclass
class Solution:
    mesh: Mesh
    A: np.ndarray
    u: np.ndarray            # (n_nodes, 2) displacements
    delta: float             # applied end displacement
    reaction: float          # total force on the loaded edge (N)

    @property
    def stiffness(self) -> float:
        """Axial stiffness of the panel, N/mm."""
        return self.reaction / self.delta

    @property
    def applied_strain(self) -> float:
        return self.delta / self.mesh.L

    # ---- strain recovery -------------------------------------------------
    def strain_at(self, pts: np.ndarray) -> np.ndarray:
        """Strain [ex, ey, gxy] evaluated directly inside the containing element."""
        m = self.mesh
        if not hasattr(self, "_finder"):
            self._tri = Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.corner_tris)
            self._finder = self._tri.get_trifinder()
        pts = np.atleast_2d(pts)
        e = self._finder(pts[:, 0], pts[:, 1])
        if np.any(e < 0):
            raise ValueError("evaluation point outside the mesh")
        X = m.nodes[m.elems[e]]
        # natural coordinates from the straight-sided (corner) triangle
        P0, P1, P2 = X[:, 0], X[:, 1], X[:, 2]
        M = np.stack([P1 - P0, P2 - P0], axis=-1)   # (n, 2, 2)
        xe = np.linalg.solve(M, (pts - P0)[..., None])[..., 0]
        B, _ = _B(X, shape_derivs(xe[:, 0], xe[:, 1]))
        ue = self.u[m.elems[e]].reshape(-1, 12)
        return np.einsum("nij,nj->ni", B, ue)

    def nodal_strain(self, node_ids: np.ndarray) -> np.ndarray:
        """Strain at corner nodes, averaged over the elements that share them."""
        m = self.mesh
        node_ids = np.atleast_1d(node_ids)
        out = np.zeros((len(node_ids), 3))
        corners = np.array([[0, 0], [1, 0], [0, 1]], float)
        for k, nid in enumerate(node_ids):
            es, loc = np.nonzero(m.corner_tris == nid)
            X = m.nodes[m.elems[es]]
            B, _ = _B(X, shape_derivs(corners[loc, 0], corners[loc, 1]))
            ue = self.u[m.elems[es]].reshape(-1, 12)
            out[k] = np.einsum("nij,nj->ni", B, ue).mean(0)
        return out


def solve(mesh: Mesh, A: np.ndarray, delta: float | None = None) -> Solution:
    """Assemble and solve the pull test. Default delta = 0.1 % nominal strain."""
    if delta is None:
        delta = 1e-3 * mesh.L
    X = mesh.nodes[mesh.elems]                       # (ne, 6, 2)
    ne = len(X)
    Ke = np.zeros((ne, 12, 12))
    for (xi, eta), w in zip(GP, GW):
        dN = np.broadcast_to(shape_derivs(np.array([xi]), np.array([eta])), (ne, 2, 6))
        B, detJ = _B(X, dN)
        if np.any(detJ <= 0):
            raise RuntimeError("inverted element")
        Ke += w * detJ[:, None, None] * np.einsum("nki,kl,nlj->nij", B, A, B)
    dofs = _dofs(mesh.elems)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 2 * len(mesh.nodes)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

    x = mesh.nodes[:, 0]
    tol = 1e-9 * mesh.L
    left = np.nonzero(x < tol)[0]
    right = np.nonzero(x > mesh.L - tol)[0]
    anchor = left[np.argmin(mesh.nodes[left, 1])]   # bottom-left corner

    fixed = np.concatenate([2 * left, 2 * right, [2 * anchor + 1]])
    vals = np.concatenate([np.zeros(len(left)), np.full(len(right), delta), [0.0]])
    free = np.setdiff1d(np.arange(n), fixed)

    u = np.zeros(n)
    u[fixed] = vals
    rhs = -K[free][:, fixed] @ vals
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    reaction = float((K[2 * right] @ u).sum())
    return Solution(mesh=mesh, A=A, u=u.reshape(-1, 2), delta=delta, reaction=reaction)
