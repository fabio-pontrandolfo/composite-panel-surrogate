"""Graded triangular mesh of a rectangular panel with an optional circular hole.

No external mesher: points are placed on rings around the hole (spacing grows
geometrically with radius), on a background grid away from it, and along the
edges with matching spacing; scipy's Delaunay connects them and triangles
inside the hole are removed. Linear triangles are then upgraded to 6-node
quadratic triangles (T6), with mid-side nodes on the hole snapped to the circle.

Panel occupies x in [0, L], y in [0, W].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class Hole:
    cx: float
    cy: float
    r: float


@dataclass
class Mesh:
    nodes: np.ndarray       # (n_nodes, 2)
    elems: np.ndarray       # (n_elems, 6) T6 connectivity, corners first, CCW
    L: float
    W: float
    hole: Hole | None

    @property
    def corner_tris(self) -> np.ndarray:
        return self.elems[:, :3]


def _edge_points(p0, p1, size_fn, n_fine=4000):
    """Points from p0 to p1 (both included) with spacing following size_fn."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = np.linspace(0.0, 1.0, n_fine)
    pts = p0 + u[:, None] * (p1 - p0)
    length = np.linalg.norm(p1 - p0)
    density = 1.0 / size_fn(pts)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1])
                                           * np.diff(u) * length)])
    n = max(int(np.ceil(cum[-1])), 1)
    u_k = np.interp(np.linspace(0, cum[-1], n + 1), cum, u)
    return p0 + u_k[:, None] * (p1 - p0)


def make_mesh(L: float, W: float, hole: Hole | None = None,
              n_theta: int = 64, h_bg: float = 6.0) -> Mesh:
    """Build the T6 mesh.

    n_theta: nodes around the hole (controls refinement at the hole edge).
    h_bg:    element size far from the hole (mm).
    """
    if hole is not None:
        lig = min(hole.cx - hole.r, L - hole.cx - hole.r,
                  hole.cy - hole.r, W - hole.cy - hole.r)
        if lig <= 0.5 * 2 * np.pi * hole.r / n_theta:
            raise ValueError("hole too close to the panel edge for this mesh")

        def size_fn(p):
            d = np.hypot(p[:, 0] - hole.cx, p[:, 1] - hole.cy)
            return np.clip(2 * np.pi * d / n_theta, 1e-3, h_bg)
    else:
        def size_fn(p):
            return np.full(len(p), h_bg)

    pts = []
    # 1. boundary
    corners = [(0, 0), (L, 0), (L, W), (0, W)]
    for i in range(4):
        e = _edge_points(corners[i], corners[(i + 1) % 4], size_fn)
        pts.append(e[:-1])
    boundary = np.vstack(pts)

    def inside_with_margin(p, margin):
        return ((p[:, 0] > margin) & (p[:, 0] < L - margin)
                & (p[:, 1] > margin) & (p[:, 1] < W - margin))

    rho_last = 0.0
    if hole is not None:
        # 2. rings around the hole, staggered every other ring
        rho, k = hole.r, 0
        while True:
            h = 2 * np.pi * rho / n_theta
            if h >= h_bg:
                break
            th = 2 * np.pi * (np.arange(n_theta) + 0.5 * (k % 2)) / n_theta
            ring = np.c_[hole.cx + rho * np.cos(th), hole.cy + rho * np.sin(th)]
            keep = inside_with_margin(ring, 0.5 * h) if k > 0 else np.ones(n_theta, bool)
            pts.append(ring[keep])
            rho_last = rho
            rho += 0.87 * h
            k += 1

    # 3. background grid (hex-like), clear of the edges and of the ring zone
    ny = int(np.ceil(W / (0.87 * h_bg)))
    ys = np.linspace(0, W, ny + 1)[1:-1]
    bg = []
    for j, y in enumerate(ys):
        nx = int(np.ceil(L / h_bg))
        xs = np.linspace(0, L, nx + 1)[1:-1] + (0.5 * L / nx if j % 2 else 0.0)
        bg.append(np.c_[xs, np.full_like(xs, y)])
    bg = np.vstack(bg)
    bg = bg[inside_with_margin(bg, 0.5 * h_bg)]
    if hole is not None:
        d = np.hypot(bg[:, 0] - hole.cx, bg[:, 1] - hole.cy)
        bg = bg[d > rho_last + 0.6 * h_bg]
    pts.append(bg)

    P = np.vstack(pts)
    P = np.unique(np.round(P, 9), axis=0)
    tri = Delaunay(P).simplices

    # 4. drop degenerate triangles and triangles inside the hole
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = np.abs(area2) > 1e-10 * L * W
    if hole is not None:
        cen = (a + b + c) / 3
        keep &= np.hypot(cen[:, 0] - hole.cx, cen[:, 1] - hole.cy) > hole.r
    tri = tri[keep]
    area2 = area2[keep]
    tri[area2 < 0] = tri[area2 < 0][:, [0, 2, 1]]   # make all CCW

    # drop unused points and renumber
    used = np.unique(tri)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, tri = P[used], remap[tri]

    # 5. upgrade to T6: one mid-side node per unique edge
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    uniq, inv = np.unique(edges_sorted, axis=0, return_inverse=True)
    mids = 0.5 * (P[uniq[:, 0]] + P[uniq[:, 1]])
    if hole is not None:
        on_hole = lambda q: np.abs(np.hypot(q[:, 0] - hole.cx, q[:, 1] - hole.cy) - hole.r) < 1e-6 * hole.r
        both = on_hole(P[uniq[:, 0]]) & on_hole(P[uniq[:, 1]])
        v = mids[both] - [hole.cx, hole.cy]
        mids[both] = [hole.cx, hole.cy] + hole.r * v / np.linalg.norm(v, axis=1)[:, None]
    ne = len(tri)
    mid_ids = len(P) + inv.reshape(3, ne).T
    nodes = np.vstack([P, mids])
    elems = np.hstack([tri, mid_ids])
    return Mesh(nodes=nodes, elems=elems, L=L, W=W, hole=hole)


def min_angle_deg(mesh: Mesh) -> float:
    """Smallest interior angle over the corner triangles (mesh quality)."""
    P = mesh.nodes[mesh.corner_tris]
    angs = []
    for i in range(3):
        u = P[:, (i + 1) % 3] - P[:, i]
        v = P[:, (i + 2) % 3] - P[:, i]
        cosang = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angs.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return float(np.min(angs))
