"""Fast checks of the solver against closed-form answers.

Run with `pytest` or `python tests/test_fea.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from panel.analytic import kt_infinite_orthotropic  # noqa: E402
from panel.clt import A_matrix, engineering_constants, unnotched_failure_strain  # noqa: E402
from panel.fea import solve  # noqa: E402
from panel.mesh import Hole, make_mesh, min_angle_deg  # noqa: E402


def test_quasi_isotropic_is_isotropic():
    A = A_matrix(0.25, 0.5, 0.25, 1.0)
    assert np.isclose(A[0, 0], A[1, 1])
    assert np.isclose(A[2, 2], (A[0, 0] - A[0, 1]) / 2)
    assert np.isclose(kt_infinite_orthotropic(A), 3.0)   # isotropic Kt = 3


def test_no_hole_matches_clt():
    A = A_matrix(0.5, 0.3, 0.2, 3.0)
    sol = solve(make_mesh(200, 100, None), A)
    clt = 100 / (np.linalg.inv(A)[0, 0] * 200)
    assert abs(sol.stiffness / clt - 1) < 1e-10
    # uniform strain everywhere, with free Poisson contraction
    eps = sol.strain_at(np.array([[30.0, 20.0], [150.0, 80.0]]))
    c = engineering_constants(A, 3.0)
    assert np.allclose(eps[:, 0], 1e-3)
    assert np.allclose(eps[:, 1], -c["nuxy"] * 1e-3)


def test_failure_strain_independent_of_thickness():
    a = unnotched_failure_strain(0.4, 0.4, 0.2)
    assert 1e-3 < a < 1e-2
    from panel.analysis import run_case
    r1 = run_case(0.4, 0.4, 0.2, 2.0, 8.0, 100, 60)
    r2 = run_case(0.4, 0.4, 0.2, 5.0, 8.0, 100, 60)
    assert abs(r1["fail_strain"] / r2["fail_strain"] - 1) < 1e-9
    assert abs(r2["stiffness"] / r1["stiffness"] - 2.5) < 1e-9


def test_mesh_quality():
    m = make_mesh(200, 100, Hole(100, 85, 8), n_theta=64)
    assert min_angle_deg(m) > 15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
