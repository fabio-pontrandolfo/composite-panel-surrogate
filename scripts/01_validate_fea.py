"""Step 5: validate the FEA the way an analyst would.

1. No hole      -> stiffness must equal classical laminate theory.
2. Central hole -> stress concentration factor vs Lekhnitskii + Tan's
                   finite-width correction, several layups and hole sizes.
3. Mesh convergence around the hole (SCF, failure strain, stiffness).

Writes results/validation.json and figures in results/figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plotstyle import INK2, SERIES, plt  # noqa: E402
from panel.analysis import point_stress_failure  # noqa: E402
from panel.analytic import kt_infinite_orthotropic, tan_finite_width_factor  # noqa: E402
from panel.clt import A_matrix  # noqa: E402
from panel.fea import solve  # noqa: E402
from panel.mesh import Hole, make_mesh  # noqa: E402

FIG = ROOT / "results" / "figures"
LAYUPS = {
    "quasi-isotropic [25/50/25]": (0.25, 0.50, 0.25),
    "0-dominated [60/30/10]": (0.60, 0.30, 0.10),
    "+/-45-dominated [10/80/10]": (0.10, 0.80, 0.10),
    "90-dominated [10/30/60]": (0.10, 0.30, 0.60),
}
T = 4.0
W, L = 100.0, 400.0     # long panel so the loaded ends don't disturb the hole


def fea_scf(A, r, n_theta=64, h_bg=6.0):
    """Gross-section SCF: N_x at the top of the hole / (reaction / width)."""
    mesh = make_mesh(L, W, Hole(L / 2, W / 2, r), n_theta=n_theta, h_bg=h_bg)
    sol = solve(mesh, A)
    top = np.argmin(np.hypot(mesh.nodes[:, 0] - L / 2, mesh.nodes[:, 1] - (W / 2 + r)))
    Nx = (A @ sol.nodal_strain(top)[0])[0]
    return Nx / (sol.reaction / W), sol


def plot_convergence(rows):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, r in enumerate((4.0, 16.0)):
        sub = [row for row in rows if row["r"] == r][:-1]
        h = [row["h_edge"] for row in sub]
        ax.plot(h, [100 * abs(row["fail_strain_change_vs_finest"]) for row in sub], "o-",
                color=SERIES[k], label=f"failure strain at d0 = 1 mm, r = {r:.0f} mm")
        ax.plot(h, [100 * abs(row["scf_err_vs_analytic"]) for row in sub], "s--",
                color=SERIES[k], mfc="none", lw=1.2, label=f"peak SCF vs analytic, r = {r:.0f} mm")
    ax.axvline(0.8, color=INK2, lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.xaxis.set_major_locator(FixedLocator([3, 2, 1.2, 0.8, 0.5, 0.35]))
    ax.set_yticks([0.1, 0.3, 1, 3, 10])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.text(0.78, 0.06, "dataset: 0.8 mm", color=INK2, fontsize=8)
    ax.set_xlabel("element size at the hole edge (mm), finer to the right")
    ax.set_ylabel("error (%)")
    ax.set_title("Failure strain converges much faster than the peak SCF")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(FIG / "mesh_convergence.png")
    plt.close(fig)


def main():
    results = {}

    # 1. no hole --------------------------------------------------------------
    rows = []
    for name, (p0, p45, p90) in LAYUPS.items():
        A = A_matrix(p0, p45, p90, T)
        sol = solve(make_mesh(L, W, None), A)
        clt = W / (np.linalg.inv(A)[0, 0] * L)
        rows.append(dict(layup=name, fea=sol.stiffness, clt=clt,
                         rel_err=abs(sol.stiffness - clt) / clt))
    results["no_hole"] = rows
    print("No hole, max relative error:", max(r["rel_err"] for r in rows))

    # 2. Lekhnitskii + finite width ------------------------------------------------
    radii = [2.5, 5.0, 10.0, 15.0, 20.0]
    rows = []
    for name, (p0, p45, p90) in LAYUPS.items():
        A = A_matrix(p0, p45, p90, T)
        kinf = kt_infinite_orthotropic(A)
        for r in radii:
            ref = kinf * tan_finite_width_factor(kinf, 2 * r / W)
            scf, _ = fea_scf(A, r, n_theta=192)   # fine mesh for the peak value
            rows.append(dict(layup=name, r=r, two_r_over_w=2 * r / W, kt_inf=kinf,
                             analytic=ref, fea=scf, rel_err=(scf - ref) / ref))
    results["lekhnitskii"] = rows
    errs = np.array([r["rel_err"] for r in rows])
    print(f"SCF vs Lekhnitskii+Tan: mean |err| {np.abs(errs).mean():.2%}, max {np.abs(errs).max():.2%}")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, name in enumerate(LAYUPS):
        sub = [r for r in rows if r["layup"] == name]
        x = [r["two_r_over_w"] for r in sub]
        ax.plot(x, [r["analytic"] for r in sub], color=SERIES[k], label=name)
        ax.plot(x, [r["fea"] for r in sub], "o", color=SERIES[k], mfc="none", mew=1.5)
    ax.set_xlabel("hole diameter / panel width, 2r/W")
    ax.set_ylabel("stress concentration factor (gross)")
    ax.set_title("FEA (circles) vs Lekhnitskii + Tan finite-width (lines)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / "validation_scf.png")
    plt.close(fig)

    # 3. mesh convergence -------------------------------------------------------------
    # Worst case for convergence: the 0-dominated layup has the steepest gradient.
    # Refinement is expressed as element size at the hole edge, h_edge = 2*pi*r/n_theta.
    A = A_matrix(0.60, 0.30, 0.10, T)
    kinf = kt_infinite_orthotropic(A)
    rows = []
    for r in (4.0, 16.0):
        ref = kinf * tan_finite_width_factor(kinf, 2 * r / W)
        for h_edge in (3.2, 1.6, 1.2, 0.8, 0.5, 0.35, 0.25):
            n = int(4 * round(2 * np.pi * r / (4 * h_edge)))
            if n < 16:
                continue
            scf, sol = fea_scf(A, r, n_theta=max(n, 16))
            rows.append(dict(r=r, h_edge=2 * np.pi * r / max(n, 16), n_theta=max(n, 16),
                             n_dof=int(sol.u.size), scf=scf, scf_ref=ref,
                             fail_strain=point_stress_failure(sol)["fail_strain"],
                             stiffness=sol.stiffness))
    for r in (4.0, 16.0):
        sub = [row for row in rows if row["r"] == r]
        fine = sub[-1]
        for row in sub:
            row["fail_strain_change_vs_finest"] = row["fail_strain"] / fine["fail_strain"] - 1
            row["stiffness_change_vs_finest"] = row["stiffness"] / fine["stiffness"] - 1
            row["scf_err_vs_analytic"] = row["scf"] / row["scf_ref"] - 1
    results["convergence"] = rows
    for row in rows:
        print(f"r={row['r']:4.0f} h_edge={row['h_edge']:.2f} dof={row['n_dof']:6d} "
              f"SCF err={row['scf_err_vs_analytic']:+.2%} "
              f"fail strain vs finest={row['fail_strain_change_vs_finest']:+.3%} "
              f"stiffness vs finest={row['stiffness_change_vs_finest']:+.4%}")

    plot_convergence(rows)

    # mesh picture for the README
    mesh = make_mesh(200, 100, Hole(100, 50, 10), n_theta=64)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.triplot(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.corner_tris, lw=0.3, color=SERIES[0])
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"Graded T6 mesh, {len(mesh.elems)} elements")
    fig.savefig(FIG / "mesh.png")
    plt.close(fig)

    (ROOT / "results" / "validation.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    if "--plots-only" in sys.argv:
        plot_convergence(json.loads((ROOT / "results" / "validation.json").read_text())["convergence"])
    else:
        main()
