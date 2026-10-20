"""Step 6: generate the dataset with a Latin hypercube design.

Design variables (6, sampled together with an LHS):
  ply fractions p0, p45, p90  - symmetric balanced layups, each >= 10 % (10 % rule)
  t   total thickness   2 - 6 mm
  r   hole radius       4 - 16 mm
  cx  hole x position   60 - 140 mm (panel is 200 x 100 mm)
  cy  hole y position   edge ligament e = distance hole edge -> panel edge >= 10 mm

Two files:
  data/train.csv   800 designs inside that space (used for training and CV)
  data/extrap.csv  120 designs with the hole closer to the edge (3 <= e < 10 mm),
                   never seen in training: the extrapolation test.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel.analysis import PANEL_W, run_case  # noqa: E402

E_MIN_TRAIN = 10.0


def sample(n, seed, e_range):
    """Map an LHS in the unit cube to designs."""
    u = qmc.LatinHypercube(d=6, seed=seed).random(n)
    a, b = u[:, 0].copy(), u[:, 1].copy()
    fold = a + b > 1                      # fold the square onto the triangle
    a[fold], b[fold] = 1 - a[fold], 1 - b[fold]
    p0 = 0.1 + 0.7 * a
    p45 = 0.1 + 0.7 * b
    p90 = 1.0 - p0 - p45
    t = 2.0 + 4.0 * u[:, 2]
    r = 4.0 + 12.0 * u[:, 3]
    cx = 60.0 + 80.0 * u[:, 4]
    e_lo, e_hi = e_range
    e_hi = np.minimum(e_hi, PANEL_W / 2 - r)   # e_hi=None -> centred hole allowed
    e = e_lo + (e_hi - e_lo) * u[:, 5]
    cy = PANEL_W - r - e                       # hole in the upper half (symmetry)
    return pd.DataFrame(dict(p0=p0, p45=p45, p90=p90, t=t, r=r, cx=cx, cy=cy))


def _run(row):
    return run_case(**row)


def run_all(designs, name):
    tic = time.time()
    with Pool() as pool:
        rows = pool.map(_run, designs.to_dict("records"), chunksize=4)
    df = pd.DataFrame(rows)
    df["ligament"] = PANEL_W - df["cy"] - df["r"]
    df.to_csv(ROOT / "data" / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} runs in {time.time() - tic:.0f} s "
          f"(mean {df.runtime_s.mean():.2f} s/run, max dof {df.n_dof.max()})")
    return df


if __name__ == "__main__":
    run_all(sample(800, seed=0, e_range=(E_MIN_TRAIN, np.inf)), "train")
    run_all(sample(120, seed=1, e_range=(3.0, E_MIN_TRAIN)), "extrap")
