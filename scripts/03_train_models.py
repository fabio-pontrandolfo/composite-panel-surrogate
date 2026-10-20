"""Step 7: train and stress-test surrogate models.

Targets:  stiffness (N/mm) and failure strain (far-field strain at first-ply
failure, point-stress criterion, reported in microstrain).

Models:   Gaussian process (ARD Matern), gradient-boosted trees, MLP.
Modes:    pure     - raw design variables -> target
          hybrid   - laminate theory gives the no-hole value y0; the model only
                     learns the knock-down caused by the hole, y / y0, from the
                     layup, the hole geometry and one physics feature (Lekhnitskii
                     Kt_inf). Thickness is dropped: both stiffness / y0 and the
                     failure strain are independent of thickness in this
                     membrane model, so the hybrid does not have to learn that.

Experiments
  1. 5-fold cross-validation on 600 designs, then a 200-design holdout test.
  2. Learning curve: holdout error vs training-set size.
  3. Active learning with the GP (add the most uncertain design) vs random.
  4. Extrapolation: holes closer to the edge than anything in training,
     including whether the GP's uncertainty warns you.

All errors are reported in physical units and as % of the true value.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plotstyle import INK2, SERIES, plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter  # noqa: E402

NAMES = {"stiffness": "Stiffness", "fail_strain": "Failure strain"}


def clean_log(ax, axes="xy"):
    """Plain numbers on log axes (1, 10, 0.1), no minor tick labels."""
    for a in axes:
        axis = ax.xaxis if a == "x" else ax.yaxis
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
from panel.analytic import kt_infinite_orthotropic  # noqa: E402
from panel.clt import A_matrix  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
FIG = ROOT / "results" / "figures"
TARGETS = {"stiffness": "stiffness_nohole", "fail_strain": "fail_strain_nohole"}
UNITS = {"stiffness": ("kN/mm", 1e-3), "fail_strain": ("µε", 1e6)}
RAW = ["p0", "p45", "t", "r", "cx", "cy"]
HYB = ["p0", "p45", "r", "cx", "ligament", "kt_inf"]


# ---- features and models ---------------------------------------------------------
def add_physics(df):
    df = df.copy()
    df["kt_inf"] = [kt_infinite_orthotropic(A_matrix(p0, p45, p90, 1.0))
                    for p0, p45, p90 in df[["p0", "p45", "p90"]].values]
    return df


def make_model(kind, n_features, seed=0):
    if kind == "GP":
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(np.ones(n_features), (1e-2, 1e3), nu=2.5)
                  + WhiteKernel(1e-5, (1e-9, 1e-1)))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed))
    if kind == "GBT":
        return GradientBoostingRegressor(n_estimators=800, learning_rate=0.03, max_depth=4,
                                         subsample=0.8, random_state=seed)
    if kind == "MLP":
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", alpha=1e-4,
                           learning_rate_init=3e-3, max_iter=4000, tol=1e-8,
                           n_iter_no_change=100, random_state=seed)
        return TransformedTargetRegressor(make_pipeline(StandardScaler(), mlp),
                                          transformer=StandardScaler())
    raise ValueError(kind)


class Surrogate:
    """Fits log(y) (pure) or log(y / y0) (hybrid); predicts y in physical units."""

    def __init__(self, kind, mode, target, seed=0):
        self.kind, self.mode, self.target = kind, mode, target
        self.cols = RAW if mode == "pure" else HYB
        self.model = make_model(kind, len(self.cols), seed)

    def _base(self, df):
        return df[TARGETS[self.target]].values if self.mode == "hybrid" else 1.0

    def fit(self, df):
        self.model.fit(df[self.cols].values, np.log(df[self.target].values / self._base(df)))
        return self

    def predict(self, df, return_std=False):
        X = df[self.cols].values
        if return_std:
            mu, sd = self.model.predict(X, return_std=True)
            return self._base(df) * np.exp(mu), mu, sd
        return self._base(df) * np.exp(self.model.predict(X))


def errors(y_true, y_pred, target):
    unit, scale = UNITS[target]
    pct = 100 * np.abs(y_pred - y_true) / y_true
    return dict(mape_pct=float(pct.mean()), p95_pct=float(np.percentile(pct, 95)),
                max_pct=float(pct.max()),
                rmse=float(np.sqrt(np.mean((y_pred - y_true) ** 2)) * scale), unit=unit)


# ---- experiments ----------------------------------------------------------------
def cross_validate(dev, kinds, modes):
    rows = []
    kf = KFold(5, shuffle=True, random_state=0)
    for target in TARGETS:
        for mode in modes:
            for kind in kinds:
                tic = time.time()
                m = []
                for tr, va in kf.split(dev):
                    s = Surrogate(kind, mode, target).fit(dev.iloc[tr])
                    m.append(errors(dev[target].values[va], s.predict(dev.iloc[va]), target))
                rows.append(dict(target=target, mode=mode, model=kind,
                                 cv_mape_pct=np.mean([e["mape_pct"] for e in m]),
                                 cv_mape_std=np.std([e["mape_pct"] for e in m]),
                                 cv_rmse=np.mean([e["rmse"] for e in m]),
                                 unit=m[0]["unit"], seconds=time.time() - tic))
                print(f"CV  {target:11s} {mode:6s} {kind:3s} "
                      f"MAPE {rows[-1]['cv_mape_pct']:.2f} ± {rows[-1]['cv_mape_std']:.2f} %")
    return rows


def holdout(dev, test, extrap, kinds, modes):
    rows, fitted = [], {}
    for target in TARGETS:
        for mode in modes:
            for kind in kinds:
                s = Surrogate(kind, mode, target).fit(dev)
                fitted[(target, mode, kind)] = s
                r = dict(target=target, mode=mode, model=kind)
                for name, df in (("test", test), ("extrap", extrap)):
                    e = errors(df[target].values, s.predict(df), target)
                    r.update({f"{name}_{k}": v for k, v in e.items()})
                rows.append(r)
                print(f"OUT {target:11s} {mode:6s} {kind:3s} test MAPE {r['test_mape_pct']:.2f} % "
                      f"| extrap MAPE {r['extrap_mape_pct']:.2f} %  max {r['extrap_max_pct']:.1f} %")
    return rows, fitted


def learning_curve(dev, test, sizes, configs, repeats=5):
    rows = []
    for target in TARGETS:
        for kind, mode in configs:
            for n in sizes:
                m = []
                for rep in range(repeats if n < len(dev) else 1):
                    sub = dev.sample(n, random_state=rep)
                    s = Surrogate(kind, mode, target, seed=rep).fit(sub)
                    m.append(errors(test[target].values, s.predict(test), target)["mape_pct"])
                rows.append(dict(target=target, model=kind, mode=mode, n=n,
                                 mape_pct=float(np.mean(m)), mape_std=float(np.std(m))))
            print(f"LC  {target:11s} {mode:6s} {kind:3s}",
                  [round(r["mape_pct"], 2) for r in rows[-len(sizes):]])
    return rows


def active_learning(dev, test, target, mode, n0=15, n_max=150, step=5, seeds=5):
    """Pool-based: at each step add the `step` pool designs with the largest GP std."""
    out = {"active": [], "random": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        start = list(rng.choice(len(dev), n0, replace=False))
        for strategy in out:
            idx = list(start)
            curve = []
            while True:
                s = Surrogate("GP", mode, target, seed=seed).fit(dev.iloc[idx])
                curve.append((len(idx), errors(test[target].values, s.predict(test),
                                               target)["mape_pct"]))
                if len(idx) >= n_max:
                    break
                pool = np.setdiff1d(np.arange(len(dev)), idx)
                if strategy == "active":
                    _, _, sd = s.predict(dev.iloc[pool], return_std=True)
                    pick = pool[np.argsort(sd)[-step:]]
                else:
                    pick = rng.choice(pool, step, replace=False)
                idx += list(pick)
            out[strategy].append(curve)
    res = {}
    for strategy, curves in out.items():
        arr = np.array([[e for _, e in c] for c in curves])
        res[strategy] = dict(n=[n for n, _ in curves[0]], mape_mean=arr.mean(0).tolist(),
                             mape_std=arr.std(0).tolist())
    print("AL  final MAPE active {:.2f} %, random {:.2f} %".format(
        res["active"]["mape_mean"][-1], res["random"]["mape_mean"][-1]))
    return res


def gp_calibration(s, df):
    """Share of designs whose true value lies inside the GP's 95 % interval."""
    _, mu, sd = s.predict(df, return_std=True)
    z = (np.log(df[s.target].values / s._base(df)) - mu) / sd
    return float(np.mean(np.abs(z) < 1.96)), sd


# ---- plots ---------------------------------------------------------------------------
def plot_parity(fitted, test, extrap, kind, mode):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, target in zip(axes, TARGETS):
        unit, scale = UNITS[target]
        s = fitted[(target, mode, kind)]
        for df, label, c in ((test, "holdout (in range)", SERIES[0]),
                             (extrap, "extrapolation (hole near edge)", SERIES[1])):
            ax.plot(df[target] * scale, s.predict(df) * scale, "o", ms=4, alpha=0.7,
                    color=c, label=label, mec="none")
        lo = min(test[target].min(), extrap[target].min()) * scale
        hi = max(test[target].max(), extrap[target].max()) * scale
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        ax.set_xlabel(f"FEA {NAMES[target].lower()} ({unit})")
        ax.set_ylabel(f"predicted ({unit})")
        ax.set_title(f"{NAMES[target]}, {mode} {kind}")
    axes[0].legend(fontsize=8)
    fig.savefig(FIG / "parity.png")
    plt.close(fig)


def plot_learning_curve(rows, configs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for ax, target in zip(axes, TARGETS):
        for k, (kind, mode) in enumerate(configs):
            sub = [r for r in rows if r["target"] == target and r["model"] == kind and r["mode"] == mode]
            ax.errorbar([r["n"] for r in sub], [r["mape_pct"] for r in sub],
                        yerr=[r["mape_std"] for r in sub], marker="o", ms=4, capsize=2,
                        color=SERIES[k], label=f"{mode} {kind}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([25, 50, 100, 200, 400, 600])
        if target == "fail_strain":
            ax.set_yticks([0.3, 0.5, 1, 2, 4])
        clean_log(ax)
        ax.set_xlabel("training designs")
        ax.set_ylabel("holdout mean abs. error (%)")
        ax.set_title(NAMES[target])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.savefig(FIG / "learning_curve.png")
    plt.close(fig)


def plot_active(res):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, strategy in enumerate(("random", "active")):
        r = res[strategy]
        m, s = np.array(r["mape_mean"]), np.array(r["mape_std"])
        ax.plot(r["n"], m, color=SERIES[k], label=f"{strategy} sampling")
        ax.fill_between(r["n"], m - s, m + s, color=SERIES[k], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3])
    clean_log(ax, "y")
    ax.set_xlabel("training designs")
    ax.set_ylabel("failure strain, holdout mean abs. error (%)\npure GP, mean ± 1 sd over 5 seeds")
    ax.set_title("Active learning gives a small gain over random sampling")
    ax.legend()
    fig.savefig(FIG / "active_learning.png")
    plt.close(fig)


def plot_extrap(fitted, extrap, test):
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    target = "fail_strain"
    both = pd.concat([test, extrap])
    ax = axes[0]
    for k, (kind, mode) in enumerate((("GP", "pure"), ("GP", "hybrid"), ("GBT", "hybrid"), ("MLP", "hybrid"))):
        s = fitted[(target, mode, kind)]
        err = 100 * (s.predict(both) - both[target]) / both[target]
        ax.plot(both["ligament"], err, "o", ms=3, alpha=0.6, mec="none",
                color=SERIES[k], label=f"{mode} {kind}")
    ax.axvline(10, color=INK2, lw=1, ls="--")
    ax.text(10.6, -5.5, "training range →", color=INK2, fontsize=8)
    ax.text(9.4, -5.5, "← extrapolation", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 40]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("hole edge to panel edge, ligament (mm)")
    ax.set_ylabel("failure strain prediction error (%)")
    ax.set_title("Errors grow ~10x once the hole leaves the training range")
    ax.legend(fontsize=8, markerscale=2)

    ax = axes[1]
    s = fitted[(target, "hybrid", "GP")]
    for k, (df, label) in enumerate(((test, "holdout, in range"), (extrap, "extrapolation"))):
        _, mu, sd = s.predict(df, return_std=True)
        z = np.abs(np.log(df[target].values / s._base(df)) - mu)
        ax.plot(100 * sd, 100 * z, "o", ms=3.5, alpha=0.7, mec="none", color=SERIES[k], label=label)
    ax.plot([0.05, 5], [0.1, 10], color=INK2, lw=1)
    ax.text(3.2, 4.2, "error = 2σ", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    clean_log(ax)
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("GP predicted std (%)")
    ax.set_ylabel("actual error (%)")
    ax.set_title("Out of range, GP errors sit far above its 2σ band")
    ax.legend(fontsize=8, markerscale=2, loc="upper left")
    fig.savefig(FIG / "extrapolation.png")
    plt.close(fig)


def main():
    data = add_physics(pd.read_csv(ROOT / "data" / "train.csv"))
    extrap = add_physics(pd.read_csv(ROOT / "data" / "extrap.csv"))
    dev, test = train_test_split(data, test_size=200, random_state=0)
    kinds, modes = ("GP", "GBT", "MLP"), ("pure", "hybrid")

    results = dict(n_dev=len(dev), n_test=len(test), n_extrap=len(extrap))
    results["cv"] = cross_validate(dev, kinds, modes)
    results["holdout"], fitted = holdout(dev, test, extrap, kinds, modes)

    cal = {}
    for target in TARGETS:
        for mode in modes:
            s = fitted[(target, mode, "GP")]
            cal[f"{target}_{mode}"] = dict(test=gp_calibration(s, test)[0],
                                           extrap=gp_calibration(s, extrap)[0])
    results["gp_95pct_coverage"] = cal
    print("GP 95% interval coverage:", cal)

    configs = [("GP", "pure"), ("GBT", "pure"), ("GP", "hybrid"), ("GBT", "hybrid"), ("MLP", "hybrid")]
    results["learning_curve"] = learning_curve(dev, test, [25, 50, 100, 200, 400, 600], configs)
    results["active_learning"] = active_learning(dev, test, "fail_strain", "pure")

    plot_parity(fitted, test, extrap, "GP", "hybrid")
    plot_learning_curve(results["learning_curve"], configs)
    plot_active(results["active_learning"])
    plot_extrap(fitted, extrap, test)
    (ROOT / "results" / "ml_results.json").write_text(json.dumps(results, indent=2, default=float))


def plots_only():
    """Redraw figures from results/ml_results.json, refitting only the models they show."""
    results = json.loads((ROOT / "results" / "ml_results.json").read_text())
    data = add_physics(pd.read_csv(ROOT / "data" / "train.csv"))
    extrap = add_physics(pd.read_csv(ROOT / "data" / "extrap.csv"))
    dev, test = train_test_split(data, test_size=200, random_state=0)
    fitted = {}
    for target in TARGETS:
        for kind, mode in (("GP", "pure"), ("GP", "hybrid"), ("GBT", "hybrid"), ("MLP", "hybrid")):
            fitted[(target, mode, kind)] = Surrogate(kind, mode, target).fit(dev)
    configs = [("GP", "pure"), ("GBT", "pure"), ("GP", "hybrid"), ("GBT", "hybrid"), ("MLP", "hybrid")]
    plot_parity(fitted, test, extrap, "GP", "hybrid")
    plot_learning_curve(results["learning_curve"], configs)
    plot_active(results["active_learning"])
    plot_extrap(fitted, extrap, test)


if __name__ == "__main__":
    plots_only() if "--plots-only" in sys.argv else main()
