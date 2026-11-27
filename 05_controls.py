"""Section V-E: the rule-recovery diagnostic on positive and negative controls (Table V)."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine
from common import load, modelled, target, RESULTS
from rule_recovery import diagnose, round_to

W_GRID = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]       # weights on the 0.05 grid
W_OFF = [0.27, 0.21, 0.17, 0.14, 0.13, 0.08]        # weights off the 0.05 grid


def features(n, seed, p=23):
    return np.round(np.random.default_rng(seed).uniform(0.3, 1.0, (n, p)), 2)


def quantile_cuts(score, q=(0.05, 0.80)):
    return np.quantile(score, q)


def to_class(s, cuts):
    return (s > cuts[0]).astype(int) + (s > cuts[1]).astype(int)


def rule_data(n, w, seed, grid_cuts=True, noise=0.0, flip=0.0):
    X = features(n, seed)
    rng = np.random.default_rng(seed + 100)
    s = X[:, :len(w)] @ np.array(w)
    cuts = quantile_cuts(features(200000, 9)[:, :len(w)] @ np.array(w))
    if grid_cuts:
        cuts = round_to(cuts, 0.05)
    if noise:
        s = s + rng.logistic(0, noise, n)
    y = to_class(s, cuts)
    if flip:   # move a share of labels to an adjacent class
        for i in rng.choice(n, int(round(flip * n)), replace=False):
            y[i] = 1 if y[i] != 1 else rng.choice([0, 2])
    return X, y


def nonlinear_data(n, seed):
    X = features(n, seed)
    s = 0.4 * X[:, 0] * X[:, 1] + 0.3 * (X[:, 2] - 0.65) ** 2 * 10 + 0.3 * X[:, 3]
    return X, to_class(s, quantile_cuts(s))


def tertiles(v):
    return np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))


d = load()
controls = {
    "This dataset": (d[modelled(d)].to_numpy(float), target(d)),
    "Positive control": rule_data(5000, W_GRID, 1),
    "Rule + score noise": rule_data(5000, W_GRID, 1, noise=0.02),
    "Non-linear rule": nonlinear_data(5000, 1),
    "Diabetes (real)": (load_diabetes().data, tertiles(load_diabetes().target)),
    "Breast cancer (real)": (load_breast_cancer().data, load_breast_cancer().target),
    "Rule + 1% label flips": rule_data(5000, W_GRID, 3, flip=0.01),
    "Off-grid exact rule": rule_data(5000, W_OFF, 4, grid_cuts=False),
    "Positive control, n = 500": rule_data(500, W_GRID, 2),
    "Positive control, n = 1000": rule_data(1000, W_GRID, 5),
    "Wine (real)": (load_wine().data, load_wine().target),
}

rows = []
for name, (X, y) in controls.items():
    r = diagnose(X, y)
    rows.append(dict(dataset=name, **r))
    print(f"{name:28s} n={r['n']:5d}  linear={r['linear']:6.2f}  rounded={r['rounded']:6.2f}  "
          f"rules={r['n_rules']:2d}  LGBM={r['lgbm']:5.1f}  exact={r['exact_rule']}")
pd.DataFrame(rows).to_csv(RESULTS / "table5_controls.csv", index=False)
