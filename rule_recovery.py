"""Out-of-fold rule-recovery diagnostic (Section IV-E of the paper).

1) Screening: L1 logits (standardised inputs) for P(y>=1) and P(y>=2);
   keep indicators with a non-zero coefficient at either boundary.
2) Weights: mildly L2-penalised logit per boundary on the retained indicators
   in raw units; w_j = beta_j / ||beta_j||_1; w = (w_1 + w_2) / 2.
3) Thresholds: one-dimensional logits on the score w^T x.
4) Rounding: weights and thresholds rounded to a grid; scored on the held-out fold.
Decision criterion: rounded-rule held-out accuracy >= 99.5% and the same rule in every fold.
"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def round_to(v, grid):
    return np.round(np.asarray(v, float) / grid) * grid


def fit_rule(X, y, C_screen=0.05, C_bound=1e3):
    Z = StandardScaler().fit_transform(X)
    bounds = np.unique(y)[1:]          # cumulative boundaries P(y >= j)
    keep = np.zeros(X.shape[1], bool)
    for j in bounds:
        m = LogisticRegression(penalty="l1", C=C_screen, solver="liblinear")
        keep |= m.fit(Z, (y >= j).astype(int)).coef_[0] != 0
    idx = np.flatnonzero(keep)
    w = np.zeros(X.shape[1])
    if idx.size == 0:
        return w, np.full(len(bounds), np.inf)
    ws = []
    for j in bounds:
        b = LogisticRegression(C=C_bound, max_iter=5000).fit(X[:, idx], (y >= j).astype(int)).coef_[0]
        ws.append(b / np.abs(b).sum())
    w[idx] = np.mean(ws, axis=0)
    s = X @ w
    tau = []
    for j in bounds:
        m = LogisticRegression(C=1e6, max_iter=5000).fit(s[:, None], (y >= j).astype(int))
        tau.append(-m.intercept_[0] / m.coef_[0][0])
    return w, np.array(tau)


def predict(X, w, tau, exact=False):
    """Classify by the latent score. With exact=True the score is rounded to 8 decimals,
    i.e. evaluated in exact decimal arithmetic for grid weights and 2-decimal inputs,
    so that floating-point error cannot move a record across a cut-point."""
    s = X @ w
    if exact:
        s = np.round(s, 8)
    return sum((s > t).astype(int) for t in tau)


def diagnose(X, y, grid=0.05, seeds=(42, 43, 44), C_screen=0.05, C_bound=1e3,
             with_lgbm=True, return_folds=False):
    """Run the diagnostic over 5 folds x len(seeds). Returns a summary dict."""
    if with_lgbm:
        from lightgbm import LGBMClassifier
    lin, rnd, lgb, rules, folds = [], [], [], set(), []
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            w, tau = fit_rule(X[tr], y[tr], C_screen, C_bound)
            lin.append((predict(X[te], w, tau) == y[te]).mean())
            wr, tr_ = round_to(w, grid), round_to(tau, grid)
            rules.add(tuple(np.round(np.r_[wr, tr_], 8)))
            rnd.append((predict(X[te], wr, tr_, exact=True) == y[te]).mean())
            folds.append((w, tau))
            if with_lgbm:
                m = LGBMClassifier(verbose=-1, random_state=sd).fit(X[tr], y[tr])
                lgb.append((m.predict(X[te]) == y[te]).mean())
    out = dict(n=len(y), linear=100 * np.mean(lin), rounded=100 * np.mean(rnd),
               n_rules=len(rules),
               lgbm=100 * np.mean(lgb) if with_lgbm else np.nan)
    out["exact_rule"] = bool(out["rounded"] >= 99.5 and out["n_rules"] == 1)
    if return_folds:
        out["folds"] = folds
        out["rules"] = rules
    return out
