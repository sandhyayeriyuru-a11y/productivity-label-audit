"""Section V-C: EBM shape functions, ALE, permutation importance and dimension ablation.

EBM (main effects) is refitted on a stratified 80% split in raw units; ALE and permutation
importance use tuned XGBoost on the remaining 20%. Hyper-parameters default to the values
most often selected by 02_benchmark.py; override them on the command line if yours differ.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from common import load, modelled, target, RESULTS, RULE, DIMENSIONS

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ebm_lr", type=float, default=0.05)
ap.add_argument("--ebm_leaves", type=int, default=3)
ap.add_argument("--xgb_depth", type=int, default=2)
ap.add_argument("--xgb_lr", type=float, default=0.1)
ap.add_argument("--lgbm_leaves", type=int, default=7)
args = ap.parse_args()

d = load()
y = target(d)
feat = modelled(d)
X = d[feat]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
signal = [c for _, c in RULE]

# ---------------- (1) EBM: term importance and linearity of shape functions
ebm = ExplainableBoostingClassifier(interactions=0, learning_rate=args.ebm_lr,
                                    max_leaves=args.ebm_leaves, random_state=0)
ebm.fit(Xtr, ytr)
imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)

rows = []
for i, name in enumerate(ebm.term_names_):
    cuts = np.asarray(ebm.bins_[i][0])
    sc = ebm.term_scores_[i][1:-1]               # drop missing / unknown bins
    wt = ebm.bin_weights_[i][1:-1]
    lo, hi = X[name].min(), X[name].max()
    edges = np.r_[lo, cuts, hi]
    centre = (edges[:-1] + edges[1:]) / 2
    contrast = sc[:, 2] - sc[:, 0]              # High-versus-Low log-odds
    ok = wt > 0
    A = np.c_[np.ones(ok.sum()), centre[ok]]
    W = np.sqrt(wt[ok])
    coef, *_ = np.linalg.lstsq(A * W[:, None], contrast[ok] * W, rcond=None)
    fit = A @ coef
    ss_res = (wt[ok] * (contrast[ok] - fit) ** 2).sum()
    ss_tot = (wt[ok] * (contrast[ok] - np.average(contrast[ok], weights=wt[ok])) ** 2).sum()
    rows.append(dict(indicator=name, importance=imp[name], slope=coef[1],
                     r2_lin=1 - ss_res / ss_tot if ss_tot > 0 else np.nan))
ebm_tab = pd.DataFrame(rows).set_index("indicator")
w_ebm = ebm_tab.loc[signal, "slope"] / ebm_tab.loc[signal, "slope"].abs().sum()
ebm_tab["w_EBM"] = w_ebm
ebm_tab = ebm_tab.sort_values("importance", ascending=False)
ebm_tab.to_csv(RESULTS / "ebm_terms.csv")
print(ebm_tab.head(10).to_string(float_format="%.3f"))
rule_w = pd.Series({c: w for w, c in RULE})
print(f"corr(w_EBM, recovered rule) = {np.corrcoef(w_ebm[signal], rule_w[signal])[0, 1]:.3f}")

# ---------------- (2) ALE of tuned XGBoost on P(High), 20 quantile bins
xgb = XGBClassifier(n_estimators=400, max_depth=args.xgb_depth, learning_rate=args.xgb_lr,
                    random_state=0, verbosity=0).fit(Xtr, ytr)


def ale_range(model, data, col, cls=2, bins=20):
    x = data[col].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    eff = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        lo_, hi_ = data[m].copy(), data[m].copy()
        lo_[col], hi_[col] = edges[b], edges[b + 1]
        eff[b] = (model.predict_proba(hi_)[:, cls] - model.predict_proba(lo_)[:, cls]).mean()
        cnt[b] = m.sum()
    ale = np.cumsum(eff)
    ale -= np.average(ale, weights=cnt)
    return ale.max() - ale.min(), ale[-1] - ale[0]


ale = pd.DataFrame([dict(indicator=c, ale_range=ale_range(xgb, Xte, c)[0],
                         direction=np.sign(ale_range(xgb, Xte, c)[1])) for c in feat])
ale = ale.sort_values("ale_range", ascending=False)
ale.to_csv(RESULTS / "ale_ranges.csv", index=False)
print(ale.head(9).to_string(index=False, float_format="%.3f"))

# ---------------- (3) permutation importance (macro-F1, 10 repeats)
pi = permutation_importance(xgb, Xte, yte, n_repeats=10, random_state=0, scoring="f1_macro")
pi = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)
pi.to_csv(RESULTS / "permutation_importance.csv", header=["importance"])
print(f"Permutation importance: signal {pi[signal].min():.3f}-{pi[signal].max():.3f}; "
      f"max other {pi.drop(signal).max():.3f}")

# ---------------- (4) dimension ablation with LightGBM (5-fold CV)
def cv_f1(cols):
    m = LGBMClassifier(n_estimators=400, num_leaves=args.lgbm_leaves, verbose=-1, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return cross_val_score(m, X[cols], y, cv=cv, scoring="f1_macro").mean()

full = cv_f1(feat)
abl = []
for dim, cols in DIMENSIONS.items():
    cols = [c for c in cols if c in feat]
    abl.append(dict(dimension=dim, removed_loss=full - cv_f1([c for c in feat if c not in cols]),
                    alone=cv_f1(cols)))
abl = pd.DataFrame(abl)
abl.to_csv(RESULTS / "ablation.csv", index=False)
print(f"Full model macro-F1 = {full:.3f}; seven signal indicators only = {cv_f1(signal):.3f}")
print(abl.to_string(index=False, float_format="%.3f"))
