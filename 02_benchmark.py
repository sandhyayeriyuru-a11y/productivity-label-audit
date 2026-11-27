"""Section V-B: eleven classifiers tuned by nested cross-validation (Table III).

Outer: stratified 5-fold, seeds 42, 43, 44. Inner: stratified 3-fold grid search on macro-F1.
LR is compared with each competitor by fold wins and a Bayesian correlated t-test
(rho = 1/5, region of practical equivalence 0.01).

Usage:  python 02_benchmark.py            # full run (slow: EBM and SVM dominate)
        python 02_benchmark.py --quick    # one seed, for a smoke test
        python 02_benchmark.py --models LR XGBoost
"""
import argparse
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from common import load, modelled, target, RESULTS, SEEDS

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--models", nargs="*")
ap.add_argument("--jobs", type=int, default=-1)
args = ap.parse_args()

d = load()
y = target(d)
num = modelled(d)
X = d[num + ["Industry", "Year"]].copy()
X["Year"] = X["Year"].astype(str)


def pipe(model):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(drop="first"), ["Industry", "Year"])])
    return Pipeline([("pre", pre), ("m", model)])


MODELS = {
    "LR": (LogisticRegression(max_iter=5000),
           {"m__C": [0.1, 1, 10, 100], "m__class_weight": [None, "balanced"]}),
    "EBM": (ExplainableBoostingClassifier(interactions=0, random_state=0, n_jobs=1),
            {"m__learning_rate": [0.015, 0.05], "m__max_leaves": [2, 3]}),
    "LightGBM": (LGBMClassifier(n_estimators=400, verbose=-1, random_state=0, n_jobs=1),
                 {"m__num_leaves": [7, 15, 31], "m__class_weight": [None, "balanced"]}),
    "XGBoost": (XGBClassifier(n_estimators=400, random_state=0, n_jobs=1, verbosity=0),
                {"m__max_depth": [2, 4], "m__learning_rate": [0.05, 0.1]}),
    "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=0), {"m__C": [1, 10, 100]}),
    "MLP": (MLPClassifier(max_iter=1000, random_state=0),
            {"m__alpha": [1e-4, 1e-2], "m__hidden_layer_sizes": [(64, 32), (128,)]}),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1),
                      {"m__max_features": ["sqrt", 0.5], "m__min_samples_leaf": [1, 5]}),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=300, random_state=0, n_jobs=1),
                    {"m__max_features": ["sqrt", 0.5, 1.0]}),
    "Gaussian NB": (GaussianNB(), None),
    "Decision Tree": (DecisionTreeClassifier(random_state=0),
                      {"m__max_depth": [4, 8, None], "m__class_weight": [None, "balanced"]}),
    "k-NN": (KNeighborsClassifier(), {"m__n_neighbors": [5, 15, 31]}),
}
names = args.models or list(MODELS)
seeds = SEEDS[:1] if args.quick else SEEDS

scores, best = {}, {}
for name in names:
    model, grid = MODELS[name]
    t0, rows, params = time.time(), [], []
    for sd in seeds:
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y)):
            est = pipe(model)
            if grid:
                est = GridSearchCV(est, grid, scoring="f1_macro", n_jobs=args.jobs,
                                   cv=StratifiedKFold(3, shuffle=True, random_state=sd))
            est.fit(X.iloc[tr], y[tr])
            if grid:
                params.append(str(est.best_params_))
            p, pr = est.predict(X.iloc[te]), est.predict_proba(X.iloc[te])
            rows.append(dict(seed=sd, fold=k,
                             accuracy=accuracy_score(y[te], p),
                             balanced_acc=balanced_accuracy_score(y[te], p),
                             macro_f1=f1_score(y[te], p, average="macro"),
                             mcc=matthews_corrcoef(y[te], p),
                             qwk=cohen_kappa_score(y[te], p, weights="quadratic"),
                             auc_ovr=roc_auc_score(y[te], pr, multi_class="ovr")))
            if sd == 42 and name in ("LR", "XGBoost"):
                best.setdefault(name, []).append((te, p))
    scores[name] = pd.DataFrame(rows)
    mode = pd.Series(params).mode()[0] if params else "-"
    print(f"{name:14s} macro-F1 {scores[name].macro_f1.mean():.4f} ± {scores[name].macro_f1.std():.4f}"
          f"  ({time.time() - t0:.0f}s)  most selected: {mode}", flush=True)

# ---- Table III with the Bayesian correlated t-test against LR
def bayes_corr_ttest(diff, rho=0.2, rope=0.01):
    n = len(diff)
    scale = np.sqrt((1 / n + rho / (1 - rho)) * diff.var(ddof=1))
    return 1 - stats.t.cdf((rope - diff.mean()) / scale, n - 1)

table = []
for name, s in scores.items():
    row = {"model": name}
    for m in ["accuracy", "balanced_acc", "macro_f1", "mcc", "qwk", "auc_ovr"]:
        row[m] = f"{s[m].mean():.4f} ± {s[m].std():.4f}"
    if "LR" in scores and name != "LR":
        diff = scores["LR"].macro_f1.to_numpy() - s.macro_f1.to_numpy()
        row["P(LR better)"] = round(bayes_corr_ttest(diff), 4)
        row["LR fold wins"] = int((diff > 0).sum())
    table.append(row)
tab = pd.DataFrame(table).sort_values("macro_f1", ascending=False)
tab.to_csv(RESULTS / "table3_benchmark.csv", index=False)
pd.concat(scores, names=["model"]).to_csv(RESULTS / "benchmark_per_fold.csv")
print(tab.to_string(index=False))

# pooled out-of-fold confusion matrices, seed-42 split (Fig. 6)
for name, parts in best.items():
    pred = np.empty_like(y)
    for te, p in parts:
        pred[te] = p
    cm = confusion_matrix(y, pred)
    print(f"{name} pooled confusion matrix (Low, Medium, High); errors = {len(y) - np.trace(cm)}\n{cm}")
