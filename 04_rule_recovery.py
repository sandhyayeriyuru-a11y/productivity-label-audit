"""Section V-D: rule recovery (Table IV, Eq. 2), exact reproduction and sensitivity."""
import itertools
import numpy as np
import pandas as pd
from common import load, modelled, target, RULE, CUTS, RESULTS, SEEDS
from rule_recovery import diagnose, round_to

d = load()
y = target(d)
feat = modelled(d)
X = d[feat].to_numpy(float)

res = diagnose(X, y, return_folds=True)
W = np.array([w for w, _ in res["folds"]])
T = np.array([t for _, t in res["folds"]])
sel = (W != 0).sum(axis=0)
tab = pd.DataFrame({"indicator": feat, "selected_folds": sel,
                    "w_mean": W.mean(0), "w_sd": W.std(0, ddof=1),
                    "w_rounded": round_to(W.mean(0), 0.05)})
tab = tab[tab.selected_folds > 0].sort_values("w_mean", key=abs, ascending=False)
tab.to_csv(RESULTS / "table4_recovered_rule.csv", index=False)
print(tab.to_string(index=False, float_format="%.4f"))
print(f"Thresholds: tau1 = {T[:, 0].mean():.3f}, tau2 = {T[:, 1].mean():.3f}")
print(f"Held-out accuracy: unrounded {res['linear']:.2f}%, rounded {res['rounded']:.2f}%; "
      f"distinct rounded rules across {len(SEEDS) * 5} folds: {res['n_rules']}; "
      f"exact rule: {res['exact_rule']}")

# ---- exact reproduction of all labels
def score(order):
    s = np.zeros(len(d))
    for w, c in order:
        s = s + w * d[c].to_numpy()
    return s

cls = lambda s: (s > CUTS[0]).astype(int) + (s > CUTS[1]).astype(int)
s4 = np.round(score(RULE), 4)
print(f"Eq. (2) at four decimals matches {(cls(s4) == y).sum()} of {len(y)} labels")
orders = list(itertools.permutations(RULE))
ok = sum((cls(score(o)) == y).all() for o in orders)
print(f"Floating point: all labels matched in {ok} of {len(orders)} summation orders")
s = score(RULE)
near = (np.minimum(abs(s - CUTS[0]), abs(s - CUTS[1])) < 1e-9).sum()
print(f"Records within 1e-9 of a cut-point: {near}")

# ---- sensitivity (Section V-D, second paragraph)
rows = []
for g in (0.01, 0.025, 0.05, 0.10):
    r = diagnose(X, y, grid=g, with_lgbm=False)
    rows.append(dict(setting="grid", value=g, rounded=r["rounded"], n_rules=r["n_rules"]))
for C in (0.005, 0.02, 0.05, 0.1, 0.5, 1.0):
    r = diagnose(X, y, C_screen=C, with_lgbm=False)
    rows.append(dict(setting="C_screen", value=C, rounded=r["rounded"], n_rules=r["n_rules"]))
sens = pd.DataFrame(rows)
sens.to_csv(RESULTS / "rule_sensitivity.csv", index=False)
print(sens.to_string(index=False, float_format="%.2f"))
