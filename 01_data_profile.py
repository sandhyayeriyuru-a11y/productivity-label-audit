"""Section V-A: generator tests (Table I), independence check, Kruskal-Wallis (Table II)."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from common import load, indicators, target, EQUITY, RESULTS

d = load()
y = target(d)
ind = indicators(d)
N = len(d)

# ---------------- Class distribution and industry association (Fig. 1)
print("Class counts:", d.Productivity_Class.value_counts().to_dict())
ct = pd.crosstab(d.Industry, d.Productivity_Class)
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"Industry x class: chi2({dof}) = {chi2:.2f}, p = {p:.2f}, "
      f"Cramer's V = {np.sqrt(chi2 / (N * (min(ct.shape) - 1))):.3f}")

# ---------------- Generator tests (Table I)
CONTINUOUS = {"R&D_Expenditure": (1, 10), "Digital_Investment": (1, 8),
              "Revenue_per_Employee": (50, 300)}


def rounded_uniform_chi2(x, step=0.01):
    k = np.round(x / step).astype(int)
    lo, hi = k.min(), k.max()
    grid = np.arange(lo, hi + 1)
    obs = pd.Series(k).value_counts().reindex(grid, fill_value=0).to_numpy()
    w = np.ones(len(grid)); w[0] = w[-1] = 0.5          # end points at half weight
    exp = w / w.sum() * len(x)
    return stats.chisquare(obs, exp).pvalue, (lo * step, hi * step)


rows = []
for c in ind:
    if c in EQUITY:
        continue
    x = d[c].to_numpy()
    if c in CONTINUOUS:
        a, b = CONTINUOUS[c]
        p = stats.kstest(x, "uniform", args=(a, b - a)).pvalue
        rng, test = (a, b), "KS vs continuous uniform"
    elif c == "Patent_Count":
        vals = np.arange(x.min(), x.max() + 1)
        obs = pd.Series(x).value_counts().reindex(vals, fill_value=0).to_numpy()
        p = stats.chisquare(obs).pvalue
        rng, test = (x.min(), x.max()), "chi2 vs discrete uniform"
    else:
        p, rng = rounded_uniform_chi2(x)
        test = "chi2 vs rounded uniform"
    rows.append(dict(indicator=c, range=f"[{rng[0]:g}, {rng[1]:g}]", test=test,
                     p=p, excess_kurtosis=stats.kurtosis(x)))
gen = pd.DataFrame(rows)
gen.to_csv(RESULTS / "table1_generator_tests.csv", index=False)
print(gen.to_string(index=False, float_format="%.3f"))

# equity rows = four integers in 0..100 divided by their sum
E = d[EQUITY].to_numpy()
exact = 0
for r in E:
    for S in range(1, 401):
        v = r * S
        if np.allclose(v, np.round(v), atol=1e-4) and np.round(v).max() <= 100:
            exact += 1
            break
print(f"Equity rows with an exact integer/sum representation: {exact} of {N}")
print("Records with a zero equity share:", int((E == 0).any(axis=1).sum()))

# ---------------- Independence vs. a simulated reference
eq_idx = {ind.index(c) for c in EQUITY}
pairs = [(i, j) for i, j in itertools.combinations(range(len(ind)), 2)
         if not (i in eq_idx and j in eq_idx)]
rho = stats.spearmanr(d[ind].to_numpy()).statistic
obs_max = max(abs(rho[i, j]) for i, j in pairs)
rng = np.random.default_rng(0)
sim = []
for _ in range(300):
    X = np.round(rng.uniform(0.3, 1.0, (N, len(ind))), 2)
    r = stats.spearmanr(X).statistic
    sim.append(max(abs(r[i, j]) for i, j in pairs))
q = np.percentile(sim, [50, 75, 95])
print(f"{len(pairs)} pairs: observed max |rho| = {obs_max:.3f}; simulated median {q[0]:.3f}, "
      f"75th pct {q[1]:.3f}, 95th pct {q[2]:.3f}")

# ---------------- Kruskal-Wallis across classes (Table II)
k = 3
kw = []
for c in ind:
    groups = [d.loc[y == g, c] for g in (0, 1, 2)]
    H, p = stats.kruskal(*groups)
    kw.append(dict(indicator=c, mean_low=groups[0].mean(), mean_med=groups[1].mean(),
                   mean_high=groups[2].mean(), H=H, p_bonf=min(1.0, p * len(ind)),
                   eta2_H=(H - k + 1) / (N - k)))
kw = pd.DataFrame(kw).sort_values("eta2_H", ascending=False)
kw.to_csv(RESULTS / "table2_kruskal_wallis.csv", index=False)
print(kw.head(9).to_string(index=False, float_format="%.4g"))

# centred log-ratio of equity shares, multiplicative zero replacement
delta = 0.0023
Ez = E.copy()
for i, r in enumerate(Ez):
    z = r == 0
    if z.any():
        r[~z] *= 1 - z.sum() * delta
        r[z] = delta
clr = np.log(Ez) - np.log(Ez).mean(axis=1, keepdims=True)
for j, c in enumerate(EQUITY):
    p = stats.kruskal(*[clr[y == g, j] for g in (0, 1, 2)]).pvalue
    print(f"clr({c}): Kruskal-Wallis p = {p:.2f}")
