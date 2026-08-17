"""Statistical analysis for the two under-powered claims.

(1) Optimiser comparison: paired Wilcoxon signed-rank, CMA-ES vs each rival,
    over all available seeds, plus a bootstrap CI on the mean paired difference.
(2) Repeatability: dispersion of MAE/RMSE across RP-disjoint partitions, with a
    Brown-Forsythe test for unequal spread and a bootstrap CI on the SD ratio.
"""
import json
import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)


def boot_ci(fn, *arrays, n=20000, alpha=0.05):
    """Percentile bootstrap CI, resampling paired observations together."""
    k = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ------------------------------------------------------------ 1. optimisers
import glob, os

recs = json.load(open("out_optimisers.json"))            # seeds 0-2, sequential
for f in ["out_opt_carry.json"] + sorted(glob.glob("out_opt_seed*.json")):
    if os.path.exists(f):
        recs += json.load(open(f))                       # seed 3, then 4-9

by = {}
for r in recs:
    by.setdefault(r["optimiser"], {})[r["seed"]] = r["MAE"]

print("=" * 70)
print("OPTIMISER COMPARISON")
print("=" * 70)
print("  seeds available per optimiser:")
for k in sorted(by):
    print(f"    {k:8s} n={len(by[k]):2d}  seeds {sorted(by[k])}")

print("\n  marginal means (over each optimiser's own seeds):")
for k in sorted(by, key=lambda k: np.mean(list(by[k].values()))):
    v = np.array(list(by[k].values()))
    print(f"    {k:8s} MAE {v.mean():6.2f} +/- {v.std(ddof=1):5.2f}  "
          f"[{v.min():.2f}, {v.max():.2f}]")

# Pairwise-complete: each comparison uses every seed where BOTH finished.
# Seed 4's PSO/DE runs died on a 60-min refit, so those pairs have one seed
# fewer; CMA-ES vs random search is unaffected and uses all seeds.
print("\n  paired tests vs CMA-ES (pairwise-complete):")
for k in sorted(by):
    if k == "CMA-ES":
        continue
    seeds = sorted(set(by["CMA-ES"]) & set(by[k]))
    cma = np.array([by["CMA-ES"][s] for s in seeds])
    o = np.array([by[k][s] for s in seeds])
    d = o - cma                      # positive => CMA-ES better
    wins = int((d > 0).sum())
    try:
        p = f"p={stats.wilcoxon(o, cma, alternative='greater').pvalue:.4f}"
    except ValueError:
        p = "p=n/a"
    lo, hi = boot_ci(lambda a, b: float(np.mean(a - b)), o, cma)
    sig = "SIGNIFICANT" if "p=" in p and float(p[2:]) < 0.05 else "not significant"
    print(f"    vs {k:8s} n={len(seeds):2d}  mean diff {d.mean():+5.2f} m  "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]  CMA-ES better on {wins}/{len(seeds)}  "
          f"{p}  -> {sig}")

# dispersion across seeds: an optimiser that occasionally lands badly is a risk
print("\n  spread across seeds (worst-case behaviour):")
for k in sorted(by, key=lambda k: np.std(list(by[k].values()), ddof=1)):
    v = np.array(list(by[k].values()))
    print(f"    {k:8s} SD {v.std(ddof=1):5.2f}  worst {v.max():6.2f} m")

# ------------------------------------------------------------ 2. repeatability
try:
    rep = json.load(open("out_repeat10.json"))
except FileNotFoundError:
    print("\n(repeatability run not finished)")
    raise SystemExit

print("\n" + "=" * 66)
print(f"REPEATABILITY  ({len(rep)} reference-point-disjoint partitions)")
print("=" * 66)
for field in ("MAE", "RMSE", "p90"):
    s = np.array([r["svr"][field] for r in rep])
    n = np.array([r["knn"][field] for r in rep])
    ratio = n.std(ddof=1) / s.std(ddof=1)
    bf = stats.levene(s, n, center="median")   # Brown-Forsythe
    lo, hi = boot_ci(lambda a, b: float(b.std(ddof=1) / a.std(ddof=1)), s, n)
    print(f"\n  {field}")
    print(f"    SVR  {s.mean():6.2f} +/- {s.std(ddof=1):5.3f}")
    print(f"    kNN  {n.mean():6.2f} +/- {n.std(ddof=1):5.3f}")
    print(f"    SD ratio kNN/SVR = {ratio:5.1f}x   95% CI [{lo:.1f}, {hi:.1f}]   "
          f"Brown-Forsythe p={bf.pvalue:.4f}")

print("\nnote: SD ratio CI is a percentile bootstrap over partitions; with 10")
print("partitions it is wide, and is reported as such rather than as a point value.")
