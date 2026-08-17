"""The region-conditioned localiser under all three protocols."""
import json
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from region_svr import RegionSVR, BUILDING, FLOOR
from uji_data import load_uji, features, targets, error_2d, metrics

P = dict(C=10, gamma=2e-5, C_clf=100, gamma_clf=2e-5)

tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr); xv, yv = targets(va)
b = tr[:, BUILDING]; f0 = np.zeros(len(tr))          # building-only routing
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

out = {}

# A -- random division
i, j = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
m = RegionSVR(**P).fit(X[i], xt[i], yt[i], b[i], f0[i])
px, py = m.predict(X[j])
out["A"] = metrics(error_2d(xt[j], yt[j], px, py))
print("A random     :", {k: round(v, 2) for k, v in out["A"].items() if k != "n"}, flush=True)

# C -- reference-point-disjoint, ten partitions
acc = []
for s in range(10):
    i, j = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(X, xt, g))
    m = RegionSVR(**P).fit(X[i], xt[i], yt[i], b[i], f0[i])
    px, py = m.predict(X[j])
    acc.append(metrics(error_2d(xt[j], yt[j], px, py)))
    print(f"  C partition {s}: MAE={acc[-1]['MAE']:.2f}", flush=True)
out["C"] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
out["C_sd"] = {k: float(np.std([a[k] for a in acc], ddof=1)) for k in acc[0]}

# B -- device and time holdout
m = RegionSVR(**P).fit(X, xt, yt, b, f0)
px, py = m.predict(Xv)
out["B"] = metrics(error_2d(xv, yv, px, py))
print("B holdout    :", {k: round(v, 2) for k, v in out["B"].items() if k != "n"}, flush=True)

json.dump(out, open("out_region_protocols.json", "w"), indent=2)
print("\nrow format  p25 & MAE & med & p75 & p90 & RMSE")
for k in ("A", "C", "B"):
    d = out[k]
    print("  %s: %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" %
          (k, d["p25"], d["MAE"], d["median"], d["p75"], d["p90"], d["RMSE"]))
