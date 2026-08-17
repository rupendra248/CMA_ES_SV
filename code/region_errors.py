"""Per-sample errors of the region-conditioned localiser under each protocol.

The CDF figure must plot the same instrument that Table 2 reports. The earlier
make_figure.py used the unconditioned SVR, whose errors are larger throughout.
Protocol C is shown for a single reference-point-disjoint division (seed 0),
whose MAE of 9.26 m lies within 0.05 m of the ten-division mean.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from region_svr import RegionSVR, BUILDING
from uji_data import load_uji, features, targets, error_2d

P = dict(C=10, gamma=2e-5, C_clf=100, gamma_clf=2e-5)

tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr); xv, yv = targets(va)
b = tr[:, BUILDING]; f0 = np.zeros(len(tr))
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

err = {}

def knn(Xa, xa, ya, Xb, xb, yb):
    k = KNeighborsRegressor(3).fit(Xa, np.c_[xa, ya]).predict(Xb)
    return error_2d(xb, yb, k[:, 0], k[:, 1])

# A -- random division (same seed as Table 2)
i, j = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
m = RegionSVR(**P).fit(X[i], xt[i], yt[i], b[i], f0[i])
px, py = m.predict(X[j])
err["SVR_A"] = error_2d(xt[j], yt[j], px, py)
err["kNN_A"] = knn(X[i], xt[i], yt[i], X[j], xt[j], yt[j])
print(f"A  SVR {np.mean(err['SVR_A']):6.2f}   kNN {np.mean(err['kNN_A']):6.2f}", flush=True)

# C -- reference-point-disjoint, division 0
i, j = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, xt, g))
assert not (set(g[j]) & set(g[i]))
m = RegionSVR(**P).fit(X[i], xt[i], yt[i], b[i], f0[i])
px, py = m.predict(X[j])
err["SVR_C"] = error_2d(xt[j], yt[j], px, py)
err["kNN_C"] = knn(X[i], xt[i], yt[i], X[j], xt[j], yt[j])
print(f"C  SVR {np.mean(err['SVR_C']):6.2f}   kNN {np.mean(err['kNN_C']):6.2f}", flush=True)

# B -- device and time holdout
m = RegionSVR(**P).fit(X, xt, yt, b, f0)
px, py = m.predict(Xv)
err["SVR_B"] = error_2d(xv, yv, px, py)
err["kNN_B"] = knn(X, xt, yt, Xv, xv, yv)
print(f"B  SVR {np.mean(err['SVR_B']):6.2f}   kNN {np.mean(err['kNN_B']):6.2f}", flush=True)

json.dump({k: np.asarray(v).tolist() for k, v in err.items()},
          open("out_region_errors.json", "w"))
print("\nwrote out_region_errors.json")
