"""Two single-fit checks reported in the article but not covered elsewhere.

1. Per-column standardised features under Protocol B (Table 1, second row):
   expected MAE 17.57 m, median 13.57 m, RMSE 25.85 m.
2. Floor-conditioned routing under Protocol B (Section 3): expected MAE 11.35 m
   against 10.85 m for building-only routing, floor identified for 90.0% of the
   holdout; the building classifier of the building-only instrument identifies
   the building for 99.8%.

Run from the directory containing uji/UJIndoorLoc, as the other scripts are.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from region_svr import RegionSVR, BUILDING, FLOOR
from uji_data import load_uji, features, targets, error_2d, metrics

P = dict(C=10, gamma=2e-5, C_clf=100, gamma_clf=2e-5)
tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr); xv, yv = targets(va)
b, bv = tr[:, BUILDING], va[:, BUILDING]
f, fv = tr[:, FLOOR], va[:, FLOOR]
f0 = np.zeros(len(tr))

# 1 -- per-column standardised features, building-only routing
s = StandardScaler().fit(X)
m = RegionSVR(**P).fit(s.transform(X), xt, yt, b, f0)
px, py = m.predict(s.transform(Xv))
print("standardised, Protocol B:",
      {k: round(v, 2) for k, v in metrics(error_2d(xv, yv, px, py)).items()})

# 2 -- floor-conditioned routing, raw features
m = RegionSVR(**P).fit(X, xt, yt, b, f)
px, py = m.predict(Xv)
bacc, facc = m.region_accuracy(Xv, bv, fv)
print("floor-conditioned, Protocol B:",
      {k: round(v, 2) for k, v in metrics(error_2d(xv, yv, px, py)).items()},
      f"building acc {bacc:.4f}, floor acc {facc:.4f}")

# building accuracy of the building-only instrument
m = RegionSVR(**P).fit(X, xt, yt, b, f0)
bacc, _ = m.region_accuracy(Xv, bv, np.zeros(len(va)))
print(f"building-only instrument: building acc {bacc:.4f}")
