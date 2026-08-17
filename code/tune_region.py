"""Tune the region-conditioned localiser, honouring the article's own discipline.

Selection is performed upon an inner partition that is reference-point-disjoint
from the fitting fold, and drawn entirely from the training data. The held-out
set is not consulted until the configuration is fixed.
"""
import itertools, json, sys, time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from region_svr import RegionSVR, BUILDING, FLOOR
from uji_data import load_uji, features, targets, error_2d

tr, _ = load_uji()
X = features(tr, "raw")
xt, yt = targets(tr)
b, f = tr[:, BUILDING], tr[:, FLOOR]
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

i_fit, i_val = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, xt, g))
assert not (set(g[i_val]) & set(g[i_fit]))
print(f"inner fit {len(i_fit)}  inner val {len(i_val)}  (reference-point-disjoint)\n", flush=True)

GRID = list(itertools.product(
    [10, 100, 1000],                # C for the regressors
    [2e-5, 1e-4, 5e-4],             # gamma for the regressors
))
CLF = [(100, 2e-5), (1000, 1e-4)]   # classifier settings

best = (1e9, None)
for (C, gam) in GRID:
    for (Cc, gc) in CLF:
        t0 = time.time()
        m = RegionSVR(C=C, gamma=gam, C_clf=Cc, gamma_clf=gc).fit(
            X[i_fit], xt[i_fit], yt[i_fit], b[i_fit], f[i_fit])
        px, py = m.predict(X[i_val])
        mae = float(np.mean(error_2d(xt[i_val], yt[i_val], px, py)))
        ab, af = m.region_accuracy(X[i_val], b[i_val], f[i_val])
        if mae < best[0]:
            best = (mae, dict(C=C, gamma=gam, C_clf=Cc, gamma_clf=gc))
        print(f"  C={C:<5g} g={gam:.0e} | clf C={Cc:<5g} g={gc:.0e} | "
              f"inner MAE={mae:6.2f}  bldg={ab:.3f} floor={af:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

print("\nbest on inner (RP-disjoint) partition:", best)
json.dump({"inner_mae": best[0], "params": best[1]}, open("out_region_tuned.json", "w"), indent=2)
