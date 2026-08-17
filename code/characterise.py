"""Definitive characterisation on the full UJI training set.

Reports each configuration under TWO protocols:
  A) random 80/20 split  -- what the submitted paper used
  B) official validation set -- different users, different phones, months later

The gap between A and B is the generalisation claim the AE asked about.
"""
import json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from corrected_pipeline import fit_predict
from uji_data import load_uji, features, targets, error_2d, metrics

tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr)
xv, yv = targets(va)

i_tr, i_te = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42)

CONFIGS = [(10, 5e-6, 0.1), (10, 1e-5, 0.1), (10, 2e-5, 0.1),
           (100, 1e-5, 0.1), (1, 2e-5, 0.1), (100, 5e-6, 0.1)]

rows = []
for C, g, e in CONFIGS:
    t = time.time()
    # A: random split, train on 80%
    ax = fit_predict(X[i_tr], xt[i_tr], X[i_te], C, g, e)
    ay = fit_predict(X[i_tr], yt[i_tr], X[i_te], C, g, e)
    mA = metrics(error_2d(xt[i_te], yt[i_te], ax, ay))
    # B: train on ALL training data, test on official validation set
    bx = fit_predict(X, xt, Xv, C, g, e)
    by = fit_predict(X, yt, Xv, C, g, e)
    mB = metrics(error_2d(xv, yv, bx, by))
    rows.append({"C": C, "gamma": g, "eps": e, "A_random": mA, "B_official": mB})
    print(f"C={C:<5g} g={g:.1e} | A(random) MAE={mA['MAE']:6.2f} med={mA['median']:5.2f} "
          f"| B(official) MAE={mB['MAE']:6.2f} med={mB['median']:5.2f} "
          f"| gap x{mB['MAE']/mA['MAE']:.2f}  ({time.time()-t:.0f}s)", flush=True)
    json.dump(rows, open("out_characterise.json", "w"), indent=2)

# kNN reference under both protocols
print("\nkNN k=3 reference:")
kA = KNeighborsRegressor(3).fit(X[i_tr], np.c_[xt[i_tr], yt[i_tr]]).predict(X[i_te])
mA = metrics(error_2d(xt[i_te], yt[i_te], kA[:, 0], kA[:, 1]))
kB = KNeighborsRegressor(3).fit(X, np.c_[xt, yt]).predict(Xv)
mB = metrics(error_2d(xv, yv, kB[:, 0], kB[:, 1]))
print(f"  A(random) MAE={mA['MAE']:.2f} med={mA['median']:.2f} | "
      f"B(official) MAE={mB['MAE']:.2f} med={mB['median']:.2f} | gap x{mB['MAE']/mA['MAE']:.2f}")
json.dump({"svr": rows, "knn": {"A": mA, "B": mB}},
          open("out_characterise.json", "w"), indent=2)
