"""Floor-aware sensitivity analysis (Part 11).

The main measurand is planar. This asks whether the protocol-induced effect
survives when the vertical dimension is admitted, WITHOUT adopting an arbitrary
floor penalty as the primary answer: floor identification accuracy is itself a
protocol-sensitive quantity and needs no constant. A penalised error is then
reported across a range of penalties to show the direction is not an artefact
of any one choice.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.svm import SVC
from region_svr import RegionSVR, BUILDING
from uji_data import load_uji, features, targets, error_2d

FLOOR = 522
P = dict(C=10, gamma=2e-5, C_clf=100, gamma_clf=2e-5)

tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr); xv, yv = targets(va)
b = tr[:, BUILDING]; f0 = np.zeros(len(tr))
fl = tr[:, FLOOR]; flv = va[:, FLOOR]
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

def run(Xa, xa, ya, ba, fa, Xb, xb, yb, fb, tag):
    m = RegionSVR(**P).fit(Xa, xa, ya, ba, np.zeros(len(Xa)))
    px, py = m.predict(Xb)
    e = error_2d(xb, yb, px, py)
    clf = SVC(kernel="rbf", C=100, gamma=2e-5, cache_size=500).fit(Xa, fa)
    fhat = clf.predict(Xb)
    acc = float(np.mean(fhat == fb))
    out = {"tag": tag, "n": int(len(e)), "MAE": float(np.mean(e)), "floor_acc": acc}
    for pen in (0, 2, 4, 6, 8):
        out[f"pen{pen}"] = float(np.mean(e + pen * np.abs(fhat - fb)))
    print(f"  {tag:14s} n={len(e):5d} MAE={out['MAE']:6.2f}  floor acc={acc*100:5.1f}%  "
          + "  ".join(f"p{p}={out[f'pen{p}']:6.2f}" for p in (0,4,8)), flush=True)
    return out

res = []
# Protocol A -- random division
i, j = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
res.append(run(X[i], xt[i], yt[i], b[i], fl[i], X[j], xt[j], yt[j], fl[j], "A random"))
# Protocol C -- reference-point-disjoint, division 0
i, j = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, xt, g))
res.append(run(X[i], xt[i], yt[i], b[i], fl[i], X[j], xt[j], yt[j], fl[j], "C RP-disjoint"))
# Protocol B -- device/time holdout
res.append(run(X, xt, yt, b, fl, Xv, xv, yv, flv, "B holdout"))

json.dump(res, open("out_floor.json", "w"), indent=2)
print("\nwrote out_floor.json")
