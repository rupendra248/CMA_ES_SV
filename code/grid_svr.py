"""Coarse grid over SVR hyperparameters on the FULL UJI training set,
scored on an inner split (never on the official validation set),
to locate the region CMA-ES should be searching.
"""
import json, time, itertools
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from uji_data import load_uji, features, targets, error_2d, metrics

tr, va = load_uji()
xt, yt = targets(tr)

# inner split of the TRAINING data only -- tuning never sees the official val set
idx_tr, idx_in = train_test_split(np.arange(len(tr)), test_size=0.25, random_state=0)


def evaluate(mode, C, gamma, eps, itr, iin):
    X = features(tr, mode)
    out = []
    for tgt in (xt, yt):
        s = StandardScaler()
        ys = s.fit_transform(tgt[itr].reshape(-1, 1)).ravel()
        m = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps).fit(X[itr], ys)
        p = m.predict(X[iin])
        out.append(s.inverse_transform(p.reshape(-1, 1)).ravel())
    return metrics(error_2d(xt[iin], yt[iin], out[0], out[1]))


results = []
grid = list(itertools.product(["powed", "raw"], [1, 10, 100], ["scale", 0.3, 1.0, 3.0]))
print(f"{len(grid)} configs on {len(idx_tr)} train / {len(idx_in)} inner-val samples\n")
for mode, C, gamma in grid:
    t = time.time()
    m = evaluate(mode, C, gamma, 0.1, idx_tr, idx_in)
    dt = time.time() - t
    results.append({"mode": mode, "C": C, "gamma": gamma, **m, "sec": dt})
    print(f"{mode:5s} C={C:<5g} gamma={str(gamma):6s} "
          f"MAE={m['MAE']:6.2f} RMSE={m['RMSE']:6.2f} med={m['median']:6.2f} "
          f"p75={m['p75']:6.2f}  ({dt:.0f}s)", flush=True)

results.sort(key=lambda r: r["MAE"])
print("\nbest by MAE:")
for r in results[:5]:
    print(f"  {r['mode']:5s} C={r['C']:<5g} gamma={str(r['gamma']):6s} MAE={r['MAE']:.2f} m")
json.dump(results, open("out_grid.json", "w"), indent=2)
