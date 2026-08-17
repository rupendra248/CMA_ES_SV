"""The same thinning experiment upon XJTLUIndoorLoc.

If reference-point spacing alone governs the inflation, the points obtained from
this database should fall upon the curve traced by UJIIndoorLoc, despite the two
surveys differing in building, hardware, and access point count.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from corrected_pipeline import fit_predict
from uji_data import error_2d

C, GAMMA, EPS = 10, 2e-5, 0.1
SENT = -110.0


def load(name):
    d = pd.read_csv(f"xjtlu/Dataset/{name}.csv")
    wap = [c for c in d.columns if c.startswith("WAP")]
    X = d[wap].values.astype(np.float64)
    x, y = d["Loc_x"].values.astype(float), d["Loc_y"].values.astype(float)
    uniq, g = np.unique(np.c_[x, y], axis=0, return_inverse=True)
    return X, x, y, g, uniq


def thin(uniq, min_d, seed):
    if min_d <= 0:
        return np.arange(len(uniq))
    rs = np.random.RandomState(seed)
    chosen = []
    for o in rs.permutation(len(uniq)):
        if all(np.hypot(*(uniq[o] - uniq[c])) >= min_d for c in chosen):
            chosen.append(o)
    return np.array(sorted(chosen))


def spacing(uniq, keep):
    P = uniq[keep]
    if len(P) < 3:
        return float("nan")
    d, _ = cKDTree(P).query(P, k=2)
    return float(np.median(d[:, 1]))


def inflation(X, xt, yt, g, seed):
    i, j = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    a = float(np.mean(error_2d(xt[j], yt[j],
        fit_predict(X[i], xt[i], X[j], C, GAMMA, EPS),
        fit_predict(X[i], yt[i], X[j], C, GAMMA, EPS))))
    i, j = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, xt, g))
    c = float(np.mean(error_2d(xt[j], yt[j],
        fit_predict(X[i], xt[i], X[j], C, GAMMA, EPS),
        fit_predict(X[i], yt[i], X[j], C, GAMMA, EPS))))
    return a, c


out = []
for src in ("Dataset_4floor_HuaWei", "Dataset_5floor_HuaWei"):
    X, xt, yt, g, uniq = load(src)
    for d in (0, 2, 3, 4):
        for seed in range(10):
            keep = thin(uniq, d, seed)
            if len(keep) < 25:
                continue
            m = np.isin(g, keep)
            a, c = inflation(X[m], xt[m], yt[m], g[m], seed)
            sp = spacing(uniq, keep)
            out.append({"src": src, "target_d": d, "seed": seed, "spacing": sp,
                        "n": int(m.sum()), "rps": len(keep),
                        "A": a, "C": c, "inflation": c / a})
            print(f"  {src[-12:]} d={d} seed {seed}: rp={len(keep):4d} n={m.sum():5d} "
                  f"spacing={sp:5.2f} A={a:5.2f} C={c:5.2f} infl={c/a:5.3f}", flush=True)
json.dump(out, open("out_density_xjtlu.json", "w"), indent=2)
print("wrote out_density_xjtlu.json")
