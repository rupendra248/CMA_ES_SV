"""Third database: Tampere. A falsification test of the f^c prediction.

TIE1 and SAH1 record exactly one capture at each of their positions, so that
f^c = 0.2 and the prediction of the article is that a random division leaks
nothing and Protocols A and C must coincide. If they do not, the account of the
mechanism given in the article is wrong.

The database also allows the underlying physical claim to be tested separately.
The bias is attributed to the worth of the leaked information, which should fall
as neighbouring reference points come closer together. Thinning the positions of
this survey, whose native spacing is 0.40 m, tests whether the error under
spatial generalisation grows with spacing as that account requires.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from corrected_pipeline import fit_predict
from uji_data import error_2d

BASE = ("tampere/WiFi RSS measurements in Tampere University multi-building "
        "campus 2017 - Zenodo 5174851/")
SENT_IN, SENT_OUT = 100.0, -95.0
C, GAMMA, EPS = 10, 2e-5, 0.1


def load(tag):
    r = pd.read_csv(BASE + tag + "_training_rss.csv", header=None).values.astype(float)
    c = pd.read_csv(BASE + tag + "_training_coordinates.csv", header=None).values
    r[r == SENT_IN] = SENT_OUT
    x, y, fl = c[:, 0], c[:, 1], c[:, 3]
    _, g = np.unique(np.c_[np.round(x, 3), np.round(y, 3), fl], axis=0, return_inverse=True)
    return r, x, y, fl, g


def spacing(x, y, fl, keep):
    sp = []
    for f in np.unique(fl[keep]):
        P = np.unique(np.c_[x[keep], y[keep]][fl[keep] == f], axis=0)
        if len(P) < 3:
            continue
        d, _ = cKDTree(P).query(P, k=2)
        sp.append(np.median(d[:, 1]))
    return float(np.median(sp)) if sp else float("nan")


def thin(x, y, fl, min_d, seed):
    if min_d <= 0:
        return np.arange(len(x))
    rs = np.random.RandomState(seed)
    keep = []
    for f in np.unique(fl):
        idx = np.where(fl == f)[0]
        P = np.c_[x[idx], y[idx]]
        chosen = []
        for o in rs.permutation(len(idx)):
            if all(np.hypot(*(P[o] - P[c])) >= min_d for c in chosen):
                chosen.append(o)
        keep += list(idx[chosen])
    return np.array(sorted(keep))


def AC(X, x, y, g, seed):
    i, j = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    a = float(np.mean(error_2d(x[j], y[j],
        fit_predict(X[i], x[i], X[j], C, GAMMA, EPS),
        fit_predict(X[i], y[i], X[j], C, GAMMA, EPS))))
    i, j = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, x, g))
    c = float(np.mean(error_2d(x[j], y[j],
        fit_predict(X[i], x[i], X[j], C, GAMMA, EPS),
        fit_predict(X[i], y[i], X[j], C, GAMMA, EPS))))
    return a, c


out = []
for tag in ("TIE1", "SAH1"):
    X, x, y, fl, g = load(tag)
    print(f"\n{tag}: {len(X)} samples, {X.shape[1]} APs, {len(np.unique(g))} positions, "
          f"{len(X)/len(np.unique(g)):.2f} captures each", flush=True)
    for d in (0, 1, 2, 4, 8):
        for seed in (0, 1, 2):
            keep = thin(x, y, fl, d, seed)
            if len(keep) < 400:
                continue
            m = np.isin(np.arange(len(X)), keep)
            a, c = AC(X[m], x[m], y[m], g[m], seed)
            sp = spacing(x, y, fl, keep)
            out.append({"db": tag, "target_d": d, "seed": seed, "spacing": sp,
                        "n": int(m.sum()), "A": a, "C": c, "inflation": c / a})
            print(f"  d={d} seed {seed}: n={m.sum():5d} spacing={sp:5.2f} "
                  f"A={a:6.2f} C={c:6.2f} infl={c/a:5.3f}", flush=True)
    json.dump(out, open("out_tampere.json", "w"), indent=2)
print("\nwrote out_tampere.json")
