"""Second database: XJTLUIndoorLoc. Does reference-point leakage generalise?

XJTLU offers a decomposition UJIIndoorLoc cannot. The 4th floor was surveyed at
the same 306 reference points with two different handsets (Huawei EVA-AL10 and
Xiaomi MIX 2), so device transfer can be isolated from spatial generalisation
instead of being confounded with it:

  A  random split of the Huawei survey        -> re-identification
  C  RP-disjoint split of the Huawei survey   -> spatial generalisation, one device
  D  Huawei -> MIX 2, same reference points   -> device generalisation, no new positions
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from corrected_pipeline import fit_predict
from uji_data import error_2d, metrics

SENTINEL = -110.0        # XJTLU "not detected"
ROOT = "xjtlu/Dataset"


def load(name):
    d = pd.read_csv(f"{ROOT}/{name}.csv")
    wap = [c for c in d.columns if c.startswith("WAP")]
    X = d[wap].values.astype(np.float64)
    x, y = d["Loc_x"].values.astype(float), d["Loc_y"].values.astype(float)
    _, g = np.unique(np.c_[x, y], axis=0, return_inverse=True)
    return X, x, y, g


def evaluate(Xtr, xtr, ytr, Xte, xte, yte, C, gamma, eps):
    px = fit_predict(Xtr, xtr, Xte, C, gamma, eps)
    py = fit_predict(Xtr, ytr, Xte, C, gamma, eps)
    m_svr = metrics(error_2d(xte, yte, px, py))
    k = KNeighborsRegressor(3).fit(Xtr, np.c_[xtr, ytr]).predict(Xte)
    m_knn = metrics(error_2d(xte, yte, k[:, 0], k[:, 1]))
    return m_svr, m_knn


Xh, xh, yh, gh = load("Dataset_4floor_HuaWei")     # 4th floor, Huawei
Xm, xm, ym, gm = load("Dataset_4thFloor_MIX2")    # 4th floor, MIX 2, same RPs

# ---------------------------------------------------------------- leakage audit
print("=" * 70)
print("LEAKAGE AUDIT -- XJTLUIndoorLoc, 4th floor (Huawei)")
print("=" * 70)
print(f"  fingerprints                    : {len(Xh)}")
print(f"  distinct reference points       : {len(np.unique(gh))}")
print(f"  mean captures per reference pt  : {len(Xh)/len(np.unique(gh)):.1f}")
i_tr, i_te = train_test_split(np.arange(len(Xh)), test_size=0.2, random_state=42)
shared = set(gh[i_te]) & set(gh[i_tr])
leaked = sum(1 for i in i_te if gh[i] in set(gh[i_tr]))
print(f"  test RPs also in training       : {len(shared)}/{len(set(gh[i_te]))} "
      f"({100*len(shared)/len(set(gh[i_te])):.1f}%)")
print(f"  test SAMPLES whose RP is in train: {leaked}/{len(i_te)} "
      f"({100*leaked/len(i_te):.1f}%)")

# hyperparameters: small dataset, coordinates already in metres
C, GAMMA, EPS = 10, 2e-5, 0.1

out = {}
print("\n" + "=" * 70)
print("PROTOCOL COMPARISON (2D positioning error, metres)")
print("=" * 70)

# A -- random split
s, k = evaluate(Xh[i_tr], xh[i_tr], yh[i_tr], Xh[i_te], xh[i_te], yh[i_te], C, GAMMA, EPS)
out["A"] = {"svr": s, "knn": k}
print(f"  A random      SVR MAE={s['MAE']:5.2f} med={s['median']:5.2f} | "
      f"kNN MAE={k['MAE']:5.2f} med={k['median']:5.2f}")

# C -- RP-disjoint, averaged over 5 partitions
accS, accK = [], []
for seed in range(5):
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xh, xh, gh))
    assert not (set(gh[b]) & set(gh[a]))
    s, k = evaluate(Xh[a], xh[a], yh[a], Xh[b], xh[b], yh[b], C, GAMMA, EPS)
    accS.append(s); accK.append(k)
mean = lambda L, f: float(np.mean([d[f] for d in L]))
out["C"] = {"svr": {f: mean(accS, f) for f in accS[0]},
            "knn": {f: mean(accK, f) for f in accK[0]}}
print(f"  C RP-disjoint SVR MAE={mean(accS,'MAE'):5.2f} med={mean(accS,'median'):5.2f} | "
      f"kNN MAE={mean(accK,'MAE'):5.2f} med={mean(accK,'median'):5.2f}   (5 partitions)")

# D -- device transfer, same reference points
s, k = evaluate(Xh, xh, yh, Xm, xm, ym, C, GAMMA, EPS)
out["D"] = {"svr": s, "knn": k}
print(f"  D device      SVR MAE={s['MAE']:5.2f} med={s['median']:5.2f} | "
      f"kNN MAE={k['MAE']:5.2f} med={k['median']:5.2f}   (Huawei -> MIX 2)")

print("\n  inflation of the random protocol:")
for m in ("svr", "knn"):
    a, c, d = out["A"][m]["MAE"], out["C"][m]["MAE"], out["D"][m]["MAE"]
    print(f"    {m.upper():4s}  A={a:5.2f}  C={c:5.2f} ({c/a:.1f}x)  D={d:5.2f} ({d/a:.1f}x)")

json.dump(out, open("out_xjtlu.json", "w"), indent=2)
print("\nwrote out_xjtlu.json")
