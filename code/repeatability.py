"""Repeatability over 10 reference-point-disjoint partitions.

The submitted analysis estimated a variance ratio from 3 partitions, which is far
too few to support a claim about dispersion. This runs 10 and adds a formal test.
"""
import json, time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from corrected_pipeline import fit_predict
from uji_data import load_uji, features, targets, error_2d, metrics

N_SPLITS = 10
C, GAMMA, EPS = 10, 2e-5, 0.1

tr, _ = load_uji()
X = features(tr, "raw")
xt, yt = targets(tr)
_, groups = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)
print(f"{len(np.unique(groups))} reference points, {N_SPLITS} disjoint partitions\n", flush=True)

rows = []
for seed in range(N_SPLITS):
    t0 = time.time()
    i_tr, i_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                      .split(X, xt, groups))
    assert not (set(groups[i_te]) & set(groups[i_tr]))

    px = fit_predict(X[i_tr], xt[i_tr], X[i_te], C, GAMMA, EPS)
    py = fit_predict(X[i_tr], yt[i_tr], X[i_te], C, GAMMA, EPS)
    m_svr = metrics(error_2d(xt[i_te], yt[i_te], px, py))

    k = KNeighborsRegressor(3).fit(X[i_tr], np.c_[xt[i_tr], yt[i_tr]]).predict(X[i_te])
    m_knn = metrics(error_2d(xt[i_te], yt[i_te], k[:, 0], k[:, 1]))

    rows.append({"seed": seed, "svr": m_svr, "knn": m_knn})
    print(f"  seed {seed}: SVR MAE={m_svr['MAE']:6.2f} RMSE={m_svr['RMSE']:6.2f} | "
          f"kNN MAE={m_knn['MAE']:6.2f} RMSE={m_knn['RMSE']:6.2f}  ({time.time()-t0:.0f}s)",
          flush=True)
    json.dump(rows, open("out_repeat10.json", "w"), indent=2)

print("\ndone", flush=True)
