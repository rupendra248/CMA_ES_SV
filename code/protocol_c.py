"""Protocol C: reference-point-disjoint split.

Protocol A (random) leaks 100% of test RPs into training, so it measures
re-identification, not localisation. Protocol B (official validation set)
changes users, phones AND time all at once. Protocol C isolates the spatial
question: same users, same devices, same sessions -- but positions the model
has never seen.
"""
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from corrected_pipeline import fit_predict
from uji_data import load_uji, features, targets, error_2d, metrics

tr, _ = load_uji()
X = features(tr, "raw")
xt, yt = targets(tr)

# group id = physical reference point
_, groups = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)
print(f"{len(np.unique(groups))} distinct reference points\n")

rows = []
for seed in (0, 1, 2):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    i_tr, i_te = next(gss.split(X, xt, groups))
    leak = len(set(groups[i_te]) & set(groups[i_tr]))
    assert leak == 0, leak

    px = fit_predict(X[i_tr], xt[i_tr], X[i_te], 10, 2e-5, 0.1)
    py = fit_predict(X[i_tr], yt[i_tr], X[i_te], 10, 2e-5, 0.1)
    m_svr = metrics(error_2d(xt[i_te], yt[i_te], px, py))

    k = KNeighborsRegressor(3).fit(X[i_tr], np.c_[xt[i_tr], yt[i_tr]]).predict(X[i_te])
    m_knn = metrics(error_2d(xt[i_te], yt[i_te], k[:, 0], k[:, 1]))

    rows.append({"seed": seed, "svr": m_svr, "knn": m_knn,
                 "n_train": len(i_tr), "n_test": len(i_te)})
    print(f"seed {seed} (RP-disjoint, {len(i_tr)} train / {len(i_te)} test, 0 leaked RPs)")
    print(f"   SVR   MAE={m_svr['MAE']:6.2f} RMSE={m_svr['RMSE']:6.2f} "
          f"med={m_svr['median']:5.2f} p75={m_svr['p75']:6.2f} p90={m_svr['p90']:6.2f}")
    print(f"   kNN3  MAE={m_knn['MAE']:6.2f} RMSE={m_knn['RMSE']:6.2f} "
          f"med={m_knn['median']:5.2f} p75={m_knn['p75']:6.2f} p90={m_knn['p90']:6.2f}",
          flush=True)
    json.dump(rows, open("out_protocol_c.json", "w"), indent=2)

for tag in ("svr", "knn"):
    a = [r[tag]["MAE"] for r in rows]
    print(f"\n{tag.upper()} MAE across seeds: {np.mean(a):.2f} +/- {np.std(a):.2f} m")
