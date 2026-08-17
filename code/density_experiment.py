"""Controlled experiment: does reference-point spacing govern leakage inflation?

The article observes that a random division inflates the reported accuracy by
1.32x upon UJIIndoorLoc (2.98 m spacing) and 1.03x upon XJTLUIndoorLoc (1.00 m),
and attributes the difference to survey geometry. Two databases cannot establish
that. Here the geometry is varied deliberately within a single database, by
thinning the reference points to a prescribed minimum separation, so that the
relationship may be traced rather than inferred.

Thinning also reduces the number of samples, which would degrade accuracy on its
own. Every thinned condition is therefore paired with a SIZE-MATCHED CONTROL that
retains all reference points and discards captures at random to the same sample
count. If inflation grows with spacing in the thinned arm but not in the control,
the cause is the geometry and not the loss of data.
"""
import json, sys, time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from corrected_pipeline import fit_predict
from uji_data import load_uji, features, targets, error_2d

C, GAMMA, EPS = 10, 2e-5, 0.1
SEEDS = (0, 1, 2)


def thin_rps(uniq, min_d, seed):
    """Greedily keep reference points at least min_d apart within each floor plane."""
    if min_d <= 0:
        return np.arange(len(uniq))
    rs = np.random.RandomState(seed)
    keep = []
    for plane in np.unique(uniq[:, 2]):
        idx = np.where(uniq[:, 2] == plane)[0]
        pts = uniq[idx][:, :2]
        chosen = []
        for o in rs.permutation(len(idx)):
            if all(np.hypot(*(pts[o] - pts[c])) >= min_d for c in chosen):
                chosen.append(o)
        keep += list(idx[chosen])
    return np.array(sorted(keep))


def median_spacing(uniq, keep):
    sp = []
    sub = uniq[keep]
    for plane in np.unique(sub[:, 2]):
        P = sub[sub[:, 2] == plane][:, :2]
        if len(P) < 3:
            continue
        d, _ = cKDTree(P).query(P, k=2)
        sp.append(np.median(d[:, 1]))
    return float(np.median(sp)) if sp else float("nan")


def inflation(X, xt, yt, g, seed):
    """Mean error under a random division and under an RP-disjoint one."""
    i_tr, i_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    px = fit_predict(X[i_tr], xt[i_tr], X[i_te], C, GAMMA, EPS)
    py = fit_predict(X[i_tr], yt[i_tr], X[i_te], C, GAMMA, EPS)
    a = float(np.mean(error_2d(xt[i_te], yt[i_te], px, py)))

    j_tr, j_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                      .split(X, xt, g))
    px = fit_predict(X[j_tr], xt[j_tr], X[j_te], C, GAMMA, EPS)
    py = fit_predict(X[j_tr], yt[j_tr], X[j_te], C, GAMMA, EPS)
    c = float(np.mean(error_2d(xt[j_te], yt[j_te], px, py)))
    return a, c


def run(target_d):
    tr, _ = load_uji()
    X_all = features(tr, "raw")
    xt_all, yt_all = targets(tr)
    uniq, g_all = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

    out = []
    for seed in SEEDS:
        keep = thin_rps(uniq, target_d, seed)
        mask = np.isin(g_all, keep)
        n = int(mask.sum())
        spac = median_spacing(uniq, keep)

        # thinned arm: fewer reference points, wider apart
        a, c = inflation(X_all[mask], xt_all[mask], yt_all[mask], g_all[mask], seed)
        out.append({"arm": "thinned", "target_d": target_d, "seed": seed,
                    "spacing": spac, "n": n, "rps": len(keep),
                    "A": a, "C": c, "inflation": c / a})

        # size-matched control: all reference points, same number of samples
        rs = np.random.RandomState(100 + seed)
        sub = rs.choice(len(X_all), n, replace=False)
        a2, c2 = inflation(X_all[sub], xt_all[sub], yt_all[sub], g_all[sub], seed)
        out.append({"arm": "control", "target_d": target_d, "seed": seed,
                    "spacing": median_spacing(uniq, np.unique(g_all[sub])),
                    "n": n, "rps": len(np.unique(g_all[sub])),
                    "A": a2, "C": c2, "inflation": c2 / a2})

        print(f"  d={target_d:>4} seed {seed}: thinned n={n:5d} rp={len(keep):4d} "
              f"spacing={spac:5.2f} A={a:6.2f} C={c:6.2f} infl={c/a:5.3f} | "
              f"control infl={c2/a2:5.3f}", flush=True)
    json.dump(out, open(f"out_density_d{target_d}.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    t0 = time.time()
    run(int(sys.argv[1]))
    print(f"  d={sys.argv[1]} done in {time.time()-t0:.0f}s", flush=True)
