"""Corrected CMA-ES-SVR evaluation harness.

Fixes, relative to the submitted pipeline:
  1. Targets are scaled for training and inverse-transformed before scoring,
     so every error is in METRES. (Original reported unitless quantities.)
  2. RMSE is sqrt(mean(e^2)). (Original reported MSE under an RMSE heading.)
  3. Error is the 2D Euclidean positioning error from BOTH coordinate models.
     (Original scored longitude only, label_col=520.)
  4. Hyperparameters are tuned on an inner split of the TRAINING data and never
     on the reported test set. (Original made test MSE the CMA-ES objective.)
  5. Search is in log space over (C, gamma, epsilon), so the optimum is interior
     rather than pinned to a boundary as in the submitted Table IV.
  6. Every optimiser gets an IDENTICAL evaluation budget.
  7. Random search is included as a baseline: a metaheuristic must beat it to
     have earned its place.
"""
import json
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from uji_data import load_uji, features, targets, error_2d, metrics

# log10 bounds for (C, gamma, epsilon).
#
# Features are raw dBm and deliberately NOT per-column standardised: most WAP
# columns are near-constant at "not detected", so StandardScaler inflates their
# rare detections and wrecks the RBF distance (measured: 67 m vs 9 m MAE).
# With var(X) ~ 29 the sklearn "scale" heuristic sits at 1/(520*29) = 6.6e-5,
# so the gamma range below brackets it with two decades of headroom either way.
#
# The submitted paper searched gamma linearly over [0.01, 100] and reported an
# optimum of exactly 0.01, i.e. pinned to the lower bound -- a clear sign the
# true optimum lay outside the searched region. It does: it is ~1e-5.
BOUNDS = np.array([[-1.0, 4.0], [-7.0, -2.0], [-3.0, 0.0]])
PARAM_NAMES = ("C", "gamma", "epsilon")


def boundary_report(z, tol=0.02):
    """Flag any parameter that converged onto a search-space boundary."""
    z = np.clip(np.asarray(z, float), 0, 1)
    return [PARAM_NAMES[i] for i in range(3) if z[i] < tol or z[i] > 1 - tol]


def decode(z):
    """Map an unconstrained/unit vector to (C, gamma, epsilon)."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return tuple(10.0 ** (lo + z * (hi - lo)))


def fit_predict(Xtr, ytr, Xte, C, gamma, eps, max_iter=-1):
    """Train on scaled targets, return predictions in the ORIGINAL units (metres).

    The inverse_transform is the step the submitted pipeline omitted.

    max_iter bounds libsvm's solver during hyperparameter search. Some corners of
    the space (large C with ill-matched gamma) take minutes per fit and would
    dominate the whole budget; capping the solver lets those points return a poor
    objective quickly, which is the correct outcome -- they are bad settings. The
    final model is always refit with max_iter=-1 (unbounded).
    """
    s = StandardScaler()
    ys = s.fit_transform(ytr.reshape(-1, 1)).ravel()
    m = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps,
            cache_size=500, max_iter=max_iter).fit(Xtr, ys)
    p = m.predict(Xte).reshape(-1, 1)
    return s.inverse_transform(p).ravel()


def standardise(X_a, X_b):
    """Kept for the ablation only -- see BOUNDS note. Not used in the pipeline."""
    s = StandardScaler()
    return s.fit_transform(X_a), s.transform(X_b)


TUNE_MAX_ITER = 300_000


def score_2d(X_a, xa, ya, X_b, xb, yb, C, gamma, eps, max_iter=-1):
    """Mean 2D positioning error in metres, fitting on a and scoring on b."""
    px = fit_predict(X_a, xa, X_b, C, gamma, eps, max_iter)
    py = fit_predict(X_a, ya, X_b, C, gamma, eps, max_iter)
    return float(np.mean(error_2d(xb, yb, px, py)))


class Objective:
    """Counts evaluations so every optimiser is held to the same budget."""

    def __init__(self, X_a, xa, ya, X_b, xb, yb, budget):
        self.args = (X_a, xa, ya, X_b, xb, yb)
        self.budget = budget
        self.n = 0
        self.best = (np.inf, None)
        self.history = []

    def __call__(self, z):
        if self.n >= self.budget:
            return self.best[0] if self.best[1] is not None else 1e9
        C, g, e = decode(z)
        v = score_2d(*self.args, C, g, e, TUNE_MAX_ITER)
        self.n += 1
        if v < self.best[0]:
            self.best = (v, (C, g, e))
        self.history.append(v)
        return v


# ---------------------------------------------------------------- optimisers
def opt_random(obj, seed):
    rs = np.random.RandomState(seed)
    while obj.n < obj.budget:
        obj(rs.rand(3))
    return obj.best


def opt_cmaes(obj, seed):
    import cma
    es = cma.CMAEvolutionStrategy(
        [0.5, 0.5, 0.5], 0.25,
        {"bounds": [0, 1], "popsize": 8, "seed": seed + 1,
         "verbose": -9, "maxfevals": obj.budget},
    )
    while obj.n < obj.budget and not es.stop():
        sols = es.ask()
        es.tell(sols, [obj(s) for s in sols])
    return obj.best


def opt_pso(obj, seed):
    rs = np.random.RandomState(seed)
    n, w, c1, c2 = 8, 0.7, 1.5, 1.7
    pos = rs.rand(n, 3)
    vel = rs.rand(n, 3) * 0.1 - 0.05
    pbest = pos.copy()
    pval = np.array([obj(p) for p in pos])
    g = pbest[int(np.argmin(pval))].copy()
    while obj.n < obj.budget:
        for i in range(n):
            if obj.n >= obj.budget:
                break
            vel[i] = (w * vel[i] + c1 * rs.rand(3) * (pbest[i] - pos[i])
                      + c2 * rs.rand(3) * (g - pos[i]))
            pos[i] = np.clip(pos[i] + vel[i], 0, 1)
            v = obj(pos[i])
            if v < pval[i]:
                pval[i], pbest[i] = v, pos[i].copy()
                if v < obj.best[0]:
                    g = pos[i].copy()
    return obj.best


def opt_de(obj, seed):
    rs = np.random.RandomState(seed)
    n, F, CR = 8, 0.8, 0.9
    pop = rs.rand(n, 3)
    val = np.array([obj(p) for p in pop])
    while obj.n < obj.budget:
        for i in range(n):
            if obj.n >= obj.budget:
                break
            a, b, c = pop[rs.choice([j for j in range(n) if j != i], 3, replace=False)]
            mut = np.clip(a + F * (b - c), 0, 1)
            cross = rs.rand(3) < CR
            if not cross.any():
                cross[rs.randint(3)] = True
            trial = np.where(cross, mut, pop[i])
            v = obj(trial)
            if v < val[i]:
                pop[i], val[i] = trial, v
    return obj.best


OPTIMISERS = {"random": opt_random, "CMA-ES": opt_cmaes, "PSO": opt_pso, "DE": opt_de}


# ---------------------------------------------------------------- experiment
def run(seeds=(0, 1, 2), budget=48, tune_n=5000, mode="raw", out="out_corrected.json"):
    tr, va = load_uji()
    X_tr_all, X_va_all = features(tr, mode), features(va, mode)
    xt, yt = targets(tr)
    xv, yv = targets(va)

    # group id = physical reference point, so the tuning objective is measured at
    # positions the tuning-fit fold has never seen. A random inner split leaks
    # 100% of reference points and would select hyperparameters that favour
    # re-identification over localisation.
    _, groups = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

    records = []
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        i_fit, i_val = next(gss.split(X_tr_all, xt, groups))
        assert not (set(groups[i_val]) & set(groups[i_fit]))
        rs = np.random.RandomState(seed)
        i_fit_s = rs.choice(i_fit, min(tune_n, len(i_fit)), replace=False)
        i_val_s = rs.choice(i_val, min(tune_n // 3, len(i_val)), replace=False)
        Xa, Xb = X_tr_all[i_fit_s], X_tr_all[i_val_s]

        for name, fn in OPTIMISERS.items():
            t0 = time.time()
            obj = Objective(Xa, xt[i_fit_s], yt[i_fit_s],
                            Xb, xt[i_val_s], yt[i_val_s], budget)
            _, best = fn(obj, seed)
            tune_s = time.time() - t0

            # refit on the FULL training set, evaluate ONCE on the official
            # validation set (different users, devices, and months)
            C, g, e = best
            z = (np.log10([C, g, e]) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
            t1 = time.time()
            px = fit_predict(X_tr_all, xt, X_va_all, C, g, e)
            py = fit_predict(X_tr_all, yt, X_va_all, C, g, e)
            m = metrics(error_2d(xv, yv, px, py))
            rec = {"seed": seed, "optimiser": name, "C": C, "gamma": g,
                   "epsilon": e, "tune_best_MAE": obj.best[0], "evals": obj.n,
                   "at_boundary": boundary_report(z),
                   "tune_sec": tune_s, "final_sec": time.time() - t1, **m}
            records.append(rec)
            bflag = f" BOUNDARY:{rec['at_boundary']}" if rec["at_boundary"] else ""
            print(f"seed {seed} {name:7s} C={C:9.3f} g={g:9.2e} e={e:7.4f} | "
                  f"tuneMAE={obj.best[0]:6.2f} | TEST MAE={m['MAE']:6.2f} "
                  f"RMSE={m['RMSE']:6.2f} med={m['median']:5.2f} p75={m['p75']:6.2f} "
                  f"({tune_s:.0f}s){bflag}", flush=True)
            json.dump(records, open(out, "w"), indent=2)
    return records


if __name__ == "__main__":
    run()
