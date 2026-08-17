"""Optimiser comparison re-run with the region-conditioned localiser."""
import sys, json, time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from region_svr import RegionSVR, BUILDING
from uji_data import load_uji, features, targets, error_2d, metrics
import corrected_pipeline as cp

seed = int(sys.argv[1])
tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr); xv, yv = targets(va)
b = tr[:, BUILDING]; f0 = np.zeros(len(tr))
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)

gss = GroupShuffleSplit(1, test_size=0.25, random_state=seed)
i_fit, i_val = next(gss.split(X, xt, g))
rs = np.random.RandomState(seed)
i_fit = rs.choice(i_fit, min(6000, len(i_fit)), replace=False)
i_val = rs.choice(i_val, min(2000, len(i_val)), replace=False)

def score(z):
    C, gam, eps = cp.decode(z)
    m = RegionSVR(C=C, gamma=gam, eps=eps, C_clf=100, gamma_clf=2e-5).fit(
        X[i_fit], xt[i_fit], yt[i_fit], b[i_fit], f0[i_fit])
    px, py = m.predict(X[i_val])
    return float(np.mean(error_2d(xt[i_val], yt[i_val], px, py)))

class Obj:
    def __init__(s, budget): s.n=0; s.budget=budget; s.best=(np.inf,None)
    def __call__(s, z):
        if s.n>=s.budget: return s.best[0]
        v=score(z); s.n+=1
        if v<s.best[0]: s.best=(v,cp.decode(z))
        return v

out=[]
for name, fn in [(k,v) for k,v in cp.OPTIMISERS.items() if k in ("CMA-ES","random")]:
    t0=time.time(); obj=Obj(24); fn(obj, seed)
    C,gam,eps = obj.best[1]
    m = RegionSVR(C=C, gamma=gam, eps=eps, C_clf=100, gamma_clf=2e-5).fit(X, xt, yt, b, f0)
    px,py = m.predict(Xv)
    r = metrics(error_2d(xv, yv, px, py))
    out.append({"seed":seed,"optimiser":name,"C":C,"gamma":gam,"epsilon":eps,
                "tune_best_MAE":obj.best[0],"sec":time.time()-t0,**r})
    print(f"seed {seed} {name:7s} C={C:9.3f} g={gam:.2e} inner={obj.best[0]:6.2f} "
          f"TEST MAE={r['MAE']:6.2f} RMSE={r['RMSE']:6.2f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"out_optregion2_seed{seed}.json","w"), indent=2)
