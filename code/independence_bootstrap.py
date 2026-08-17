"""Sensitivity of the finite-sample component to the assumed sampling structure.

u_1 = s/sqrt(n) presumes the holdout measurements independent. The holdout carries
no reference-point labels (SPACEID is null throughout, and 1042 of 1111 longitudes
are distinct), so an RP-level cluster bootstrap is impossible. Two structures the
data do record admit one: the handset, and the building-and-floor plane.
"""
import json
import numpy as np
from uji_data import load_uji

B, PHONE, FLOOR, BUILD = 20000, 527, 522, 523
tr, va = load_uji()
e = np.asarray(json.load(open("out_region_errors.json"))["SVR_B"])
assert len(e) == len(va)
rng = np.random.RandomState(0)

def clustered(labels):
    u = np.unique(labels)
    m = [e[np.concatenate([np.where(labels == c)[0]
         for c in rng.choice(u, len(u), replace=True)])].mean() for _ in range(B)]
    return float(np.std(m, ddof=1)), len(u)

u1 = float(e.std(ddof=1) / np.sqrt(len(e)))
out = {"u1_analytic": u1, "n": int(len(e))}
out["sample"] = float(np.std([e[rng.randint(0, len(e), len(e))].mean() for _ in range(B)], ddof=1))
out["plane"], out["n_planes"] = clustered(va[:, BUILD] * 10 + va[:, FLOOR])
out["phone"], out["n_phones"] = clustered(va[:, PHONE])
json.dump(out, open("out_independence_raw.json", "w"), indent=2)
for k, v in out.items():
    print(f"  {k}: {v}")
