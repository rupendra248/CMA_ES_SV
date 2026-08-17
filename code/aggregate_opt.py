"""Aggregate all region-conditioned optimiser runs into the Table 7 rows."""
import glob, json, numpy as np
from scipy import stats

tab = {}
for f in glob.glob("/Users/rupendra/VS Code/CMA_ES_SV_github/results/out_optregion*.json") + \
         glob.glob("out_optregion3_seed*.json"):
    d = json.load(open(f))
    for r in (d if isinstance(d, list) else [d]):
        if isinstance(r, dict) and "optimiser" in r:
            tab.setdefault(r["optimiser"], {})[r["seed"]] = r["MAE"]

U1, U2, U4 = 0.257, 0.423, 0.019
print(f"{'Procedure':14s} {'n':>3s} {'Mean':>7s} {'Worst':>7s} {'u3':>7s} {'uc':>7s} {'U':>7s}")
rows = {}
for o in ("CMA-ES", "DE", "random", "PSO"):
    if o not in tab: continue
    ks = sorted(tab[o]); v = np.array([tab[o][k] for k in ks])
    u3 = v.std(ddof=1)
    uc = np.sqrt(U1**2 + U2**2 + u3**2 + U4**2)
    nu = uc**4 / (U1**4/1110 + U2**4/9 + u3**4/(len(v)-1) + U4**4/39)
    U = stats.t.ppf(0.975, nu) * uc
    rows[o] = (len(v), v.mean(), v.max(), u3, uc, U, ks)
    print(f"{o:14s} {len(v):3d} {v.mean():7.2f} {v.max():7.2f} {u3:7.3f} {uc:7.3f} {U:7.2f}   seeds={ks}")
json.dump({k: {"n": r[0], "mean": r[1], "worst": r[2], "u3": r[3], "uc": r[4], "U": r[5],
               "seeds": r[6], "values": [tab[k][s] for s in r[6]]} for k, r in rows.items()},
          open("out_optimisers_final.json", "w"), indent=2)
print("\nwrote out_optimisers_final.json")
