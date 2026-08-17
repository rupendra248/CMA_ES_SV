"""Type B components of the uncertainty budget.

The reported RSS is quantised to 1 dBm, so the value recorded during the online
phase differs from the quantity it represents by up to half a least significant
bit. That interval is propagated to the measurand by Monte Carlo, in the manner
of GUM Supplement 1: the instrument is fitted once, and the online measurement is
then perturbed repeatedly by a rectangular deviate of half-width 0.5 dBm.

A second component arises from the coordinates of the reference points, which the
databases report without any statement of their own uncertainty. Its magnitude is
therefore evaluated conditionally, by displacing the reference coordinates by a
prescribed amount and observing the effect upon the measurand.
"""
import json
import numpy as np
from region_svr import RegionSVR, BUILDING
from uji_data import load_uji, features, targets, error_2d

P = dict(C=10, gamma=2e-5, C_clf=100, gamma_clf=2e-5)
N = 40

tr, va = load_uji()
X, Xv = features(tr, "raw"), features(va, "raw")
xt, yt = targets(tr)
xv, yv = targets(va)
b = tr[:, BUILDING]
f0 = np.zeros(len(tr))

m = RegionSVR(**P).fit(X, xt, yt, b, f0)
px, py = m.predict(Xv)
base = float(np.mean(error_2d(xv, yv, px, py)))
print(f"unperturbed measurand: {base:.4f} m\n", flush=True)

# ---- B1: quantisation of the online RSS measurement (rectangular, a = 0.5 dBm)
rs = np.random.RandomState(0)
vals = []
for i in range(N):
    Xp = Xv + rs.uniform(-0.5, 0.5, Xv.shape)
    p1, p2 = m.predict(Xp)
    vals.append(float(np.mean(error_2d(xv, yv, p1, p2))))
    if (i + 1) % 10 == 0:
        print(f"  B1 {i+1}/{N}: running mean {np.mean(vals):.4f}, sd {np.std(vals, ddof=1):.4f}",
              flush=True)
uB1 = float(np.std(vals, ddof=1))

# ---- B2: uncertainty of the reference-point coordinates, evaluated conditionally
#      a displacement of standard deviation s is applied to the survey coordinates
uB2 = {}
for s in (0.10, 0.25, 0.50, 1.00):
    out = []
    for i in range(8):
        r2 = np.random.RandomState(100 + i)
        xd = xt + r2.normal(0, s, len(xt))
        yd = yt + r2.normal(0, s, len(yt))
        md = RegionSVR(**P).fit(X, xd, yd, b, f0)
        q1, q2 = md.predict(Xv)
        out.append(float(np.mean(error_2d(xv, yv, q1, q2))))
    uB2[s] = (float(np.mean(out)), float(np.std(out, ddof=1)))
    print(f"  B2 survey sd {s:.2f} m -> measurand {np.mean(out):.3f} m "
          f"(shift {np.mean(out)-base:+.3f})", flush=True)

json.dump({"base": base, "uB1": uB1, "B1_runs": vals,
           "uB2": {str(k): v for k, v in uB2.items()}},
          open("out_typeb.json", "w"), indent=2)
print(f"\nu(B1), RSS quantisation      = {uB1:.4f} m")
print("u(B2) is conditional upon the assumed survey uncertainty; see table.")
