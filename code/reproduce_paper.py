"""Reproduce Table III's headline number using the repo's own pipeline,
then show what that number is once units and protocol are corrected.

Repo pipeline (CMA_ES.py + settings/CMA_ES.config), verbatim:
  data/train.csv, feature_cols=520, label_col=520
  train_test_split(test_size=0.2, random_state=42)
  StandardScaler on X only; y left as-is (already min-max normalised in the CSV)
  SVR(kernel='rbf', epsilon=0.1, C=<tuned>, gamma=<tuned>)
  MSE = mean((Y_test - Y_pred)**2)   <-- reported, and also the CMA-ES objective
"""
import csv, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Paper Table IV, optimised values for D1 (UJI)
C_PAPER, GAMMA_PAPER, EPS_PAPER = 10.759, 0.01, 0.1
PAPER_RMSE = 0.01132     # Table III, CMA-ES*, D1
PAPER_MAE = 0.01294

print("loading repo data/train.csv ...")
rows = np.array([[float(v) for v in r] for r in csv.reader(open("repo/data/train.csv"))])
X, Y = rows[:, :520], rows[:, 520]          # label_col = 520 -> longitude only
print(f"  X {X.shape}  Y {Y.shape}   Y range [{Y.min():.4f}, {Y.max():.4f}]  <- unitless")

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
sc = StandardScaler()
Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

print("fitting SVR with the paper's Table IV hyperparameters ...")
m = SVR(kernel="rbf", C=C_PAPER, gamma=GAMMA_PAPER, epsilon=EPS_PAPER).fit(Xtr, Ytr)
pred = m.predict(Xte)

err = np.abs(Yte - pred)                      # 1-D error, normalised units
rmse_n = float(np.sqrt(np.mean(err ** 2)))
mae_n = float(np.mean(err))

print("\n--- as the paper computes it (normalised units, longitude only) ---")
print(f"  RMSE = {rmse_n:.5f}   (paper Table III: {PAPER_RMSE})")
print(f"  MAE  = {mae_n:.5f}   (paper Table III: {PAPER_MAE})")
print(f"  RMSE >= MAE ? {rmse_n >= mae_n}   <- always true for real data")

# Rescale to metres. Repo normalisation basis = min-max over train+validation,
# verified to match repo/data/train.csv to full float precision.
def read_uji(p):
    r = csv.reader(open(p)); next(r)
    return np.array([[float(v) for v in row] for row in r])

both = np.vstack([read_uji("uji/UJIndoorLoc/trainingData.csv"),
                  read_uji("uji/UJIndoorLoc/validationData.csv")])
lon_span = float(both[:, 520].max() - both[:, 520].min())
lat_span = float(both[:, 521].max() - both[:, 521].min())

print(f"\n--- rescaled to metres (lon span {lon_span:.2f} m, lat span {lat_span:.2f} m) ---")
print(f"  longitude-only RMSE = {rmse_n * lon_span:.2f} m")
print(f"  longitude-only MAE  = {mae_n * lon_span:.2f} m")
print(f"  inflation factor claimed vs actual: {lon_span:.0f}x")

json.dump({"rmse_normalised": rmse_n, "mae_normalised": mae_n,
           "lon_span_m": lon_span, "lat_span_m": lat_span,
           "rmse_metres_lon_only": rmse_n * lon_span},
          open("out_reproduce.json", "w"), indent=2)
print("\nwrote out_reproduce.json")
