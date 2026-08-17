"""UJIIndoorLoc loading with correct units and correct handling of 'no signal'.

Key points the original pipeline got wrong:
  * targets were min-max normalised and never inverse-transformed, so all
    reported errors were dimensionless rather than metres;
  * RSSI value 100 means "AP not detected" but was fed to the model as a
    raw feature value, i.e. as the strongest possible signal.
"""
import csv
import numpy as np

NO_SIGNAL = 100.0      # UJI sentinel for "AP not detected"
RSSI_FLOOR = -105.0    # one dB below the weakest real reading (-104 dBm)
N_WAP = 520
LON, LAT, FLOOR, BUILDING, USERID, PHONEID = 520, 521, 522, 523, 526, 527


def _read(path):
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        return np.array([[float(v) for v in row] for row in r], dtype=np.float64)


def load_uji(root="uji/UJIndoorLoc"):
    return _read(f"{root}/trainingData.csv"), _read(f"{root}/validationData.csv")


def features(raw, mode="powed"):
    """Convert the 520 raw RSSI columns into model features.

    raw    : signal in dBm, 100 -> RSSI_FLOOR. Keeps physical units.
    powed  : Torres-Sospedra et al. representation, ((rssi - min)/ -min)**e,
             0 for undetected. Standard for UJI and markedly better than raw.
    """
    x = raw[:, :N_WAP].copy()
    x[x == NO_SIGNAL] = RSSI_FLOOR
    if mode == "raw":
        return x
    if mode == "powed":
        v = np.clip(x, RSSI_FLOOR, None)
        pos = (v - RSSI_FLOOR) / (-RSSI_FLOOR)   # 0 at floor, ~1 at 0 dBm
        return pos ** np.e
    raise ValueError(mode)


def targets(raw):
    """(x, y) in metres, origin shifted to the dataset corner for readability."""
    return raw[:, LON], raw[:, LAT]


def error_2d(x_true, y_true, x_pred, y_pred):
    """Per-sample 2D Euclidean positioning error in metres."""
    return np.hypot(x_pred - x_true, y_pred - y_true)


def metrics(err):
    """Positioning-error summary. RMSE >= MAE always holds by construction."""
    err = np.asarray(err, dtype=np.float64)
    return {
        "p25": float(np.percentile(err, 25)),
        "MAE": float(np.mean(err)),
        "median": float(np.percentile(err, 50)),
        "p75": float(np.percentile(err, 75)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.max(err)),
        "n": int(err.size),
    }
