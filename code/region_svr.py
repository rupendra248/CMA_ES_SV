"""Region-conditioned SVR.

A single global regressor must fit a target function that is discontinuous at the
boundaries between buildings, which is why the localiser of the article is not
competitive upon UJIIndoorLoc. Here the building and the floor are classified
first, and a pair of regressors is fitted within each region, so that every
regressor sees a contiguous and comparatively smooth coordinate surface.

A misclassified building costs a very large error, so the classifier matters as
much as the regressor; both are support vector machines, and the whole remains a
support vector method.
"""
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

BUILDING, FLOOR = 523, 522


class RegionSVR:
    def __init__(self, C=10, gamma=2e-5, eps=0.1, C_clf=100, gamma_clf=2e-5,
                 min_region=40):
        self.p = dict(C=C, gamma=gamma, eps=eps)
        self.C_clf, self.gamma_clf = C_clf, gamma_clf
        self.min_region = min_region

    # ---------------------------------------------------------------- fitting
    def _fit_pair(self, X, x, y):
        out = []
        for t in (x, y):
            s = StandardScaler()
            ts = s.fit_transform(t.reshape(-1, 1)).ravel()
            m = SVR(kernel="rbf", C=self.p["C"], gamma=self.p["gamma"],
                    epsilon=self.p["eps"], cache_size=400).fit(X, ts)
            out.append((m, s))
        return out

    def fit(self, X, x, y, b, f):
        # a database covering a single building needs no classifier, and SVC
        # cannot be fitted to one class; the model then degenerates to a single
        # global regressor pair, which is the correct behaviour.
        self.clf_b = (SVC(kernel="rbf", C=self.C_clf, gamma=self.gamma_clf,
                          cache_size=400).fit(X, b)
                      if len(np.unique(b)) > 1 else float(b[0]))
        # floor is classified within each building, floors being building-specific
        self.clf_f = {}
        for bb in np.unique(b):
            m = b == bb
            if len(np.unique(f[m])) > 1:
                self.clf_f[bb] = SVC(kernel="rbf", C=self.C_clf,
                                     gamma=self.gamma_clf, cache_size=400).fit(X[m], f[m])
            else:
                self.clf_f[bb] = float(f[m][0])
        # one regressor pair per region, with a per-building and a global fallback
        self.reg, self.reg_b = {}, {}
        for bb in np.unique(b):
            mb = b == bb
            self.reg_b[bb] = self._fit_pair(X[mb], x[mb], y[mb])
            for ff in np.unique(f[mb]):
                m = mb & (f == ff)
                if m.sum() >= self.min_region:
                    self.reg[(bb, ff)] = self._fit_pair(X[m], x[m], y[m])
        self.reg_g = self._fit_pair(X, x, y)
        return self

    # -------------------------------------------------------------- inference
    @staticmethod
    def _apply(pair, X):
        return np.column_stack([s.inverse_transform(m.predict(X).reshape(-1, 1)).ravel()
                                for m, s in pair])

    def predict(self, X):
        bh = (self.clf_b.predict(X) if hasattr(self.clf_b, "predict")
              else np.full(len(X), self.clf_b))
        fh = np.empty(len(X))
        for bb in np.unique(bh):
            m = bh == bb
            c = self.clf_f.get(bb)
            fh[m] = c.predict(X[m]) if hasattr(c, "predict") else (c if c is not None else 0)
        out = np.empty((len(X), 2))
        for i in range(len(X)):
            pair = self.reg.get((bh[i], fh[i])) or self.reg_b.get(bh[i]) or self.reg_g
            out[i] = self._apply(pair, X[i:i + 1])[0]
        return out[:, 0], out[:, 1]

    def region_accuracy(self, X, b, f):
        bh = (self.clf_b.predict(X) if hasattr(self.clf_b, "predict")
              else np.full(len(X), self.clf_b))
        fh = np.empty(len(X))
        for bb in np.unique(bh):
            m = bh == bb
            c = self.clf_f.get(bb)
            fh[m] = c.predict(X[m]) if hasattr(c, "predict") else (c if c is not None else 0)
        return float((bh == b).mean()), float((fh == f).mean())
