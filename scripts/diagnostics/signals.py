"""Diagnostic: how much each relevance signal knows about the human label, on a fully labelled testbed.

For the endpoint targets u1 = 1[y>=1] and u0 = 1[y=2] reports AUC and the squared-error R^2 of the best
monotone (isotonic, query-grouped 5-fold) calibration of each signal: each judge's label, the judges' joint
state, the cross-encoder probability, and their logistic combination (query-grouped 5-fold).  R^2 is what PPI-type
corrections convert into variance reduction (Var(u - f) = (1 - R^2) Var(u)), so it bounds their gain.
Also prints the joint-state x human-grade table.
usage: python scripts/diagnostics/signals.py <testbed.yaml> <eval.yaml> [extra_predictor.npy]
"""
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
fr = D.in_union
J, y, qi = D.J[:, fr], D.y[fr], D.qi[fr]
lce = np.log(np.clip(D.ce[fr], 1e-4, 1 - 1e-4) / (1 - np.clip(D.ce[fr], 1e-4, 1 - 1e-4)))
state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
levels = np.unique(state)
onehot = (state[:, None] == levels[None, :]).astype(float)
folds = list(GroupKFold(5).split(lce, groups=qi))
print(f"frame items={fr.sum()} grade shares={np.bincount(y, minlength=3) / len(y)}")


def cv_pred(X, t, model):
    p = np.empty(len(t))
    for tr, te in folds:
        if model == "iso":
            p[te] = IsotonicRegression(out_of_bounds="clip").fit(X[tr], t[tr]).predict(X[te])
        else:
            p[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return p


for name, t in (("u1=1[y>=1]", (y >= 1).astype(float)), ("u0=1[y=2]", (y == 2).astype(float))):
    print(f"== target {name}: mean {t.mean():.3f}")
    sigs = {f"judge_{j}": (J[i].astype(float), "iso") for i, j in enumerate(D.judges)}
    sigs["judge_mean"] = (np.where(J >= 0, J, 0).mean(0), "iso")
    sigs["ce"] = (lce, "iso")
    sigs["joint_state"] = (onehot, "lr")
    sigs["joint_state+ce"] = (np.column_stack([onehot, lce]), "lr")
    sigs["joint_state+ce+prov"] = (np.column_stack([onehot, lce, D.h[fr], D.n_top[fr]]), "lr")
    # retrieval pattern: every evaluated system's top-k hit and rank discount
    pattern = np.column_stack([(D.disc[:, fr] > 0).T, D.disc[:, fr].T])
    sigs["joint_state+ce+pattern"] = (np.column_stack([onehot, lce, pattern]), "lr")
    if len(sys.argv) > 3:  # optional extra predictor, [n, 3] class probabilities aligned with items.parquet
        ft = np.load(sys.argv[3])[fr]
        ftl = np.log(np.clip(ft, 1e-6, 1)) - np.log(np.clip(ft[:, :1], 1e-6, 1))  # log-odds against class 0
        sigs["ft"] = (ftl[:, 1:], "lr")
        sigs["joint_state+ce+ft"] = (np.column_stack([onehot, lce, ftl[:, 1:]]), "lr")
        sigs["joint_state+ce+ft+pattern"] = (np.column_stack([onehot, lce, ftl[:, 1:], pattern]), "lr")
    for s, (X, model) in sigs.items():
        p = cv_pred(X, t, model)
        auc = roc_auc_score(t, X if X.ndim == 1 else p)
        r2 = 1 - np.mean((t - p) ** 2) / t.var()
        print(f"  {s:22s} AUC={auc:.3f}  calibrated R2={r2:.3f}")
print("== joint state (flash,pro) x human grade counts")
for lev in levels:
    a, b = (lev % 4) - 1, (lev // 4) - 1
    print(f"  ({a:+d},{b:+d}): " + " ".join(f"y={k}:{int(((state == lev) & (y == k)).sum())}" for k in range(3)))
print("DONE")
