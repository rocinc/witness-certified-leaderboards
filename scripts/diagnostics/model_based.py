"""Diagnostic: prediction (model-based) estimators of the pairwise leaderboard vs design-based ones.

A design-based estimator (HT/GREG/PPI) pays (1-pi)/pi * sum_d g^2 e_d^2 for the residual e; a prediction estimator
  Delta_hat = (1/Q)[ sum_{labelled} g u + sum_{unlabelled} g x^T beta_hat ]
pays only (1-pi) * sum_d g^2 sigma_d^2 plus the estimation variance of beta_hat, i.e. a factor ~1/pi less, but is
biased by (1/Q) sum_d g (E[u|x] - x^T beta) when the outcome model is wrong for the pair's contrast g.
Feature sets (linear in x, fitted by OLS):
  cells        one-hot joint judge state
  cells_ce     + cross-encoder logit
  fam          + BM25 / dense / reranker hit shares (provenance)
  sys          + per-system top-k hit and rank discount (each system's 'judge-invisible quality' offset)
Oracle part (all labels): bias of the best fit, B_p = (1/Q) sum_d g (x^T beta* - u), against |Delta|.
Sampled part (uniform Poisson, n labels, `reps` draws): coverage of the model-based 95% CI (HC1 sandwich for beta,
plus the residual term), pairs with t > 3 and their sign errors; the GREG design-based estimator with the same x.
usage: python scripts/diagnostics/model_based.py <testbed.yaml> <eval.yaml> <rho> <n_labels> <reps>
"""
import os
import sys

import numpy as np
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
rho, n_lab, reps = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
fr = D.in_union
g = D.g[:, fr]  # [P, N]
N, P, Q = int(fr.sum()), D.P, D.n_q
u = ((1 - rho) * D.u0 + rho * D.u1)[fr]
delta = g @ u / Q
J = D.J[:, fr]
state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
cells = (state[:, None] == np.unique(state)[None, :]).astype(float)  # [N, C], includes every state (no intercept)
lce = np.log(np.clip(D.ce[fr], 1e-4, 1 - 1e-4) / (1 - np.clip(D.ce[fr], 1e-4, 1 - 1e-4)))
fam = np.array(["bm25" if s.startswith("bm25") else "rerank" if s.startswith("rerank") else "dense" for s in D.systems])
hit = (D.disc[:, fr] > 0).astype(float)  # [S, N]
famx = np.column_stack([hit[fam == k].mean(0) for k in ("bm25", "dense", "rerank")])
X_SETS = {
    "cells": cells,
    "cells_ce": np.column_stack([cells, lce]),
    "fam": np.column_stack([cells, lce, famx]),
    "sys": np.column_stack([cells, lce, hit.T, D.disc[:, fr].T]),
}
print(f"frame N={N} pairs={P} queries={Q} rho={rho} n_labels={n_lab} (pi={n_lab / N:.4f}) reps={reps}")
print(f"|Delta| quantiles(.25,.5,.9)={np.round(np.quantile(np.abs(delta), [.25, .5, .9]), 4).tolist()}")


def ols(X, y, w=None):
    w = np.ones(len(y)) if w is None else w
    XtX = X.T @ (X * w[:, None])
    beta = np.linalg.lstsq(XtX, X.T @ (w * y), rcond=None)[0]
    return beta, XtX


for name, X in X_SETS.items():
    beta_star, _ = ols(X, u)
    bias = g @ (X @ beta_star - u) / Q
    wrong = int((np.sign(g @ (X @ beta_star) / Q) != np.sign(delta)).sum())
    print(f"[oracle {name:8s}] p={X.shape[1]:3d} R2={1 - np.mean((u - X @ beta_star) ** 2) / u.var():.3f} "
          f"|bias| q(.5,.9)={np.round(np.quantile(np.abs(bias), [.5, .9]), 4).tolist()} "
          f"|bias|/|Delta| median={np.median(np.abs(bias) / np.maximum(np.abs(delta), 1e-12)):.3f} "
          f"plug-in sign wrong={wrong}/{P}")

rng = np.random.default_rng(0)
pi = n_lab / N
res = {k: {"cov_m": [], "cert_m": [], "wrong_m": [], "cov_d": [], "cert_d": [], "wrong_d": [], "se_m": [], "se_d": []}
       for k in X_SETS}
for rep in range(reps):
    xi = rng.random(N) < pi
    for name, X in X_SETS.items():
        beta, XtX = ols(X[xi], u[xi])
        e = u[xi] - X[xi] @ beta
        p_ = X.shape[1]
        bread = np.linalg.pinv(XtX)
        meat = X[xi].T @ (X[xi] * (e ** 2)[:, None])
        cov_beta = bread @ meat @ bread * xi.sum() / max(xi.sum() - p_, 1)  # HC1
        unl = ~xi
        c = (g[:, unl] @ X[unl]) / Q  # [P, p]
        est_m = (g[:, xi] @ u[xi] + g[:, unl] @ (X[unl] @ beta)) / Q
        s2 = np.full(N, (e ** 2).mean())  # homoskedastic residual term (the estimation term is HC1)
        var_m = np.einsum("pi,ij,pj->p", c, cov_beta, c) + (g[:, unl] ** 2) @ s2[unl] / Q ** 2
        # design-based GREG with the same x (fit on the same sample, weights 1/pi), HT residual correction
        est_d = (g @ (X @ beta) + g[:, xi] @ e / pi) / Q
        var_d = (g[:, xi] ** 2) @ ((1 - pi) / pi ** 2 * e ** 2) / Q ** 2
        for tag, est, var in (("m", est_m, var_m), ("d", est_d, var_d)):
            se = np.sqrt(var)
            z = (est - delta) / se
            cert = np.abs(est) > 3 * se
            res[name][f"cov_{tag}"].append(np.mean(np.abs(z) <= 1.96))
            res[name][f"cert_{tag}"].append(cert.sum())
            res[name][f"wrong_{tag}"].append((cert & (np.sign(est) != np.sign(delta))).sum())
            res[name][f"se_{tag}"].append(np.median(se))
for name, r in res.items():
    print(f"[sampled {name:8s}] model: cover={np.mean(r['cov_m']):.3f} cert(t>3)={np.mean(r['cert_m']):.1f} "
          f"wrong={np.mean(r['wrong_m']):.2f} medSE={np.mean(r['se_m']):.4f} | design GREG: cover={np.mean(r['cov_d']):.3f} "
          f"cert={np.mean(r['cert_d']):.1f} wrong={np.mean(r['wrong_d']):.2f} medSE={np.mean(r['se_d']):.4f}")
print("DONE")
