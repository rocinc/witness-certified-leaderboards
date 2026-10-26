"""Diagnostic (oracle, full truth): where does the judge plug-in's leaderboard error come from?

Plug-in error of pair p over the frame (items in some top-k): E_p = (1/Q) sum_d g_pd b_d, b_d = f_d - u_d with f the
judges' mean gain (refusals: the cross-encoder probability) and u the true gain at rho.  With cells c = joint judge
state, exact decomposition
  E_p = T1_p + T2_p + T3_p
  T1 = (1/Q) sum_c G_pc bbar_c                 calibration: cell-mean judge error x the pair's net exposure G_pc
  T2 = (1/Q) sum_d g_pd x_d^T gamma            differential error: within-cell least-squares fit of b on the evaluated
                                               systems' retrieval pattern x_d (family hits), i.e. judge error that
                                               moves with which systems retrieved the item (circularity)
  T3 = the remainder                            item-level residual uncorrelated with x within cells
Also the variance an estimator pays: design-based (Poisson, uniform pi = n/N) charges sum_d g^2 (1-pi)/pi e_d^2 for the
residual e after calibration; a model-based estimator of the cell means charges sum_c G_pc^2 s_c^2 / n_c plus the
finite-population term sum_d g^2 s_c^2.  Their ratio is driven by the within-cell cancellation
kappa_pc = G_pc^2 / (N_c sum_{d in c} g_pd^2) in [0, 1].
usage: python scripts/diagnostics/decomposition.py <testbed.yaml> <eval.yaml> <rho> <n_labels>
"""
import os
import sys

import numpy as np
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
rho, n_lab = float(sys.argv[3]), int(sys.argv[4])
fr = D.in_union
g = D.g[:, fr]  # [P, N]
N, P, Q = fr.sum(), D.P, D.n_q
u = ((1 - rho) * D.u0 + rho * D.u1)[fr]
J = D.J[:, fr]
ans = J >= 0
gain = np.where(J == 2, 1.0, np.where(J == 1, rho, 0.0))
f = np.where(ans.sum(0) > 0, (gain * ans).sum(0) / np.maximum(ans.sum(0), 1), D.ce[fr])
b = f - u
cell = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
levels, cidx = np.unique(cell, return_inverse=True)
C = len(levels)
fam = np.array(["bm25" if s.startswith("bm25") else "rerank" if s.startswith("rerank") else "dense" for s in D.systems])
hit = D.disc[:, fr] > 0  # [S, N]
x = np.column_stack([hit[fam == k].mean(0) for k in ("bm25", "dense", "rerank")])  # [N, 3] family hit shares

onehot = np.zeros((N, C))
onehot[np.arange(N), cidx] = 1.0
bbar = np.bincount(cidx, weights=b, minlength=C) / np.bincount(cidx, minlength=C)
r1 = b - bbar[cidx]
xc = x - (onehot @ (onehot.T @ x / onehot.sum(0)[:, None]))  # within-cell centred family shares
gamma, *_ = np.linalg.lstsq(xc, r1, rcond=None)
fit2 = xc @ gamma
r2 = r1 - fit2
G = g @ onehot  # [P, C] net exposure
T1 = G @ bbar / Q
T2 = g @ fit2 / Q
T3 = g @ r2 / Q
E = g @ b / Q
assert np.allclose(E, T1 + T2 + T3)
delta = g @ u / Q
plug = g @ f / Q
print(f"frame N={N} pairs={P} queries={Q} rho={rho}; cell mean judge error by cell (count):")
for c in range(C):
    print(f"  state {levels[c]:3d}: n={int((cidx == c).sum()):6d} bbar={bbar[c]:+.3f} sd(b)={b[cidx == c].std():.3f}")
print(f"within-cell differential coefficients (bm25, dense, rerank shares): {np.round(gamma, 4).tolist()}")
q = [0.5, 0.9]
for name, t in (("|E| plug-in error", E), ("|T1| calibration", T1), ("|T2| differential", T2), ("|T3| residual", T3),
                ("|Delta| true", delta)):
    print(f"{name:20s} quantiles(.5,.9)={np.round(np.quantile(np.abs(t), q), 4).tolist()}")
print(f"plug-in sign wrong: {int((np.sign(plug) != np.sign(delta)).sum())}/{P}; "
      f"after removing T1: {int((np.sign(plug - T1) != np.sign(delta)).sum())}/{P}; "
      f"after removing T1+T2: {int((np.sign(plug - T1 - T2) != np.sign(delta)).sum())}/{P}")
# variances an estimator pays at n labels, uniform within cells (pi = n/N)
pi = n_lab / N
s2 = np.bincount(cidx, weights=r1 ** 2, minlength=C) / np.bincount(cidx, minlength=C)  # within-cell residual var
n_c = n_lab * np.bincount(cidx, minlength=C) / N
v_design = (g ** 2 * ((1 - pi) / pi) * (r1 ** 2)[None, :]).sum(1) / Q ** 2
v_model = ((G ** 2) @ (s2 / n_c) + (g ** 2) @ s2[cidx]) / Q ** 2
kap = (G ** 2) / np.maximum(np.bincount(cidx, minlength=C)[None, :] * ((g ** 2) @ onehot), 1e-30)
print(f"n={n_lab}: SE design quantiles(.5,.9)={np.round(np.quantile(np.sqrt(v_design), q), 4).tolist()}  "
      f"SE model={np.round(np.quantile(np.sqrt(v_model), q), 4).tolist()}  "
      f"t>3 design {(np.abs(delta) > 3 * np.sqrt(v_design)).sum()}/{P}  model {(np.abs(delta) > 3 * np.sqrt(v_model)).sum()}/{P}")
w = (g ** 2) @ onehot
print(f"exposure-weighted cancellation kappa: median over pairs {np.median((kap * w).sum(1) / w.sum(1)):.4f}")
# label complexity under the Neyman-optimal design, with sigma = the cell-calibrated residual's within-cell SD
from witness import core as W  # noqa: E402

sig = np.sqrt(s2[cidx])
for t in (2.0, 3.0):
    nstar = W.label_complexity(g, sig, delta, Q, t)
    print(f"label complexity t={t}: n* quantiles(.1,.25,.5,.75)={np.round(np.quantile(nstar, [.1, .25, .5, .75])).tolist()}  "
          + "  ".join(f"pairs with n*<={b}: {(nstar <= b).sum()}/{P}" for b in (800, 3200, 12800)))
# joint leaderboard budget: labels needed to certify the m easiest pairs (by single-pair n*) together, t = 3
order = np.argsort(W.label_complexity(g, sig, delta, Q, 3.0))
curve = []
for m in sorted({max(1, int(round(P * f))) for f in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75)}):
    sel = order[:m]
    bud, _, viol = W.leaderboard_budget(g[sel], sig, delta[sel], Q, 3.0)
    curve.append((m, bud, viol))
print("joint budget to certify the m easiest pairs at t=3 (m: labels, max rel. violation): "
      + "; ".join(f"{m}: {b:.0f} ({v:+.3f})" for m, b, v in curve))
# query-cluster design: label every frame item of a query sampled with prob pi_q = n/N (same expected labels).
# Both systems list k items per query, so sum_{d in q} g_pd = 0 and any query-level component of the residual cancels.
qf = D.qi[fr]
print(f"max |sum_(d in q) g_pd| over queries and pairs = {np.abs(np.stack([np.bincount(qf, weights=g[p], minlength=Q) for p in range(P)])).max():.2e}")
for name, e in (("raw judge error b", b), ("cell-calibrated r1", r1), ("cell+family r2", r2)):
    per_q = np.stack([np.bincount(qf, weights=g[p] * e, minlength=Q) for p in range(P)])  # [P, Q]
    v_item = (g ** 2 * ((1 - pi) / pi) * (e ** 2)[None, :]).sum(1) / Q ** 2
    v_clu = ((1 - pi) / pi) * (per_q ** 2).sum(1) / Q ** 2
    print(f"  residual={name:20s} SE item quantiles(.5,.9)={np.round(np.quantile(np.sqrt(v_item), q), 4).tolist()}  "
          f"SE query-cluster={np.round(np.quantile(np.sqrt(v_clu), q), 4).tolist()}  "
          f"median ratio var_clu/var_item={np.median(v_clu / v_item):.3f}  "
          f"t>3 item {(np.abs(delta) > 3 * np.sqrt(v_item)).sum()}/{P} cluster {(np.abs(delta) > 3 * np.sqrt(v_clu)).sum()}/{P}")
# the gain itself (no predictor): classical designs
per_q = np.stack([np.bincount(qf, weights=g[p] * u, minlength=Q) for p in range(P)])
v_item = (g ** 2 * ((1 - pi) / pi) * (u ** 2)[None, :]).sum(1) / Q ** 2
v_clu = ((1 - pi) / pi) * (per_q ** 2).sum(1) / Q ** 2
print(f"  no predictor (u itself)    SE item={np.round(np.quantile(np.sqrt(v_item), q), 4).tolist()}  "
      f"SE query-cluster={np.round(np.quantile(np.sqrt(v_clu), q), 4).tolist()}  "
      f"t>3 item {(np.abs(delta) > 3 * np.sqrt(v_item)).sum()}/{P} cluster {(np.abs(delta) > 3 * np.sqrt(v_clu)).sum()}/{P}")
print("DONE")
