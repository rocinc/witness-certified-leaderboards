"""Monte Carlo prediction of official-leaderboard (false) certification under any qrels mechanism (witness.mech).

Complements scripts/theory/official_flips_general.py, whose refined prediction treats the query-CLT SE s_q as fixed at
sqrt(E[s_q^2]).  Here the mechanism itself is drawn n_mc times (QrelsMech.draw, the evaluator's own sampler) and every
draw runs the official test exactly as scripts/eval/evaluate.py (passive 'official' + certify): Delta_off = G o / Q,
var = per-query variance (ddof = 1) / Q, one-sided endpoint p-values (identical at rho = 0 and 1 for binary qrels),
BH over the 2P one-sided nulls; a certification is wrong under the endpoint rule.  No human labels are involved, so
this is the exact expectation of the evaluator's official rows up to MC error.
Also printed per pair: E[Delta_off], sd(Delta_off), E[s_q], sd(s_q), corr(Delta_off, s_q), and P(certified), P(wrong).
usage: python scripts/theory/official_flips_mc.py <testbed.yaml> <eval.yaml> <n_mc> <seed>
"""
import os
import sys

import numpy as np
from scipy.stats import norm
from witness import core as W  # noqa: E402
from witness import mech as WM  # noqa: E402
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[2])
D = load_testbed(load_config(sys.argv[1]), cfg)
N_MC, SEED = int(sys.argv[3]), int(sys.argv[4])
# the judge/CE content shift is built inside scripts/eval/evaluate.py and not reproduced here
assert WM.kind_of(cfg["mnar"]) != "logistic" or cfg["mnar"]["shift"] is None
M = WM.QrelsMech(cfg["mnar"], D.y, D.h, D.qi, D.n_q, D.systems, cfg["out_dir"], None)
Q, P = D.n_q, D.P
t0, t1 = D.g @ D.u0 / Q, D.g @ D.u1 / Q
plus_true, minus_true = np.minimum(t0, t1) <= 0, np.maximum(t0, t1) >= 0
print("mnar:", cfg["mnar"])
print(f"kind={M.kind} pairs={P} queries={Q} n_mc={N_MC} seed={SEED}")
rng = np.random.default_rng(SEED)
n_cert, n_wrong = np.zeros(N_MC), np.zeros(N_MC)
cert_p, wrong_p = np.zeros(P), np.zeros(P)
est_all, sq_all = np.zeros((N_MC, P)), np.zeros((N_MC, P))
for t in range(N_MC):
    o, _ = M.draw(rng)
    x = np.zeros((P, Q))
    np.add.at(x.T, D.qi[o], D.g[:, o].T)  # [P, Q] per-query official contrasts
    est = x.sum(1) / Q
    var = x.var(1, ddof=1) / Q
    se = np.sqrt(np.maximum(var, 1e-30))
    p_plus, p_minus = norm.sf(est / se), norm.cdf(est / se)
    rej = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"])
    rp, rm = rej[:P], rej[P:]
    wrong = (rp & plus_true) | (rm & minus_true)
    n_cert[t], n_wrong[t] = (rp | rm).sum(), wrong.sum()
    cert_p += rp | rm
    wrong_p += wrong
    est_all[t], sq_all[t] = est, se
se_mc = lambda v: v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731
print(f"MC expected certified {n_cert.mean():.3f} +- {se_mc(n_cert):.3f}; expected wrongly certified "
      f"{n_wrong.mean():.3f} +- {se_mc(n_wrong):.3f}; P(any wrong) {(n_wrong > 0).mean():.3f}; "
      f"fdp {np.mean(n_wrong / np.maximum(n_cert, 1)):.3f}")
# fixed-SE counterpart computed from the same draws: s_q replaced by its MC root-mean-square (the general script's
# sqrt(E[s_q^2]) up to MC error), Delta_off kept random; isolates the randomness of s_q
s_fix = np.sqrt((sq_all ** 2).mean(0))
n_fix = 0.0
for t in range(N_MC):
    zt = est_all[t] / s_fix
    rej = W.bh(np.concatenate([norm.sf(zt), norm.cdf(zt)]), cfg["alpha"])
    n_fix += rej.sum()
print(f"same draws with s_q fixed at its rms: expected certified {n_fix / N_MC:.3f}")
corr = np.array([np.corrcoef(est_all[:, p], sq_all[:, p])[0, 1] for p in range(P)])
print("per pair: A, B, Delta(1/2), E[Delta_off], sd(Delta_off), rms(s_q), mean(s_q), sd(s_q), corr(Delta_off, s_q), "
      "P(cert), P(wrong)")
order = np.argsort(-cert_p)
for p in order:
    a, b = D.pairs[p]
    print(f"  {D.systems[a]:14s} {D.systems[b]:14s} {(t0[p] + t1[p]) / 2:+.4f} {est_all[:, p].mean():+.4f} "
          f"{est_all[:, p].std():.4f} {s_fix[p]:.4f} {sq_all[:, p].mean():.4f} {sq_all[:, p].std():.4f} {corr[p]:+.3f} "
          f"{cert_p[p] / N_MC:.3f} {wrong_p[p] / N_MC:.3f}")
print("DONE")
