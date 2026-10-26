"""Per-pair coupling test: with a small uniform human sample, estimate each pair's judge plug-in bias
and test it, then check against the oracle plug-in error.

Plug-in Delta_plug = (1/Q) sum_d g f (judges' mean gain on every frame item, the usual LLM-judge leaderboard).
Its error B = Delta_plug - Delta = (1/Q) sum g (f - u).  HT estimate from n labels: B_hat = (1/Q) sum_{xi} g (f - u)/pi,
Var_hat = (1/Q^2) sum_{xi} g^2 (1-pi)/pi^2 (f-u)^2.  Reported over `reps` samples: correlation of B_hat with B,
the share of pairs whose plug-in sign is wrong that the test flags (|B_hat| > 2 SE and B_hat pushes the plug-in past 0),
and the share of correct-sign pairs falsely flagged.
usage: python scripts/diagnostics/coupling_test.py <testbed.yaml> <eval.yaml> <rho> <n_labels> <reps>
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
g = D.g[:, fr]
N, P, Q = int(fr.sum()), D.P, D.n_q
u = ((1 - rho) * D.u0 + rho * D.u1)[fr]
J = D.J[:, fr]
ans = J >= 0
gain = np.where(J == 2, 1.0, np.where(J == 1, rho, 0.0))
f = np.where(ans.sum(0) > 0, (gain * ans).sum(0) / np.maximum(ans.sum(0), 1), D.ce[fr])
delta = g @ u / Q
plug = g @ f / Q
B = plug - delta
wrong = np.sign(plug) != np.sign(delta)
print(f"frame N={N} pairs={P} rho={rho} n={n_lab}: plug-in sign wrong {wrong.sum()}/{P}; |B| median {np.median(np.abs(B)):.4f}")
pi = n_lab / N
rng = np.random.default_rng(0)
corr, hit, false_flag = [], [], []
for _ in range(reps):
    xi = rng.random(N) < pi
    bh = (g[:, xi] @ (f - u)[xi]) / (pi * Q)
    se = np.sqrt((g[:, xi] ** 2) @ ((1 - pi) / pi ** 2 * (f - u)[xi] ** 2)) / Q
    # flag: the bias is significant and large enough to move the plug-in across zero
    flag = (np.abs(bh) > 2 * se) & (np.sign(plug - bh) != np.sign(plug))
    corr.append(np.corrcoef(bh, B)[0, 1])
    hit.append(flag[wrong].mean() if wrong.any() else np.nan)
    false_flag.append(flag[~wrong].mean())
print(f"corr(B_hat, B)={np.mean(corr):.3f}  wrong-sign pairs flagged={np.nanmean(hit):.3f}  "
      f"correct-sign pairs flagged={np.mean(false_flag):.3f}")
print("DONE")
