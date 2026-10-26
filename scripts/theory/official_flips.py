"""Theory check of the pool-familiarity prediction: predicted sign flips of the official leaderboard as a function of pool-bias strength.

Under the simulated mechanism P(O_d = 1) = e_d(b) = sigmoid(a_y + b h_d + c 1[y=2]) (a_y = a for y >= 1, a_fp for y = 0),
the official contrast has expectation E[Delta_off_p](b) = (1/Q) sum_d g_pd e_d(b) (binary qrels: gain 1 per entry).
A pair is predicted to be officially mis-ordered at b when sign E[Delta_off_p](b) != sign Delta_p(rho = 1/2).
For each b on a grid: the number of predicted flips among the pairs, and the predicted wrong share among pairs whose
|E[Delta_off]| / sd(Delta_off) exceeds 3 (the ones an official significance test would 'certify'), with
Var(Delta_off_p) = (1/Q^2) sum_d g_pd^2 e_d (1 - e_d) from the Bernoulli labelling (the query-CLT the official test uses
also includes between-query spread; this is the labelling part only).  Compared offline with the simulated
false-certification proportions in d2/eval_sens_*_raw.csv.
usage: python scripts/theory/official_flips.py <testbed.yaml> <eval.yaml> <b_grid comma-separated>
"""
import os
import sys

import numpy as np
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[2])
D = load_testbed(load_config(sys.argv[1]), cfg)
grid = [float(x) for x in sys.argv[3].split(",")]
mn = cfg["mnar"]
delta = D.g @ (0.5 * D.u0 + 0.5 * D.u1) / D.n_q
print(f"pairs={D.P}  pool={cfg['pool_systems']}  a={mn['a']} a_fp={mn['a_fp']} c={mn['c']}")
for b in grid:
    logit = np.where(D.y >= 1, mn["a"], mn["a_fp"]) + b * D.h + mn["c"] * (D.y == 2)
    e = 1.0 / (1.0 + np.exp(-logit))
    m = D.g @ e / D.n_q
    sd = np.sqrt((D.g ** 2) @ (e * (1 - e))) / D.n_q
    flip = np.sign(m) != np.sign(delta)
    strong = np.abs(m) > 3 * sd
    print(f"b={b:4.1f}: mean qrels recall among relevant {e[D.y >= 1].mean():.3f}; predicted flips {flip.sum()}/{D.P}; "
          f"'certifiable' (|E|>3sd) {strong.sum()}, of which flipped {(flip & strong).sum()} "
          f"(share {(flip & strong).sum() / max(strong.sum(), 1):.3f})")
    # Refined prediction: the official test divides by the query-CLT SE, whose expectation includes between-query
    # spread: E[s_q^2] ~ (sum_q E[X_q^2] - Q m^2) / (Q (Q - 1)), X_q = sum_{d in q} g O, E[X_q^2] = (sum g e)^2 +
    # sum g^2 e (1 - e).  The statistic Delta_off / s_q passes z when Delta_off > z s_q; Delta_off ~ N(m, sd_bern^2)
    # over the labelling.  z is the BH cut for the expected number of rejections (fixed point over 2P one-sided nulls).
    from scipy.stats import norm

    ex = np.stack([np.bincount(D.qi, weights=D.g[p] * e, minlength=D.n_q) for p in range(D.P)])  # [P, Q]
    vx = np.stack([np.bincount(D.qi, weights=D.g[p] ** 2 * e * (1 - e), minlength=D.n_q) for p in range(D.P)])
    s_q = np.sqrt(np.maximum(((ex ** 2 + vx).sum(1) - D.n_q * m ** 2) / (D.n_q * (D.n_q - 1)), 1e-30))
    k = 1.0
    for _ in range(50):
        z = norm.isf(cfg["alpha"] * max(k, 1.0) / (2 * D.P))
        p_pos = norm.sf((z * s_q - m) / np.maximum(sd, 1e-30))   # certify 'first > second'
        p_neg = norm.cdf((-z * s_q - m) / np.maximum(sd, 1e-30))  # certify 'second > first'
        k_new = float((p_pos + p_neg).sum())
        if abs(k_new - k) < 1e-6:
            break
        k = k_new
    wrong = np.where(delta > 0, p_neg, p_pos)  # probability of certifying the direction opposite to the truth
    print(f"         refined (query-CLT SE, BH cut z={z:.2f}): expected certified {k:.2f}, expected wrongly certified "
          f"{wrong.sum():.2f}, expected false-certification share {wrong.sum() / max(k, 1e-9):.3f}")
print("DONE")
