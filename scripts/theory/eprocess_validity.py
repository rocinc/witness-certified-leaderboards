"""Validity check of the M4 evidence processes on a skewed-increment counterexample.

Zero-mean increments z_j = -(B_j / q - 1), B_j ~ Bernoulli(q): a small positive drift most of the time and a rare
large negative jump, the shape of an HT increment whose rare item has g (u - H) < 0.  H0: E[z] <= 0 holds with
equality.  For each q: P(sup over checkpoints of E >= 1/a) for a in {.05, 1/3120} (1/3120 = e-BH's single-rejection
threshold at m = 156), for (i) the Gaussian-mixture process with running variance (witness.core.log_e_onesided) and
(ii) the predictable plug-in betting process (witness.core.bet_step) with the sure lower bound lo = -(1/q - 1).
Ville's inequality bounds the betting process's probabilities by a; the mixture has no such guarantee.
usage: python scripts/theory/eprocess_validity.py
"""
import os
import sys

import numpy as np
from witness import core as W  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
CHECK = [800, 1600, 3200, 4800, 6400, 9600, 12800]  # configs/eval/sequential_d1.yaml checkpoints
N_SIM, N_TARGET, CAP = 4000, 3200, 0.5
rng = np.random.default_rng(0)
for q in (0.01, 0.003, 0.001):
    lo = np.full(N_SIM, -(1.0 / q - 1.0))
    logk, s1, s2 = np.zeros(N_SIM), np.zeros(N_SIM), np.zeros(N_SIM)
    S, S2 = np.zeros(N_SIM), np.zeros(N_SIM)
    sup_mix, sup_bet = np.full(N_SIM, -np.inf), np.full(N_SIM, -np.inf)
    sup_bet_all = np.full(N_SIM, -np.inf)  # sup over every step, not only the checkpoints
    for j in range(1, CHECK[-1] + 1):
        z = 1.0 - (rng.random(N_SIM) < q) / q
        logk, s1, s2 = W.bet_step(logk, s1, s2, z, lo, CAP)
        S += z
        S2 += z ** 2
        sup_bet_all = np.maximum(sup_bet_all, logk)
        if j in CHECK:
            var = np.maximum(S2 / j - (S / j) ** 2, 1e-30)
            sup_mix = np.maximum(sup_mix, W.log_e_onesided(S, j, var, N_TARGET))
            sup_bet = np.maximum(sup_bet, logk)
    for a in (0.05, 1 / 3120):
        print(f"q={q}: a={a:.2e}  P(sup_ck E_mix >= 1/a)={np.mean(sup_mix >= np.log(1 / a)):.4f}  "
              f"P(sup_ck K_bet >= 1/a)={np.mean(sup_bet >= np.log(1 / a)):.4f}  "
              f"P(sup_all K_bet >= 1/a)={np.mean(sup_bet_all >= np.log(1 / a)):.4f}")
    print(f"q={q}: mean K_bet at n={CHECK[-1]} = {np.exp(logk).mean():.4f}  (supermartingale: <= 1 in expectation)")
print("DONE")
