"""Theory prediction of official-leaderboard false certification for any qrels mechanism.

Generalises scripts/theory/official_flips.py from the logistic propensity to every kind of
witness.mech.  Binary official gain: Delta_off_p = (1/Q) sum_q X_pq, X_pq = sum_{d in q} g_pd O_d.
  E[Delta_off_p]   = (1/Q) sum_q mu_pq,            mu_pq = E[X_pq] = sum_{d in q} g_pd e_d      (any mechanism)
  Var(Delta_off_p) = (1/Q^2) sum_q V_pq,           V_pq = Var(X_pq)  (queries are labelled independently)
  E[s_q^2]         ~ (sum_q (mu_pq^2 + V_pq) - Q m_p^2) / (Q (Q - 1)),  the query-CLT variance the official test uses
V_pq comes from QrelsMech.second_moments: Bernoulli for the logistic kind; for the hidden engine with stop_after = 1 at
most one item per query is marked, so E[X_pq^2] = sum g^2 e exactly.  As in scripts/theory/official_flips.py, Delta_off ~ N(m, sd^2)
over the labelling and the official test certifies 'A > B' when Delta_off > z s_q, z the BH cut at the expected
number of rejections (fixed point over the 2P one-sided nulls).  A certification is wrong as in the evaluator
(scripts/eval/evaluate.py certify): 'A > B' is wrong when Delta(rho) <= 0 at some endpoint rho in {0, 1}.
Also printed: the rho = 1/2 sign version of scripts/theory/official_flips.py, and per system the discount-weighted mean propensity
of its relevant retrieved items, e_bar_s (in expectation the official leaderboard ranks DCG^(1)_s * e_bar_s).
usage: python scripts/theory/official_flips_general.py <testbed.yaml> <eval.yaml> [engine1,engine2,...]
  the optional list sweeps the hidden engine: each engine is removed from the testbed's dense systems and used as
  mnar.hidden.engine (hidden / qdepth kinds only).
"""
import copy
import os
import sys

import numpy as np
from scipy.stats import norm
from witness import mech as WM  # noqa: E402
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
BASE = load_config(sys.argv[1])
CFG = load_config(sys.argv[2])
print("mnar:", CFG["mnar"])


def predict(base, cfg):
    D = load_testbed(base, cfg)
    # the judge/CE content shift is built inside scripts/eval/evaluate.py and not reproduced here
    assert WM.kind_of(cfg["mnar"]) != "logistic" or cfg["mnar"]["shift"] is None
    M = WM.QrelsMech(cfg["mnar"], D.y, D.h, D.qi, D.n_q, D.systems, cfg["out_dir"], None)
    Q, P = D.n_q, D.P
    t0, th, t1 = (D.g @ u / Q for u in (D.u0, 0.5 * D.u0 + 0.5 * D.u1, D.u1))
    mu, V = M.second_moments(D.g)  # [P, Q] each
    m = mu.sum(1) / Q
    sd = np.sqrt(V.sum(1)) / Q
    s_q = np.sqrt(np.maximum(((mu ** 2 + V).sum(1) - Q * m ** 2) / (Q * (Q - 1)), 1e-30))
    k = 1.0
    for _ in range(100):
        z = norm.isf(cfg["alpha"] * max(k, 1.0) / (2 * P))
        p_pos = norm.sf((z * s_q - m) / np.maximum(sd, 1e-30))
        p_neg = norm.cdf((-z * s_q - m) / np.maximum(sd, 1e-30))
        k_new = float((p_pos + p_neg).sum())
        if abs(k_new - k) < 1e-8:
            break
        k = k_new
    plus_true, minus_true = np.minimum(t0, t1) <= 0, np.maximum(t0, t1) >= 0
    wrong_end = p_pos * plus_true + p_neg * minus_true
    wrong_half = np.where(th > 0, p_neg, p_pos)
    rel = D.y >= 1
    print(f"systems={D.systems}")
    print(f"pairs={P} queries={Q}; expected qrels per query {M.e.sum() / Q:.3f}; expected recall among relevant in "
          f"frame {M.e[D.in_union & rel].mean():.3f}; expected qrels error share {M.e[D.y == 0].sum() / M.e.sum():.3f}")
    flip = np.sign(m) != np.sign(th)
    print(f"predicted sign flips of E[Delta_off] vs Delta(1/2): {int(flip.sum())}/{P}")
    print(f"refined (query-CLT SE, BH cut z={z:.2f}): expected certified {k:.3f}, expected wrongly certified "
          f"(endpoint rule) {wrong_end.sum():.3f}, share {wrong_end.sum() / max(k, 1e-9):.3f}; "
          f"(rho=1/2 sign rule) {wrong_half.sum():.3f}")
    dcg1 = D.disc @ D.u1 / Q
    e_bar = (D.disc * D.u1) @ M.e / np.maximum((D.disc * D.u1).sum(1), 1e-30)
    if M.kind != "logistic":
        top_h = M.rank_h > 0
        share = [(D.disc[s] > 0)[top_h].sum() / max((D.disc[s] > 0).sum(), 1) for s in range(len(D.systems))]
    else:
        share = [np.nan] * len(D.systems)
    print("per system: true DCG(1/2), DCG(1), e_bar, DCG(1)*e_bar, share of top-k inside the engine list")
    for s, name in enumerate(D.systems):
        print(f"  {name:14s} {D.disc[s] @ (0.5 * D.u0 + 0.5 * D.u1) / Q:.4f} {dcg1[s]:.4f} {e_bar[s]:.4f} "
              f"{dcg1[s] * e_bar[s]:.4f} {share[s]:.3f}")
    order = np.argsort(-wrong_end)[:8]
    print("pairs with the largest wrong-certification probability (A, B, Delta(1/2), E[Delta_off], sd, s_q, P(wrong)):")
    for p in order:
        a, b = D.pairs[p]
        print(f"  {D.systems[a]:14s} {D.systems[b]:14s} {th[p]:+.4f} {m[p]:+.4f} {sd[p]:.4f} {s_q[p]:.4f} "
              f"{wrong_end[p]:.3f}")
    return k, wrong_end.sum()


if len(sys.argv) > 3:
    for eng in sys.argv[3].split(","):
        b2, c2 = copy.deepcopy(BASE), copy.deepcopy(CFG)
        assert WM.kind_of(c2["mnar"]) in ("hidden", "qdepth") and eng in b2["dense"]
        del b2["dense"][eng]
        c2["mnar"]["hidden"]["engine"] = eng
        print(f"\n=== engine {eng}")
        k, w = predict(b2, c2)
        print(f"SWEEP engine={eng} expected_certified={k:.3f} expected_wrong={w:.3f}")
else:
    predict(BASE, CFG)
print("DONE")
