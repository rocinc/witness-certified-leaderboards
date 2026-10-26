"""Exact checks of the guarantees of src/witness/core.py that the paper relies on.

Symbols (as in core.py)
  G [R, n]   contrast rows g;  u [n] gain in [0, 1];  H [R, n] row predictor;  pi [n] Poisson inclusion probabilities
  xi [n]     Bernoulli(pi) draw;  Q = n_q;  Delta = (1/Q) G u
Design expectations are computed exactly by enumerating every draw of the frame {pi > 0} with its probability.
usage: PYTHONPATH=src python3 -m pytest -q tests
"""
import itertools

import numpy as np

from witness import core as W

TOL = 1e-10


def all_draws(pi):
    """Every Bernoulli(pi) draw on the frame {pi > 0} and its probability: xi [2^n_fr, n], prob [2^n_fr]."""
    fr = np.flatnonzero(pi > 0)
    bits = np.array(list(itertools.product([False, True], repeat=len(fr))))  # [2^n_fr, n_fr]
    xi = np.zeros((len(bits), len(pi)), bool)
    xi[:, fr] = bits
    prob = np.where(bits, pi[fr], 1 - pi[fr]).prod(1)
    assert abs(prob.sum() - 1.0) < 1e-12
    return xi, prob


def test_stage_estimate_design_unbiased():
    rng = np.random.default_rng(0)
    n, R, n_q = 10, 3, 4
    G = rng.normal(size=(R, n))
    u = rng.random(n)
    H = rng.random((R, n))
    pi = rng.uniform(0.15, 0.9, n)
    # items 0, 1 were labelled at an earlier stage (pi = 0, gain known); items 2, 3 enter through the plug-in
    pi[:4] = 0.0
    known_u = np.zeros(n)
    known_u[:2] = u[:2]
    plug = np.zeros(n, bool)
    plug[2:4] = True
    fr = pi > 0
    xi_all, prob = all_draws(pi)
    est_all = np.empty((len(prob), R))  # [2^n_fr, R]
    var_all = np.empty((len(prob), R))
    for j, xi in enumerate(xi_all):
        u_seen = np.where(xi, u, np.nan)  # u may be read only where xi is True
        est_all[j], var_all[j] = W.stage_estimate(G, known_u, plug, H, u_seen, xi, pi, n_q)
    target = (G @ known_u + (G[:, plug] * H[:, plug]).sum(1) + G[:, fr] @ u[fr]) / n_q
    mean = prob @ est_all
    assert np.abs(mean - target).max() < TOL, (mean, target)
    true_var = prob @ (est_all - mean) ** 2
    design_var = W.predicted_var(G, pi, (u[None, :] - H) ** 2, n_q)
    assert np.abs(true_var - design_var).max() < TOL, (true_var, design_var)
    assert np.abs(prob @ var_all - true_var).max() < TOL, (prob @ var_all, true_var)


def test_stage_estimate_census_case():
    """Without known or plug items the target is exactly Delta = G u / Q."""
    rng = np.random.default_rng(1)
    n, R, n_q = 8, 2, 3
    G = rng.normal(size=(R, n))
    u = rng.random(n)
    H = rng.random((R, n))
    pi = rng.uniform(0.1, 0.95, n)
    xi_all, prob = all_draws(pi)
    est_all = np.array([W.stage_estimate(G, np.zeros(n), np.zeros(n, bool), H, np.where(xi, u, np.nan), xi, pi,
                                         n_q)[0] for xi in xi_all])  # [2^n, R]
    assert np.abs(prob @ est_all - G @ u / n_q).max() < TOL


def test_xfit_estimate_design_unbiased():
    rng = np.random.default_rng(2)
    n, R, m, n_q, n_folds = 9, 3, 2, 3, 3
    G = rng.normal(size=(R, n))
    u = rng.random(n)
    Zc = np.column_stack([np.ones(n), rng.random(n)])  # [n, m]
    beta0 = np.array([0.4, 0.2])
    pi = rng.uniform(0.2, 0.9, n)
    fold = np.arange(n) % n_folds  # fixed before the draw
    plug = np.zeros(n, bool)
    xi_all, prob = all_draws(pi)
    est_all = np.empty((len(prob), R))  # [2^n, R]
    for j, xi in enumerate(xi_all):
        u_seen = np.where(xi, u, np.nan)  # u may be read only where xi is True
        est_all[j] = W.xfit_estimate(G, Zc, beta0, u_seen, xi, pi, fold, n_folds, 1.0, n_q, plug)[0]
    mean = prob @ est_all
    assert np.abs(mean - G @ u / n_q).max() < TOL, (mean, G @ u / n_q)
    # the in-sample (non-cross-fitted) predictor is biased on the same instance: the test has power
    est_in = np.empty_like(est_all)
    for j, xi in enumerate(xi_all):
        lab = np.flatnonzero(xi)
        H = W.greg_fit(G, Zc, np.where(xi, u, 0.0), lab, pi[lab], beta0, 1.0) @ Zc.T  # [R, n]
        est_in[j] = W.stage_estimate(G, np.zeros(n), plug, H, np.where(xi, u, 0.0), xi, pi, n_q)[0]
    assert np.abs(prob @ est_in - G @ u / n_q).max() > 1e3 * TOL


def test_bet_step_one_step_supermartingale():
    cap = 0.5
    for lo_val, b_val in [(-1.0, 1.0), (-99.0, 1.0), (-3.0, 0.2)]:
        for s1_val, s2_val in [(0.0, 0.0), (-5.0, 10.0), (0.3, 1.0), (50.0, 1.0), (2.0, 4.0)]:
            s1, s2, lo = np.array([s1_val]), np.array([s2_val]), np.array([lo_val])
            # two-point increment z in {lo, b} with mean exactly 0, and a mean-negative three-point one
            laws = [(np.array([lo_val, b_val]), np.array([b_val, -lo_val]) / (b_val - lo_val)),
                    (np.array([lo_val, 0.0, b_val]), np.array([0.5, 0.3, 0.2]))]
            for zs, pz in laws:
                factor = np.array([np.exp(W.bet_step(np.zeros(1), s1, s2, np.array([z]), lo, cap)[0][0])
                                   for z in zs])  # [n_atoms] = 1 + lam z
                assert (factor >= 1 - cap - 1e-12).all()
                ez = pz @ zs
                assert pz @ factor <= 1 + 1e-12
                if abs(ez) < 1e-15:
                    assert abs(pz @ factor - 1) < 1e-12
            _, s1_new, s2_new = W.bet_step(np.zeros(1), s1, s2, np.array([b_val]), lo, cap)
            assert s1_new[0] == s1_val + b_val and s2_new[0] == s2_val + b_val ** 2


def test_bet_step_ville_heavy_tailed():
    """Counterexample of scripts/theory/eprocess_validity.py: z = 1 - B/q, B ~ Bernoulli(q), sure bound lo = 1 - 1/q."""
    rng = np.random.default_rng(0)
    n_sim, n_steps, q, cap, a = 4000, 12800, 0.01, 0.5, 0.05
    lo = np.full(n_sim, 1.0 - 1.0 / q)
    logk, s1, s2 = np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim)
    sup = np.full(n_sim, -np.inf)
    for _ in range(n_steps):
        z = 1.0 - (rng.random(n_sim) < q) / q
        logk, s1, s2 = W.bet_step(logk, s1, s2, z, lo, cap)
        sup = np.maximum(sup, logk)
    p_hat = np.mean(sup >= np.log(1 / a))
    assert p_hat <= a + 3 * np.sqrt(a * (1 - a) / n_sim), p_hat


def bh_reference(p, alpha, dependent):
    """Step-up: k* = max{k : p_(k) <= alpha k / (m c(m))}; reject every p <= p_(k*)."""
    m = len(p)
    c = sum(1.0 / i for i in range(1, m + 1)) if dependent else 1.0
    ps = sorted(p)
    k_star = 0
    for k in range(m, 0, -1):
        if ps[k - 1] <= alpha * k / (m * c):
            k_star = k
            break
    if k_star == 0:
        return np.zeros(m, bool)
    return p <= ps[k_star - 1]


def test_bh_matches_reference():
    rng = np.random.default_rng(3)
    n_reject = 0
    for trial in range(400):
        m = int(rng.integers(1, 60))
        n_alt = int(rng.integers(0, m + 1))
        p = np.concatenate([rng.random(m - n_alt), rng.beta(0.05, 1.0, n_alt)])
        if trial % 4 == 0:  # exercise ties
            p = np.round(p, 2)
        rng.shuffle(p)
        for alpha in (0.05, 0.2):
            for dependent in (False, True):
                rej = W.bh(p, alpha, dependent)
                assert (rej == bh_reference(p, alpha, dependent)).all(), (p, alpha, dependent)
                n_reject += rej.sum()
    assert n_reject > 0


def test_discounts_and_pair_coefs():
    rng = np.random.default_rng(4)
    n_q, S, k = 4, 4, 3
    qi = rng.permutation(np.repeat(np.arange(n_q), [2, 5, 3, 6]))  # queries with fewer and more items than k
    n = len(qi)
    scores = rng.integers(0, 3, size=(S, n)).astype(float)  # many ties
    disc = W.discounts(qi, scores, k=k)
    assert disc.shape == (S, n)
    ref = np.zeros((S, n))
    for s in range(S):
        for q in range(n_q):
            items = sorted(np.flatnonzero(qi == q), key=lambda d: (-scores[s, d], d))
            for r, d in enumerate(items[:k], start=1):
                ref[s, d] = 1.0 / np.log2(r + 1)
    assert np.array_equal(disc, ref)
    pairs, g = W.pair_coefs(disc)
    P = S * (S - 1) // 2
    assert pairs.shape == (P, 2) and g.shape == (P, n)
    assert (pairs[:, 0] < pairs[:, 1]).all() and len({tuple(r) for r in pairs}) == P
    assert np.array_equal(g, disc[pairs[:, 0]] - disc[pairs[:, 1]])
    # reversing the system order turns pair (i, j) into (S-1-j, S-1-i) with the opposite orientation
    pairs_rev, g_rev = W.pair_coefs(disc[::-1])
    row = {tuple(r): i for i, r in enumerate(pairs)}
    for i, (a, b) in enumerate(pairs_rev):
        assert np.array_equal(g_rev[i], -g[row[(S - 1 - b, S - 1 - a)]])


def test_sampling_probs_clip():
    rng = np.random.default_rng(5)
    for _ in range(50):
        N = int(rng.integers(5, 200))
        w = rng.exponential(size=N) ** 2
        w[rng.random(N) < 0.2] = 0.0
        pi_min = float(rng.uniform(0.0, 0.5 / N * 10)) if rng.random() < 0.8 else 0.0
        pi_min = min(pi_min, 0.2)
        n_max = (w > 0).sum() + pi_min * (w == 0).sum()
        n_min = N * pi_min
        if n_max <= n_min + 1e-9:
            continue
        n = float(rng.uniform(n_min, n_max))
        pi = W.sampling_probs_clip(w, n, pi_min)
        assert abs(pi.sum() - n) <= 1e-6 * n
        assert (pi >= pi_min).all() and (pi <= 1.0).all()
        order = np.argsort(w, kind="stable")
        assert (np.diff(pi[order]) >= -1e-12).all()
        # Prop. 3.1: interior items share one kappa = pi_d / w_d
        inner = (pi > pi_min + 1e-9) & (pi < 1 - 1e-9)
        if inner.sum() >= 2:
            kappa = pi[inner] / w[inner]
            assert kappa.max() - kappa.min() <= 1e-9 * kappa.max()
