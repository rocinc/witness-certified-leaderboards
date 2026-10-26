"""Sparse-qrels mechanism family for the WITNESS simulator.

Kinds (cfg["mnar"]["kind"]; an mnar block with exactly the keys {a, b, a_fp, c, shift} is the logistic kind):
  logistic  P(O_d = 1) = sigmoid(a_y + b h_d + c 1[y=2] + shift_d), independent over d (witness.core.simulate_qrels,
            called directly, so a logistic config reproduces its draws exactly).
  hidden    MS MARCO style: a production ranker H that is NOT an evaluated system lists its top-k_H candidates per
            query; the annotator scans them in rank order r = 1..k_H and marks item d (rank r_d) with prob
            p_d = p_y * decay^(r_d - 1)  (p_y = p1 for y = 1, p2 for y = 2, p_err for y = 0, the qrels errors),
            stopping after `stop_after` marks.  Items outside H's list are never in qrels.
  qdepth    per-query depth heterogeneity: each query is judged deeply (the logistic kind on its items) with prob
            s_deep, otherwise through the hidden engine as above.
Symbols
  rank_H[d]   1-based rank of d in H's top-k_H list for its query, 0 outside the list
  slots[q, r] item index at rank r + 1 of query q (-1 when the query has fewer candidates)
  p[d]        per-item mark probability given (y, rank) when the annotator reaches d
  e[d]        exact marginal propensity P(O_d = 1) (dynamic programme over the number of marks so far)
  mu_q, V_q   E and Var of X_q = sum_{d in q} g_d O_d for a contrast g (exact for stop_after = 1, see second_moments)
usage: imported by scripts/eval/evaluate.py and scripts/theory/official_flips_general.py
"""
import numpy as np

from witness import core as W

LOGISTIC_KEYS = {"a", "b", "a_fp", "c", "shift"}


def kind_of(mnar):
    """The original configs carry exactly the logistic keys and no 'kind'; every other block must name its kind."""
    if set(mnar) == LOGISTIC_KEYS:
        return "logistic"
    assert mnar["kind"] in ("logistic", "hidden", "qdepth"), mnar["kind"]
    return mnar["kind"]


def engine_slots(qi, n_q, score, k_h):
    """Rank the candidates of each query by the hidden engine's score (ties by item index, as W.discounts).
    Returns slots [n_q, k_h] (item index or -1) and rank_H [n] (1-based, 0 outside the list)."""
    order = np.lexsort((np.arange(len(qi)), -score, qi))  # by query, then score descending, then index
    qs = qi[order]
    start = np.searchsorted(qs, np.arange(n_q))
    pos = np.arange(len(qi)) - start[qs]  # 0-based rank within the query
    keep = pos < k_h
    slots = np.full((n_q, k_h), -1)
    slots[qs[keep], pos[keep]] = order[keep]
    rank_h = np.zeros(len(qi), int)
    rank_h[order[keep]] = pos[keep] + 1
    return slots, rank_h


def mark_probs(y, rank_h, hid):
    """p[d] = p_y decay^(rank - 1) inside H's list, 0 outside."""
    p_y = np.select([y == 2, y == 1], [hid["p2"], hid["p1"]], hid["p_err"])
    return np.where(rank_h > 0, p_y * hid["decay"] ** np.maximum(rank_h - 1, 0), 0.0)


def hidden_propensity(slots, p, stop_after):
    """Exact e[d] = p[d] * P(fewer than stop_after marks before d's rank): DP over the mark count, all queries at once."""
    n_q, k_h = slots.shape
    ps = np.where(slots >= 0, p[np.maximum(slots, 0)], 0.0)  # [n_q, k_h]
    alive = np.zeros((n_q, stop_after))
    alive[:, 0] = 1.0  # distribution of the number of marks so far, restricted to < stop_after
    e_slot = np.zeros((n_q, k_h))
    for r in range(k_h):
        pr = ps[:, r:r + 1]
        e_slot[:, r] = alive.sum(1) * ps[:, r]
        moved = alive * pr
        alive = alive * (1 - pr)
        alive[:, 1:] += moved[:, :-1]  # moved[:, -1] reaches stop_after: the annotator stops
    e = np.zeros(len(p))
    ok = slots >= 0
    e[slots[ok]] = e_slot[ok]
    return e


def hidden_draw(rng, slots, p, stop_after, n):
    """One annotation pass: would-mark coins in rank order; the first stop_after of them are kept."""
    ps = np.where(slots >= 0, p[np.maximum(slots, 0)], 0.0)
    hit = rng.random(slots.shape) < ps  # [n_q, k_h]
    hit &= np.cumsum(hit, axis=1) <= stop_after
    o = np.zeros(n, bool)
    o[slots[hit]] = True
    return o


class QrelsMech:
    """Fixed parts of a mechanism (engine ranking, propensities) built once; draw(rng) returns (o [n] bool, e [n])."""

    def __init__(self, mnar, y, h, qi, n_q, systems, out_dir, shift):
        self.kind = kind_of(mnar)
        self.mnar, self.y, self.h, self.qi, self.n_q, self.shift = mnar, y, h, qi, n_q, shift
        if self.kind == "logistic":
            t = np.where(y >= 1, mnar["a"], mnar["a_fp"]) + mnar["b"] * h + mnar["c"] * (y == 2)
            self.e = 1.0 / (1.0 + np.exp(-(t + (0.0 if shift is None else np.where(y >= 1, shift, 0.0)))))
            return
        assert shift is None  # the content shift is defined for the logistic kind only
        hid = mnar["hidden"]
        assert hid["engine"] not in systems  # the annotation engine is never an evaluated system
        score = np.load(f"{out_dir}/scores/{hid['engine']}.npy").astype(float)
        assert score.shape == y.shape
        self.slots, self.rank_h = engine_slots(qi, n_q, score, hid["k"])
        self.p = mark_probs(y, self.rank_h, hid)
        self.stop_after = hid["stop_after"]
        self.e_hid = hidden_propensity(self.slots, self.p, self.stop_after)
        if self.kind == "hidden":
            self.e = self.e_hid
        else:
            t = np.where(y >= 1, mnar["a"], mnar["a_fp"]) + mnar["b"] * h + mnar["c"] * (y == 2)
            self.e_log = 1.0 / (1.0 + np.exp(-t))
            self.e = mnar["s_deep"] * self.e_log + (1 - mnar["s_deep"]) * self.e_hid

    def draw(self, rng):
        m, n = self.mnar, len(self.y)
        if self.kind == "logistic":
            return W.simulate_qrels(rng, self.y, self.h, m["a"], m["b"], m["a_fp"], m["c"], shift=self.shift)
        if self.kind == "hidden":
            return hidden_draw(rng, self.slots, self.p, self.stop_after, n), self.e
        deep_q = rng.random(self.n_q) < m["s_deep"]
        o_log = rng.random(n) < self.e_log
        o_hid = hidden_draw(rng, self.slots, self.p, self.stop_after, n)
        return np.where(deep_q[self.qi], o_log, o_hid), self.e

    def second_moments(self, g):
        """Per-query mean mu [R, Q] and variance V [R, Q] of X_q = sum_{d in q} g_d O_d for rows g [R, n].

        logistic: independent Bernoulli, V = sum g^2 e (1 - e).  hidden with stop_after = 1: at most one mark per
        query, so E[X_q^2] = sum g^2 e exactly.  qdepth: the query's depth coin mixes the two conditional laws."""
        def per_q(w):
            out = np.zeros((g.shape[0], self.n_q))
            np.add.at(out.T, self.qi, w.T)
            return out

        if self.kind == "logistic":
            return per_q(g * self.e), per_q(g ** 2 * self.e * (1 - self.e))
        assert self.stop_after == 1  # exact pairwise marks for stop_after > 1 are not implemented
        mu_h, m2_h = per_q(g * self.e_hid), per_q(g ** 2 * self.e_hid)
        if self.kind == "hidden":
            return mu_h, m2_h - mu_h ** 2
        s = self.mnar["s_deep"]
        mu_l = per_q(g * self.e_log)
        m2_l = mu_l ** 2 + per_q(g ** 2 * self.e_log * (1 - self.e_log))
        mu = s * mu_l + (1 - s) * mu_h
        return mu, s * m2_l + (1 - s) * m2_h - mu ** 2
