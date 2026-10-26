"""WITNESS core: pool-biased qrels, dependent relevance evidence, design-unbiased active correction, anytime-valid
leaderboard certification.  Pure functions; scripts hold the state.

Symbols
  n items d (query, document), Q queries, S systems.
  qi[d]        query index of item d
  y[d]         true graded relevance in {0, 1, 2}
  u_rho(y)     gain: 1[y=2] + rho * 1[y=1], in [0, 1]
  disc[s, d]   1/log2(rank+1) if d is in system s's top-k for its query, else 0
  DCG_s        (1/Q) sum_d disc[s, d] u(y_d)                       (linear in u)
  g^{AB}[d]    disc[A, d] - disc[B, d];  Delta_AB = (1/Q) sum_d g^{AB}[d] u(y_d)
"""
import itertools

import numpy as np
from scipy.stats import norm


def gain(y, rho):
    return (y == 2) + rho * (y == 1)


def discounts(qi, scores, k=10):
    """disc [S, n]: rank each query's items by score (descending, ties broken by item index)."""
    S, n = scores.shape
    disc = np.zeros((S, n))
    order_q = np.argsort(qi, kind="stable")
    bounds = np.flatnonzero(np.diff(qi[order_q])) + 1
    for grp in np.split(order_q, bounds):
        for s in range(S):
            r = grp[np.lexsort((grp, -scores[s, grp]))][:k]
            disc[s, r] = 1.0 / np.log2(np.arange(2, len(r) + 2))
    return disc


def pair_coefs(disc):
    """pairs [P, 2] and g [P, n] = disc[A] - disc[B]."""
    pairs = np.array(list(itertools.combinations(range(disc.shape[0]), 2)))
    return pairs, disc[pairs[:, 0]] - disc[pairs[:, 1]]


def official_deltas(g, o, n_q):
    """Delta under sparse qrels: labelled items gain 1, the rest 0."""
    return g @ o.astype(float) / n_q


def plugin_deltas(g, f, n_q):
    return g @ f / n_q


def query_bootstrap_ci(g, f, qi, n_q, rng, reps=1000, alpha=0.05):
    """Percentile CI of the plug-in Delta over a query bootstrap that treats f as the truth (the usual practice)."""
    per_q = np.zeros((g.shape[0], n_q))
    np.add.at(per_q.T, qi, (g * f).T)
    boots = np.empty((reps, g.shape[0]))
    for r in range(reps):
        boots[r] = per_q[:, rng.integers(0, n_q, n_q)].mean(1)
    return np.quantile(boots, alpha / 2, 0), np.quantile(boots, 1 - alpha / 2, 0)


def poisson_sample(rng, pi):
    assert ((pi > 0) & (pi <= 1)).all()
    return rng.random(len(pi)) < pi


def corrected_deltas(g, f, u, xi, pi, n_q, lam=1.0):
    """Design-unbiased Delta for every pair and its variance estimate under Poisson sampling.

    est = (1/Q)[lam * sum_d g f + sum_{d in sample} g (u - lam f) / pi]
    Unbiased for any f and lam that do not depend on the sampled labels; u is read only on the sample.
    """
    r = np.where(xi, (u - lam * f) / pi, 0.0)
    est = (lam * (g @ f) + g @ r) / n_q
    w = np.where(xi, (1 - pi) / pi ** 2 * (u - lam * f) ** 2, 0.0)
    var = (g ** 2) @ w / n_q ** 2
    return est, var


def optimal_lambda(g, f, u, xi, pi):
    """Per-pair lam minimising the Poisson-sampling variance sum (1-pi)/pi g^2 (u - lam f)^2, from the sample."""
    w = np.where(xi, (1 - pi) / pi ** 2, 0.0)
    num = (g ** 2) @ (w * u * f)
    den = (g ** 2) @ (w * f * f)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def sampling_probs(budget, weight, floor=1e-3):
    """pi proportional to weight with sum pi = budget, capped at 1 (water-filling), floored for positivity."""
    w = np.maximum(weight, 0.0) + floor * weight.max()
    pi = np.zeros_like(w)
    free = np.ones(len(w), bool)
    left = float(budget)
    for _ in range(50):
        p = left * w[free] / w[free].sum()
        if (p <= 1).all():
            pi[free] = p
            break
        over = np.flatnonzero(free)[p > 1]
        pi[over] = 1.0
        free[over] = False
        left = budget - pi.sum()
    assert np.isclose(pi.sum(), budget, rtol=1e-6), (pi.sum(), budget)
    return pi


def sampling_probs_clip(weight, n, pi_min):
    """Unique optimum of the convex allocation problem with a floor: pi_d = clip(kappa * weight_d, pi_min, 1) with
    sum_d pi_d = n; kappa by bisection."""
    w = np.asarray(weight, dtype=float)
    assert (w >= 0).all() and len(w) * pi_min <= n
    assert n <= (w > 0).sum() + pi_min * (w == 0).sum()

    def total(k):
        return np.clip(k * w, pi_min, 1.0).sum()

    lo, hi = 0.0, 1.0
    while total(hi) < n:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < n:
            lo = mid
        else:
            hi = mid
    pi = np.clip(hi * w, pi_min, 1.0)
    assert np.isclose(pi.sum(), n, rtol=1e-6), (pi.sum(), n)
    return pi


def cert_weights(delta_mean, delta_var, v_final, z, n_nodes=7):
    """Pair weights of the expected-number-of-certified-pairs objective, w = E[t phi(t - z)] / (2 V_final),
    t = |Delta| / sqrt(V_final), averaged over the predictive Delta ~ N(delta_mean, delta_var) (Gauss-Hermite); also
    returns the predicted certification probability E[Phi(t - z)].  All inputs broadcast to the same shape."""
    x, wq = np.polynomial.hermite_e.hermegauss(n_nodes)  # weight exp(-x^2/2)
    wq = wq / wq.sum()
    d = delta_mean[..., None] + np.sqrt(delta_var)[..., None] * x
    t = np.abs(d) / np.sqrt(v_final)[..., None]
    w = (wq * t * norm.pdf(t - z)).sum(-1) / (2.0 * v_final)
    p_cert = (wq * norm.cdf(t - z)).sum(-1)
    return w, p_cert


def greg_fit(G, Zc, u, lab, pi_lab, beta0, ridge):
    """Per-row GREG coefficients (a-weighted least squares) from the labelled items `lab`.

    beta_r = argmin sum_{d in lab} a_rd (u_d - z_d^T beta)^2 + r_r ||beta - beta0||^2,  a_rd = g_rd^2 (1-pi_d)/pi_d^2,
    with r_r = ridge * tr(A_r)/(m k), i.e. the plain predictor beta0 acts as `ridge` pseudo-labels.
    G [R, n], Zc [n, m], u [n], lab [k] indices, pi_lab [k] inclusion probabilities at the time each label was bought,
    beta0 [m].  Returns beta [R, m].
    """
    m = Zc.shape[1]
    a = G[:, lab] ** 2 * ((1 - pi_lab) / pi_lab ** 2)  # [R, k]
    Z = Zc[lab]  # [k, m]
    A = np.einsum("rk,ki,kj->rij", a, Z, Z)  # [R, m, m]
    b = np.einsum("rk,ki,k->ri", a, Z, u[lab])  # [R, m]
    # the prior is worth `ridge` average labelled items, so shrinkage fades as labels accumulate; rows that no labelled
    # item touches (A = 0) return beta0 exactly
    r = ridge * np.maximum(np.trace(A, axis1=1, axis2=2) / (m * max(len(lab), 1)), 1e-12)  # [R]
    A = A + r[:, None, None] * np.eye(m)
    b = b + r[:, None] * beta0
    return np.linalg.solve(A, b[..., None])[..., 0]


def stage_estimate(G, known_u, plug, H, u, xi, pi, n_q):
    """One Poisson stage of the design-unbiased estimator with row-specific predictor values H [R, n].

    Frame = {pi > 0}.  Items with pi = 0 enter either through known_u (labels already bought; 0 elsewhere) or, where
    plug is True, through their predictor value without correction (only the 'qrels stratum not sampled' ablation
    uses plug; it is biased by construction).  u is read only where xi is True.
    Returns est [R] and the unbiased variance estimate [R].
    """
    fr = pi > 0
    assert not (fr & plug).any() and not (xi & ~fr).any()
    Gf, Hf, pf, s = G[:, fr], H[:, fr], pi[fr], xi[fr]
    uf = np.where(s, u[fr], 0.0)
    r = np.where(s, (uf - Hf) / pf, 0.0)  # [R, n_fr]
    est = (G @ known_u + (G[:, plug] * H[:, plug]).sum(1) + (Gf * (Hf + r)).sum(1)) / n_q
    var = (Gf ** 2 * np.where(s, (1 - pf) / pf ** 2 * (uf - Hf) ** 2, 0.0)).sum(1) / n_q ** 2
    return est, var


def xfit_estimate(G, Zc, beta0, u, xi, pi, fold, n_folds, ridge, n_q, plug):
    """Frame-partition cross-fitting of the single-stage GREG estimator: exactly design-unbiased.

    Every frame item gets a fold label independent of the sampling draw.  For fold k the predictor H^(-k) is a GREG
    fit on the labelled items of the other folds, so given everything outside fold k the fold-k draws are still
    independent Bernoulli(pi):  E[sum_{d in k} g H^(-k) + sum_{d in k, xi} g (u - H^(-k)) / pi] = sum_{d in k} g u.
    Summing over folds gives Q Delta exactly (no O(1/n) term, unlike the in-sample fit).  Variance estimate: sum of the
    folds' HT variance estimates (folds treated as independent given their predictors).
    plug items (pi = 0 by design) enter with the all-label fit; only the biased L-not-sampled ablation uses them.
    G [R, n], Zc [n, m], beta0 [m], u [n] (read only where xi), xi/pi/fold [n].  Returns est [R], var [R]."""
    fr = pi > 0
    est = np.zeros(G.shape[0])
    var = np.zeros(G.shape[0])
    for k in range(n_folds):
        lab = np.flatnonzero(xi & (fold != k))
        H = greg_fit(G, Zc, u, lab, pi[lab], beta0, ridge) @ Zc.T  # [R, n]
        mk = fr & (fold == k)
        sk = mk & xi
        r = (u[sk][None, :] - H[:, sk])  # [R, n_sk]
        est += (G[:, mk] * H[:, mk]).sum(1) + (G[:, sk] * r / pi[sk][None, :]).sum(1)
        var += (G[:, sk] ** 2 * ((1 - pi[sk]) / pi[sk] ** 2)[None, :] * r ** 2).sum(1)
    if plug.any():
        lab = np.flatnonzero(xi)
        H = greg_fit(G, Zc, u, lab, pi[lab], beta0, ridge) @ Zc.T
        est += (G[:, plug] * H[:, plug]).sum(1)
    return est / n_q, var / n_q ** 2


def stage_xfit_estimate(G, known_u, Zc, beta0, u, lab_prev, pi_prev, xi, pi, fold, n_folds, ridge, n_q):
    """One Poisson stage of a multi-stage design with within-stage cross-fitting: exactly conditionally unbiased.

    Frame of the stage = {pi > 0} (items not labelled earlier); known_u carries the earlier stages' labels (0
    elsewhere).  For fold k the predictor is a GREG fit on every earlier label (lab_prev, weights from pi_prev) plus
    the stage's own labels outside fold k.  Given the past, the stage's draws are independent Bernoulli(pi) and the
    fold labels are independent of them, so for fold k, conditionally on everything outside it,
      E[sum_{d in k} g H^(-k) + sum_{d in k, xi} g (u - H^(-k)) / pi] = sum_{d in k} g u,
    and E[est | past] = Q Delta (design unbiasedness per stage; stages combine with weights fixed before each draw,
    so the stage errors are martingale differences).
    Zc, beta0 may themselves depend on earlier labels (e.g. a PU predictor refitted on them): they are fixed given
    the past.  Variance: sum of the folds' HT variance estimates (folds treated as independent, as xfit_estimate).
    G [R, n], known_u/u/xi/pi/fold [n], Zc [n, m], lab_prev [k0], pi_prev [k0].  Returns est [R], var [R]."""
    fr = pi > 0
    assert not (xi & ~fr).any() and not (fr[lab_prev]).any()
    est = G @ known_u
    var = np.zeros(G.shape[0])
    for k in range(n_folds):
        cur = np.flatnonzero(xi & (fold != k))
        lab = np.concatenate([lab_prev, cur])
        H = greg_fit(G, Zc, u, lab, np.concatenate([pi_prev, pi[cur]]), beta0, ridge) @ Zc.T  # [R, n]
        mk = fr & (fold == k)
        sk = mk & xi
        r = u[sk][None, :] - H[:, sk]  # [R, n_sk]
        est += (G[:, mk] * H[:, mk]).sum(1) + (G[:, sk] * r / pi[sk][None, :]).sum(1)
        var += (G[:, sk] ** 2 * ((1 - pi[sk]) / pi[sk] ** 2)[None, :] * r ** 2).sum(1)
    return est / n_q, var / n_q ** 2


def predicted_var(G, pi, m2, n_q):
    """Design variance (1/Q^2) sum_d g^2 (1-pi)/pi m2 over the frame {pi > 0}; m2 [R, n] = E[(u - H)^2 | x]."""
    fr = pi > 0
    return (G[:, fr] ** 2 * ((1 - pi[fr]) / pi[fr]) * m2[:, fr]).sum(1) / n_q ** 2


def label_complexity(g, sigma, delta, n_q, t):
    """Label complexity: minimal expected number of human labels for pair p to reach |Delta_p| / SE_p >= t under the
    Neyman-optimal Poisson design (ignoring the cap pi <= 1), with the exact finite-population correction:
      n*_p = t^2 (sum_d |g_pd| sigma_d)^2 / (Q^2 Delta_p^2 + t^2 sum_d g_pd^2 sigma_d^2).
    g [P, N] contrasts over the frame, sigma [N] residual SD after calibration, delta [P]."""
    a = np.abs(g) * sigma[None, :]
    return t ** 2 * a.sum(1) ** 2 / (n_q ** 2 * delta ** 2 + t ** 2 * (a ** 2).sum(1))


def leaderboard_budget(g, sigma, delta, n_q, t, iters=3000, step=0.5):
    """Joint leaderboard budget: minimal expected labels to certify every pair in g at |Delta|/SE >= t jointly,
      min sum_d pi_d  s.t.  V_p(pi) = (1/Q^2) sum_d g_pd^2 sigma_d^2 (1-pi_d)/pi_d <= Delta_p^2 / t^2,  0 < pi_d <= 1.
    Convex; KKT gives pi_d = min(1, sqrt(sum_p mu_p g_pd^2 sigma_d^2) / Q).  Multipliers mu by multiplicative updates
    mu_p <- mu_p (V_p / target_p)^step.  Returns (budget, pi [N], max relative violation)."""
    target = delta ** 2 / t ** 2  # [P]
    gs = g ** 2 * (sigma ** 2)[None, :]  # [P, N]
    mu = np.ones(len(delta)) * (1.0 / np.maximum(target, 1e-30)) / gs.sum(1).clip(1e-30)
    for _ in range(iters):
        pi = np.clip(np.sqrt(np.maximum(mu @ gs, 1e-300)) / n_q, 1e-12, 1.0)
        V = gs @ ((1 - pi) / pi) / n_q ** 2
        mu = mu * (V / target) ** step
    viol = (V / target - 1).max()
    return pi.sum(), pi, viol


def log_e_onesided(s_sum, n, sigma2, n_target):
    """log of the one-sided Gaussian-mixture e-process for H: mean <= 0 of martingale increments with conditional
    variance sigma2 (Robbins' normal mixture over lambda >= 0, half-normal prior with variance rho^2,
    rho^2 = 1 / (n_target sigma2)); with sigma2 replaced by the running variance this is the asymptotic
    confidence-sequence construction of Waudby-Smith et al. (2021), valid in the large-sample sense only: it is NOT
    an e-process (E[E_n] <= 1 fails for skewed increments, scripts/theory/eprocess_validity.py); bet_step is the exact one.
      E_n = 2 / sqrt(1 + n rho^2 sigma2) * exp(S^2 rho^2 / (2 (1 + n rho^2 sigma2))) * Phi(S rho / sqrt(1 + n rho^2 sigma2))
    s_sum, sigma2 broadcast; returns log E_n."""
    from scipy.special import log_ndtr

    r2 = 1.0 / (n_target * sigma2)
    v = 1.0 + n * r2 * sigma2
    return np.log(2.0) - 0.5 * np.log(v) + s_sum ** 2 * r2 / (2 * v) + log_ndtr(s_sum * np.sqrt(r2) / np.sqrt(v))


def bet_step(logk, s1, s2, z, lo, cap):
    """One step of the predictable plug-in betting e-process for H0: E[z_j | past] <= 0, with z_j >= lo_j and lo_j
    known before step j (Waudby-Smith & Ramdas 2024, aGRAPA): K_j = prod_i (1 + lam_i z_i),
      lam_j = clip(mu / (var + mu^2), 0, cap / (-lo_j)) = clip(s1 / s2, 0, cap / (-lo_j))
    with s1, s2 the sums of z and z^2 over the earlier steps only.  lam_j >= 0 and 1 + lam_j z_j >= 1 - cap > 0, so
    E[1 + lam_j z_j | past] = 1 + lam_j E[z_j | past] <= 1 under H0: K is a nonnegative supermartingale and
    P(sup_j K_j >= 1/a) <= a (Ville), at any stopping time.  Arrays broadcast; returns updated (logk, s1, s2)."""
    lam = np.divide(s1, s2, out=np.zeros_like(s1), where=s2 > 0)
    lim = np.divide(cap, -lo, out=np.full_like(lo, np.inf), where=lo < 0)
    lam = np.clip(lam, 0.0, lim)
    return logk + np.log1p(lam * z), s1 + z, s2 + z ** 2


def ebh(e, alpha):
    """e-BH (Wang & Ramdas 2022): reject the k* largest e-values, k* = max{k : e_(k) >= m / (alpha k)}."""
    m = len(e)
    order = np.argsort(-e, kind="stable")
    ok = e[order] >= m / (alpha * np.arange(1, m + 1))
    rej = np.zeros(m, bool)
    if ok.any():
        rej[order[: np.flatnonzero(ok).max() + 1]] = True
    return rej


def decision_weights(delta_hat, se_hat, delta_floor):
    """Pair weights for decision-aware sampling: a pair needs SE below |Delta|/z, so its variance counts with weight
    1/max(|Delta|, floor)^2; pairs already certified (|Delta| > 3 SE) get no weight."""
    w = 1.0 / np.maximum(np.abs(delta_hat), delta_floor) ** 2
    return np.where(np.abs(delta_hat) > 3 * se_hat, 0.0, w)


def allocation_weight(g, v, pair_w=None):
    """Optimal Poisson inclusion weight sqrt(sum_p w_p g_pd^2 v_d) for the objective sum_p w_p Var_p (T3)."""
    pw = np.ones(g.shape[0]) if pair_w is None else pair_w
    return np.sqrt(pw @ (g ** 2) * v)


def bh(p, alpha, dependent=False):
    """Benjamini-Hochberg (Benjamini-Yekutieli when dependent=True) rejections."""
    m = len(p)
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / (m * c)
    rej = np.zeros(m, bool)
    if ok.any():
        rej[order[: np.flatnonzero(ok).max() + 1]] = True
    return rej


def z_pvalues(est, var):
    se = np.sqrt(var)
    return np.where(se > 0, 2 * norm.sf(np.abs(est) / np.where(se > 0, se, 1.0)), np.where(est != 0, 0.0, 1.0))


def simulate_qrels(rng, y, pool_hits, a, b, a_fp, c=0.0, min_rel=1, shift=None):
    """MNAR sparse qrels: a relevant item (y >= min_rel) is labelled with prob sigmoid(a + b * pool_hits + c 1[y=2]
    + shift_d) (c > 0: annotators label highly relevant documents more readily; shift [n]: a content-dependent logit
    term, e.g. calibrated on real qrels from judge and encoder scores, which breaks within-stratum SCAR); a
    non-relevant item is labelled (a qrels error) with prob sigmoid(a_fp + b * pool_hits).  pool_hits[d] in [0, 1] is
    the share of pool systems retrieving d."""
    s = np.zeros(len(y)) if shift is None else np.where(y >= min_rel, shift, 0.0)
    e = 1.0 / (1.0 + np.exp(-(np.where(y >= min_rel, a, a_fp) + b * pool_hits + c * (y == 2) + s)))
    return rng.random(len(y)) < e, e


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def pu_fuse(o, z, x, lab_idx, lab_y, rho, n_iter=30, w0=None, clamp_out=True, lab_w=None, C=1.0):
    """M1+M2 presence-only fusion (Ward et al. 2009 EM; shadow variables z, Miao & Tchetgen Tchetgen 2016).

    Model: P(Y=k | z) = softmax(theta_k^T z), k in {0,1,2};  P(O=1 | x, Y=k) = e(x,k) = sigmoid(phi^T x + psi_k),
    e(x,0) = 0 (qrels errors are left to the design-based correction).  Items with a human label have Y fixed.
    z [n, p] and x [n, q] carry no intercept column.  Returns f = E[u | z, x, O], Var(u | z, x, O), and e(x,k) [n, 3].
    clamp_out=False returns the model posterior for the labelled items too (not their observed Y): a predictor used in
    a design-based estimator must not equal the label it is corrected against.
    lab_w [k] (None: all 1) weights labelled item d's complete-data term; the EM ascends the penalised
      l(theta) = sum_{d unlabelled} log P(O_d | z, x) + sum_{d labelled} lam_d log P(Y_d, O_d | z, x).
    lam = 1 is the observed-data likelihood, valid only if human sampling is ignorable given (z, x, O).  lam = 1/pi
    is a design-weighted composite likelihood: over the Poisson draw E[l] = sum_d log P(Y_d, O_d) +
    sum_d (1 - pi_d) log P(O_d), a census complete-data term plus an O-marginal term, neither of which depends on
    which items were drawn.  All EM weights stay >= 0, so the EM remains monotone.
    C: inverse L2 strength of both M-step regressions.
    """
    from sklearn.linear_model import LogisticRegression

    n = len(o)
    known = np.full(n, -1)
    known[lab_idx] = lab_y
    kn = known >= 0
    lam = np.ones(n)
    if lab_w is not None:
        assert lab_w.shape == (len(lab_idx),) and (lab_w >= 1.0).all()
        lam[lab_idx] = lab_w
    lab_rows = np.zeros((int(kn.sum()), 3))  # [k, 3] lam_d 1[Y_d = k]: the complete-data weights of labelled items
    lab_rows[np.arange(int(kn.sum())), known[kn]] = lam[kn]
    if w0 is None:  # initial responsibilities
        w = np.zeros((n, 3))
        w[o.astype(bool), 1:] = 0.5
        w[~o.astype(bool)] = [0.8, 0.1, 0.1]
    else:  # warm start from a previous fit (refits after each stage of human labels)
        w = w0.copy()
    ks = np.arange(3)
    for _ in range(n_iter):
        w[kn] = lab_rows  # one-hot labels when lam = 1
        # M step, theta: weighted multinomial regression of Y on z
        zz = np.repeat(z, 3, axis=0)
        kk = np.tile(ks, n)
        ww = w.reshape(-1)
        keep = ww > 1e-8
        m_p = LogisticRegression(C=C, max_iter=500).fit(zz[keep], kk[keep], sample_weight=ww[keep])
        p = m_p.predict_proba(z)  # [n, 3], classes 0,1,2
        assert list(m_p.classes_) == [0, 1, 2]
        # M step, phi/psi: weighted logistic regression of O on (x, 1[k=2]) over the relevant states
        xx = np.vstack([np.column_stack([x, np.zeros(n)]), np.column_stack([x, np.ones(n)])])  # k=1 rows, k=2 rows
        oo = np.concatenate([o, o]).astype(int)
        we = np.concatenate([w[:, 1], w[:, 2]])
        keep = we > 1e-8
        m_e = LogisticRegression(C=C, max_iter=500).fit(xx[keep], oo[keep], sample_weight=we[keep])
        e = np.zeros((n, 3))
        e[:, 1] = m_e.predict_proba(np.column_stack([x, np.zeros(n)]))[:, 1]
        e[:, 2] = m_e.predict_proba(np.column_stack([x, np.ones(n)]))[:, 1]
        # E step
        lik = np.where(o[:, None].astype(bool), p * e, p * (1 - e))
        w = lik / lik.sum(1, keepdims=True)
    if clamp_out:
        w[kn] = lab_rows / lam[kn, None]
    gk = np.array([0.0, rho, 1.0])
    f = w @ gk
    return f, w @ gk ** 2 - f ** 2, e, w
