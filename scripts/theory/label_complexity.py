"""Is the label-complexity theory (single-pair and joint label budgets under the optimal design) predictive of how
many pairs a given budget certifies?

Prediction under the realised design pi of a single-stage cross-fitted method (witness_xfit, active_xfit of
scripts/eval/evaluate.py).  pi is captured from the evaluator's own run() (the last W.sampling_probs_clip call of a
single-stage run is the draw's pi), so the design is reproduced exactly, not re-implemented.
Oracle ingredients, per pair p and endpoint b in {rho=0, rho=1}:
  e_pb   = u_b - Z beta_pb, beta_pb the population limit of the fold-out GREG fit (weights g^2 (1-pi)/pi over the frame,
           same ridge rule as witness.core.greg_fit with (K-1)/K of the frame per fold);
  V_pb   = (1/Q^2) sum_d g_pd^2 (1-pi_d)/pi_d e_pbd^2,  C_p = (1/Q^2) sum_d g_pd^2 (1-pi_d)/pi_d e_p0d e_p1d,
  mu_pb  = true Delta_p(rho_b).
CLT: (Dhat_p0, Dhat_p1) ~ N(mu_p, [[V0, C], [C, V1]]); SE treated as known (= sqrt V).
Certification (certify of scripts/eval/evaluate.py): A>B iff min(z_p0, z_p1) >= z*, B>A iff max(z_p0, z_p1) <= -z*,
z* = Phi^{-1}(1 - alpha k / (2P)) with k the number of BH rejections = the number of certified pairs.
  pcert_p(z) = P2(z - mu0/s0, z - mu1/s1; r) + P2(z + mu0/s0, z + mu1/s1; r),  P2(a, b; r) = P(Z0 >= a, Z1 >= b)
  E[K] ~ K(z*(k)) solved as the largest fixed point k = sum_p pcert_p(z*(k))  (as scripts/theory/official_flips.py).
Ladder of predictions (each changes one ingredient): 'old det' (the single-endpoint rule: rho=1/2, t=3, count of |Delta| >= 3 SE) -> det two
endpoints t=3 -> det two endpoints z* fixed point -> E[K] independent endpoints -> E[K] bivariate -> Monte Carlo of the
full 2P-dimensional Gaussian (cross-pair dependence) with exact BH.  Realised: the replayed estimator on fresh draws and
the evaluator's raw csv.
Modes:
  sim <testbed.yaml> <eval.yaml> <rep_lo> <rep_hi> <out.pkl>    replay + predictions per rep/budget/method
  opt <testbed.yaml> <eval.yaml> <out.pkl>                       joint-budget optimal-design oracles (single- and two-endpoint)
  power <testbed.yaml> <eval.yaml>                               power-corrected joint budget (t = z*(m) + z_beta)
  d3  <configs/d3_duretrieval.yaml> <batch1.xlsx> <batch2.xlsx>              D3 ex-ante prediction from batch 1 and post-hoc plug-in
  report <raw_csv_d1> <raw_csv_d2> <sim/opt pkl ...>             tables
"""
import hashlib
import os
import pickle
import sys

import numpy as np
from scipy.stats import norm
from witness.config import load_config  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1]
METHODS = ("witness_xfit", "active_xfit")
GL_X, GL_W = np.polynomial.legendre.leggauss(400)


def bvn_upper(a, b, r):
    """P(Z0 >= a, Z1 >= b), standard bivariate normal with correlation r; broadcast arrays.
    = int_a^inf phi(x) Phi((r x - b) / sqrt(1 - r^2)) dx, Gauss-Legendre on [max(a, -9), max(a, -9) + 18]."""
    a, b, r = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.clip(np.asarray(r, float), -.999, .999))
    lo = np.maximum(a, -9.0)
    hi = lo + 18.0
    x = (lo + hi)[..., None] / 2 + (hi - lo)[..., None] / 2 * GL_X
    f = norm.pdf(x) * norm.cdf((r[..., None] * x - b[..., None]) / np.sqrt(1 - r[..., None] ** 2))
    return (f * GL_W).sum(-1) * (hi - lo) / 2


def pcert(z, m0, m1, s0, s1, r, joint=True):
    """Per-pair certification probability at threshold z (either direction), endpoints jointly Gaussian."""
    t0, t1 = m0 / s0, m1 / s1
    if joint:
        return bvn_upper(z - t0, z - t1, r) + bvn_upper(z + t0, z + t1, r)
    return norm.sf(z - t0) * norm.sf(z - t1) + norm.cdf(-z - t0) * norm.cdf(-z - t1)


def bh_fixed_point(prob_of_z, P, alpha, iters=200):
    """Largest fixed point of k = sum_p prob_p(z*(k)), z*(k) = Phi^{-1}(1 - alpha max(k, 1) / (2P)); iterate down from k = P."""
    k = float(P)
    for _ in range(iters):
        z = norm.isf(alpha * max(k, 1.0) / (2 * P))
        k_new = float(prob_of_z(z).sum())
        if abs(k_new - k) < 1e-7:
            break
        k = k_new
    return k, z


def det_fixed_point(tmin, P, alpha):
    """Deterministic analogue: k = #{p: tmin_p >= z*(k)}, largest fixed point."""
    k = P
    for _ in range(4 * P):
        z = norm.isf(alpha * max(k, 1) / (2 * P))
        k_new = int((tmin >= z).sum())
        if k_new == k:
            break
        k = k_new
    return k, z


def bh_count(z0, z1, alpha):
    """Exact certification rule of certify in scripts/eval/evaluate.py for arrays [..., P] of endpoint z-scores (vectorised BH)."""
    P = z0.shape[-1]
    p_plus = np.maximum(norm.sf(z0), norm.sf(z1))
    p_minus = np.maximum(norm.cdf(z0), norm.cdf(z1))
    p = np.sort(np.concatenate([p_plus, p_minus], -1), -1)
    ok = p <= alpha * np.arange(1, 2 * P + 1) / (2 * P)
    return np.where(ok.any(-1), 2 * P - np.argmax(ok[..., ::-1], -1), 0)  # number of rejections = certified pairs


def pop_greg(Gp, Zc, u, pi, beta0, ridge, n_folds):
    """Population limit of the fold-out GREG fit (witness.core.greg_fit on (K-1)/K of the frame) for rows Gp [P, N].
    Returns residuals e [P, N] = u - Z beta_p."""
    N, m = Zc.shape
    c = (n_folds - 1) / n_folds
    w = Gp ** 2 * ((1 - pi) / pi)[None, :]  # E over the draw of a_rd 1[labelled]
    Z2 = (Zc[:, :, None] * Zc[:, None, :]).reshape(N, m * m)
    A = (w @ Z2).reshape(-1, m, m)
    bvec = w @ (Zc * u[:, None])
    r = ridge * np.maximum(np.trace(A, axis1=1, axis2=2) / (m * pi.sum()), 1e-12)  # ridge*tr(cA)/(m * c n)
    beta = np.linalg.solve(c * A + r[:, None, None] * np.eye(m), (c * bvec + r[:, None] * beta0)[..., None])[..., 0]
    return u[None, :] - beta @ Zc.T


def oracle_moments(Gp, pi, e0, e1, n_q):
    k = Gp ** 2 * ((1 - pi) / pi)[None, :] / n_q ** 2
    return (k * e0 ** 2).sum(1), (k * e1 ** 2).sum(1), (k * e0 * e1).sum(1)


def predictions(mu0, mu1, muh, V0, V1, C, Vh, alpha, rng, n_mc, Mfull=None):
    """The ladder of predictions for one design.  Mfull [2P, N]: rows g_p e_pb sqrt((1-pi)/pi)/Q for the MC with
    cross-pair dependence (covariance Mfull Mfull^T)."""
    P = len(mu0)
    s0, s1 = np.sqrt(V0), np.sqrt(V1)
    r = C / (s0 * s1)
    out = {}
    out["old_det_t3"] = int((np.abs(muh) >= 3 * np.sqrt(Vh)).sum())
    same = np.sign(mu0) == np.sign(mu1)
    tmin = np.where(same, np.minimum(np.abs(mu0) / s0, np.abs(mu1) / s1), 0.0)
    out["det2_t3"] = int((tmin >= 3).sum())
    out["det2_bh"], out["z_det2_bh"] = det_fixed_point(tmin, P, alpha)
    out["EK_rhalf_bh"], _ = bh_fixed_point(lambda z: norm.sf(z - np.abs(muh) / np.sqrt(Vh)), P, alpha)
    out["EK_t3_joint"] = float(pcert(3.0, mu0, mu1, s0, s1, r).sum())
    out["EK_indep"], out["z_indep"] = bh_fixed_point(lambda z: pcert(z, mu0, mu1, s0, s1, r, joint=False), P, alpha)
    out["EK_joint"], out["z_joint"] = bh_fixed_point(lambda z: pcert(z, mu0, mu1, s0, s1, r), P, alpha)
    out["p_pair"] = pcert(out["z_joint"], mu0, mu1, s0, s1, r)
    out["r_end"] = r
    # MC: (a) pairs independent, endpoints correlated; (b) full covariance (shared items across pairs)
    L2 = np.stack([np.stack([s0, np.zeros(P)]), np.stack([r * s1, np.sqrt(np.maximum(1 - r ** 2, 0)) * s1])])  # [2,2,P]
    eps = rng.standard_normal((n_mc, 2, P))
    d0 = mu0 + L2[0, 0] * eps[:, 0]
    d1 = mu1 + L2[1, 0] * eps[:, 0] + L2[1, 1] * eps[:, 1]
    k_ind = bh_count(d0 / s0, d1 / s1, alpha)
    out["MC_pairs_indep"], out["MC_pairs_indep_sd"] = float(k_ind.mean()), float(k_ind.std())
    if Mfull is not None:
        S = Mfull @ Mfull.T
        Lf = np.linalg.cholesky(S + 1e-14 * np.eye(2 * P) * np.trace(S) / (2 * P))
        dd = np.concatenate([mu0, mu1]) + rng.standard_normal((n_mc, 2 * P)) @ Lf.T
        kf = bh_count(dd[:, :P] / s0, dd[:, P:] / s1, alpha)
        out["MC_full"], out["MC_full_sd"] = float(kf.mean()), float(kf.std())
    return out


if MODE in ("sim", "opt", "power"):
    testbed, evalcfg = sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]
    sys.argv = [sys.argv[0], testbed, evalcfg]
    __file__ = os.path.join(here, "..", "eval", "evaluate.py")
    _src = open(__file__).read()
    exec(compile(_src[: _src.index("\nrows = []\n")], __file__, "exec"))
    print("md5 witness_complexity_check.py", hashlib.md5(open(os.path.abspath(sys.argv[0]), "rb").read()).hexdigest())
    FR = D.in_union  # noqa: F821
    Gp = G[:P][:, FR]  # noqa: F821
    MU = [TRUTH[b][:P] for b in range(3)]  # noqa: F821
    _last = {}
    _spc = W.sampling_probs_clip  # noqa: F821

    def _capture(weight, n_, pi_min):
        out = _spc(weight, n_, pi_min)
        _last["pi"] = out
        return out

    W.sampling_probs_clip = _capture  # noqa: F821

    def oracle_for(o, spec, pu_w, pi_fr):
        F, _ = base_predictors(spec["f"], o, pu_w)  # noqa: F821
        es = []
        for b in range(3):
            Zc, b0 = design(F[b], o, spec["strata"], spec["extras"])  # noqa: F821
            es.append(pop_greg(Gp, Zc[FR], U_END[b][FR], pi_fr, b0, cfg["ridge"], cfg["xfit_folds"]))  # noqa: F821
        return es

if MODE == "sim":
    rep_lo, rep_hi, out_path = int(rest[0]), int(rest[1]), rest[2]
    recs = []
    for rep in range(rep_lo, rep_hi):
        rng = np.random.default_rng([cfg["seed"], rep])  # noqa: F821
        o, _ = MECH.draw(rng)  # noqa: F821  (same rng stream as the evaluator's main loop)
        _, _, _, pu_w = W.pu_fuse(o, Z_PU, X_PU, np.array([], int), np.array([], int), 0.5)  # noqa: F821
        rng_mc = np.random.default_rng([cfg["seed"], rep, 777])  # noqa: F821
        for budget in cfg["budgets"]:  # noqa: F821
            for m in METHODS:
                spec = SPECS[m]  # noqa: F821
                est, var, nl = run(o, budget, spec, pu_w, rng)  # noqa: F821
                pi_fr = _last["pi"]
                assert len(pi_fr) == FR.sum() and np.isclose(pi_fr.sum(), budget * spec["fracs"][0], rtol=1e-5)
                e0, eh, e1 = oracle_for(o, spec, pu_w, pi_fr)
                V0, V1, C = oracle_moments(Gp, pi_fr, e0, e1, n_q)  # noqa: F821
                Vh = oracle_moments(Gp, pi_fr, eh, eh, n_q)[0]  # noqa: F821
                sq = np.sqrt((1 - pi_fr) / pi_fr)[None, :] / n_q  # noqa: F821
                pr = predictions(MU[0], MU[2], MU[1], V0, V1, C, Vh, cfg["alpha"], rng_mc, 4000,  # noqa: F821
                                 Mfull=np.vstack([Gp * e0 * sq, Gp * e1 * sq]))
                c = certify(est, var, TRUTH)  # noqa: F821
                z0 = est[0][:P] / np.sqrt(np.maximum(var[0][:P], 1e-30))  # noqa: F821
                z1 = est[2][:P] / np.sqrt(np.maximum(var[2][:P], 1e-30))  # noqa: F821
                k_real = int(bh_count(z0, z1, cfg["alpha"]))  # noqa: F821
                assert abs(k_real - c["cert_bh"] * P) < 1e-6  # noqa: F821
                # which pairs were certified (for reliability): recompute BH cut
                zc = norm.isf(cfg["alpha"] * max(k_real, 1) / (2 * P))  # noqa: F821
                cert_pair = (np.minimum(z0, z1) >= zc) | (np.maximum(z0, z1) <= -zc) if k_real > 0 else np.zeros(P, bool)  # noqa: F821
                recs.append(dict(rep=rep, budget=budget, method=m, n_labels=nl, k_real=k_real, cert_pair=cert_pair,
                                 est0=est[0][:P], est1=est[2][:P], var0=var[0][:P], var1=var[2][:P],  # noqa: F821
                                 V0=V0, V1=V1, C=C, mu0=MU[0], mu1=MU[2], **pr))
                print(f"rep {rep} n={budget} {m:13s} realised {k_real:3d} | old det(t=3,rho=.5) {pr['old_det_t3']:3d} "
                      f"det2 t3 {pr['det2_t3']:3d} det2 BH {pr['det2_bh']:3d} | EK indep {pr['EK_indep']:6.2f} "
                      f"joint {pr['EK_joint']:6.2f} (z*={pr['z_joint']:.2f}) MC pairs-indep {pr['MC_pairs_indep']:6.2f} "
                      f"MC full {pr['MC_full']:6.2f} | median Vhat/V {np.median(np.r_[var[0][:P] / V0, var[2][:P] / V1]):.3f}",  # noqa: F821
                      flush=True)
        pickle.dump(recs, open(out_path, "wb"))
    print("DONE")

if MODE in ("opt", "power"):
    # joint-budget optimal-design oracles at each budget: the largest m (easiest pairs first) whose joint budget fits n.
    out_path = rest[0] if MODE == "opt" else None
    rng = np.random.default_rng([cfg["seed"], 0])  # noqa: F821
    o, _ = MECH.draw(rng)  # noqa: F821
    _, _, _, pu_w = W.pu_fuse(o, Z_PU, X_PU, np.array([], int), np.array([], int), 0.5)  # noqa: F821
    # single-endpoint sigma (scripts/diagnostics/decomposition.py): cell-calibrated judge plug-in residual SD at rho = 1/2
    J = D.J[:, FR]  # noqa: F821
    ansj = J >= 0
    gain = np.where(J == 2, 1.0, np.where(J == 1, 0.5, 0.0))
    f = np.where(ansj.sum(0) > 0, (gain * ansj).sum(0) / np.maximum(ansj.sum(0), 1), D.ce[FR])  # noqa: F821
    bres = f - U_END[1][FR]  # noqa: F821
    _, cidx = np.unique(((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0), return_inverse=True)
    bbar = np.bincount(cidx, weights=bres) / np.bincount(cidx)
    sig_old = np.sqrt((np.bincount(cidx, weights=(bres - bbar[cidx]) ** 2) / np.bincount(cidx))[cidx])
    alpha = cfg["alpha"]  # noqa: F821
    res = {}

    def max_m(rows_g, rows_d, order_groups, t_of_m, ms):
        """Joint budget curve [(m, budget, max rel. violation)] for the first m groups at level t(m)."""
        curve = []
        for mm in ms:
            sel = np.concatenate(order_groups[:mm])
            bud, pi_s, viol = W.leaderboard_budget(rows_g[sel], np.ones(rows_g.shape[1]), rows_d[sel], n_q, t_of_m(mm),  # noqa: F821
                                                   iters=2000)
            curve.append((mm, float(bud), float(viol)))
        return curve

    def ek_at_budget(rows_g, rows_d, order_groups, t_of_m, ms, n_budget, e0, e1):
        """E[K] (joint endpoints, BH fixed point, oracle residuals e0/e1 held fixed) under the joint-budget-optimal pi of the
        largest m in ms whose joint budget fits n, after scaling that pi up to spend exactly n (clip at 1)."""
        best = None
        for mm in ms:
            sel = np.concatenate(order_groups[:mm])
            bud, pi_s, viol = W.leaderboard_budget(rows_g[sel], np.ones(rows_g.shape[1]), rows_d[sel], n_q, t_of_m(mm),  # noqa: F821
                                                   iters=2000)
            if bud <= n_budget:
                best = (mm, pi_s)
        if best is None:
            return 0, float("nan")
        mm, pi_s = best
        pi_n = W.sampling_probs_clip(pi_s, n_budget, 0.0) if pi_s.sum() < n_budget else pi_s  # noqa: F821
        V0, V1, C = oracle_moments(Gp, pi_n, e0, e1, n_q)  # noqa: F821
        s0, s1 = np.sqrt(V0), np.sqrt(V1)
        k, _ = bh_fixed_point(lambda z: pcert(z, MU[0], MU[2], s0, s1, C / (s0 * s1)), P, alpha)  # noqa: F821
        return mm, k

    if MODE == "power":
        # power-corrected joint budget: per endpoint SE <= |Delta| / (z*(m) + z_beta), so a target pair certifies with
        # probability about (1 - beta) per endpoint instead of 1/2; m_max is then a count prediction to compare with
        # the expected count E[K] at that design and with the realised count of the evaluator's design.
        for budget in cfg["budgets"]:  # noqa: F821
            spec = SPECS["witness_xfit"]  # noqa: F821
            run(o, budget, spec, pu_w, rng)  # noqa: F821
            pi_fr = _last["pi"]
            e0, eh, e1 = oracle_for(o, spec, pu_w, pi_fr)
            mu0, mu1 = MU[0], MU[2]
            lc = lambda ge, d, t: W.label_complexity(ge, np.ones(ge.shape[1]), d, n_q, t)  # noqa: E731,F821
            n2 = np.maximum(lc(Gp * np.abs(e0), mu0, 2.5), lc(Gp * np.abs(e1), mu1, 2.5))
            n2[~((np.sign(mu0) == np.sign(mu1)) & (mu0 != 0) & (mu1 != 0))] = np.inf
            o2 = np.argsort(n2)
            rows2 = np.vstack([Gp * np.abs(e0), Gp * np.abs(e1)])
            d2 = np.r_[mu0, mu1]
            groups2 = [np.array([p, P + p]) for p in o2[: np.isfinite(n2).sum()]]  # noqa: F821
            n_fin = int(np.isfinite(n2).sum())
            ms = list(range(1, n_fin + 1)) if n_fin <= 25 else sorted({int(x) for x in np.linspace(1, n_fin, 25).round()})
            V0, V1, C = oracle_moments(Gp, pi_fr, e0, e1, n_q)  # noqa: F821
            k_real_design, _ = bh_fixed_point(lambda z: pcert(z, mu0, mu1, np.sqrt(V0), np.sqrt(V1), C / np.sqrt(V0 * V1)), P, alpha)  # noqa: F821
            for zb in (0.0, 0.8416, 1.2816):
                m_max, ek = ek_at_budget(rows2, d2, groups2, lambda mm: norm.isf(alpha * mm / (2 * P)) + zb, ms, budget, e0, e1)  # noqa: F821
                print(f"n={budget} z_beta={zb:.2f}: label-complexity feasible m_max={m_max}  E[K] at that design (scaled to n)={ek:.2f}  "
                      f"| E[K] at the WITNESS design={k_real_design:.2f}", flush=True)
        print("DONE")
        sys.exit(0)
    for budget in cfg["budgets"]:  # noqa: F821
        spec = SPECS["witness_xfit"]  # noqa: F821
        run(o, budget, spec, pu_w, rng)  # noqa: F821  (captures the realised pi)
        pi_fr = _last["pi"]
        e0, eh, e1 = oracle_for(o, spec, pu_w, pi_fr)
        mu0, mu1, muh = MU[0], MU[2], MU[1]
        ok2 = (np.sign(mu0) == np.sign(mu1)) & (mu0 != 0) & (mu1 != 0)
        okh = muh != 0
        # single-row interior label complexities for the ordering
        lc = lambda ge, d, t: W.label_complexity(ge, np.ones(ge.shape[1]), d, n_q, t)  # noqa: E731,F821
        n2 = np.maximum(lc(Gp * np.abs(e0), mu0, 2.5), lc(Gp * np.abs(e1), mu1, 2.5))
        n2[~ok2] = np.inf
        nh_old = lc(Gp * sig_old[None, :], muh, 3.0)
        nh_old[~okh] = np.inf
        nh_new = lc(Gp * np.abs(eh), muh, 3.0)
        nh_new[~okh] = np.inf
        # every m for small leaderboards, about 30 points otherwise
        ms_all = lambda nn: sorted({int(x) for x in np.linspace(1, np.isfinite(nn).sum(), min(np.isfinite(nn).sum(), 30)).round()})  # noqa: E731
        o2 = np.argsort(n2)
        rows2 = np.vstack([Gp * np.abs(e0), Gp * np.abs(e1)])
        d2 = np.r_[mu0, mu1]
        groups2 = [np.array([p, P + p]) for p in o2[: np.isfinite(n2).sum()]]  # noqa: F821
        zstar = lambda mm: norm.isf(alpha * mm / (2 * P))  # noqa: E731,F821
        cur_new = max_m(rows2, d2, groups2, zstar, ms_all(n2))
        cur_new_t3 = max_m(rows2, d2, groups2, lambda mm: 3.0, ms_all(n2))
        if budget == cfg["budgets"][0]:  # noqa: F821  (the single-endpoint oracle does not depend on the realised pi)
            oh_old = np.argsort(nh_old)
            g_old = [np.array([p]) for p in oh_old[: np.isfinite(nh_old).sum()]]
            cur_old = max_m(Gp * sig_old[None, :], muh, g_old, lambda mm: 3.0, ms_all(nh_old))
        oh_new = np.argsort(nh_new)
        g_new = [np.array([p]) for p in oh_new[: np.isfinite(nh_new).sum()]]
        cur_oldprog_newsig = max_m(Gp * np.abs(eh), muh, g_new, lambda mm: 3.0, ms_all(nh_new))
        m_opt, ek_opt = ek_at_budget(rows2, d2, groups2, zstar, ms_all(n2), budget, e0, e1)
        print(f"n={budget} E[K] under the two-endpoint optimal-design pi (m={m_opt} target pairs, scaled to n): {ek_opt:.2f}", flush=True)
        res[budget] = dict(two_end_bh=cur_new, two_end_t3=cur_new_t3, old=cur_old, old_prog_new_sigma=cur_oldprog_newsig,
                           realised_pi_sum=float(pi_fr.sum()), ek_opt=(m_opt, ek_opt))
        for name, cur in res[budget].items():
            if name not in ("realised_pi_sum", "ek_opt"):
                print(f"n={budget} {name:20s} " + "; ".join(f"{mm}:{b:.0f}" for mm, b, v in cur), flush=True)
        pickle.dump(res, open(out_path, "wb"))
    print("DONE")

if MODE == "d3":
    # D3 ex-ante prediction after batch 1 for the two-batch certified count, and post-hoc checks of its ingredients.
    # Ingredients exactly as scripts/d3/make_next_batch.py: batch-1 design_fam estimate (est, var), batch-1
    # cell-calibrated working variance s2, the saved batch-2 design pi2; V_fin = (1/var + 1/V_new)^-1.
    from witness import core as W
    from witness import d3 as WD

    cfg = load_config(sys.argv[2])
    sheets = sys.argv[3:]
    assert len(sheets) == 2
    print("md5 witness_complexity_check.py", hashlib.md5(open(os.path.abspath(sys.argv[0]), "rb").read()).hexdigest())
    D = WD.load_frame(cfg)
    b1, b2 = WD.read_batch(D, 1, sheets[0]), WD.read_batch(D, 2, sheets[1])
    P, G, n_q, alpha = D.P, D.G, D.n_q, cfg["alpha"]
    B = cfg["next_batch"]
    est1, var1 = WD.design_estimate(D, [b1], B["extras"], cfg["ridge"], D.F)
    s2 = [WD.working_var(D, b, [b1]) for b in (0, 2)]
    pi2_plan = WD.batch_pi(D, 2)
    v_new = [W.predicted_var(G, pi2_plan, np.broadcast_to(s, G.shape), n_q)[:P] for s in s2]
    v_fin = [1.0 / (1.0 / np.maximum(var1[b][:P], 1e-30) + 1.0 / np.maximum(v, 1e-30)) for b, v in zip((0, 2), v_new)]
    m0, m1 = est1[0][:P], est1[2][:P]
    pv0, pv1 = var1[0][:P], var1[2][:P]
    # endpoint correlation of the batch-1 HT terms (residual vs the judge plug-in), used for the joint law
    it = b1.items
    u1 = WD.gains(b1.y)
    r0, r1 = u1[0] - D.F[0][it], u1[2] - D.F[2][it]
    kk = G[:P][:, it] ** 2 * ((1 - b1.pi[it]) / b1.pi[it] ** 2)[None, :]
    c01 = (kk * r0 * r1).sum(1) / np.sqrt((kk * r0 ** 2).sum(1) * (kk * r1 ** 2).sum(1))
    print(f"batch-1 endpoint correlation of HT terms, quantiles (.1,.5,.9) = {np.round(np.quantile(c01, [.1, .5, .9]), 3).tolist()}")
    # predictive Delta ~ N(est1, var1) per endpoint (correlation c01), Gauss-Hermite 9 x 9
    gx, gw = np.polynomial.hermite_e.hermegauss(9)
    gw = gw / gw.sum()
    X0 = gx[:, None] + 0 * gx[None, :]
    X1 = gx[None, :] + 0 * gx[:, None]
    WW = gw[:, None] * gw[None, :]

    def pc_pred(z, joint, avg=True):
        """Predicted per-pair certification probability after batch 2 at threshold z."""
        s0, s1 = np.sqrt(v_fin[0]), np.sqrt(v_fin[1])
        if not avg:
            return pcert(z, m0, m1, s0, s1, c01, joint)
        rr = c01[:, None, None]
        d0 = m0[:, None, None] + np.sqrt(pv0)[:, None, None] * X0
        d1 = m1[:, None, None] + np.sqrt(pv1)[:, None, None] * (rr * X0 + np.sqrt(1 - rr ** 2) * X1)
        p = pcert(z, d0, d1, s0[:, None, None], s1[:, None, None], rr + 0 * d0, joint)
        return (p * WW).sum((1, 2))

    # planner reproduction: independent endpoints, each endpoint averaged separately (W.cert_weights), z = 2.5
    _, pc0 = W.cert_weights(m0, pv0, v_fin[0], B["z_plan"])
    _, pc1 = W.cert_weights(m1, pv1, v_fin[1], B["z_plan"])
    # the planner's final pass uses pi from the last plan iteration = the saved pi2, so this must reproduce its 14.3
    print(f"planner reproduction (indep endpoints, z={B['z_plan']}, separate averaging): {np.sum(pc0 * pc1):.2f}")
    final_est, final_var = WD.design_estimate(D, [b1, b2], B["extras"], cfg["ridge"], D.F)
    z0f = final_est[0][:P] / np.sqrt(final_var[0][:P])
    z1f = final_est[2][:P] / np.sqrt(final_var[2][:P])
    k_real = int(bh_count(z0f, z1f, alpha))
    zc = norm.isf(alpha * max(k_real, 1) / (2 * P))
    cert = (np.minimum(z0f, z1f) >= zc) | (np.maximum(z0f, z1f) <= -zc)
    print(f"realised two-batch certified (design_fam, BH): {k_real}  (BH cut z={zc:.3f})")
    s2h = WD.working_var(D, 1, [b1])  # rho = 1/2 block, same cell calibration
    vh_fin = 1.0 / (1.0 / var1[1][:P] + 1.0 / W.predicted_var(G, pi2_plan, np.broadcast_to(s2h, G.shape), n_q)[:P])
    det_old = np.abs(est1[1][:P]) >= 3 * np.sqrt(vh_fin)
    pr = {"old det rho=1/2 t=3": det_old.astype(float),
          "det two endpoints t=3": ((np.sign(m0) == np.sign(m1)) & (np.minimum(np.abs(m0) / np.sqrt(v_fin[0]), np.abs(m1) / np.sqrt(v_fin[1])) >= 3)).astype(float),
          "indep z=2.5 (planner)": pc0 * pc1,
          "indep z=2.5 joint-avg": pc_pred(2.5, False),
          "joint z=2.5 joint-avg": pc_pred(2.5, True)}
    tm = np.where(np.sign(m0) == np.sign(m1), np.minimum(np.abs(m0) / np.sqrt(v_fin[0]), np.abs(m1) / np.sqrt(v_fin[1])), 0.0)
    kdet, zdet = det_fixed_point(tm, P, alpha)
    pr["det two endpoints BH fixed point"] = (tm >= zdet).astype(float)
    for name, joint, avg in (("indep BH fixed point, no avg", False, False), ("joint BH fixed point, no avg", True, False),
                             ("indep BH fixed point, avg", False, True), ("joint BH fixed point, avg", True, True)):
        k, z = bh_fixed_point(lambda zz: pc_pred(zz, joint, avg), P, alpha)
        pr[name + f" (z*={z:.2f})"] = pc_pred(z, joint, avg)
    print(f"{'prediction':44s} {'E[K]':>6s}  Brier  (realised {k_real})")
    for name, p in pr.items():
        print(f"{name:44s} {p.sum():6.2f}  {np.mean((p - cert) ** 2):.3f}")
    # post-hoc ingredient checks: realised final SE vs the predicted V_fin; batch-2 stage variance vs V_new
    for b, bi in ((0, 0), (2, 1)):
        rat = final_var[b][:P] / v_fin[bi]
        print(f"endpoint rho={b // 2}: final Vhat / predicted V_fin quantiles (.1,.5,.9) = {np.round(np.quantile(rat, [.1, .5, .9]), 3).tolist()}")
    s2_post = [WD.working_var(D, b, [b1, b2]) for b in (0, 2)]
    for b, bi in ((0, 0), (2, 1)):
        rat = W.predicted_var(G, pi2_plan, np.broadcast_to(s2_post[bi], G.shape), n_q)[:P] / v_new[bi]
        print(f"endpoint rho={b // 2}: V_new with two-batch s2 / with batch-1 s2, quantiles (.1,.5,.9) = {np.round(np.quantile(rat, [.1, .5, .9]), 3).tolist()}")
    # Delta drift toward zero: sign(batch1) (final - batch1) / se1 (plug-in |Delta| inflation diagnostic)
    for b in (0, 2):
        dz = np.sign(est1[b][:P]) * (final_est[b][:P] - est1[b][:P]) / np.sqrt(var1[b][:P])
        print(f"endpoint rho={b // 2}: sign(batch1) (final - batch1) / se1 mean {dz.mean():+.3f} sd {dz.std():.3f}; "
              f"|final| < |batch1| for {(np.abs(final_est[b][:P]) < np.abs(est1[b][:P])).sum()}/{P} pairs")
    # is the rho=0 shrinkage a grade-boundary shift between batches?  Within (stratum x judge state) cells, the share
    # of y=2 among y>=1 per batch; batch-2 cells weighted by batch-2 counts (a within-cell comparison, so the different
    # cell mix of the two designs does not enter)
    cell = D.state * 2 + D.o.astype(int)
    diffs, wts = [], []
    for c in np.unique(cell):
        y1c = b1.y[(cell[b1.items] == c) & (b1.y >= 1)]
        y2c = b2.y[(cell[b2.items] == c) & (b2.y >= 1)]
        if len(y1c) >= 5 and len(y2c) >= 5:
            diffs.append((y2c == 2).mean() - (y1c == 2).mean())
            wts.append(len(y2c))
    print(f"share of y=2 among y>=1, batch 2 minus batch 1 within cells (cells with >=5 relevant per batch: {len(diffs)}): "
          f"weighted mean {np.average(diffs, weights=wts):+.3f}, per cell {np.round(diffs, 2).tolist()}")
    # the same with 1/pi (Hajek) weights: each batch's ratio estimates the population cell share, so a difference
    # here is not explained by batch 2 sampling different items within a cell
    hd, hw = [], []
    for c in np.unique(cell):
        est_c = []
        for bt in (b1, b2):
            mc = (cell[bt.items] == c) & (bt.y >= 1)
            wc = 1.0 / bt.pi[bt.items][mc]
            est_c.append((wc * (bt.y[mc] == 2)).sum() / wc.sum() if mc.sum() >= 5 else np.nan)
        if np.isfinite(est_c).all():
            hd.append(est_c[1] - est_c[0])
            hw.append(((cell[b2.items] == c) & (b2.y >= 1)).sum())
    print(f"Hajek-weighted share of y=2 among y>=1, batch 2 minus batch 1 within cells: weighted mean "
          f"{np.average(hd, weights=hw):+.3f}, per cell {np.round(hd, 2).tolist()}")
    print(f"raw shares y=2 among y>=1: batch 1 {(b1.y == 2).sum() / (b1.y >= 1).sum():.3f}, batch 2 {(b2.y == 2).sum() / (b2.y >= 1).sum():.3f}")
    pp = pc0 * pc1
    print(f"predictive sd of K under the planner's per-pair probabilities (pairs independent): {np.sqrt(np.sum(pp * (1 - pp))):.2f}")
    # post-hoc plug-in E[K]: Delta = final estimate, V = final variance (what the measured SEs imply)
    s0f, s1f = np.sqrt(final_var[0][:P]), np.sqrt(final_var[2][:P])
    kpost, zpost = bh_fixed_point(lambda zz: pcert(zz, final_est[0][:P], final_est[2][:P], s0f, s1f, c01), P, alpha)
    print(f"post-hoc plug-in (Delta=final est, V=final var, joint, BH): E[K]={kpost:.2f} z*={zpost:.2f}")
    # 2 x 2 decomposition of the ex-ante error: Delta in {batch-1 est, final est} x V in {predicted V_fin, final var}
    for dn, (d0, d1) in (("batch-1 Delta", (m0, m1)), ("final Delta", (final_est[0][:P], final_est[2][:P]))):
        for vn, (q0, q1) in (("predicted V_fin", (v_fin[0], v_fin[1])), ("final Vhat", (final_var[0][:P], final_var[2][:P]))):
            k, z = bh_fixed_point(lambda zz: pcert(zz, d0, d1, np.sqrt(q0), np.sqrt(q1), c01), P, alpha)
            print(f"   decomposition: {dn:14s} x {vn:16s}: E[K] (joint, BH, no averaging) = {k:.2f} (z*={z:.2f})")
    for p in np.argsort(-pr["joint z=2.5 joint-avg"]):
        a, b_ = D.pairs[p]
        print(f"   {D.systems[a]:14s} vs {D.systems[b_]:14s} planner {pc0[p] * pc1[p]:.2f} joint-avg-z2.5 {pr['joint z=2.5 joint-avg'][p]:.2f} "
              f"r={c01[p]:+.2f} t0={m0[p] / np.sqrt(v_fin[0][p]):+.2f} t1={m1[p] / np.sqrt(v_fin[1][p]):+.2f} realised {int(cert[p])} "
              f"final z0={z0f[p]:+.2f} z1={z1f[p]:+.2f}")
    print("DONE")

if MODE == "report":
    # report <alpha> <testbed name> <sim.pkl> <eval_raw.csv> [<testbed name> <sim.pkl> <eval_raw.csv> ...]
    import pandas as pd

    alpha = float(sys.argv[2])
    args = sys.argv[3:]
    assert len(args) % 3 == 0
    PRED = ["old_det_t3", "det2_t3", "det2_bh", "EK_rhalf_bh", "EK_t3_joint", "EK_indep", "EK_joint", "MC_pairs_indep",
            "MC_full"]
    table, cells_err = [], []
    for name, pkl, raw in zip(args[0::3], args[1::3], args[2::3]):
        recs = pickle.load(open(pkl, "rb"))
        csv = pd.read_csv(raw)
        P = len(recs[0]["mu0"])
        for (budget, m), grp in pd.DataFrame(recs).groupby(["budget", "method"]):
            rr = grp.to_dict("records")
            k = np.array([r["k_real"] for r in rr])
            c = csv[(csv.method == m) & (csv.budget == budget)]
            kc = (c.cert_bh * P).round().to_numpy()
            # realised count with the oracle SE in place of the estimated one (same point estimates)
            k_or = np.array([bh_count(r["est0"] / np.sqrt(r["V0"]), r["est1"] / np.sqrt(r["V1"]), alpha) for r in rr])
            zs = np.concatenate([np.r_[(r["est0"] - r["mu0"]) / np.sqrt(r["V0"]), (r["est1"] - r["mu1"]) / np.sqrt(r["V1"])] for r in rr])
            zh = np.concatenate([np.r_[(r["est0"] - r["mu0"]) / np.sqrt(r["var0"]), (r["est1"] - r["mu1"]) / np.sqrt(r["var1"])] for r in rr])
            vr = np.concatenate([np.r_[r["var0"] / r["V0"], r["var1"] / r["V1"]] for r in rr])
            row = dict(testbed=name, method=m, budget=budget, P=P, reps=len(rr), real=k.mean(), real_sd=k.std(ddof=1),
                       real_csv=kc.mean(), real_csv_sd=kc.std(ddof=1), real_oracleSE=k_or.mean(),
                       MC_full_sd=np.mean([r["MC_full_sd"] for r in rr]),
                       z_sd_oracle=zs.std(), z_sd_hat=zh.std(), vhat_over_v_med=np.median(vr),
                       vhat_over_v_q10=np.quantile(vr, .1), vhat_over_v_q90=np.quantile(vr, .9),
                       r_end_med=np.median(np.concatenate([r["r_end"] for r in rr])))
            for key in PRED:
                row[key] = np.mean([r[key] for r in rr])
            table.append(row)
            # pair-level reliability of p_pair (joint, z* fixed point) against realised certification
            for r in rr:
                cells_err.append(pd.DataFrame(dict(testbed=name, method=m, budget=budget, p=r["p_pair"],
                                                   y=r["cert_pair"].astype(float))))
    T = pd.DataFrame(table)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("== certified pairs per (testbed, method, budget): realised (replay, csv) vs predictions (means over reps)")
    print(T[["testbed", "method", "budget", "P", "reps", "real", "real_sd", "real_csv", "real_csv_sd"] + PRED + ["MC_full_sd"]].round(2).to_string(index=False))
    print("== calibration over cells (prediction - realised replay mean); combined realised = replay and csv pooled")
    T["real_pool"] = (T.real * T.reps + T.real_csv * T.reps) / (2 * T.reps)
    for key in PRED:
        err = T[key] - T.real_pool
        rel = err / T.P
        print(f"   {key:15s} mean err {err.mean():+6.2f}  MAE {err.abs().mean():5.2f}  max|err| {err.abs().max():5.2f}  "
              f"MAE/P {rel.abs().mean():.3f}  "
              + " ".join(f"{t}:{e:+.1f}" for t, e in zip(T.testbed + '/' + T.method.str[:3] + '/' + T.budget.astype(str), err)))
    print("== diagnostics: sd of (est - mu)/sqrt(V_oracle) and /sqrt(Vhat); Vhat/V quantiles; endpoint corr; realised with oracle SE")
    print(T[["testbed", "method", "budget", "z_sd_oracle", "z_sd_hat", "vhat_over_v_q10", "vhat_over_v_med",
             "vhat_over_v_q90", "r_end_med", "real", "real_oracleSE", "MC_full"]].round(3).to_string(index=False))
    E = pd.concat(cells_err)
    print("== pair-level reliability of p_pair (joint, BH fixed point) vs realised certification (replay)")
    for name, e in E.groupby("testbed"):
        bins = np.digitize(e.p, [.05, .2, .4, .6, .8, .95])
        print(f"   {name}: Brier {np.mean((e.p - e.y) ** 2):.4f}  (n={len(e)}) " + "  ".join(
            f"[{lo:.2f},{hi:.2f}) p={e.p[bins == i].mean():.2f} obs={e.y[bins == i].mean():.2f} n={int((bins == i).sum())}"
            for i, (lo, hi) in enumerate(zip([0, .05, .2, .4, .6, .8, .95], [.05, .2, .4, .6, .8, .95, 1.0001])) if (bins == i).any()))
    print("DONE")
