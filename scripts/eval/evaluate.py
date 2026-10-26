"""WITNESS evaluator on a fully labelled testbed with simulated MNAR sparse qrels.

Truth: endpoint gains u0 = 1[y=2], u1 = 1[y>=1]; u_rho = (1-rho) u0 + rho u1, so certifying the sign of Delta at
rho = 0 and rho = 1 certifies it on [0, 1] (intersection-union test).  A third row block, rho = 1/2, is the
point metric used for coverage, RMSE and Kendall tau.
Sparse qrels (binary): an item is labelled with prob sigmoid(a_y + b h + c 1[y=2]) (witness.core.simulate_qrels), or
by a hidden annotation engine / per-query depth mixture (witness.mech, cfg mnar.kind);
the official metric gives every qrels entry gain 1 at every rho.
Every estimator with human labels is a sum of Poisson stages, design-unbiased for any predictor: frame = items in
some top-k list, both the qrels stratum L and the rest U; row-specific predictors H = Zc beta from GREG on earlier
stages' labels (exactly unbiased; "prev"), or one final fit on all labels applied to every stage (PPI++-style, O(1/n)
bias; "final"); stages combined with variance-stabilising weights 1/sqrt(pvar) fixed before each draw, so the stage
errors are martingale differences.  Certification: one-sided endpoint p-values, max over endpoints, BH and BY over
the 2P one-sided nulls; the reported error rate is the false-certification proportion.
usage: python scripts/eval/evaluate.py <testbed.yaml> <eval.yaml> [rep_lo rep_hi]   (repetitions rep_lo..rep_hi-1)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
from witness import rivals as RV  # noqa: E402
from witness import core as W  # noqa: E402
from witness import mech as WM  # noqa: E402
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
base = load_config(sys.argv[1])
cfg = load_config(sys.argv[2])
print("config:", cfg)
REP_LO, REP_HI = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 3 else (0, cfg["reps"])
assert 0 <= REP_LO < REP_HI <= cfg["reps"]
SUFFIX = f"{cfg['tag']}" if (REP_LO, REP_HI) == (0, cfg["reps"]) else f"{cfg['tag']}_r{REP_LO}-{REP_HI}"
import hashlib  # noqa: E402

for path in (os.path.abspath(__file__), W.__file__, RV.__file__, WM.__file__):
    print("md5", os.path.basename(path), hashlib.md5(open(path, "rb").read()).hexdigest())
D = load_testbed(base, cfg)
G, P, n_q, qi = D.G, D.P, D.n_q, D.qi
R = G.shape[0]
n = len(D.y)
U_END = [D.u0, 0.5 * D.u0 + 0.5 * D.u1, D.u1]  # endpoint rho = 0, point rho = 1/2, endpoint rho = 1
TRUTH = [G @ ue / n_q for ue in U_END]
print(f"systems={len(D.systems)} pairs={P} items={n} queries={n_q} frame={int(D.in_union.sum())}")
print("true DCG@k at rho=1/2:", dict(zip(D.systems, np.round(TRUTH[1][P:], 4))))
ans = D.J >= 0
print("judges:", D.judges, "exact agreement:", [round(float((D.J[i] == D.y)[ans[i]].mean()), 3) for i in range(len(D.J))],
      "refused:", (~ans).sum(1).tolist())
HB = np.digitize(D.h, cfg["h_bins"])  # witness strata; per-stratum slopes absorb the scale of the qrels propensity
N_HB = len(cfg["h_bins"]) + 1
LCE = np.log(np.clip(D.ce, 1e-4, 1 - 1e-4) / (1 - np.clip(D.ce, 1e-4, 1 - 1e-4)))
JPAT = ((D.J + 1) * 4 ** np.arange(len(D.J))[:, None]).sum(0)  # joint judge state
JPAT_LEVELS = np.unique(JPAT)
Z_PU = np.column_stack([(JPAT[:, None] == JPAT_LEVELS[None, 1:]).astype(float), LCE, D.h, D.n_top / len(D.systems)])
X_PU = np.column_stack([D.h, D.n_top / len(D.systems)])
# shadow variables without the provenance columns: h and n_top also enter the propensity x, which breaks the exclusion
# restriction and leaves the zero-label EM drifting toward e -> 1
Z_PU_EX = np.column_stack([(JPAT[:, None] == JPAT_LEVELS[None, 1:]).astype(float), LCE])
# Content-dependent qrels propensity (breaks within-stratum SCAR and the shadow-variable exclusion of Z_PU):
# logit shift = gamma_judge * z(mean judge gain) + gamma_ce * z(logit CE) on relevant items, z = standardised over the
# frame.  The D3 real-qrels fit (scripts/d3/pu_real_qrels.py) gives +.65 / +.72 per SD within pool-share strata.
SHIFT = None
if cfg["mnar"]["shift"] is not None:
    _fj = np.where(ans, D.J, 0).sum(0) / np.maximum(ans.sum(0), 1)
    _zs = lambda v: (v - v[D.in_union].mean()) / v[D.in_union].std()  # noqa: E731
    SHIFT = cfg["mnar"]["shift"]["judge"] * _zs(_fj) + cfg["mnar"]["shift"]["ce"] * _zs(LCE)  # [n]
    print(f"qrels shift: judge {cfg['mnar']['shift']['judge']} ce {cfg['mnar']['shift']['ce']} "
          f"sd on frame {SHIFT[D.in_union].std():.3f}")
# qrels mechanism (witness.mech): logistic mnar blocks draw through W.simulate_qrels
MECH = WM.QrelsMech(cfg["mnar"], D.y, D.h, D.qi, D.n_q, D.systems, cfg["out_dir"], SHIFT)
print(f"qrels mechanism: {MECH.kind}; expected qrels recall among relevant in frame "
      f"{MECH.e[D.in_union & (D.y >= 1)].mean():.3f}, expected qrels per query {MECH.e.sum() / n_q:.2f}")
FAMILY = np.array(["bm25" if s.startswith("bm25") else "rerank" if s.startswith("rerank") else "dense" for s in D.systems])
FAMX = np.column_stack([(D.disc[FAMILY == k] > 0).mean(0) for k in ("bm25", "dense", "rerank")])  # [n, 3] hit shares
FT_PATH = f"{cfg['out_dir']}/predictors/ft_ce.npy"
if os.path.exists(FT_PATH):  # optional extra predictor; specs that use it assert it exists via FTL
    _ft = np.load(FT_PATH)
    assert _ft.shape == (len(D.y), 3)
    FTL = np.log(np.clip(_ft, 1e-6, 1)) - np.log(np.clip(_ft[:, :1], 1e-6, 1))
Z_PLAN = cfg["z_plan"]
# Rival methods (witness.rivals) see the judges only as a label distribution: the vote shares of the two judges,
# the analogue of the LLM label distribution both papers read from one LLM.  Their predicted gain is E_pi[u].
PI_VOTE = RV.vote_shares(D.J)  # [n, 3]
F_VOTE = [PI_VOTE[:, 2], 0.5 * PI_VOTE[:, 2] + 0.5 * (PI_VOTE[:, 1] + PI_VOTE[:, 2]), PI_VOTE[:, 1] + PI_VOTE[:, 2]]
FRAME_PER_Q = np.bincount(qi[D.in_union], minlength=n_q)  # human labels needed to score one query for every system


def endpoint_f(F0, F1):
    """Predictors and working variances for the three rows blocks from P(y=2), P(y>=1); u0, u1 binary."""
    Fh = 0.5 * F0 + 0.5 * F1
    v0, v1 = F0 * (1 - F0), F1 * (1 - F1)
    vh = 0.25 * v0 + 0.25 * v1 + 0.5 * F0 * (1 - F1)  # Cov(u0, u1 | x) = F0 (1 - F1) since u0 <= u1
    return [F0, Fh, F1], [v0, vh, v1]


def base_predictors(kind, o, pu_w):
    if kind == "zero":
        z = np.zeros(n)
        return endpoint_f(z, z)
    if kind == "qrels":
        return endpoint_f(o.astype(float), o.astype(float))
    if kind == "judge":  # the usual practice: qrels count as relevant, holes patched by the judges
        k = np.maximum(ans.sum(0), 1)
        f1 = np.where(ans.sum(0) > 0, ((D.J >= 1) & ans).sum(0) / k, D.ce)
        f0 = np.where(ans.sum(0) > 0, ((D.J == 2) & ans).sum(0) / k, D.ce)
        return endpoint_f(np.where(o, 1.0, f0), np.where(o, 1.0, f1))
    assert kind == "pu"
    return endpoint_f(pu_w[:, 2], pu_w[:, 1] + pu_w[:, 2])


def design(F, o, strata, extras):
    """Calibration columns Zc [n, m] and the prior beta0 [m] with Zc beta0 = F (GREG calibration; per-stratum slopes)."""
    cols, b0 = [], []
    U = ~o
    if strata == "global":
        cols, b0 = [F], [1.0]
    else:
        cols += [F * o, o.astype(float)]
        b0 += [1.0, 0.0]
        if strata == "LU":
            cols, b0 = cols + [F * U], b0 + [1.0]
        else:
            assert strata == "LUh"
            for k in range(N_HB):
                cols.append(F * (U & (HB == k)))
                b0.append(1.0)
    for x in extras:
        if x == "ce":
            cols.append(np.where(U, LCE - LCE[U].mean(), 0.0))
            b0.append(0.0)
        elif x == "jpat":
            for lev in JPAT_LEVELS[1:]:
                cols.append((U & (JPAT == lev)).astype(float))
                b0.append(0.0)
        elif x == "fam":
            # provenance calibration: the judges' error moves with which system family
            # retrieved the item, so the family hit shares enter as auxiliary variables, centred within U
            for k in range(FAMX.shape[1]):
                cols.append(np.where(U, FAMX[:, k] - FAMX[U, k].mean(), 0.0))
                b0.append(0.0)
        elif x == "ft":
            # fine-tuned cross-encoder (scripts/data/finetune_ce.py): log-odds of y=1 and y=2 against y=0, centred within U
            for k in (1, 2):
                cols.append(np.where(U, FTL[:, k] - FTL[U, k].mean(), 0.0))
                b0.append(0.0)
        else:
            # per-system top-k hit and discount: the contrast g of every pair is a linear combination of these
            # columns, so the calibrated predictor carries no pair-specific bias (the 'sys' oracle diagnostic)
            assert x == "sys"
            for s in range(len(D.systems)):
                for col in (D.disc[s] > 0, D.disc[s]):
                    cols.append(np.where(U, col - col[U].mean(), 0.0))
                    b0.append(0.0)
    return np.column_stack(cols), np.array(b0)


def query_var(contrib):
    per_q = np.zeros((contrib.shape[0], n_q))
    np.add.at(per_q.T, qi, contrib.T)
    return per_q.var(1, ddof=1) / n_q


def certify(est, var, truth):
    """est/var/truth: lists over the three row blocks (rho=0, 1/2, 1), each [R]; certification uses the endpoints."""
    se = [np.sqrt(np.maximum(v[:P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:P] / se[0]), norm.sf(est[2][:P] / se[2]))  # H+: Delta(rho) <= 0 somewhere
    p_minus = np.maximum(norm.cdf(est[0][:P] / se[0]), norm.cdf(est[2][:P] / se[2]))
    t0, t1 = truth[0][:P], truth[2][:P]
    plus_true = np.minimum(t0, t1) <= 0  # H+ true: "A > B for every rho" is false
    minus_true = np.maximum(t0, t1) >= 0
    decidable = ~plus_true | ~minus_true
    out = {}
    for name, dep in (("bh", False), ("by", True)):
        rej = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"], dependent=dep)
        rp, rm = rej[:P], rej[P:]
        wrong = (rp & plus_true) | (rm & minus_true)
        n_rej = int(rp.sum() + rm.sum())
        out[f"cert_{name}"] = (rp | rm).mean()
        out[f"cert_dec_{name}"] = ((rp & ~plus_true) | (rm & ~minus_true)).sum() / max(decidable.sum(), 1)
        out[f"fdp_{name}"] = wrong.sum() / max(n_rej, 1)
        out[f"nwrong_{name}"] = int(wrong.sum())
    e, v, t = est[1], var[1], truth[1]
    z = (e - t) / np.sqrt(np.maximum(v, 1e-30))
    out.update(cover_pairs=(np.abs(z[:P]) <= 1.96).mean(), cover_sys=(np.abs(z[P:]) <= 1.96).mean(),
               z_mean=z[:P].mean(), z_sd=z[:P].std(), rmse=np.sqrt(np.mean((e[:P] - t[:P]) ** 2)),
               tau=kendalltau(e[P:], t[P:])[0], n_decidable=int(decidable.sum()))
    return out


def passive(o, kind):
    F, _ = base_predictors(kind, o, None) if kind != "official" else ([o.astype(float)] * 3, None)
    est = [G @ f / n_q for f in F]
    return est, [query_var(G * f) for f in F]




def cells(o):
    """Calibration cells for the working variance: qrels stratum x joint judge state."""
    return (JPAT * 2 + o.astype(int)).astype(int)


def working_m2(b, F, V, H, o, lab, pi_lab):
    """E[(u - H_r)^2 | x] = s2(x) + (F - H_r)^2 with s2 = Var(u | x) under the working model.

    The base predictor's own variance F(1-F) is overconfident where the judges agree (it is 0 at F in {0, 1}), so s2
    is calibrated per cell (qrels stratum x judge state) from the labels bought so far: the 1/pi-weighted mean of
    (u - F)^2, shrunk toward max(V, v_floor) with `cell_prior` pseudo-items.  Before any label s2 = max(V, v_floor)."""
    s2 = np.maximum(V, cfg["v_floor"])
    if len(lab) > 0:
        c = cells(o)
        wl = 1.0 / pi_lab
        num = np.bincount(c[lab], weights=wl * (U_END[b][lab] - F[lab]) ** 2, minlength=c.max() + 1)
        den = np.bincount(c[lab], weights=wl, minlength=c.max() + 1)
        cnt = np.bincount(c[lab], minlength=c.max() + 1)
        prior = np.bincount(c, weights=s2, minlength=c.max() + 1) / np.maximum(np.bincount(c, minlength=c.max() + 1), 1)
        k = cfg["cell_prior"]
        cal = (cnt * np.where(den > 0, num / np.maximum(den, 1e-30), 0.0) + k * prior) / (cnt + k)
        s2 = np.maximum(cal[c], cfg["v_floor"])
    return s2[None, :] + (F[None, :] - H) ** 2


def plan_weights(c_rows, w_rows, Gsq, pair_mask):
    """c_d = sum over endpoint blocks and pair rows of w * g^2 * m2."""
    return sum((w[:, None] * Gsq[pair_mask] * m2[pair_mask]).sum(0) for w, m2 in zip(w_rows, c_rows))


def run(o, budget, spec, pu_w, rng):
    fracs = spec["fracs"]  # share of the budget spent in each stage
    T = len(fracs)
    assert np.isclose(sum(fracs), 1.0)
    F, V = base_predictors(spec["f"], o, pu_w)
    frame0 = D.in_union.copy() if spec["sample_L"] else D.in_union & ~o
    plug = D.in_union & o & ~frame0
    designs = [design(F[b], o, spec["strata"], spec["extras"]) for b in range(3)]
    assert spec["lam"] in ("none", "prev", "final", "xfit", "xstage")
    assert spec["lam"] != "xfit" or T == 1
    assert not spec["adapt"] or (spec["f"] == "pu" and spec["lam"] in ("prev", "xstage") and spec["sample_L"])
    assert not spec["adapt"] or spec["pu_z"] == "main"  # the refit uses Z_PU
    labelled = np.zeros(n, bool)
    pi_buy = np.zeros(n)
    acc = [dict(num=np.zeros(R), den=np.zeros(R), vnum=np.zeros(R)) for _ in range(3)]
    stages = []  # (labelled before the stage, pi, xi, weight) for the final refit
    Gsq = G ** 2
    pair_mask = np.arange(R) < P
    for t in range(T):
        lab = np.flatnonzero(labelled)
        if spec["adapt"] and t > 0:
            # adaptive witness (M1): refit the PU model with the human labels bought so far (their Y fixed in the EM),
            # then rebuild the predictor and calibration columns.  Both depend on earlier stages only, so the stage's
            # estimator stays exactly conditionally unbiased (stage_xfit_estimate).  With unit label weights the
            # refit is dominated by the label-free O likelihood; adapt_w weights each
            # label by 1/pi (design-weighted composite likelihood).  Either costs efficiency only, never validity.
            _, _, _, pu_w = W.pu_fuse(o, Z_PU, X_PU, lab, D.y[lab], 0.5, w0=pu_w, clamp_out=False,
                                      lab_w=(1.0 / pi_buy[lab]) if spec["adapt_w"] else None)
            F, V = base_predictors(spec["f"], o, pu_w)
            designs = [design(F[b], o, spec["strata"], spec["extras"]) for b in range(3)]
        # planning always uses the GREG fit on the labels bought so far (it sets pi only)
        betas = []
        for b in range(3):
            Zc, b0 = designs[b]
            if spec["lam"] != "none" and len(lab) > 0:
                betas.append(W.greg_fit(G, Zc, U_END[b], lab, pi_buy[lab], b0, cfg["ridge"]))
            else:
                betas.append(np.tile(b0, (R, 1)))
        H = [betas[b] @ designs[b][0].T for b in range(3)]  # [R, n] per block
        fr = frame0 & ~labelled
        n_t = budget * fracs[t]
        # working residual second moments E[(u - H)^2 | x]; they set pi only, never validity (design unbiasedness holds for any working model)
        m2 = [working_m2(b, F[b], V[b], H[b], o, lab, pi_buy[lab]) for b in range(3)]
        pi_min = cfg["pi_min_frac"] * n_t / fr.sum()
        if spec["alloc"] == "uniform":
            wd = fr.astype(float)
        else:
            w_rows = [np.ones(P), np.ones(P)]
            for _ in range(cfg["plan_iters"] if spec["alloc"] == "decision" else 1):
                cd = plan_weights([m2[0], m2[2]], w_rows, Gsq, pair_mask)
                if spec["alloc"] == "single":
                    break
                # project with the designed remaining budget, not the realised count: Poisson draws can overshoot,
                # and the plan must not depend on that noise (nor shrink below the floor it implies)
                n_rem = min(budget * sum(fracs[t:]), fr.sum() * 0.999)
                pi_rem = np.zeros(n)
                pi_rem[fr] = W.sampling_probs_clip(np.sqrt(cd[fr]), n_rem, cfg["pi_min_frac"] * n_rem / fr.sum())
                newv = []
                for bi, b in enumerate((0, 2)):
                    v_rem = W.predicted_var(G, pi_rem, m2[b], n_q)[:P]
                    if acc[b]["den"].any():
                        e_now = acc[b]["num"][:P] / acc[b]["den"][:P]
                        v_now = acc[b]["vnum"][:P] / acc[b]["den"][:P] ** 2
                        v_fin = 1.0 / (1.0 / np.maximum(v_now, 1e-30) + 1.0 / np.maximum(v_rem, 1e-30))
                    else:
                        e_now = (G[:P] @ (labelled * U_END[b]) + (G[:P] * H[b][:P] * ~labelled).sum(1)) / n_q
                        v_now, v_fin = v_rem, v_rem
                    newv.append(W.cert_weights(e_now, v_now, np.maximum(v_fin, 1e-30), Z_PLAN))
                (w0, pc0), (w1, pc1) = newv
                w_rows = [w0 * pc1, w1 * pc0]  # a pair counts only if both endpoints certify
            wd = np.sqrt(np.maximum(cd, 0.0)) * fr
        pi = np.zeros(n)
        pi[fr] = W.sampling_probs_clip(wd[fr], n_t, pi_min)
        xi = fr & (rng.random(n) < pi)
        # stage weights fixed before the draw (martingale-difference combination): stable = 1/sd (Hadad et al. 2021), invvar = 1/var,
        # equal = the stage's budget share (deterministic)
        om = [{"stable": 1.0 / np.sqrt(np.maximum(W.predicted_var(G, pi, m2[b], n_q), 1e-16)),
               "invvar": 1.0 / np.maximum(W.predicted_var(G, pi, m2[b], n_q), 1e-16),
               "equal": np.full(R, fracs[t])}[spec["comb"]] for b in range(3)]
        stages.append((labelled.copy(), pi, xi, om))
        if spec["lam"] == "xfit":
            # frame-partition cross-fitting (witness.core.xfit_estimate): exactly design-unbiased single stage
            fold = rng.integers(0, cfg["xfit_folds"], n)
            for b in range(3):
                est_t, var_t = W.xfit_estimate(G, designs[b][0], designs[b][1], U_END[b], xi, pi, fold,
                                               cfg["xfit_folds"], cfg["ridge"], n_q, plug)
                acc[b]["num"] += om[b] * est_t
                acc[b]["den"] += om[b]
                acc[b]["vnum"] += om[b] ** 2 * var_t
        elif spec["lam"] == "xstage":
            # within-stage cross-fitting on top of all earlier labels: exactly conditionally unbiased per stage
            fold = rng.integers(0, cfg["xfit_folds"], n)
            for b in range(3):
                est_t, var_t = W.stage_xfit_estimate(G, labelled * U_END[b], designs[b][0], designs[b][1], U_END[b], lab,
                                                     pi_buy[lab], xi, pi, fold, cfg["xfit_folds"], cfg["ridge"], n_q)
                acc[b]["num"] += om[b] * est_t
                acc[b]["den"] += om[b]
                acc[b]["vnum"] += om[b] ** 2 * var_t
        elif spec["lam"] != "final":  # the stage's predictor is fixed before its draw: exactly unbiased
            for b in range(3):
                est_t, var_t = W.stage_estimate(G, labelled * U_END[b], plug, H[b], U_END[b], xi, pi, n_q)
                acc[b]["num"] += om[b] * est_t
                acc[b]["den"] += om[b]
                acc[b]["vnum"] += om[b] ** 2 * var_t
        pi_buy[xi] = pi[xi]
        labelled |= xi
    if spec["lam"] == "final":
        # one GREG fit on every label, applied to every stage: the PPI++ in-sample choice, O(1/n) bias
        lab = np.flatnonzero(labelled)
        H = [W.greg_fit(G, designs[b][0], U_END[b], lab, pi_buy[lab], designs[b][1], cfg["ridge"]) @ designs[b][0].T
             for b in range(3)]
        for known, pi, xi, om in stages:
            for b in range(3):
                est_t, var_t = W.stage_estimate(G, known * U_END[b], plug, H[b], U_END[b], xi, pi, n_q)
                acc[b]["num"] += om[b] * est_t
                acc[b]["den"] += om[b]
                acc[b]["vnum"] += om[b] ** 2 * var_t
    est = [a["num"] / a["den"] for a in acc]
    var = [a["vnum"] / a["den"] ** 2 for a in acc]
    return est, var, int(labelled.sum())


def run_rival(o, budget, spec, rng):
    """Prior LLM-assisted IR evaluation methods (witness.rivals), same frame, qrels draw, truth and certification.

    oost_ppi (Oosterhuis et al. KDD'24, PPI): the human budget buys every frame item of n = round(budget / mean frame
      items per query) queries drawn uniformly without replacement (official datasets.py::sample); the interval of
      every pair row is their PPI interval for the per-query difference metric.  f = "vote": E_pi[u] under the judges'
      vote shares (their predictor, no qrels); f = "judge": our ppi_tuned predictor (qrels count as relevant, holes
      patched by the judges), an adaptation that isolates their query-cluster design.
    lara (Takehi et al. SIGIR'25): LARA(n=1) completes the frame's labels (human where bought, calibrated argmax
      elsewhere; qrels unused, as in the paper); the completed labels are then used the way a test collection is
      used, Delta = plug-in and Var from the paired per-query differences (as judge_plugin)."""
    rc = cfg["rivals"]
    if spec["kind"] == "oost_ppi":
        F = F_VOTE if spec["f"] == "vote" else base_predictors(spec["f"], o, None)[0]
        n_lab_q = max(2, int(round(budget / FRAME_PER_Q.mean())))
        lab_q = rng.choice(n_q, n_lab_q, replace=False)
        est, var = zip(*[RV.oosterhuis_ppi(G, F[b], U_END[b], qi, n_q, lab_q) for b in range(3)])
        return list(est), list(var), int(FRAME_PER_Q[lab_q].sum())
    assert spec["kind"] == "lara" and spec["f"] == "vote"
    fr = np.flatnonzero(D.in_union)
    lc = rc["lara"]
    y_hat_fr, lab_fr = RV.lara(PI_VOTE[fr], D.y[fr], budget, rng, spec["mode"], lc["lr"], lc["batch"], lc["epochs"],
                               lc["transition_w"], lc["diff_w"], lc["lbfgs_iter"])
    y_hat = np.zeros(n, int)  # items outside the frame have G = 0 in every row
    y_hat[fr] = y_hat_fr
    U_HAT = [(y_hat == 2).astype(float), 0.5 * (y_hat == 2) + 0.5 * (y_hat >= 1), (y_hat >= 1).astype(float)]
    return [G @ f / n_q for f in U_HAT], [query_var(G * f) for f in U_HAT], int(lab_fr.sum())


# Main method: single Poisson stage planned from the zero-label PU predictor, one
# GREG fit on all labels (PPI++-style). Two-stage variants are ablations.
BASE = dict(f="pu", strata="LUh", extras=["ce", "jpat"], alloc="decision", fracs=[1.0], lam="final",
            comb="stable", sample_L=True, adapt=False, adapt_w=False, pu_z="main")
ONE = dict(fracs=[1.0], lam="final", extras=[], comb="stable", sample_L=True, adapt=False, adapt_w=False, pu_z="main")
SPECS = {
    "classical": dict(ONE, f="zero", strata="global", alloc="uniform", lam="none"),
    "qrels_rectified": dict(ONE, f="qrels", strata="LU", alloc="uniform"),
    "ppi": dict(ONE, f="judge", strata="global", alloc="uniform", lam="none"),
    "ppi_tuned": dict(ONE, f="judge", strata="global", alloc="uniform"),
    "active": dict(ONE, f="judge", strata="global", alloc="single"),
    "witness": BASE,
    "witness_fam": dict(BASE, extras=["ce", "jpat", "fam"]),
    "witness_ft": dict(BASE, extras=["ce", "jpat", "ft"]),
    "abl_f_judge": dict(BASE, f="judge"),
    "abl_no_strata": dict(BASE, strata="global", extras=[]),
    "abl_no_extras": dict(BASE, extras=[]),
    "abl_alloc_single": dict(BASE, alloc="single"),
    "abl_alloc_uniform": dict(BASE, alloc="uniform"),
    "abl_L_not_sampled": dict(BASE, sample_L=False),
    "abl_two_stage_final": dict(BASE, fracs=cfg["fracs"]),
    "abl_two_stage_prev": dict(BASE, fracs=cfg["fracs"], lam="prev"),
    "abl_two_stage_invvar": dict(BASE, fracs=cfg["fracs"], comb="invvar"),
    "witness_xfit": dict(BASE, lam="xfit"),
    # zero-label PU with the exclusion restriction (z = judge states + CE), EM run cfg["pu_ex_iter"] iterations
    "witness_xfit_ex": dict(BASE, lam="xfit", pu_z="ex"),
    "witness_fam_xfit": dict(BASE, extras=["ce", "jpat", "fam"], lam="xfit"),
    "ppi_tuned_xfit": dict(ONE, f="judge", strata="global", alloc="uniform", lam="xfit"),
    "active_xfit": dict(ONE, f="judge", strata="global", alloc="single", lam="xfit"),
    # module ablations under cross-fitting: in-sample GREG optimism grows with the number of calibration columns
    # (O(m / n_eff) for an in-sample GREG fit), so between-method comparisons are made xfit vs xfit
    "abl_f_judge_xfit": dict(BASE, f="judge", lam="xfit"),
    "abl_no_extras_xfit": dict(BASE, extras=[], lam="xfit"),
    "abl_no_strata_xfit": dict(BASE, strata="global", extras=[], lam="xfit"),
    "abl_alloc_single_xfit": dict(BASE, alloc="single", lam="xfit"),
    "abl_alloc_uniform_xfit": dict(BASE, alloc="uniform", lam="xfit"),
    # adaptive witness: stage 1 plans with the zero-label PU model; stage 2 refits PU on stage-1 human labels and
    # re-plans; every stage cross-fitted (exactly unbiased).  The non-adaptive two-stage twin isolates the refit.
    "witness_adapt": dict(BASE, extras=["ce", "jpat", "fam"], fracs=cfg["fracs"], lam="xstage", adapt=True),
    "witness_2stage_xs": dict(BASE, extras=["ce", "jpat", "fam"], fracs=cfg["fracs"], lam="xstage"),
    # adaptive witness with the 1/pi-weighted refit (design-weighted composite likelihood)
    "witness_adapt_w": dict(BASE, extras=["ce", "jpat", "fam"], fracs=cfg["fracs"], lam="xstage", adapt=True,
                            adapt_w=True),
    # prior methods (run_rival)
    "oost_ppi": dict(kind="oost_ppi", f="vote"),
    "oost_ppi_qrels": dict(kind="oost_ppi", f="judge"),
    "lara": dict(kind="lara", f="vote", mode="paper"),
    "lara_code": dict(kind="lara", f="vote", mode="code"),
}
SPECS = {m: SPECS[m] for m in cfg["methods"]}
rows = []
for rep in range(REP_LO, REP_HI):
    t0 = time.time()
    rng = np.random.default_rng([cfg["seed"], rep])
    o, e_true = MECH.draw(rng)
    pu_w = pu_w_ex = None
    if any(sp["f"] == "pu" and sp["pu_z"] == "ex" for sp in SPECS.values() if "kind" not in sp):
        pu_w_ex = W.pu_fuse(o, Z_PU_EX, X_PU, np.array([], int), np.array([], int), 0.5, n_iter=cfg["pu_ex_iter"])[3]
        print(f"rep {rep}: pu_ex ({cfg['pu_ex_iter']} it) mean P(y>=1) on U frame "
              f"{(pu_w_ex[:, 1] + pu_w_ex[:, 2])[D.in_union & ~o].mean():.4f} vs truth {D.u1[D.in_union & ~o].mean():.4f}")
    if any(sp["f"] == "pu" for sp in SPECS.values()):
        _, _, e_hat, pu_w = W.pu_fuse(o, Z_PU, X_PU, np.array([], int), np.array([], int), 0.5)
        rel = D.y >= 1
        e_hat_d = np.where(D.y == 2, e_hat[:, 2], e_hat[:, 1])
        f1 = pu_w[:, 1] + pu_w[:, 2]
        print(f"rep {rep}: pu (30 it) mean P(y>=1) on U frame {f1[D.in_union & ~o].mean():.4f}")
        print(f"rep {rep}: qrels recall={o[rel].mean():.3f} qrels error share={(D.y[o] == 0).mean():.3f} "
              f"pu: max|e_hat-e| on relevant={np.abs(e_hat_d - e_true)[rel].max():.3f} "
              f"corr(P(y>=1), u1) on U={np.corrcoef(f1[~o], D.u1[~o])[0, 1]:.3f}", flush=True)
    for kind in ("official", "judge"):
        est, var = passive(o, kind)
        rows.append(dict(rep=rep, method="judge_plugin" if kind == "judge" else "official", budget=0, n_labels=0,
                         **certify(est, var, TRUTH)))
    for b in cfg["budgets"]:
        for m, spec in SPECS.items():
            est, var, nl = run_rival(o, b, spec, rng) if "kind" in spec else run(o, b, spec, pu_w_ex if spec["pu_z"] == "ex" else pu_w, rng)
            rows.append(dict(rep=rep, method=m, budget=b, n_labels=nl, **certify(est, var, TRUTH)))
    print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{cfg['out_dir']}/eval_{SUFFIX}_raw.csv", index=False)
tab = df.groupby(["method", "budget"]).mean(numeric_only=True).drop(columns="rep")
pd.set_option("display.width", 250)
print(tab.round(4).to_string())
tab.to_csv(f"{cfg['out_dir']}/eval_{SUFFIX}.csv")
print("DONE")
