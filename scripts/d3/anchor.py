"""D3 anchor batch: identify the annotator-standard drift between batches 1 and 2 and re-certify under one standard.

The anchor sheet (scripts/d3/make_anchor.py) re-labels a simple random sample of n_per labelled items from each
batch, blinded and shuffled together, under one current standard A.  For an anchor item of batch t: y_orig (label under
batch t's standard), y_A (label under A).  Within batch t the anchor is an SRS of the labelled items, which are themselves a
Poisson sample of batch t's frame, so 1/pi_t weights estimate frame-level quantities of batch t.
[1] Agreement: exact and binary agreement of y_A with y_orig per batch, the 3x3 transition tables, and the repeat
    agreement inside the anchor sheet.
[2] Offsets: d_t[b] = weighted mean of u_b(y_A) - u_b(y_orig) over batch t's anchor items, per block b (rho = 0, 1/2, 1),
    with bootstrap CIs; d_2 - d_1 is the drift between the two batch standards measured under one standard (compare
    with the indirect estimate from scripts/d3/drift.py, batch 2 lower by .060 at rho = 1/2).
[3] Re-certification under standard A by regression calibration: each batch's gains replaced by
    T_t[b][k] = weighted mean of u_b(y_A) among batch t's anchor items with y_orig = k (non-differential
    misclassification: given y_orig and the batch, y_A does not depend on the item).  The design-based estimator
    (design_fam, as the main D3 analysis) is re-run; uncertainty of T is propagated by a bootstrap over anchor items
    (per pair: share of bootstrap replicates in which it is certified in the same direction).
[4] Proper multiple imputation of the standard-A labels (Rubin's rules); its estimate, variance and BH certification are the
    final D3 result and are written to <out_dir>/d3_results_anchor.json (entry design_fam_anchor_mi, schema of d3_results.json).
usage: python scripts/d3/anchor.py configs/d3_duretrieval.yaml <anchor_annotated.xlsx> <annotated_1.xlsx> <annotated_2.xlsx>
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sys.argv[3:5])]
key = json.load(open(f"{cfg['out_dir']}/anchor_key.json", encoding="utf8"))
lab = WD.sheet_labels(sys.argv[2])
assert len(lab) == key["n_rows"]
B_REPS = 2000
rng = np.random.default_rng(0)

# one record per unique anchor item (first occurrence); repeats only for the repeat agreement
first, rep_agree = {}, []
for k in key["rows"]:
    it = D.pos[(k["qi"], k["di"])]
    if k["repeat"]:
        rep_agree.append(first[it]["y_A"] == lab[k["row"]])
        continue
    first[it] = dict(batch=k["batch"], y_orig=k["y_orig"], y_A=lab[k["row"]], w=1.0 / k["pi"])
recs = list(first.values())
assert len(recs) == key["n_unique"]
bt_ = np.array([r["batch"] for r in recs])
yo = np.array([r["y_orig"] for r in recs])
ya = np.array([r["y_A"] for r in recs])
wt = np.array([r["w"] for r in recs])
# the anchor labels agree with the stored batch labels for these items (key integrity)
for r_, it in zip(recs, first):
    bt = batches[r_["batch"] - 1]
    assert bt.y[np.searchsorted(bt.items, it)] == r_["y_orig"]
print(f"anchor: {len(recs)} unique items (batch 1 {np.sum(bt_ == 1)}, batch 2 {np.sum(bt_ == 2)}); "
      f"repeat agreement {np.mean(rep_agree):.3f} over {len(rep_agree)}")

# ---- [1] agreement and transition tables
for t in (1, 2):
    m = bt_ == t
    tab = np.zeros((3, 3), int)
    np.add.at(tab, (yo[m], ya[m]), 1)
    print(f"[1] batch {t}: exact agreement {np.average(yo[m] == ya[m], weights=wt[m]):.3f} (weighted), "
          f"binary {np.average((yo[m] >= 1) == (ya[m] >= 1), weights=wt[m]):.3f}; rows y_orig 0/1/2, cols y_A 0/1/2:")
    for k in range(3):
        print(f"      {k}: {tab[k].tolist()}")


# ---- [2] offsets per batch and their difference
def gain_b(y, b):
    return WD.gains(y)[b]


def offsets(idx):
    out = np.zeros((2, 3))
    for t in (1, 2):
        m = idx[bt_[idx] == t]
        for b in range(3):
            out[t - 1, b] = np.average(gain_b(ya[m], b) - gain_b(yo[m], b), weights=wt[m])
    return out


groups = [np.flatnonzero(bt_ == 1), np.flatnonzero(bt_ == 2)]


def boot_index():
    return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


pt = offsets(np.arange(len(recs)))
bs = np.array([offsets(boot_index()) for _ in range(B_REPS)])  # [B, 2, 3]
for b, rho in enumerate((0, 0.5, 1)):
    lo1, hi1 = np.quantile(bs[:, 0, b], [.025, .975])
    lo2, hi2 = np.quantile(bs[:, 1, b], [.025, .975])
    dd = bs[:, 1, b] - bs[:, 0, b]
    print(f"[2] rho={rho}: d_1 = {pt[0, b]:+.4f} [{lo1:+.4f}, {hi1:+.4f}]  d_2 = {pt[1, b]:+.4f} [{lo2:+.4f}, {hi2:+.4f}]  "
          f"d_2 - d_1 = {pt[1, b] - pt[0, b]:+.4f} [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")


# ---- [3] re-certification under standard A (regression calibration)
def transition(idx):
    """T[t][b][k]: mean gain under A among batch-t anchor items with y_orig = k (weighted)."""
    T = np.zeros((2, 3, 3))
    for t in (1, 2):
        for k in range(3):
            m = idx[(bt_[idx] == t) & (yo[idx] == k)]
            assert len(m) > 0, (t, k)
            for b in range(3):
                T[t - 1, b, k] = np.average(gain_b(ya[m], b), weights=wt[m])
    return T


def certify(est, var):
    se = [np.sqrt(np.maximum(v[:D.P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:D.P] / se[0]), norm.sf(est[2][:D.P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:D.P] / se[0]), norm.cdf(est[2][:D.P] / se[2]))
    rej = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"])
    return rej[:D.P], rej[D.P:]


orig_gains = WD.gains
by_id = {id(bt.y): t for t, bt in enumerate(batches)}


def run(T):
    def g(yv):
        t = by_id.get(id(yv))
        if T is None or t is None:
            return orig_gains(yv)
        return [T[t, b][yv] for b in range(3)]
    WD.gains = g  # design_estimate and working_var read WD.gains for each batch's label vector
    est, var = WD.design_estimate(D, batches, ["fam"], cfg["ridge"], D.F)
    WD.gains = orig_gains
    return est, var, certify(est, var)


est0, var0, (c0p, c0m) = run(None)
T_hat = transition(np.arange(len(recs)))
for t in (1, 2):
    print(f"[3] batch {t} calibration T (rows rho 0/.5/1, cols y_orig 0/1/2): "
          + "; ".join("[" + ", ".join(f"{v:.3f}" for v in T_hat[t - 1, b]) + "]" for b in range(3)))
estA, varA, (cAp, cAm) = run(T_hat)
print(f"[3] certified: original labels {int(c0p.sum() + c0m.sum())}/{D.P}; standard A {int(cAp.sum() + cAm.sum())}/{D.P}")
n_boot_cert = 300
agree = np.zeros((D.P, 3))  # share certified +, certified -, open across bootstrap replicates of T
for _ in range(n_boot_cert):
    _, _, (cp, cm) = run(transition(boot_index()))
    agree[:, 0] += cp
    agree[:, 1] += cm
agree /= n_boot_cert
agree[:, 2] = 1 - agree[:, 0] - agree[:, 1]
for p, (a_, b_) in enumerate(D.pairs):
    o = "+" if c0p[p] else "-" if c0m[p] else "?"
    s = "+" if cAp[p] else "-" if cAm[p] else "?"
    print(f"    {D.systems[a_]:14s} vs {D.systems[b_]:14s} original {o}  D(.5) {est0[1][p]:+.4f}  | standard A {s}  "
          f"D(.5) {estA[1][p]:+.4f} se {np.sqrt(varA[1][p]):.4f}  bootstrap P(+) {agree[p, 0]:.2f} P(-) {agree[p, 1]:.2f} "
          f"P(open) {agree[p, 2]:.2f}{'  CHANGED' if o != s else ''}")


# ---- [4] proper multiple imputation of the standard-A labels (Rubin's rules)
# Regression calibration [3] replaces each label by E[u_A | y_orig, batch] and drops the conditional spread of u_A,
# so its design variance is too small.  Here each imputation m draws the anchor sample with replacement (uncertainty
# of the misclassification model), estimates P(y_A = j | y_orig = k, batch t) from it, draws y_A for every labelled item
# of the two batches, and re-runs the design estimator.  Total variance = mean within + (1 + 1/M) between.
def cond_probs(idx):
    Pm = np.zeros((2, 3, 3))
    for t in (1, 2):
        for k in range(3):
            m = idx[(bt_[idx] == t) & (yo[idx] == k)]
            assert len(m) > 0, (t, k)
            Pm[t - 1, k] = np.bincount(ya[m], weights=wt[m], minlength=3) / wt[m].sum()
    return Pm


M = 200
draws = []
for _ in range(M):
    Pm = cond_probs(boot_index())
    imp = {}
    for t, bt in enumerate(batches):
        u = rng.random(len(bt.y))
        cum = np.cumsum(Pm[t][bt.y], axis=1)  # [k_t, 3]
        imp[id(bt.y)] = (cum < u[:, None]).sum(1).clip(0, 2)

    def g(yv, imp=imp):
        return orig_gains(imp[id(yv)]) if id(yv) in imp else orig_gains(yv)
    WD.gains = g
    est, var = WD.design_estimate(D, batches, ["fam"], cfg["ridge"], D.F)
    WD.gains = orig_gains
    draws.append((np.array(est), np.array(var)))
E = np.stack([d[0] for d in draws])  # [M, 3, R]: all R rows (pairs, then systems), as d3_results.json
V = np.stack([d[1] for d in draws])
est_mi = E.mean(0)
var_mi = V.mean(0) + (1 + 1 / M) * E.var(0, ddof=1)
frac_between = ((1 + 1 / M) * E.var(0, ddof=1) / var_mi)[1][:D.P]
cmp_, cmm_ = certify(list(est_mi), list(var_mi))
print(f"[4] multiple imputation (M={M}): certified {int(cmp_.sum() + cmm_.sum())}/{D.P}; between-imputation share of the "
      f"variance at rho=1/2: median {np.median(frac_between):.2f} [{frac_between.min():.2f}, {frac_between.max():.2f}]")
for p, (a_, b_) in enumerate(D.pairs):
    o = "+" if c0p[p] else "-" if c0m[p] else "?"
    s = "+" if cmp_[p] else "-" if cmm_[p] else "?"
    print(f"    {D.systems[a_]:14s} vs {D.systems[b_]:14s} original {o}  | MI {s}  D(.5) {est_mi[1][p]:+.4f} "
          f"se {np.sqrt(var_mi[1][p]):.4f} (regression-calibration se {np.sqrt(varA[1][p]):.4f}){'  CHANGED' if o != s else ''}")
# the final D3 result in the schema of the `methods` entries of d3_results.json (analysis/make_figures.py, Fig. 3)
assert est_mi.shape == var_mi.shape == (3, len(D.G)) and len(cmp_) == D.P
json.dump(dict(methods={"design_fam_anchor_mi": dict(est=est_mi.tolist(), var=var_mi.tolist(),
                                                     cert_plus=cmp_.tolist(), cert_minus=cmm_.tolist())}),
          open(f"{cfg['out_dir']}/d3_results_anchor.json", "w"))
print(f"wrote {cfg['out_dir']}/d3_results_anchor.json")
print("DONE")
