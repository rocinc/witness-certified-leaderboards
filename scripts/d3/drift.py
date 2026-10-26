"""D3 annotator drift between the two human-label batches: is it there, is it the annotator or the
items, and could it change a certified conclusion?

Batch t labels a Poisson sample of its frame with known pi_t (batch 2's frame = items batch 1 did not label); within a
batch the 1/pi_t-weighted relevance rate of a cell is a Hajek estimate of that cell's rate in the frame.
[1] Cell comparison: cells = qrels stratum x judge joint state.  Per cell, rate_1 - rate_2 with a within-batch bootstrap
    (labels resampled with replacement inside each batch, weights kept: the effective sample size of unequal weights
    is respected, unlike a nominal-n z test) and the Wald statistic sum over cells with >= MIN_N labels
    in both batches of (diff / boot sd)^2, df = number of such cells.
[2] Items or annotator?  Batch 2 was planned toward other system pairs, so within a cell it may hold different items.
    Weighted logistic regression of 1[y >= 1] on cell dummies + item covariates (bge_m3 cosine, family hit shares,
    n_top) + 1[batch 2], combined design weights; the batch coefficient with and without covariates.  A coefficient
    that shrinks to 0 with covariates points to item composition; one that stays points to the annotator.
[3] Grade use: the 1-vs-2 split among relevant items per batch (the gain at rho = 0 uses only grade 2), weighted.
[4] Tipping point: every batch-2 label is shifted in gain by delta (u -> clip(u + delta, 0, 1) is not linear, so the
    shift is applied to the gain directly: u_b2 + delta, which is what a systematic leniency of size delta does to the
    estimator in expectation); for delta on a grid, re-run the multi-batch design estimator (design_fam) and report
    the certified count and which pairs change.  The smallest |delta| that changes the certified set, against the
    observed drift, says whether drift can matter.
[5] The observed drift in gain units (cell- and covariate-adjusted batch coefficient per rho block) and the certified
    set with either batch aligned to the other's standard.
usage: python scripts/d3/drift.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
D = WD.load_frame(cfg)
b1, b2 = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sys.argv[2:4])]
B_REPS, MIN_N = 2000, 15
rng = np.random.default_rng(0)
lab = [b1.items, b2.items]
y = [b1.y, b2.y]
w = [1.0 / b1.pi[b1.items], 1.0 / b2.pi[b2.items]]
cell = D.state * 2 + D.o.astype(int)  # [N]
print(f"labels: batch 1 {len(lab[0])}, batch 2 {len(lab[1])}; Kish effective n: "
      f"{w[0].sum() ** 2 / (w[0] ** 2).sum():.0f}, {w[1].sum() ** 2 / (w[1] ** 2).sum():.0f}")


def boot_idx(t):
    return rng.integers(0, len(lab[t]), len(lab[t]))


# ---- [1] cell comparison with design-respecting bootstrap
def cell_rates(idx0, idx1):
    out = []
    for t, idx in ((0, idx0), (1, idx1)):
        c, r, ww = cell[lab[t][idx]], (y[t][idx] >= 1).astype(float), w[t][idx]
        num = np.bincount(c, weights=ww * r, minlength=cell.max() + 1)
        den = np.bincount(c, weights=ww, minlength=cell.max() + 1)
        out.append(np.where(den > 0, num / np.maximum(den, 1e-30), np.nan))
    return out


full = [np.arange(len(lab[0])), np.arange(len(lab[1]))]
r1, r2 = cell_rates(*full)
cnt = [np.bincount(cell[lab[t]], minlength=cell.max() + 1) for t in (0, 1)]
bs = np.array([np.subtract(*cell_rates(boot_idx(0), boot_idx(1))) for _ in range(B_REPS)])  # [B, n_cells]
comp = np.flatnonzero((cnt[0] >= MIN_N) & (cnt[1] >= MIN_N))
sd = np.nanstd(bs[:, comp], axis=0)
zc = (r1 - r2)[comp] / sd
print(f"[1] cells with >= {MIN_N} labels in both batches: {len(comp)}")
for k, c in enumerate(comp):
    st, inq = divmod(int(c), 2)
    print(f"    cell state={st} {'L' if inq else 'U'}: n {cnt[0][c]}/{cnt[1][c]}  rate b1 {r1[c]:.3f}  b2 {r2[c]:.3f}  "
          f"diff {r1[c] - r2[c]:+.3f}  boot sd {sd[k]:.3f}  z {zc[k]:+.2f}")
stat = float((zc ** 2).sum())
print(f"[1] Wald sum z^2 = {stat:.2f}, df {len(comp)}, p = {chi2.sf(stat, len(comp)):.3f}")

# ---- [2] items or annotator: batch coefficient with and without item covariates
q_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/q.npy").astype(np.float32)
p_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/p.npy", mmap_mode="r")
ids = json.load(open(f"{cfg['duretrieval_dir']}/systems/bge_m3/ids.json"))
corpus_ids = pd.read_parquet(glob.glob(f"{cfg['duretrieval_dir']}/hf/DuRetrieval/data/corpus-*.parquet")[0])["id"]
assert ids["pids"] == list(corpus_ids)
items = np.concatenate(lab)
cos = np.einsum("nd,nd->n", q_emb[D.qi[items]], np.asarray(p_emb[D.frame.di.to_numpy()[items]], np.float32))
lexm = np.array([s.startswith("bm25") for s in D.systems])
rrm = np.array([s.startswith("rerank") for s in D.systems])
fam = np.column_stack([(D.disc[m][:, items] > 0).mean(0) for m in (lexm, ~lexm & ~rrm, rrm)])
ntop = (D.disc[:, items] > 0).sum(0) / len(D.systems)
yy = (np.concatenate(y) >= 1).astype(int)
bat = np.r_[np.zeros(len(lab[0])), np.ones(len(lab[1]))]
# combined design weights: each batch normalised to its own HT frame size (the two-batch mixture of scripts/d3/pu_real_qrels.py)
a = len(lab[0]) / len(items)
ww = np.concatenate([a * w[0] + (1 - a), (1 - a) * w[1]])
levels = np.unique(cell[items])
cd = (cell[items][:, None] == levels[None, 1:]).astype(float)
covs = np.column_stack([(cos - cos.mean()) / cos.std(), fam, ntop])


def batch_coef(Xm, idx):
    m = LogisticRegression(C=1e4, max_iter=2000).fit(Xm[idx], yy[idx], sample_weight=ww[idx])
    return m.coef_[0][-1]


groups = [np.flatnonzero(bat == 0), np.flatnonzero(bat == 1)]
for name, Xm in (("cells only", np.column_stack([cd, bat])), ("cells + item covariates", np.column_stack([cd, covs, bat]))):
    pt = batch_coef(Xm, np.arange(len(items)))
    bb = np.array([batch_coef(Xm, np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]))
                   for _ in range(400)])
    lo, hi = np.quantile(bb, [0.025, 0.975])
    print(f"[2] logit coefficient of batch 2 ({name}): {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
# covariate balance between batches (weighted means), to see what the planner changed
for k, nm in enumerate(["cos (std)", "lex share", "dense share", "rerank share", "n_top/S"]):
    print(f"[2] weighted mean {nm:12s} b1 {np.average(covs[bat == 0, k], weights=ww[bat == 0]):+.3f}  "
          f"b2 {np.average(covs[bat == 1, k], weights=ww[bat == 1]):+.3f}")

# ---- [3] grade use among relevant items
for t in (0, 1):
    rel = y[t] >= 1
    print(f"[3] batch {t + 1}: weighted P(y=2 | y>=1) = {np.average(y[t][rel] == 2, weights=w[t][rel]):.3f} "
          f"(n relevant {rel.sum()})")
g2 = []
for _ in range(B_REPS):
    v = []
    for t in (0, 1):
        i = boot_idx(t)
        rel = y[t][i] >= 1
        v.append(np.average(y[t][i][rel] == 2, weights=w[t][i][rel]))
    g2.append(v[0] - v[1])
print(f"[3] difference b1 - b2: 95% [{np.quantile(g2, .025):+.3f}, {np.quantile(g2, .975):+.3f}]")


# ---- [4] tipping point of the certified set under a systematic batch-2 gain shift
def certify(est, var):
    se = [np.sqrt(np.maximum(v[:D.P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:D.P] / se[0]), norm.sf(est[2][:D.P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:D.P] / se[0]), norm.cdf(est[2][:D.P] / se[2]))
    rej = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"])
    return rej[:D.P], rej[D.P:]


orig_gains = WD.gains


def shifted_gains(delta, stage_y):
    def g(yv):
        out = orig_gains(yv)
        return [u + delta for u in out] if yv is stage_y else out
    return g


runs = []
for delta in (-0.20, -0.15, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.15, 0.20):
    WD.gains = shifted_gains(delta, b2.y)  # design_estimate reads WD.gains for every batch's labels
    est, var = WD.design_estimate(D, [b1, b2], ["fam"], cfg["ridge"], D.F)
    WD.gains = orig_gains
    cp, cm = certify(est, var)
    print(f"[4] delta {delta:+.3f}: certified {int(cp.sum() + cm.sum())}/{D.P}", flush=True)
    runs.append((delta, cp, cm))
base = next((cp, cm) for d_, cp, cm in runs if d_ == 0.0)
for delta, cp, cm in runs:
    ch = [f"{D.systems[a_]} vs {D.systems[b_]}" for p, (a_, b_) in enumerate(D.pairs)
          if (cp[p], cm[p]) != (base[0][p], base[1][p])]
    print(f"[4] delta {delta:+.3f}: pairs whose certification differs from delta=0: {len(ch)} {ch}")
# ---- [5] the observed drift in gain units, and the certified set with the two batches aligned to each other
# Weighted linear model gain_rho ~ cell dummies + item covariates + 1[batch 2] per block: the batch coefficient is the
# cell- and covariate-adjusted mean gain difference.  Aligning batch 2 to batch 1 subtracts it from batch 2's gains;
# aligning batch 1 to batch 2 adds it to batch 1's.  Which annotator standard is right is not identified without
# anchor items labelled under both standards.
from sklearn.linear_model import LinearRegression  # noqa: E402

gall = WD.gains(np.concatenate(y))
coef = []
for b in range(3):
    Xm = np.column_stack([cd, covs, bat])
    m = LinearRegression().fit(Xm, gall[b], sample_weight=ww)
    bb = []
    for _ in range(400):
        i = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        bb.append(LinearRegression().fit(Xm[i], gall[b][i], sample_weight=ww[i]).coef_[-1])
    coef.append(m.coef_[-1])
    print(f"[5] block rho={(0, 0.5, 1)[b]}: adjusted gain difference b2 - b1 = {m.coef_[-1]:+.4f} "
          f"[{np.quantile(bb, .025):+.4f}, {np.quantile(bb, .975):+.4f}]")


def aligned_gains(shift_y, sign):
    def g(yv):
        out = orig_gains(yv)
        return [u + sign * coef[b] for b, u in enumerate(out)] if yv is shift_y else out
    return g


for name, sy, sg in (("batch 2 aligned to batch 1", b2.y, -1.0), ("batch 1 aligned to batch 2", b1.y, +1.0)):
    WD.gains = aligned_gains(sy, sg)
    est, var = WD.design_estimate(D, [b1, b2], ["fam"], cfg["ridge"], D.F)
    WD.gains = orig_gains
    cp, cm = certify(est, var)
    ch = [f"{D.systems[a_]} vs {D.systems[b_]} ({'+' if base[0][p] else '-' if base[1][p] else '?'}->"
          f"{'+' if cp[p] else '-' if cm[p] else '?'})" for p, (a_, b_) in enumerate(D.pairs)
          if (cp[p], cm[p]) != (base[0][p], base[1][p])]
    print(f"[5] {name}: certified {int(cp.sum() + cm.sum())}/{D.P}; changed {ch}")
    for p, (a_, b_) in enumerate(D.pairs):
        if D.systems[a_].startswith("bm25") and D.systems[b_].startswith("bm25"):
            print(f"    {D.systems[a_]} vs {D.systems[b_]}: D(.5) {est[1][p]:+.4f} se {np.sqrt(var[1][p]):.4f}  "
                  f"D(0) {est[0][p]:+.4f} D(1) {est[2][p]:+.4f}")
print("DONE")
