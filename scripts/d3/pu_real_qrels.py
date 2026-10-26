"""D3 real-qrels test of the zero-label PU predictor.

D1/D2 found the PU predictor (witness.core.pu_fuse) to be the largest single module, but only under simulated qrels whose
mechanism depends on the pool share h by construction.  Here o = the real DuRetrieval qrels, judges = the real pro/flash
labels, truth = the human labels of all batches (Poisson draws with known inclusion probabilities).
D3 analogues of scripts/eval/evaluate.py Z_PU / X_PU:
  judge joint state      dummies of the (pro, flash) joint label, as JPAT (no refusals on D3)
  score                  bge_m3 bi-encoder cosine, standardised over the frame (D1/D2: logit of a cross-encoder
                         probability; the D3 reranker scored only bge_m3's top-20, so it does not cover the frame)
  h                      share of the 3 BM25 systems with the item in their top-10 (D1: pool share of the BM25 pool
                         systems at pool_k; the real DuRetrieval pool is unknown, so h is a proxy)
  n_top / S              number of the 7 systems with the item in their top-10, over 7
  Z_PU = [state dummies, score, h, n_top/S], X_PU = [h, n_top/S]; pu_fuse is fitted with zero human labels.
HT weights for pooling the batches (2 batches): batch-1 alone estimates a frame total by sum x/pi1; batch-1 values plus
batch 2 (whose frame is the rest) estimate it by sum_{b1} x + sum_{b2} x/pi2.  The mixture with a = n1/(n1+n2) gives
item weights a/pi1 + (1 - a) (batch 1) and (1 - a)/pi2 (batch 2); ratio statistics (AUC, calibration, regression)
use these weights.  Uncertainty: bootstrap of the labelled items with replacement within each batch, weights kept
(the convention of scripts/d3/findings.py).
[1] PU fit diagnostics.  [2] On U (not in qrels): weighted AUC, binned calibration (10 equal-width bins, ECE),
Brier and calibration-in-the-large of P(y>=1) for the PU predictor, the judge plug-in mean(J>=1), their average,
and the cosine alone.  [3] within-stratum SCAR on real data: among human-relevant items (L and U), weighted
logistic O ~ h dummies + n_top bins + 1[y=2] + judge mean grade + cosine (per SD), and the cell-stratified gap
P(O=1 | both judges >= 1) - P(O=1 | otherwise) within h x n_top cells.  [4] Design estimator
(witness.d3.design_estimate) with the judge or the PU base predictor, extras [] and [fam]: certified pairs (BH/BY)
and mean SE over the 21 pairs at rho = 0, 1/2, 1.  [5] Adaptive witness: PU refitted with the batch-1 human labels;
its calibration on batch-2 U items and the design estimator with it as the base predictor.
usage: python scripts/d3/pu_real_qrels.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
sheets = sys.argv[2:]
assert len(sheets) == 2  # the two-batch HT mixture below
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sheets)]
labelled_1 = np.zeros(D.N, bool)
labelled_1[batches[0].items] = True
assert ((batches[1].pi > 0) == ~labelled_1).all()  # batch 2's frame is everything batch 1 did not label
o, U, S = D.o, ~D.o, len(D.systems)
B_REPS = 2000
rng = np.random.default_rng(0)

# ---- D3 analogues of Z_PU / X_PU
assert (D.J >= 0).all()  # no refusals on D3
q_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/q.npy").astype(np.float32)  # [n_q, 1024], unit norm
p_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/p.npy", mmap_mode="r")  # [n_corpus, 1024]
ids = json.load(open(f"{cfg['duretrieval_dir']}/systems/bge_m3/ids.json"))
corpus_ids = pd.read_parquet(glob.glob(f"{cfg['duretrieval_dir']}/hf/DuRetrieval/data/corpus-*.parquet")[0])["id"]
assert ids["pids"] == list(corpus_ids)  # frame.di indexes the corpus in this order (scripts/d3/make_batch.py)
di = D.frame.di.to_numpy()
cos = np.einsum("nd,nd->n", q_emb[D.qi], p_emb[di].astype(np.float32)).astype(float)  # [N]
COS = (cos - cos.mean()) / cos.std()
lexm = np.array([s.startswith("bm25") for s in D.systems])
H_LEX = (D.disc[lexm] > 0).mean(0)  # [N] in {0, 1/3, 2/3, 1}
N_TOP = (D.disc > 0).sum(0)  # [N] in 1..7
assert N_TOP.min() >= 1
LEVELS = np.unique(D.state)
Z_PU = np.column_stack([(D.state[:, None] == LEVELS[None, 1:]).astype(float), COS, H_LEX, N_TOP / S])
X_PU = np.column_stack([H_LEX, N_TOP / S])
print(f"frame N={D.N} (L {o.sum()}, U {U.sum()}); judge states {len(LEVELS)}; Z_PU {Z_PU.shape}, X_PU {X_PU.shape}")
print(f"cosine bge_m3: mean L {cos[o].mean():.3f}, U {cos[U].mean():.3f}; h levels "
      f"{dict(zip(*np.unique(np.round(H_LEX, 3), return_counts=True)))}; n_top {np.bincount(N_TOP)[1:].tolist()}")

# ---- labelled items and their HT weights (two-batch mixture, see the docstring)
b1, b2 = batches
n1, n2 = len(b1.items), len(b2.items)
a = n1 / (n1 + n2)
lab = np.concatenate([b1.items, b2.items])
y = np.concatenate([b1.y, b2.y])
w = np.concatenate([a / b1.pi[b1.items] + (1 - a), (1 - a) / b2.pi[b2.items]])
bat = np.r_[np.zeros(n1, int), np.ones(n2, int)]
assert len(np.unique(lab)) == len(lab)
print(f"labels: batch 1 {n1}, batch 2 {n2}, mixture a={a:.3f}; HT frame size sum w = {w.sum():.0f} (N={D.N}); "
      f"U labels {U[lab].sum()}, L labels {o[lab].sum()}")


def boot(stat):
    """Point estimate and percentile 95% CI of stat(idx) over within-batch with-replacement resamples of the labels."""
    pt = stat(np.arange(len(lab)))
    groups = [np.flatnonzero(bat == k) for k in (0, 1)]
    bs = np.array([stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B_REPS)])
    return pt, np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)


def fmt(t, f="{:.3f}"):
    pt, lo, hi = t
    return f"{f.format(pt)} [{f.format(lo)}, {f.format(hi)}]"


# ---- [1] zero-label PU fit on the real qrels
_, _, e_hat, pu_w = W.pu_fuse(o, Z_PU, X_PU, np.array([], int), np.array([], int), 0.5)
PU1, PU0 = pu_w[:, 1] + pu_w[:, 2], pu_w[:, 2]  # P(y>=1), P(y=2) given (z, x, O)
JU1, JU0 = (D.J >= 1).mean(0), (D.J == 2).mean(0)  # judge plug-in on U
print(f"[1] PU: mean P(y>=1) on U {PU1[U].mean():.3f} -> {PU1[U].sum():.0f} implied holes; on L {PU1[o].mean():.3f}; "
      f"mean e(x,1) {e_hat[:, 1].mean():.3f}, e(x,2) {e_hat[:, 2].mean():.3f}; judge mean(J>=1) on U {JU1[U].mean():.3f}")
uu = U[lab]
hu = boot(lambda i: np.average((y[i] >= 1)[uu[i]], weights=w[i][uu[i]]))
print(f"[1] human HT P(y>=1 | U) = {fmt(hu)} -> {hu[0] * U.sum():.0f} holes")


# ---- [2] P(y>=1) on U against the human labels
def w_auc(s, yb, ww):
    """Weighted AUC: P(s_pos > s_neg) + 1/2 P(tie) with pos/neg pairs weighted by w_pos w_neg."""
    order = np.argsort(s, kind="stable")
    s, yb, ww = s[order], yb[order], ww[order]
    uniq, start = np.unique(s, return_index=True)
    w_pos = np.add.reduceat(ww * yb, start)  # [n_levels] positive weight per score level
    w_neg = np.add.reduceat(ww * ~yb, start)
    below = np.cumsum(w_neg) - w_neg  # negative weight strictly below each level
    return (w_pos * (below + 0.5 * w_neg)).sum() / (w_pos.sum() * w_neg.sum())


EDGES = np.linspace(0, 1, 11)


def calib(p, yb, ww):
    """10 equal-width bins on [0, 1]: ECE (weighted |mean p - mean y| per bin, bin share of weight), Brier, and
    calibration-in-the-large mean(p) - mean(y)."""
    k = np.clip(np.digitize(p, EDGES[1:-1]), 0, 9)
    ece = 0.0
    for b in np.unique(k):
        m = k == b
        ece += ww[m].sum() * abs(np.average(p[m], weights=ww[m]) - np.average(yb[m], weights=ww[m]))
    return np.array([ece / ww.sum(), np.average((p - yb) ** 2, weights=ww), np.average(p - yb, weights=ww)])


PRED = {"pu": PU1, "judge": JU1, "avg": 0.5 * (PU1 + JU1), "cosine": COS}
y1 = y >= 1
for name, s in PRED.items():
    sl = s[lab]
    auc = boot(lambda i, sl=sl: w_auc(sl[i][uu[i]], y1[i][uu[i]], w[i][uu[i]]))
    line = f"[2] {name:7s} weighted AUC on U {fmt(auc)}"
    if name != "cosine":  # the cosine is not a probability
        cal = boot(lambda i, sl=sl: calib(sl[i][uu[i]], y1[i][uu[i]], w[i][uu[i]]))
        line += (f" | ECE {fmt((cal[0][0], cal[1][0], cal[2][0]))} | Brier {fmt((cal[0][1], cal[1][1], cal[2][1]))}"
                 f" | mean(p - y) {fmt((cal[0][2], cal[1][2], cal[2][2]), '{:+.3f}')}")
    print(line)
for name in ("pu", "judge"):
    sl = PRED[name][lab]
    k = np.clip(np.digitize(sl, EDGES[1:-1]), 0, 9)
    rows = []
    for b in range(10):
        m = uu & (k == b)
        if m.any():
            rows.append(f"[{EDGES[b]:.1f},{EDGES[b + 1]:.1f}) n={m.sum()} p={np.average(sl[m], weights=w[m]):.3f} "
                        f"y={np.average(y1[m], weights=w[m]):.3f}")
    print(f"[2] {name} reliability on U: " + "; ".join(rows))
# the PU predictor's ranking inside judge states: does it add anything the judges do not have?
sl = PU1[lab]
st_lab = D.state[lab]
cells = [s for s in LEVELS if min((uu & (st_lab == s) & y1).sum(), (uu & (st_lab == s) & ~y1).sum()) >= 5]


def within_state_auc(i, sc):
    num = den = 0.0
    for s in cells:
        m = uu[i] & (st_lab[i] == s)
        if y1[i][m].all() or not y1[i][m].any():
            return np.nan
        wt = (w[uu & (st_lab == s)]).sum()  # fixed cell weights: HT U mass of the state
        num, den = num + wt * w_auc(sc[i][m], y1[i][m], w[i][m]), den + wt
    return num / den


for name in ("pu", "cosine"):
    sc = PRED[name][lab]
    print(f"[2] {name} AUC within judge joint state on U (states with >= 5 pos and neg labels: "
          f"{[(int(s % 4) - 1, int(s // 4) - 1) for s in cells]}): {fmt(boot(lambda i, sc=sc: within_state_auc(i, sc)))}")

# ---- [3] within-stratum SCAR among human-relevant items: O ~ provenance strata + grade + non-provenance features
JMEAN = D.J.mean(0)  # [N] mean judge grade in [0, 2]
NT_BIN = np.digitize(N_TOP, [3, 5])  # n_top in {1-2, 3-4, 5-7}
H_LEV = np.round(H_LEX * 3).astype(int)  # 0..3 BM25 systems
rel = y1  # labelled relevant items, L and U
o_lab = o[lab].astype(int)
strat = H_LEV[lab] * 3 + NT_BIN[lab]
STRATA = np.unique(strat[rel])
jz, cz = (JMEAN[lab] - JMEAN.mean()) / JMEAN.std(), COS[lab]
X_A1 = np.column_stack([(strat[:, None] == STRATA[None, 1:]).astype(float), (y == 2).astype(float), jz, cz])
A1_NAMES = ["y=2", "judge mean grade (per SD)", "bge_m3 cosine (per SD)"]


def a1_fit(i):
    m = rel[i]
    xi_, oi_, wi_ = X_A1[i][m], o_lab[i][m], w[i][m]
    keep = (xi_[:, :len(STRATA) - 1].sum(0) > 0)  # strata absent from a resample carry no information
    xi_ = np.column_stack([xi_[:, :len(STRATA) - 1][:, keep], xi_[:, len(STRATA) - 1:]])
    if oi_.all() or not oi_.any():
        return np.full(3, np.nan)
    m_ = LogisticRegression(C=1e6, max_iter=5000).fit(xi_, oi_, sample_weight=wi_ / wi_.mean())
    return m_.coef_[0, -3:]


print(f"[3] relevant labels {rel.sum()} (in qrels {o_lab[rel].sum()}); HT share of relevant frame items in qrels "
      f"{fmt(boot(lambda i: np.average(o_lab[i][rel[i]], weights=w[i][rel[i]])))}; strata (h level x n_top bin) "
      f"with relevant labels: {len(STRATA)}")
a1 = boot(a1_fit)
for k, nm in enumerate(A1_NAMES):
    print(f"[3] weighted logit O ~ strata + y=2 + judge + cosine | relevant: coef {nm} = "
          f"{fmt((a1[0][k], a1[1][k], a1[2][k]), '{:+.3f}')}")
both = (D.J >= 1).all(0)[lab]


def a1_gap(i):
    """P(O=1 | both judges >= 1) - P(O=1 | not), among relevant items within each stratum; fixed stratum weights."""
    num = den = 0.0
    for s in cells_a1:
        m1, m0 = i[rel[i] & (strat[i] == s) & both[i]], i[rel[i] & (strat[i] == s) & ~both[i]]
        if len(m1) == 0 or len(m0) == 0:
            return np.nan
        wt = w[rel & (strat == s)].sum()
        num += wt * (np.average(o_lab[m1], weights=w[m1]) - np.average(o_lab[m0], weights=w[m0]))
        den += wt
    return num / den


cells_a1 = [s for s in STRATA if min((rel & (strat == s) & both).sum(), (rel & (strat == s) & ~both).sum()) >= 5]
print(f"[3] stratified gap P(O | relevant, both judges >= 1) - P(O | relevant, not both) over "
      f"{len(cells_a1)} strata with >= 5 in each arm: {fmt(boot(a1_gap), '{:+.3f}')}")
for s in cells_a1:
    m1, m0 = rel & (strat == s) & both, rel & (strat == s) & ~both
    print(f"   stratum h={s // 3}/3 n_top bin {s % 3}: both n={m1.sum()} P(O)={np.average(o_lab[m1], weights=w[m1]):.3f}"
          f" | not both n={m0.sum()} P(O)={np.average(o_lab[m0], weights=w[m0]):.3f}")


# ---- [4] design estimator with the judge vs the PU base predictor
def certify(est, var):
    """Endpoint intersection-union + BH / BY over the 2P one-sided nulls (as scripts/d3/analyze.py)."""
    se = [np.sqrt(np.maximum(v[:D.P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:D.P] / se[0]), norm.sf(est[2][:D.P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:D.P] / se[0]), norm.cdf(est[2][:D.P] / se[2]))
    return {name: W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"], dependent=dep).reshape(2, D.P)
            for name, dep in (("BH", False), ("BY", True))}


F_PU = [PU0, 0.5 * PU0 + 0.5 * PU1, PU1]
F_AVG = [0.5 * (a_ + b_) for a_, b_ in zip(F_PU, D.F)]  # identical on L (unused there: the L column carries o)
res = {}
for pname, F in (("judge", D.F), ("pu", F_PU), ("avg", F_AVG)):
    for extras in ([], ["fam"]):
        est, var = WD.design_estimate(D, batches, extras, cfg["ridge"], F)
        c = certify(est, var)
        tag = f"{pname}{'+' + extras[0] if extras else ''}"
        res[tag] = (est, var, c)
        se = [np.sqrt(v[:D.P]) for v in var]
        print(f"[4] {tag:10s} certified BH {int(c['BH'].sum())}/{D.P}, BY {int(c['BY'].sum())}/{D.P}; mean SE over pairs "
              f"rho=0 {se[0].mean():.4f}, rho=1/2 {se[1].mean():.4f}, rho=1 {se[2].mean():.4f}")
# ---- [5] adaptive witness on real data: refit PU with the batch-1 human labels (Y fixed in the EM; the returned
# posterior is the model's, not the observed label), and use it as batch 2's base predictor only.  Batch 1 keeps a
# zero-label predictor: a predictor fitted on a batch's own labels would leak them into its HT correction (residual 0
# on those items) and understate the variance.  Final-mode GREG fit per batch, O(m/n) like every number above.
_, _, _, pu_w1 = W.pu_fuse(o, Z_PU, X_PU, b1.items, b1.y, 0.5, w0=pu_w, clamp_out=False)
PA1, PA0 = pu_w1[:, 1] + pu_w1[:, 2], pu_w1[:, 2]
uu2 = U[b2.items]
w2 = 1.0 / b2.pi[b2.items]
y2 = (b2.y >= 1).astype(float)
for nm, pr in (("PU zero-label", PU1), ("PU refit on batch 1", PA1), ("judge", JU1)):
    pb = pr[b2.items][uu2]
    print(f"[5] batch-2 U items ({uu2.sum()}): {nm:20s} mean(p - y) {np.average(pb - y2[uu2], weights=w2[uu2]):+.4f}  "
          f"Brier {np.average((pb - y2[uu2]) ** 2, weights=w2[uu2]):.4f}")
F_PA = [PA0, 0.5 * PA0 + 0.5 * PA1, PA1]
for first, nm in ((F_PU, "pu"), (D.F, "judge")):  # batch 1's (zero-label) predictor
    for extras in ([], ["fam"]):
        est, var = WD.design_estimate(D, batches, extras, cfg["ridge"], [first, F_PA])
        c = certify(est, var)
        tag = f"{nm}>pu_refit{'+' + extras[0] if extras else ''}"
        res[tag] = (est, var, c)
        se = [np.sqrt(v[:D.P]) for v in var]
        print(f"[5] {tag:18s} certified BH {int(c['BH'].sum())}/{D.P}, BY {int(c['BY'].sum())}/{D.P}; mean SE over "
              f"pairs rho=0 {se[0].mean():.4f}, rho=1/2 {se[1].mean():.4f}, rho=1 {se[2].mean():.4f}")
for base in ("judge", "judge+fam"):
    for alt in ("pu", "avg", "pu+fam", "avg+fam", "pu>pu_refit", "pu>pu_refit+fam", "judge>pu_refit",
                "judge>pu_refit+fam"):
        if base.endswith("fam") != alt.endswith("fam"):
            continue
        r = [np.sqrt(res[alt][1][b][:D.P] / res[base][1][b][:D.P]) for b in range(3)]
        flips = int((res[alt][2]["BH"] != res[base][2]["BH"]).any(0).sum())
        print(f"[4] SE ratio {alt} / {base} per pair: rho=0 median {np.median(r[0]):.3f} [{r[0].min():.3f}, {r[0].max():.3f}], "
              f"rho=1/2 median {np.median(r[1]):.3f} [{r[1].min():.3f}, {r[1].max():.3f}], rho=1 median "
              f"{np.median(r[2]):.3f} [{r[2].min():.3f}, {r[2].max():.3f}]; pairs whose BH decision differs: {flips}")
for p, (a_, b_) in enumerate(D.pairs):
    print(f"   {D.systems[a_]:14s} vs {D.systems[b_]:14s} D(.5) judge {res['judge'][0][1][p]:+.4f} "
          f"(se {np.sqrt(res['judge'][1][1][p]):.4f}) pu {res['pu'][0][1][p]:+.4f} (se {np.sqrt(res['pu'][1][1][p]):.4f}) "
          f"judge+fam {res['judge+fam'][0][1][p]:+.4f} (se {np.sqrt(res['judge+fam'][1][1][p]):.4f}) "
          f"pu+fam {res['pu+fam'][0][1][p]:+.4f} (se {np.sqrt(res['pu+fam'][1][1][p]):.4f})")

# ---- [6] why the refit drifts: refit variants of the batch-1 PU refit, each against its
# own zero-label fit (same z, same C).  lam = 1/pi1 is the design-weighted pseudo-likelihood (witness.core.pu_fuse);
# 'fam' adds the dense and rerank hit shares to z (the BM25 share is already in Z_PU); 'sys' adds every system's top-10
# hit and discount, which makes batch-1 sampling exactly ignorable given (z, O): pi1 is a function of the judges,
# the discounts and O only (scripts/d3/make_batch.py).  'cont' continues the zero-label EM for the refit's
# 30 iterations with no label (EM non-convergence check).  Calibration on batch-2 U items (1/pi2 weights; bootstrap
# of those items only, the refit is held fixed) and the census mean of p over batch 2's U frame (exact, no labels).
sysm = np.array([s.startswith("rerank") for s in D.systems])
DENSE, RER = (D.disc[~lexm & ~sysm] > 0).mean(0), (D.disc[sysm] > 0).mean(0)
Z_FAM = np.column_stack([Z_PU, DENSE, RER])
Z_SYS = np.column_stack([Z_PU] + [c for s in range(S) for c in ((D.disc[s] > 0).astype(float), D.disc[s])])
NONE = np.array([], int)
b1u = b1.items[U[b1.items]]
lam1 = 1.0 / b1.pi[b1.items]
print(f"[6] Z_FAM {Z_FAM.shape}, Z_SYS {Z_SYS.shape}; batch-1 lam = 1/pi1: min {lam1.min():.1f} median "
      f"{np.median(lam1):.1f} max {lam1.max():.1f}; batch-1 U labels {len(b1u)}")
zero_fit = {"": pu_w}
for zn, zm, C in (("fam", Z_FAM, 1.0), ("sys", Z_SYS, 1.0), ("C0.1", Z_PU, 0.1)):
    zero_fit[zn] = W.pu_fuse(o, zm, X_PU, NONE, NONE, 0.5, C=C)[3]
ZM = {"": (Z_PU, 1.0), "fam": (Z_FAM, 1.0), "sys": (Z_SYS, 1.0), "C0.1": (Z_PU, 0.1)}
# name: (z key, labelled items, their y, lam or None)
VARIANTS = {"cont (no label)": ("", NONE, NONE, None), "unw": ("", b1.items, b1.y, None),
            "w": ("", b1.items, b1.y, lam1), "unw U-only": ("", b1u, b1.y[U[b1.items]], None),
            "unw+fam": ("fam", b1.items, b1.y, None), "w+fam": ("fam", b1.items, b1.y, lam1),
            "unw+sys": ("sys", b1.items, b1.y, None), "w+sys": ("sys", b1.items, b1.y, lam1),
            "unw C0.1": ("C0.1", b1.items, b1.y, None), "w C0.1": ("C0.1", b1.items, b1.y, lam1)}
post = {}
for nm, (zk, li, ly, lw) in VARIANTS.items():
    if nm == "unw":
        post[nm] = pu_w1  # the [5] refit, same call
        continue
    zm, C = ZM[zk]
    post[nm] = W.pu_fuse(o, zm, X_PU, li, ly, 0.5, w0=zero_fit[zk], clamp_out=False, lab_w=lw, C=C)[3]
fr2 = U & ~labelled_1  # batch 2's U frame
y2u, w2u = y2[uu2], w2[uu2]


def cal_boot(pb):
    """calib() of batch-2 U predictions with a with-replacement bootstrap of those items."""
    pt = calib(pb, y2u.astype(bool), w2u)
    bs = np.array([calib(pb[j], y2u[j].astype(bool), w2u[j])
                   for j in (rng.integers(0, len(pb), len(pb)) for _ in range(B_REPS))])
    return pt, np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0)


p1 = {}
for zk, pw in zero_fit.items():
    p1[f"zero{'+' + zk if zk else ''}"] = pw[:, 1] + pw[:, 2]
for nm, pw in post.items():
    p1[nm] = pw[:, 1] + pw[:, 2]
p1["judge"] = JU1
base_of = {nm: f"zero{'+' + v[0] if v[0] else ''}" for nm, v in VARIANTS.items()}
for nm, pr in p1.items():
    cb = cal_boot(pr[b2.items][uu2])
    line = (f"[6] {nm:16s} census mean p on b2 U frame {pr[fr2].mean():.4f}"
            + (f" (vs {base_of[nm]} {pr[fr2].mean() - p1[base_of[nm]][fr2].mean():+.4f})" if nm in base_of else "")
            + f" | b2 U: mean(p - y) {fmt((cb[0][2], cb[1][2], cb[2][2]), '{:+.4f}')}"
            f" Brier {fmt((cb[0][1], cb[1][1], cb[2][1]), '{:.4f}')} ECE {fmt((cb[0][0], cb[1][0], cb[2][0]), '{:.4f}')}")
    print(line)
# selection given the judge state: batch-1 U labels, unweighted vs 1/pi1-weighted relevance rate per state.  Under
# ignorability given z the two agree within a cell of z; pi1 varies within a state only through the discounts g.
y1b = (b1.y >= 1)
st1 = D.state[b1.items]
for s in np.unique(D.state[U]):
    m = U[b1.items] & (st1 == s)
    if m.sum() < 5:
        continue
    mf = fr2 & (D.state == s)
    print(f"[6] U judge state (pro, flash) = ({int(s % 4) - 1}, {int(s // 4) - 1}): b1 n={m.sum()} rate unweighted "
          f"{y1b[m].mean():.3f} HT {np.average(y1b[m], weights=lam1[m]):.3f}; median 1/pi1 {np.median(lam1[m]):.0f}; "
          f"b2 U frame N={mf.sum()} census p: zero {p1['zero'][mf].mean():.3f} unw {p1['unw'][mf].mean():.3f} "
          f"w {p1['w'][mf].mean():.3f} unw+sys {p1['unw+sys'][mf].mean():.3f}")
# the same comparison by provenance inside the state that holds most labels: pi1 grows with the pair weights of the
# systems that retrieved the item
for nm_, col in (("BM25 share", H_LEX), ("dense share", DENSE), ("rerank hit", RER)):
    for lv in np.unique(col):
        m = U[b1.items] & (col[b1.items] == lv)
        if m.sum() < 10:
            continue
        mf = fr2 & (col == lv)
        print(f"[6] U {nm_} = {lv:.3f}: b1 n={m.sum()} rate unweighted {y1b[m].mean():.3f} HT "
              f"{np.average(y1b[m], weights=lam1[m]):.3f} | census p zero {p1['zero'][mf].mean():.3f} "
              f"unw {p1['unw'][mf].mean():.3f} w {p1['w'][mf].mean():.3f} unw+sys {p1['unw+sys'][mf].mean():.3f}")
for nm in ("unw", "w", "unw+fam", "w+fam", "unw+sys", "w+sys", "unw C0.1", "w C0.1", "unw U-only"):
    F_V = [post[nm][:, 2], 0.5 * post[nm][:, 2] + 0.5 * p1[nm], p1[nm]]
    for extras in ([], ["fam"]):
        est, var = WD.design_estimate(D, batches, extras, cfg["ridge"], [F_PU, F_V])
        c = certify(est, var)
        se = [np.sqrt(v[:D.P]) for v in var]
        r = np.sqrt(var[1][:D.P] / res[f"judge{'+fam' if extras else ''}"][1][1][:D.P])
        print(f"[6] pu>{nm + ('+fam' if extras else ''):18s} certified BH {int(c['BH'].sum())}/{D.P}, BY "
              f"{int(c['BY'].sum())}/{D.P}; mean SE rho=0 {se[0].mean():.4f}, 1/2 {se[1].mean():.4f}, 1 "
              f"{se[2].mean():.4f}; SE ratio vs judge{'+fam' if extras else ''} at rho=1/2 median {np.median(r):.3f}")

# ---- [7] EM convergence ([6]: continuing the zero-label EM with no label moves p more than the refit does).  Chain
# the zero-label EM in blocks of 30 iterations to 300 (Z_PU and Z_SYS) and report the census mean of P(y>=1) on
# batch 2's U frame after each block; then refit from the 300-iteration fit for 150 iterations (unweighted, 1/pi1,
# unweighted + sys) so the effect of the labels is measured against a converged zero-label fit.  The human target:
# HT relevance rate of U per batch (batch 1 and batch 2 alone, each over its own frame).
for bt, nm_ in ((b1, "batch 1"), (b2, "batch 2")):
    mu = U[bt.items]
    yb, wb = (bt.y[mu] >= 1), 1.0 / bt.pi[bt.items][mu]
    bs = [np.average(yb[j], weights=wb[j]) for j in (rng.integers(0, len(yb), len(yb)) for _ in range(B_REPS))]
    print(f"[7] HT P(y>=1 | U) from {nm_} alone: {np.average(yb, weights=wb):.4f} "
          f"[{np.quantile(bs, 0.025):.4f}, {np.quantile(bs, 0.975):.4f}] (n={mu.sum()})")
conv = {}
for zk, zm in (("", Z_PU), ("sys", Z_SYS)):
    pw = zero_fit[zk]
    traj = [pw[:, 1][fr2].mean() + pw[:, 2][fr2].mean()]
    for _ in range(9):
        pw = W.pu_fuse(o, zm, X_PU, NONE, NONE, 0.5, w0=pw)[3]
        traj.append(pw[:, 1][fr2].mean() + pw[:, 2][fr2].mean())
    conv[zk] = pw
    print(f"[7] zero-label EM{'+' + zk if zk else ''}: census mean P(y>=1) on b2 U frame after 30, 60, ..., 300 "
          f"iterations: {' '.join(f'{v:.4f}' for v in traj)}")
# 'scaled': every label weighted by the constant mean(1/pi1), the same total label weight as 'w' without the design
# information.  If 'scaled' ~ 'w', the 1/pi refit works by giving the labels census-scale weight against the
# 61,554 O-likelihood terms (dilution), not by undoing informative selection (under which 'scaled' would stay biased
# toward the over-sampled items).
for nm, zk, lw in (("unw", "", None), ("w", "", lam1), ("scaled", "", np.full(len(lam1), lam1.mean())),
                   ("unw+sys", "sys", None), ("w+sys", "sys", lam1)):
    pw = W.pu_fuse(o, ZM[zk][0], X_PU, b1.items, b1.y, 0.5, n_iter=150, w0=conv[zk], clamp_out=False, lab_w=lw)[3]
    pr, pz = pw[:, 1] + pw[:, 2], conv[zk][:, 1] + conv[zk][:, 2]
    cb, cz = cal_boot(pr[b2.items][uu2]), cal_boot(pz[b2.items][uu2])
    print(f"[7] converged {nm:8s}: census mean p on b2 U frame {pr[fr2].mean():.4f} (converged zero "
          f"{pz[fr2].mean():.4f}, label effect {pr[fr2].mean() - pz[fr2].mean():+.4f}) | b2 U mean(p - y) "
          f"{fmt((cb[0][2], cb[1][2], cb[2][2]), '{:+.4f}')} vs converged zero {fmt((cz[0][2], cz[1][2], cz[2][2]), '{:+.4f}')}"
          f" | Brier {cb[0][1]:.4f} vs {cz[0][1]:.4f}")
print("DONE")
