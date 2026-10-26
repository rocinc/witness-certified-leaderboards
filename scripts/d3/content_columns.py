"""D3 main analysis (pro+flash, two real batches): content-feature calibration vs the judge+fam baseline, and the
certification / SE non-monotonicity of the judge ladder.
[A] predictors / columns: judge, judge+fam (baseline), judge_cos (bge_m3 cosine as an extra GREG column centred in U,
    as the D1/D2 'ce' column), judge_cos+fam, pu_prior (PU fitted P(Y | z), selection factor removed), pu_prior+fam,
    pu_prior_cos+fam.  Certified BH / BY, mean SE at rho 0, 1/2, 1, per-pair SE ratio vs judge+fam.
    Uses WD.design_estimate with WD.design_columns temporarily wrapped to append the cosine column; checked against
    the unwrapped call when the column is absent.
[B] non-monotonicity: judge predictor, extras [], rung pro+flash vs rung bm25_char1 (scores cut to the LLM grade shares,
    as scripts/d3/judge_ladder.py).  Per pair: endpoint estimates and SEs, z, the IU p-value, BH rank and
    threshold; pairs whose decision differs.
usage: python scripts/d3/content_columns.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx>
"""
import glob
import re
import sys
from types import SimpleNamespace

import bm25s
import numpy as np
import pandas as pd
from scipy.stats import norm
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
D0 = WD.load_frame(cfg)
batches = [WD.read_batch(D0, t + 1, x) for t, x in enumerate(sys.argv[2:4])]
o, U, N, P, PD = D0.o, ~D0.o, D0.N, D0.P, cfg["duretrieval_dir"]
DI = D0.frame.di.to_numpy()


def one(pattern):
    hits = glob.glob(pattern)
    assert len(hits) == 1, pattern
    return hits[0]


q = np.load(f"{PD}/systems/bge_m3/q.npy").astype(np.float32)  # [n_q, 1024], unit norm
p = np.load(f"{PD}/systems/bge_m3/p.npy", mmap_mode="r")  # [n_corpus, 1024]
cos = np.einsum("nd,nd->n", q[D0.qi], np.asarray(p[DI], dtype=np.float32)).astype(float)  # [N]
CONTENT = (cos - cos.mean()) / cos.std()
COL = np.where(U, CONTENT - CONTENT[U].mean(), 0.0)  # centred within U, 0 on L
H_LEX = (D0.disc[np.array([s.startswith("bm25") for s in D0.systems])] > 0).mean(0)
N_TOP = (D0.disc > 0).sum(0)
X_PU = np.column_stack([H_LEX, N_TOP / len(D0.systems)])

_design_columns = WD.design_columns
ADD_COS = [False]


def design_columns_cos(D, b, extras, f):
    Zc, b0 = _design_columns(D, b, extras, f)
    if ADD_COS[0]:
        return np.column_stack([Zc, COL]), np.r_[b0, 0.0]
    return Zc, b0


WD.design_columns = design_columns_cos  # design_estimate looks the name up in its module at call time


def certify(est, var):
    """Endpoint intersection-union + BH / BY over the 2P one-sided nulls (as scripts/d3/pu_real_qrels.py)."""
    se = [np.sqrt(np.maximum(v[:P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:P] / se[0]), norm.sf(est[2][:P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:P] / se[0]), norm.cdf(est[2][:P] / se[2]))
    pv = np.concatenate([p_plus, p_minus])
    return {name: W.bh(pv, cfg["alpha"], dependent=dep).reshape(2, P) for name, dep in (("BH", False), ("BY", True))}, pv


def run(D, F, extras, add_cos):
    ADD_COS[0] = add_cos
    est, var = WD.design_estimate(D, batches, extras, cfg["ridge"], F)
    ADD_COS[0] = False
    return est, var


# ---- [A]
J = D0.J
lev = np.unique(D0.state)
Z = np.column_stack([(D0.state[:, None] == lev[None, 1:]).astype(float), CONTENT[:, None], X_PU])
_, _, e, pw = W.pu_fuse(o, Z, X_PU, np.array([], int), np.array([], int), 0.5)
pr = pw / (1 - e)
pr /= pr.sum(1, keepdims=True)
F_PRIOR = [np.where(o, 1.0, v) for v in (pr[:, 2], 0.5 * pr[:, 2] + 0.5 * (pr[:, 1] + pr[:, 2]), pr[:, 1] + pr[:, 2])]
SPECS = {"judge": (D0.F, [], False), "judge+fam": (D0.F, ["fam"], False), "judge_cos": (D0.F, [], True),
         "judge_cos+fam": (D0.F, ["fam"], True), "pu_prior": (F_PRIOR, [], False),
         "pu_prior+fam": (F_PRIOR, ["fam"], False), "pu_prior_cos+fam": (F_PRIOR, ["fam"], True)}
res = {}
for name, (F, extras, add_cos) in SPECS.items():
    est, var = run(D0, F, extras, add_cos)
    if not add_cos:
        e2, v2 = WD.design_estimate(D0, batches, extras, cfg["ridge"], F)  # wrapper is a no-op here
        assert all(np.array_equal(a_, b_) for a_, b_ in zip(est + var, e2 + v2))
    c, _ = certify(est, var)
    res[name] = (est, var, c)
    se = [np.sqrt(v[:P]).mean() for v in var]
    print(f"[A] {name:17s} certified BH {int(c['BH'].sum())}/{P}, BY {int(c['BY'].sum())}/{P}; mean SE rho=0 {se[0]:.4f}, "
          f"1/2 {se[1]:.4f}, 1 {se[2]:.4f}")
base = res["judge+fam"]
for name in SPECS:
    if name == "judge+fam":
        continue
    r = [np.sqrt(res[name][1][b][:P] / base[1][b][:P]) for b in range(3)]
    flips = [f"{D0.systems[D0.pairs[k][0]]}-{D0.systems[D0.pairs[k][1]]}"
             for k in np.flatnonzero((res[name][2]["BH"] != base[2]["BH"]).any(0))]
    print(f"[A] SE ratio {name} / judge+fam per pair: " + "; ".join(
        f"rho={rl} median {np.median(x):.3f} [{x.min():.3f}, {x.max():.3f}]" for rl, x in zip(("0", "1/2", "1"), r))
        + f"; BH decisions differing: {flips}")

# ---- [B] non-monotonicity: pro+flash vs bm25_char1 judge, extras []
corpus = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/corpus-*.parquet"))
queries = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/queries-*.parquet"))
CJK = re.compile(r"[\u4e00-\u9fff]")


def t_char1(s):
    out = []
    for run_ in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", s):
        if CJK.match(run_[0]):
            out.extend(run_)
        else:
            out.append(run_.lower())
    return out


ret = bm25s.BM25(k1=1.2, b=0.75)
ret.index([t_char1(s) for s in corpus["text"]], show_progress=False)
qtok = [t_char1(s) for s in queries["text"]]
BM = np.full(N, np.nan)
rows_of = pd.Series(np.arange(N)).groupby(D0.qi).apply(np.array)
for qq in range(D0.n_q):
    BM[rows_of[qq]] = ret.get_scores(qtok[qq])[DI[rows_of[qq]]]
assert np.isfinite(BM).all()
SHARE = np.mean([np.bincount(D0.frame[j], minlength=3) / N for j in cfg["judges"]], axis=0)
Jb = np.digitize(BM, np.quantile(BM, np.cumsum(SHARE)[:2]))[None]
F0, F1 = (Jb == 2).mean(0), (Jb >= 1).mean(0)
Db = SimpleNamespace(**{**vars(D0), "J": Jb, "state": ((Jb + 1) * 4 ** np.arange(1)[:, None]).sum(0),
                        "F": [np.where(o, 1.0, F0), np.where(o, 1.0, 0.5 * F0 + 0.5 * F1), np.where(o, 1.0, F1)]})
out = {}
for name, D in (("pro+flash", D0), ("bm25_char1", Db)):
    est, var = run(D, D.F, [], False)
    c, pv = certify(est, var)
    out[name] = (est, var, c, pv)
    print(f"[B] {name}: certified BH {int(c['BH'].sum())}; mean SE rho=1/2 {np.sqrt(var[1][:P]).mean():.4f}")
alpha, m = cfg["alpha"], 2 * P
for name, (est, var, c, pv) in out.items():
    k = int(c["BH"].sum())
    srt = np.sort(pv)
    print(f"[B] {name}: BH rejections {k}; threshold alpha*k/2P = {alpha * k / m:.5f}; p of rank k {srt[k - 1]:.2e}, "
          f"rank k+1 {srt[k]:.2e} (threshold at k+1 {alpha * (k + 1) / m:.5f})")
print("[B] per pair (D at rho=0 / rho=1, se, IU p of the favoured direction, BH certified) pro+flash | bm25_char1:")
for k in range(P):
    cells = []
    for name, (est, var, c, pv) in out.items():
        pk = min(pv[k], pv[P + k])
        cells.append(f"D0 {est[0][k]:+.3f} ({np.sqrt(var[0][k]):.3f}) D1 {est[2][k]:+.3f} ({np.sqrt(var[2][k]):.3f}) "
                     f"p {pk:.1e} {'C' if c['BH'][:, k].any() else '-'}")
    flag = " <== differs" if out["pro+flash"][2]["BH"][:, k].any() != out["bm25_char1"][2]["BH"][:, k].any() else ""
    print(f"   {D0.systems[D0.pairs[k][0]]:13s} vs {D0.systems[D0.pairs[k][1]]:13s} " + " | ".join(cells) + flag)
print("DONE")
