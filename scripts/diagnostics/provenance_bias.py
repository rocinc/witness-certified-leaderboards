"""Diagnostic (oracle): is the LLM judges' error differential by retrieval provenance?

Within each joint judge state (flash, pro), compares the true gain u and the judge error b = f - u of frame items
retrieved only by lexical systems (BM25 family), only by neural systems (dense or reranker), or by both.
A positive lexical-minus-neural gap in b means: at the same judge verdict, lexically retrieved items are less
relevant than the judges say, so the judge plug-in leaderboard favours lexical systems (a shared blind spot of the
judges and BM25).  Also reports the query-bootstrap CI of the pooled, state-weighted gap.
usage: python scripts/diagnostics/provenance_bias.py <testbed.yaml> <eval.yaml> <rho>
"""
import os
import sys

import numpy as np
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
rho = float(sys.argv[3])
fr = D.in_union
u = ((1 - rho) * D.u0 + rho * D.u1)[fr]
J = D.J[:, fr]
ans = J >= 0
gain = np.where(J == 2, 1.0, np.where(J == 1, rho, 0.0))
f = np.where(ans.sum(0) > 0, (gain * ans).sum(0) / np.maximum(ans.sum(0), 1), D.ce[fr])
b = f - u
lex = np.array([s.startswith("bm25") for s in D.systems])
hit = D.disc[:, fr] > 0  # [S, N]
by_lex, by_neu = hit[lex].any(0), hit[~lex].any(0)
grp = np.where(by_lex & ~by_neu, 0, np.where(~by_lex & by_neu, 1, 2))  # 0 lexical only, 1 neural only, 2 both
state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
qf = D.qi[fr]
names = ["lex_only", "neural_only", "both"]
print(f"frame N={fr.sum()} rho={rho}; group sizes:", {names[k]: int((grp == k).sum()) for k in range(3)})
print("state (flash,pro): per group n / mean u / mean judge error b")
rows = []
for s in np.unique(state):
    m = state == s
    if m.sum() < 30:
        continue
    a_, b_ = (s % 4) - 1, (s // 4) - 1
    cells = []
    for k in range(3):
        mk = m & (grp == k)
        cells.append(f"{names[k]}: n={int(mk.sum()):5d} u={u[mk].mean() if mk.any() else np.nan:.3f} b={b[mk].mean() if mk.any() else np.nan:+.3f}")
    print(f"  ({a_:+d},{b_:+d}) " + " | ".join(cells))
    if (m & (grp == 0)).sum() >= 10 and (m & (grp == 1)).sum() >= 10:
        rows.append((s, min((m & (grp == 0)).sum(), (m & (grp == 1)).sum())))


def pooled_gap(wq):
    """State-weighted mean of [b(lex_only) - b(neural_only)] over states (weights = min group size), with item
    weights wq[qi] = multiplicity of the item's query in a query bootstrap draw (all ones for the point estimate)."""
    wi = wq[qf]
    num = den = 0.0
    for s, w in rows:
        m0 = (state == s) & (grp == 0)
        m1 = (state == s) & (grp == 1)
        w0, w1 = wi[m0].sum(), wi[m1].sum()
        assert w0 > 0 and w1 > 0
        num += w * ((wi[m0] * b[m0]).sum() / w0 - (wi[m1] * b[m1]).sum() / w1)
        den += w
    return num / den


nq = D.n_q
gap = pooled_gap(np.ones(nq))
rng = np.random.default_rng(0)
boots = []
while len(boots) < 1000:
    cnt = np.bincount(rng.integers(0, nq, nq), minlength=nq).astype(float)
    ok = all(cnt[qf[(state == s) & (grp == k)]].sum() > 0 for s, _ in rows for k in (0, 1))
    if ok:  # a draw that leaves a (state, group) cell empty has no defined gap; with >=10 items per cell it is rare
        boots.append(pooled_gap(cnt))
print(f"pooled state-weighted gap b(lex_only) - b(neural_only) = {gap:+.4f}  "
      f"query-bootstrap 95% [{np.quantile(boots, .025):+.4f}, {np.quantile(boots, .975):+.4f}]")
print("DONE")
