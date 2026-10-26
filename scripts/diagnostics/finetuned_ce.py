"""Diagnostic (oracle): how much the train-split fine-tuned cross-encoder knows about the D1 human label.

On the D1 frame (items in some system's top-k), for the endpoint targets u1 = 1[y>=1] and u0 = 1[y=2], reports AUC
and query-grouped 5-fold calibrated R^2 (as scripts/diagnostics/signals.py) of
  (a) the fine-tuned CE: isotonic on its target probability (P(y>=1) for u1, P(y=2) for u0), and a logistic model
      on both logits [logit P(y>=1), logit P(y=2)];
  (b) the judges' joint state + Qwen CE logit (logistic);
  (c) (a) and (b) combined (logistic).
Provenance gap: within deciles of the fine-tuned P(y>=1), mean true u1 of items retrieved only by lexical (BM25)
systems minus items retrieved only by neural systems (grouping of scripts/diagnostics/provenance_bias.py), pooled over
deciles with weights = min group size (deciles with >= 10 items per group), with a query-bootstrap 95% CI.  The
same is reported for the predictor error b = P(y>=1) - u1 (positive: lexical items are less relevant than predicted).
usage: python scripts/diagnostics/finetuned_ce.py <testbed.yaml> <eval.yaml> <ft_ce.npy>
"""
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
prob = np.load(sys.argv[3]).astype(float)  # [n, 3]
assert prob.shape == (len(D.y), 3) and np.allclose(prob.sum(1), 1, atol=1e-3)
fr = D.in_union
J, y, qi = D.J[:, fr], D.y[fr], D.qi[fr]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


lce = logit(D.ce[fr])
s1, s2 = prob[fr, 1] + prob[fr, 2], prob[fr, 2]  # fine-tuned P(y>=1), P(y=2)
ft = np.column_stack([logit(s1), logit(s2)])
state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
onehot = (state[:, None] == np.unique(state)[None, :]).astype(float)
folds = list(GroupKFold(5).split(lce, groups=qi))
print(f"frame items={fr.sum()} queries={len(np.unique(qi))} grade shares={np.bincount(y, minlength=3) / len(y)}")
print(f"fine-tuned CE on frame: argmax accuracy={(prob[fr].argmax(1) == y).mean():.3f}")


def cv_pred(X, t, model):
    p = np.empty(len(t))
    for tr, te in folds:
        if model == "iso":
            p[te] = IsotonicRegression(out_of_bounds="clip").fit(X[tr], t[tr]).predict(X[te])
        else:
            p[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return p


for name, t, s in (("u1=1[y>=1]", (y >= 1).astype(float), s1), ("u0=1[y=2]", (y == 2).astype(float), s2)):
    print(f"== target {name}: mean {t.mean():.3f}")
    sigs = {"(a) ft_ce iso": (s, "iso"),
            "(a) ft_ce lr": (ft, "lr"),
            "(b) joint_state+qwen_ce": (np.column_stack([onehot, lce]), "lr"),
            "(c) ft_ce+joint_state+qwen_ce": (np.column_stack([ft, onehot, lce]), "lr")}
    for k, (X, model) in sigs.items():
        p = cv_pred(X, t, model)
        auc = roc_auc_score(t, X if X.ndim == 1 else p)
        r2 = 1 - np.mean((t - p) ** 2) / t.var()
        print(f"  {k:32s} AUC={auc:.3f}  calibrated R2={r2:.3f}")

u = (y >= 1).astype(float)
b = s1 - u
lex = np.array([s.startswith("bm25") for s in D.systems])
hit = D.disc[:, fr] > 0  # [S, N]
by_lex, by_neu = hit[lex].any(0), hit[~lex].any(0)
grp = np.where(by_lex & ~by_neu, 0, np.where(~by_lex & by_neu, 1, 2))  # 0 lexical only, 1 neural only, 2 both
names = ["lex_only", "neural_only", "both"]
edges = np.quantile(s1, np.linspace(0, 1, 11))
bucket = np.searchsorted(edges[1:-1], s1, side="right")  # [N] decile of P(y>=1), 0..9
print(f"== provenance gap within deciles of fine-tuned P(y>=1); group sizes:",
      {names[k]: int((grp == k).sum()) for k in range(3)})
rows = []
for d in range(10):
    m = bucket == d
    cells = []
    for k in range(3):
        mk = m & (grp == k)
        cells.append(f"{names[k]}: n={int(mk.sum()):5d} u={u[mk].mean() if mk.any() else np.nan:.3f} "
                     f"b={b[mk].mean() if mk.any() else np.nan:+.3f}")
    print(f"  decile {d} P in [{edges[d]:.3f},{edges[d + 1]:.3f}] " + " | ".join(cells))
    n0, n1 = (m & (grp == 0)).sum(), (m & (grp == 1)).sum()
    if n0 >= 10 and n1 >= 10:
        rows.append((d, min(n0, n1)))
print(f"  deciles used: {[d for d, _ in rows]}")


def pooled_gap(wq, v):
    """Bucket-weighted mean of [v(lex_only) - v(neural_only)] (weights = min group size), with item weights
    wq[qi] = multiplicity of the item's query in a query bootstrap draw (all ones for the point estimate)."""
    wi = wq[qi]
    num = den = 0.0
    for d, w in rows:
        m0 = (bucket == d) & (grp == 0)
        m1 = (bucket == d) & (grp == 1)
        w0, w1 = wi[m0].sum(), wi[m1].sum()
        assert w0 > 0 and w1 > 0
        num += w * ((wi[m0] * v[m0]).sum() / w0 - (wi[m1] * v[m1]).sum() / w1)
        den += w
    return num / den


nq = D.n_q
rng = np.random.default_rng(0)
draws = []
while len(draws) < 1000:
    cnt = np.bincount(rng.integers(0, nq, nq), minlength=nq).astype(float)
    if all(cnt[qi[(bucket == d) & (grp == k)]].sum() > 0 for d, _ in rows for k in (0, 1)):
        draws.append(cnt)  # a draw that empties a (decile, group) cell has no defined gap; rare with >= 10 items
for lab, v in (("u(lex_only) - u(neural_only)", u), ("b(lex_only) - b(neural_only)", b)):
    gap = pooled_gap(np.ones(nq), v)
    boots = [pooled_gap(c, v) for c in draws]
    print(f"pooled decile-weighted gap {lab} = {gap:+.4f}  "
          f"query-bootstrap 95% [{np.quantile(boots, .025):+.4f}, {np.quantile(boots, .975):+.4f}]")
print("DONE")
