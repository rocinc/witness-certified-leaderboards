"""Mechanism check on real D3 qrels: why the PU posterior P(Y | z, x, O=0) ranks worse than its own prior P(Y | z).
Per n_top bin on U: HT human P(y>=1), mean posterior, mean prior, mean e(x,1), for the pro+flash rung and the
text2vec cosine rung (pieces as in scripts/d3/judge_ladder.py).
usage: python scripts/d3/ladder_selection.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx>
"""
import sys

import numpy as np
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
D = WD.load_frame(cfg)
b1, b2 = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sys.argv[2:4])]
a = len(b1.items) / (len(b1.items) + len(b2.items))
lab = np.concatenate([b1.items, b2.items])
y1 = np.concatenate([b1.y, b2.y]) >= 1
w = np.concatenate([a / b1.pi[b1.items] + (1 - a), (1 - a) / b2.pi[b2.items]])  # two-batch HT mixture weights
o, U, PD = D.o, ~D.o, cfg["duretrieval_dir"]
uu = U[lab]


def cosine(model):
    q = np.load(f"{PD}/systems/{model}/q.npy").astype(np.float32)  # [n_q, dim], unit norm
    p = np.load(f"{PD}/systems/{model}/p.npy", mmap_mode="r")  # [n_corpus, dim]
    return np.einsum("nd,nd->n", q[D.qi], np.asarray(p[D.frame.di.to_numpy()], dtype=np.float32)).astype(float)  # [N]


cos_bge = cosine("bge_m3")
CONTENT = (cos_bge - cos_bge.mean()) / cos_bge.std()
H_LEX = (D.disc[np.array([s.startswith("bm25") for s in D.systems])] > 0).mean(0)
N_TOP = (D.disc > 0).sum(0)
X_PU = np.column_stack([H_LEX, N_TOP / len(D.systems)])
SHARE = np.mean([np.bincount(D.frame[j], minlength=3) / D.N for j in cfg["judges"]], axis=0)
c_t2v = cosine("text2vec_zh")
RUNGS = {"pro+flash": D.J, "cos_text2vec_zh": np.digitize(c_t2v, np.quantile(c_t2v, np.cumsum(SHARE)[:2]))[None]}
BINS = [(1, 2), (3, 4), (5, 7)]
for name, J in RUNGS.items():
    state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
    lev = np.unique(state)
    Z = np.column_stack([(state[:, None] == lev[None, 1:]).astype(float), CONTENT[:, None], X_PU])
    _, _, e, pw = W.pu_fuse(o, Z, X_PU, np.array([], int), np.array([], int), 0.5)
    prior = pw / (1 - e)
    prior /= prior.sum(1, keepdims=True)
    post1, prior1 = pw[:, 1] + pw[:, 2], prior[:, 1] + prior[:, 2]
    print(f"rung {name}: psi (k=2 vs k=1 logit of e) e(x,2)-e(x,1) mean on U {(e[U, 2] - e[U, 1]).mean():+.4f}")
    for lo, hi in BINS:
        m_all = U & (N_TOP >= lo) & (N_TOP <= hi)
        m = uu & (N_TOP[lab] >= lo) & (N_TOP[lab] <= hi)
        print(f"   n_top {lo}-{hi}: U items {m_all.sum()} (labelled {m.sum()}); human HT P(y>=1) "
              f"{np.average(y1[m], weights=w[m]):.3f}; mean on labelled: posterior {np.average(post1[lab][m], weights=w[m]):.3f}"
              f", prior {np.average(prior1[lab][m], weights=w[m]):.3f}; e(x,1) mean over U items {e[m_all, 1].mean():.3f}; "
              f"share of the bin's relevant frame items in qrels (labelled, L+U) "
              f"{np.average(o[lab][(N_TOP[lab] >= lo) & (N_TOP[lab] <= hi) & y1], weights=w[(N_TOP[lab] >= lo) & (N_TOP[lab] <= hi) & y1]):.3f}")
print("DONE")
