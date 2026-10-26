"""Zero-label PU EM convergence with and without the shadow-variable exclusion restriction.  The loop is a line-for-line
copy of witness.core.pu_fuse (no labels), with checkpoints.
  zkind pu  z = Z_PU of the main code (judge-state dummies, score, h, n_top/S): h and n_top also in x
  zkind ex  z = judge-state dummies + score only (D3: bge_m3 cosine; D1/D2: logit CE): z excluded from O's model
x = [h, n_top/S] in both.  D3: real qrels, target = human HT P(y>=1 | U) .189; D1/D2: rep 0 of the simulated qrels
(main eval config), truth u1 known.  Saves the posterior at each checkpoint to <out_dir>/em_<dataset>_<zkind>.npz
(out_dir of the D3 config, or of the eval config on D1/D2) for reuse.
usage: python scripts/diagnostics/em_exclusion.py d3 <pu|ex> <n_iter_max> configs/d3_duretrieval.yaml
       python scripts/diagnostics/em_exclusion.py d1 <pu|ex> <n_iter_max> configs/testbeds/d1_t2ranking.yaml \
           configs/eval/main_d1.yaml
       python scripts/diagnostics/em_exclusion.py d2 <pu|ex> <n_iter_max> configs/testbeds/d2_dl19.yaml \
           configs/eval/main_d2.yaml
"""
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from witness import core as W  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
DS, ZK, N_MAX = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert DS in ("d3", "d1", "d2") and ZK in ("pu", "ex")
CKPT = [c for c in (30, 60, 100, 200, 300, 500, 700, 1000, 1500, 2000) if c <= N_MAX]
if DS == "d3":
    from witness import d3 as WD

    assert len(sys.argv) == 5
    cfg = load_config(sys.argv[4])
    D = WD.load_frame(cfg)
    q_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/q.npy").astype(np.float32)  # [n_q, 1024], unit norm
    p_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/p.npy", mmap_mode="r")  # [n_corpus, 1024]
    cos = np.einsum("nd,nd->n", q_emb[D.qi], p_emb[D.frame.di.to_numpy()].astype(np.float32)).astype(float)  # [N]
    score = (cos - cos.mean()) / cos.std()
    lexm = np.array([s.startswith("bm25") for s in D.systems])
    h, n_top, S = (D.disc[lexm] > 0).mean(0), (D.disc > 0).sum(0), len(D.systems)
    state, o = D.state, D.o.astype(bool)
    U, truth = ~o, None
else:
    from witness.data import load_testbed

    assert len(sys.argv) == 6
    base = load_config(sys.argv[4])
    cfg = load_config(sys.argv[5])
    D = load_testbed(base, cfg)
    score = np.log(np.clip(D.ce, 1e-4, 1 - 1e-4) / (1 - np.clip(D.ce, 1e-4, 1 - 1e-4)))
    h, n_top, S = D.h, D.n_top, len(D.systems)
    state = ((D.J + 1) * 4 ** np.arange(len(D.J))[:, None]).sum(0)
    rng = np.random.default_rng([cfg["seed"], 0])
    o, _ = W.simulate_qrels(rng, D.y, D.h, cfg["mnar"]["a"], cfg["mnar"]["b"], cfg["mnar"]["a_fp"], cfg["mnar"]["c"])
    U, truth = D.in_union & ~o, D.u1
LV = np.unique(state)
dummies = (state[:, None] == LV[None, 1:]).astype(float)  # [n, n_states - 1]
z = np.column_stack([dummies, score, h, n_top / S]) if ZK == "pu" else np.column_stack([dummies, score])
x = np.column_stack([h, n_top / S])  # [n, 2]
n = len(o)
print(f"{DS} {ZK}: n {n}, L {o.sum()}, U {U.sum()}, z {z.shape[1]} cols, x {x.shape[1]} cols"
      + (f", truth P(y>=1 | U) {truth[U].mean():.4f}" if truth is not None else ", human HT P(y>=1 | U) .189 [.154, .224]"))
w = np.zeros((n, 3))  # [n, 3] responsibilities P(Y=k | z, x, O)
w[o, 1:] = 0.5
w[~o] = [0.8, 0.1, 0.1]
ks = np.arange(3)
xx = np.vstack([np.column_stack([x, np.zeros(n)]), np.column_stack([x, np.ones(n)])])  # [2n, 3] k=1 rows, k=2 rows
oo = np.concatenate([o, o]).astype(int)
zz, kk = np.repeat(z, 3, axis=0), np.tile(ks, n)
t0, snaps, prev = time.time(), {}, None
for it in range(1, N_MAX + 1):
    ww = w.reshape(-1)
    keep = ww > 1e-8
    m_p = LogisticRegression(C=1.0, max_iter=500).fit(zz[keep], kk[keep], sample_weight=ww[keep])
    p = m_p.predict_proba(z)  # [n, 3]
    we = np.concatenate([w[:, 1], w[:, 2]])
    keep = we > 1e-8
    m_e = LogisticRegression(C=1.0, max_iter=500).fit(xx[keep], oo[keep], sample_weight=we[keep])
    e1 = m_e.predict_proba(np.column_stack([x, np.zeros(n)]))[:, 1]
    e2 = m_e.predict_proba(np.column_stack([x, np.ones(n)]))[:, 1]
    e = np.column_stack([np.zeros(n), e1, e2])  # [n, 3] e(x, k), e(x, 0) = 0
    lik = np.where(o[:, None], p * e, p * (1 - e))
    w = lik / lik.sum(1, keepdims=True)
    if it in CKPT:
        r1 = w[:, 1] + w[:, 2]
        ll = np.log(lik.sum(1)).sum()
        line = (f"{DS} {ZK} it {it:5d} ({time.time() - t0:.0f}s): loglik(O) {ll:.2f}  mean e(x,1) on L {e1[o].mean():.3f}"
                f"  P(y>=1) on U {r1[U].mean():.4f}  max|dP| vs last ckpt "
                f"{np.abs(r1 - prev).max() if prev is not None else float('nan'):.4f}")
        if truth is not None:
            pu = r1[U]
            tu = truth[U]
            order = np.argsort(pu)
            ranks = np.empty(len(pu))
            ranks[order] = np.arange(len(pu))
            npos = tu.sum()
            auc = (ranks[tu > 0].sum() - npos * (npos - 1) / 2) / (npos * (len(tu) - npos))
            line += f"  truth {tu.mean():.4f}  bias {pu.mean() - tu.mean():+.4f}  Brier {((pu - tu) ** 2).mean():.4f}  AUC {auc:.3f}"
        print(line)
        snaps[f"w{it}"] = w.copy()
        prev = r1
np.savez_compressed(f"{cfg['out_dir']}/em_{DS}_{ZK}.npz", o=o, **snaps)
print("DONE")
