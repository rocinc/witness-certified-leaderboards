"""Trajectory of the zero-label PU EM (a line-for-line copy of the witness.core.pu_fuse loop, lab_w=None, no labels)
with the penalised observed-data log-likelihood, mean e(x,1), mean P(y>=1) on U, and the coefficient norms, every 30
iterations to 600.  D3 (real qrels) and D2 rep 0 (simulated qrels, mnar a=-2.5, b=4, a_fp=-6.5, c=0).
usage: python scripts/diagnostics/em_trajectory.py configs/d3_duretrieval.yaml configs/testbeds/d2_dl19.yaml \
           configs/eval/adaptive_d2.yaml
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402
from witness.data import load_testbed  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
assert len(sys.argv) == 4


def traj(o, z, x, U, tag, truth_u=None, n_blocks=20):
    """Run the EM for 30 * n_blocks iterations from the pu_fuse initial responsibilities; print every 30.
    o [n] bool qrels membership, z [n, p], x [n, q], U [n] bool unlabelled-frame mask, truth_u [n] (None on D3)."""
    n = len(o)
    w = np.zeros((n, 3))  # [n, 3] responsibilities P(Y=k | z, x, O)
    w[o, 1:] = 0.5
    w[~o] = [0.8, 0.1, 0.1]
    ks = np.arange(3)
    for it in range(1, 30 * n_blocks + 1):
        zz = np.repeat(z, 3, axis=0)
        kk = np.tile(ks, n)
        ww = w.reshape(-1)
        keep = ww > 1e-8
        m_p = LogisticRegression(C=1.0, max_iter=500).fit(zz[keep], kk[keep], sample_weight=ww[keep])
        p = m_p.predict_proba(z)  # [n, 3]
        xx = np.vstack([np.column_stack([x, np.zeros(n)]), np.column_stack([x, np.ones(n)])])
        oo = np.concatenate([o, o]).astype(int)
        we = np.concatenate([w[:, 1], w[:, 2]])
        keep = we > 1e-8
        m_e = LogisticRegression(C=1.0, max_iter=500).fit(xx[keep], oo[keep], sample_weight=we[keep])
        e = np.zeros((n, 3))  # [n, 3] e(x, k), e(x, 0) = 0
        e[:, 1] = m_e.predict_proba(np.column_stack([x, np.zeros(n)]))[:, 1]
        e[:, 2] = m_e.predict_proba(np.column_stack([x, np.ones(n)]))[:, 1]
        lik = np.where(o[:, None], p * e, p * (1 - e))
        w = lik / lik.sum(1, keepdims=True)
        if it % 30 == 0:
            ll = np.log(lik.sum(1)).sum()
            r1 = w[:, 1] + w[:, 2]
            line = (f"{tag} it {it:4d}: loglik(O) {ll:.2f}  mean e(x,1) {e[:, 1].mean():.3f} e on L {e[o, 1].mean():.3f}  "
                    f"mean P(y>=1|z) all {(p[:, 1] + p[:, 2]).mean():.4f}  posterior P(y>=1) on U {r1[U].mean():.4f}  "
                    f"|theta| {np.abs(m_p.coef_).sum():.2f} |phi| {np.abs(m_e.coef_).sum():.2f} b_e {m_e.intercept_[0]:+.2f}")
            if truth_u is not None:
                line += f"  truth P(y>=1|U) {truth_u[U].mean():.4f}"
            print(line)


cfg = load_config(sys.argv[1])
D = WD.load_frame(cfg)
q_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/q.npy").astype(np.float32)
p_emb = np.load(f"{cfg['duretrieval_dir']}/systems/bge_m3/p.npy", mmap_mode="r")
di = D.frame.di.to_numpy()
cos = np.einsum("nd,nd->n", q_emb[D.qi], p_emb[di].astype(np.float32)).astype(float)  # [N]
COS = (cos - cos.mean()) / cos.std()
lexm = np.array([s.startswith("bm25") for s in D.systems])
H = (D.disc[lexm] > 0).mean(0)
NT = (D.disc > 0).sum(0)
S = len(D.systems)
LV = np.unique(D.state)
Z = np.column_stack([(D.state[:, None] == LV[None, 1:]).astype(float), COS, H, NT / S])
X = np.column_stack([H, NT / S])
traj(D.o.astype(bool), Z, X, ~D.o, "D3")

base, c2 = load_config(sys.argv[2]), load_config(sys.argv[3])
T = load_testbed(base, c2)
JP = ((T.J + 1) * 4 ** np.arange(len(T.J))[:, None]).sum(0)
L2 = np.unique(JP)
LCE = np.log(np.clip(T.ce, 1e-4, 1 - 1e-4) / (1 - np.clip(T.ce, 1e-4, 1 - 1e-4)))
Z2 = np.column_stack([(JP[:, None] == L2[None, 1:]).astype(float), LCE, T.h, T.n_top / len(T.systems)])
X2 = np.column_stack([T.h, T.n_top / len(T.systems)])
rng = np.random.default_rng([c2["seed"], 0])
o2, _ = W.simulate_qrels(rng, T.y, T.h, -2.5, 4.0, -6.5, 0.0)
traj(o2, Z2, X2, T.in_union & ~o2, "D2", truth_u=T.u1, n_blocks=10)
print("DONE")
