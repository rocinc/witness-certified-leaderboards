"""M4: item-level sequential certification with anytime-valid e-values, against fixed-n z tests
and the naive 'peek and keep' use of z tests.

Labels are bought one item at a time by PPS without replacement from the shrinking frame (labelled items leave it).
Within a batch of `batch` draws the weights w are fixed; draw j has q_j(d) = w_d / W_j over the remaining frame.
For every row r (pair contrasts and systems, three rho blocks) the increment
  Z_j = sum_{known} g u + sum_{frame} g H + g_{d_j} (u_{d_j} - H_{d_j}) / q_j(d_j)
has E[Z_j | past] = Q Delta for any predictable q, H; H is a GREG fit on labels of earlier batches only (predictable).
Evidence per checkpoint: (a) z test with the running mean and variance of Z; (b) 'e': the predictable plug-in
betting e-process (witness.core.bet_step), an exact nonnegative supermartingale under H0 because every increment has a
sure lower bound known before its draw (below); (c) 'mix': the one-sided Gaussian-mixture process with running
variance (witness.core.log_e_onesided), an asymptotic confidence sequence only, kept for comparison.  Endpoint
intersection-union = min over rho in {0,1}; certification by e-BH.
Sure bound of draw j for row r: Z_j >= (known + frame)/Q + min(0, min_d m_d / w_d)/Q, m_d = min over u in {0,1} of
g_d (u - H_d), over the batch-start frame (a superset of the current one) and with q_j(d) = w_d / W_j >= w_d.
Peeking: a pair certified at any checkpoint stays certified ('union'); for z tests this is the usual misuse.
usage: python scripts/eval/sequential.py <testbed.yaml> <eval.yaml> <seq.yaml> [rep_lo rep_hi]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from witness.config import load_config  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
SEQ = load_config(sys.argv[3])
REPS = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, SEQ["reps"])
sys.argv = sys.argv[:3]
__file__ = os.path.join(here, "evaluate.py")
_src = open(__file__).read()
exec(compile(_src[: _src.index("\nrows = []\n")], __file__, "exec"))
G_ = G  # noqa: F821  [R, n]
assert SEQ["checkpoints"][-1] % SEQ["batch"] == 0 and all(c % SEQ["batch"] == 0 for c in SEQ["checkpoints"])


def stream(o, pu_w, rng, spec):
    """One label stream up to the last checkpoint.  Returns, per checkpoint, est/var per block and log e-values."""
    F, V = base_predictors(spec["f"], o, pu_w)  # noqa: F821
    designs = [design(F[b], o, spec["strata"], spec["extras"]) for b in range(3)]  # noqa: F821
    frame = D.in_union.copy()  # noqa: F821
    known_sum = np.zeros((3, R))  # noqa: F821  sum over labelled items of g u
    lab, pi_lab = [], []
    sZ, sZ2 = np.zeros((3, R)), np.zeros((3, R))  # noqa: F821
    # betting state per block, row and direction (+: H0 Delta <= 0 on Z; -: H0 Delta >= 0 on -Z)
    bet = {s: [np.zeros((3, R)) for _ in range(3)] for s in ("plus", "minus")}  # noqa: F821  (logK, s1, s2)
    n_drawn = 0
    out = []
    betas = [np.tile(designs[b][1], (R, 1)) for b in range(3)]  # noqa: F821
    H = cd = None
    for start in range(0, SEQ["checkpoints"][-1], SEQ["batch"]):
        if start % SEQ["refit_every"] == 0:
            # predictor and design weights change only here (both depend on earlier batches only)
            if len(lab) > 0 and spec["lam"] == "prev":
                li, pl = np.array(lab), np.array(pi_lab)
                betas = [W.greg_fit(G_, designs[b][0], U_END[b], li, pl, designs[b][1], cfg["ridge"]) for b in range(3)]  # noqa: F821
            H = [betas[b] @ designs[b][0].T for b in range(3)]  # [R, n]
            # Neyman weights on the pair rows of both endpoints
            m2 = [working_m2(b, F[b], V[b], H[b], o, np.array(lab, int), np.array(pi_lab)) for b in (0, 2)]  # noqa: F821
            cd = sum((G_[:P] ** 2 * m[:P]).sum(0) for m in m2)  # noqa: F821
        fidx = np.flatnonzero(frame)
        # this batch's weights: Neyman restricted to the remaining frame, mixed with uniform for positivity
        w = np.sqrt(np.maximum(cd[fidx], 0.0))
        w = (1 - SEQ["mix"]) * w / w.sum() + SEQ["mix"] / len(fidx)
        # PPS without replacement for one batch: exponential keys
        keys = rng.exponential(size=len(fidx)) / w
        take = np.argsort(keys)[: SEQ["batch"]]
        d = fidx[take]
        wd = w[take]
        W_before = 1.0 - np.concatenate([[0.0], np.cumsum(wd)[:-1]])  # remaining weight before each draw
        q = wd / W_before  # [b]
        for b in range(3):
            frame_sum = (G_[:, fidx] * H[b][:, fidx]).sum(1)  # [R] at batch start
            gu = G_[:, d] * U_END[b][d][None, :]  # [R, b]
            gh = G_[:, d] * H[b][:, d]  # [R, b]
            # before draw j: known += sum_{i<j} g u, frame -= sum_{i<j} g H
            known_before = known_sum[b][:, None] + np.concatenate([np.zeros((R, 1)), np.cumsum(gu, 1)[:, :-1]], 1)  # noqa: F821
            frame_before = frame_sum[:, None] - np.concatenate([np.zeros((R, 1)), np.cumsum(gh, 1)[:, :-1]], 1)  # noqa: F821
            Z = (known_before + frame_before + (gu - gh) / q[None, :]) / n_q  # noqa: F821  [R, b], mean Delta
            gf, hf = G_[:, fidx], H[b][:, fidx]  # [R, |frame|]
            m_lo = np.minimum(-gf * hf, gf * (1.0 - hf)) / w[None, :]
            m_hi = np.maximum(-gf * hf, gf * (1.0 - hf)) / w[None, :]
            base = (known_before + frame_before) / n_q  # noqa: F821  [R, b]
            lo = base + np.minimum(0.0, m_lo.min(1))[:, None] / n_q  # noqa: F821
            hi = base + np.maximum(0.0, m_hi.max(1))[:, None] / n_q  # noqa: F821
            assert (Z >= lo - 1e-9).all() and (Z <= hi + 1e-9).all()
            for j in range(Z.shape[1]):
                for s, zj, lj in (("plus", Z[:, j], lo[:, j]), ("minus", -Z[:, j], -hi[:, j])):
                    st = bet[s]
                    st[0][b], st[1][b], st[2][b] = W.bet_step(st[0][b], st[1][b], st[2][b], zj, lj, SEQ["cap"])
            sZ[b] += Z.sum(1)
            sZ2[b] += (Z ** 2).sum(1)
            known_sum[b] += gu.sum(1)
        frame[d] = False
        lab.extend(d.tolist())
        pi_lab.extend(np.minimum(SEQ["batch"] * wd, 1.0).tolist())  # approximate inclusion for the GREG fit only
        n_drawn += len(d)
        if n_drawn in SEQ["checkpoints"]:
            mean = sZ / n_drawn
            var_inc = np.maximum(sZ2 / n_drawn - mean ** 2, 1e-30)
            out.append(dict(n=n_drawn, est=mean, var=var_inc / n_drawn,
                            lb_plus=bet["plus"][0].copy(), lb_minus=bet["minus"][0].copy(),
                            le_plus=W.log_e_onesided(sZ - 0.0, n_drawn, var_inc, SEQ["n_target"]),
                            le_minus=W.log_e_onesided(-sZ, n_drawn, var_inc, SEQ["n_target"])))
    return out


def certify_seq(ck):
    """z-test BH (endpoint IUT, as certify()) and e-BH on min-over-endpoints e-values; returns certified A>B / B>A."""
    se = np.sqrt(ck["var"][:, :P])  # noqa: F821
    est = ck["est"][:, :P]  # noqa: F821
    p_plus = np.maximum(norm.sf(est[0] / se[0]), norm.sf(est[2] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0] / se[0]), norm.cdf(est[2] / se[2]))
    rz = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"])  # noqa: F821
    res = {"z": (rz[:P], rz[P:])}  # noqa: F821
    for k, kp, km in (("e", "lb_plus", "lb_minus"), ("mix", "le_plus", "le_minus")):
        e_plus = np.exp(np.minimum(ck[kp][0, :P], ck[kp][2, :P]))  # noqa: F821
        e_minus = np.exp(np.minimum(ck[km][0, :P], ck[km][2, :P]))  # noqa: F821
        re = W.ebh(np.concatenate([e_plus, e_minus]), cfg["alpha"])  # noqa: F821
        res[k] = (re[:P], re[P:])  # noqa: F821
    return res


t0_, t1_ = TRUTH[0][:P], TRUTH[2][:P]  # noqa: F821
plus_true = np.minimum(t0_, t1_) <= 0
minus_true = np.maximum(t0_, t1_) >= 0
decidable = ~plus_true | ~minus_true
rows_out = []
for rep in range(*REPS):
    tt = time.time()
    rng = np.random.default_rng([cfg["seed"], rep, 7])  # noqa: F821
    o, _ = W.simulate_qrels(rng, D.y, D.h, cfg["mnar"]["a"], cfg["mnar"]["b"], cfg["mnar"]["a_fp"], cfg["mnar"]["c"])  # noqa: F821
    _, _, _, pu_w = W.pu_fuse(o, Z_PU, X_PU, np.array([], int), np.array([], int), 0.5)  # noqa: F821
    for sname in SEQ["specs"]:
        spec = dict(BASE, **SEQ["specs"][sname])  # noqa: F821
        cks = stream(o, pu_w, rng, spec)
        union = {k: (np.zeros(P, bool), np.zeros(P, bool)) for k in ("z", "e", "mix")}  # noqa: F821
        for ck in cks:
            c = certify_seq(ck)
            for k in ("z", "e", "mix"):
                rp, rm = c[k]
                union[k] = (union[k][0] | rp, union[k][1] | rm)
                if k == "z":  # unbiasedness / variance check of the sequential estimator at rho = 1/2
                    zz = (ck["est"][1] - TRUTH[1]) / np.sqrt(ck["var"][1])  # noqa: F821
                    rows_out.append(dict(rep=rep, spec=sname, n=ck["n"], test="check", mode="rho_half",
                                         cert_dec=np.mean(np.abs(zz[:P]) <= 1.96), fdp=zz[:P].mean(), n_wrong=zz[:P].std()))  # noqa: F821
                for mode, (ap, am) in (("now", (rp, rm)), ("union", union[k])):
                    wrong = (ap & plus_true) | (am & minus_true)
                    n_rej = int(ap.sum() + am.sum())
                    rows_out.append(dict(rep=rep, spec=sname, n=ck["n"], test=k, mode=mode,
                                         cert_dec=((ap & ~plus_true) | (am & ~minus_true)).sum() / decidable.sum(),
                                         fdp=wrong.sum() / max(n_rej, 1), n_wrong=int(wrong.sum())))
    print(f"rep {rep} done {time.time() - tt:.0f}s", flush=True)
    pd.DataFrame(rows_out).to_csv(f"{cfg['out_dir']}/seq_{SEQ['tag']}_r{REPS[0]}-{REPS[1]}_raw.csv", index=False)  # noqa: F821
df = pd.DataFrame(rows_out)
tab = df.groupby(["spec", "test", "mode", "n"])[["cert_dec", "fdp", "n_wrong"]].mean()
pd.set_option("display.width", 200)
print(tab.round(4).to_string())
print("DONE")
