"""Judge-strength ladder on D3.

Question: on the same real DuRetrieval qrels, the same 843 human labels and the same two-batch Poisson design, does the
zero-label PU predictor's advantage over the judge predictor grow as the judge weakens?  Only the assessor changes, and
every rung is a real assessor (no noise injection):
  pro+flash, flash, pro   the DeepSeek LLM judges of the main analysis (both, or one of them)
  cos_<encoder>           bi-encoder cosine of (query, passage) from the stored unit-norm embeddings (systems/<encoder>)
  bm25_char1              BM25 score, char-unigram tokens, k1 1.2, b .75 (bm25s, tokenizer t_char1 below),
                          recomputed for every frame item and checked against the stored top-100 scores
  Score judges are cut into grades 0/1/2 at the global frame quantiles that reproduce the mean marginal grade shares of
  the two LLM judges over the frame, so no human label enters a threshold.
Per rung, witness.d3 is left untouched: a copy of the frame namespace carries the rung's J, judge plug-in F and
joint state (same formulas as WD.load_frame), so the GREG state dummies and the stage weights follow the rung.
  [S] strength against humans, weighted by the two-batch HT mixture of scripts/d3/pu_real_qrels.py: AUC on U of the mean
      grade and of P(y>=1) = mean(J>=1); exact / binary agreement of each component judge over all labels.
  [P] zero-label PU (witness.core.pu_fuse, rho 1/2): Z = [rung state dummies, content, h, n_top/S], X = [h, n_top/S],
      content = standardised bge_m3 cosine (e5_small cosine on the bge_m3 rung, so content is never the judge itself).
      pu_nc drops the content column.  *_prior is the fitted P(Y | z) without the selection factor: on U the
      posterior is w_k = p_k (1 - e_k) / sum_j p_j (1 - e_j), so p_k ~ w_k / (1 - e_k).  AUC and calibration on U, AUC
      within the rung's judge states.
  [D] design estimator: a replica of witness.d3.design_estimate that also returns the per-label variance terms,
      asserted to reproduce design_estimate for every predictor under the standard columns.  Predictors: judge,
      judge_cos (judge plug-in with the content column added to the GREG calibration), pu, pu_prior, pu_nc,
      pu_nc_prior; extras [] and [fam].  Certified pairs (BH / BY over the 2P one-sided endpoint nulls), mean SE at
      rho = 0, 1/2, 1, per-pair SE ratio against the judge predictor (median, range) and a bootstrap CI of the median
      (labels resampled with replacement within each batch, common across predictors; fitted GREG predictors and
      stage weights held fixed, so the CI covers the variance-estimate noise, not the refit).
  [R] dose-response across rungs: Spearman / Pearson of judge strength (AUC of the mean grade on U) with the log median
      SE ratio pu / judge, exact permutation p-value of Spearman.
usage: python scripts/d3/judge_ladder.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx>
"""
import glob
import hashlib
import itertools
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
for path in (os.path.abspath(__file__), W.__file__, WD.__file__):
    print(f"md5 {os.path.basename(path)} {hashlib.md5(open(path, 'rb').read()).hexdigest()}")
cfg = load_config(sys.argv[1])
sheets = sys.argv[2:]
assert len(sheets) == 2  # the two-batch HT mixture below
D0 = WD.load_frame(cfg)
batches = [WD.read_batch(D0, t + 1, x) for t, x in enumerate(sheets)]
o, U, S, N = D0.o, ~D0.o, len(D0.systems), D0.N
PD = cfg["duretrieval_dir"]
B_REPS = 2000
rng = np.random.default_rng(0)


def one(pattern):
    hits = glob.glob(pattern)
    assert len(hits) == 1, pattern
    return hits[0]


# ---- labelled items and their HT weights (two-batch mixture, as scripts/d3/pu_real_qrels.py)
b1, b2 = batches
n1, n2 = len(b1.items), len(b2.items)
a = n1 / (n1 + n2)
lab = np.concatenate([b1.items, b2.items])
y = np.concatenate([b1.y, b2.y])
w = np.concatenate([a / b1.pi[b1.items] + (1 - a), (1 - a) / b2.pi[b2.items]])
bat = np.r_[np.zeros(n1, int), np.ones(n2, int)]
assert len(np.unique(lab)) == len(lab)
uu, y1 = U[lab], y >= 1
GROUPS = [np.flatnonzero(bat == k) for k in (0, 1)]
BOOT_IDX = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in GROUPS]) for _ in range(B_REPS)]
MULT = np.stack([np.bincount(i, minlength=len(lab)) for i in BOOT_IDX]).astype(float)  # [B, n_lab]
print(f"frame N={N} (L {o.sum()}, U {U.sum()}); labels {len(lab)} (batch 1 {n1}, batch 2 {n2}; U {uu.sum()}, "
      f"L {(~uu).sum()}); mixture a={a:.3f}")


def boot(stat):
    """Point estimate and percentile 95% CI of stat(idx) over the shared within-batch resamples of the labels."""
    bs = np.array([stat(i) for i in BOOT_IDX])
    return stat(np.arange(len(lab))), np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)


def fmt(t, f="{:.3f}"):
    pt, lo, hi = t
    return f"{f.format(pt)} [{f.format(lo)}, {f.format(hi)}]"


def w_auc(s, yb, ww):
    """Weighted AUC (copied from scripts/d3/pu_real_qrels.py): P(s_pos > s_neg) + 1/2 P(tie), pairs weighted w_pos w_neg."""
    order = np.argsort(s, kind="stable")
    s, yb, ww = s[order], yb[order], ww[order]
    uniq, start = np.unique(s, return_index=True)
    w_pos = np.add.reduceat(ww * yb, start)
    w_neg = np.add.reduceat(ww * ~yb, start)
    below = np.cumsum(w_neg) - w_neg
    return (w_pos * (below + 0.5 * w_neg)).sum() / (w_pos.sum() * w_neg.sum())


def auc_u(score):
    """Weighted AUC of score [N] for 1[y>=1] over the labelled U items, with bootstrap CI."""
    sl = score[lab]
    return boot(lambda i: w_auc(sl[i][uu[i]], y1[i][uu[i]], w[i][uu[i]]))


# ---- features: bi-encoder cosines, BM25 char1 scores, pool proxies
corpus = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/corpus-*.parquet"))
queries = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/queries-*.parquet"))
assert len(queries) == D0.n_q
DI = D0.frame.di.to_numpy()


def cosine(model):
    ids = json.load(open(f"{PD}/systems/{model}/ids.json"))
    assert ids["pids"] == list(corpus["id"]) and ids["qids"] == list(queries["id"])
    q = np.load(f"{PD}/systems/{model}/q.npy").astype(np.float32)  # [n_q, dim], unit norm
    p = np.load(f"{PD}/systems/{model}/p.npy", mmap_mode="r")  # [n_corpus, dim]
    assert q.shape[0] == D0.n_q and p.shape[0] == len(corpus)
    return np.einsum("nd,nd->n", q[D0.qi], np.asarray(p[DI], dtype=np.float32)).astype(float)  # [N]


CJK = re.compile(r"[一-鿿]")


def t_char1(s):
    """char1 tokenizer of the stored BM25 run: CJK runs split into characters, latin/digit runs lower-cased, rest dropped."""
    out = []
    for run in re.findall(r"[一-鿿]+|[A-Za-z0-9]+", s):
        if CJK.match(run[0]):
            out.extend(run)
        else:
            out.append(run.lower())
    return out


def bm25_char1():
    import bm25s

    ret = bm25s.BM25(k1=1.2, b=0.75)
    ret.index([t_char1(s) for s in corpus["text"]], show_progress=False)
    qtok = [t_char1(s) for s in queries["text"]]
    saved = np.load(f"{PD}/systems/bm25/char1.npz")
    out = np.full(N, np.nan)
    rows_of = pd.Series(np.arange(N)).groupby(D0.qi).apply(np.array)
    for q in range(D0.n_q):
        assert len(qtok[q]) > 0
        s = ret.get_scores(qtok[q])  # [n_corpus]
        assert np.allclose(s[saved["idx"][q]], saved["score"][q], rtol=1e-4, atol=1e-4), q
        out[rows_of[q]] = s[DI[rows_of[q]]]
    assert np.isfinite(out).all()
    return out


COS = {m: cosine(m) for m in ("bge_m3", "e5_small", "text2vec_zh")}
BM25 = bm25_char1()
print("BM25 char1 recomputed for every frame item; stored top-100 scores reproduced for all queries")
STD = {k: (v - v.mean()) / v.std() for k, v in COS.items()}
H_LEX = (D0.disc[np.array([s.startswith("bm25") for s in D0.systems])] > 0).mean(0)  # [N]
N_TOP = (D0.disc > 0).sum(0)  # [N] in 1..7
X_PU = np.column_stack([H_LEX, N_TOP / S])
SHARE = np.mean([np.bincount(D0.frame[j], minlength=3) / N for j in cfg["judges"]], axis=0)
print(f"LLM mean marginal grade shares on the frame (target of the score-judge thresholds): {np.round(SHARE, 4).tolist()}")


def grade3(s):
    t = np.quantile(s, np.cumsum(SHARE)[:2])
    g = np.digitize(s, t)  # 0 below t1, 1 in [t1, t2), 2 at or above t2
    print(f"   thresholds {np.round(t, 4).tolist()} -> shares {np.round(np.bincount(g, minlength=3) / N, 4).tolist()}")
    return g


RUNGS = {}  # name -> (J [n_judges, N], content name, raw continuous score or None)
RUNGS["pro+flash"] = (D0.J, "bge_m3", None)
for j_i, j in enumerate(cfg["judges"]):
    RUNGS[j] = (D0.J[j_i:j_i + 1], "bge_m3", None)
for m in COS:
    print(f"cos_{m}:")
    RUNGS[f"cos_{m}"] = (grade3(COS[m])[None], "e5_small" if m == "bge_m3" else "bge_m3", COS[m])
print("bm25_char1:")
RUNGS["bm25_char1"] = (grade3(BM25)[None], "bge_m3", BM25)
for m, v in COS.items():
    print(f"[S] content/score {m} cosine (continuous) AUC on U {fmt(auc_u(v))}")
print(f"[S] bm25_char1 score (continuous) AUC on U {fmt(auc_u(BM25))}")


def rung_frame(J):
    """Copy of D0 with the rung's judges: plug-in F and joint state as in WD.load_frame."""
    F0, F1 = (J == 2).mean(0), (J >= 1).mean(0)
    F = [np.where(o, 1.0, F0), np.where(o, 1.0, 0.5 * F0 + 0.5 * F1), np.where(o, 1.0, F1)]
    state = ((J + 1) * 4 ** np.arange(len(J))[:, None]).sum(0)
    return SimpleNamespace(**{**vars(D0), "J": J, "F": F, "state": state})


_chk = rung_frame(D0.J)
assert (_chk.state == D0.state).all() and all((a_ == b_).all() for a_, b_ in zip(_chk.F, D0.F))


def endpoints(p2, p1):
    return [p2, 0.5 * p2 + 0.5 * p1, p1]


def pu_fit(Dr, content):
    """Zero-label PU with the rung's state dummies.  Returns posterior and prior endpoint predictors and e(x, k)."""
    lev = np.unique(Dr.state)
    cols = [(Dr.state[:, None] == lev[None, 1:]).astype(float)]
    if content is not None:
        cols.append(content[:, None])
    Z = np.column_stack(cols + [X_PU])
    _, _, e, pw = W.pu_fuse(o, Z, X_PU, np.array([], int), np.array([], int), 0.5)
    assert (e[:, 0] == 0).all() and (e[:, 1:] < 1).all()
    pr = pw / (1 - e)  # valid on U only (O = 0); L values are unused by the design (its column is o)
    pr /= pr.sum(1, keepdims=True)
    post = endpoints(pw[:, 2], pw[:, 1] + pw[:, 2])
    prior = [np.where(o, f_post, f_pr) for f_post, f_pr in zip(post, endpoints(pr[:, 2], pr[:, 1] + pr[:, 2]))]
    return post, prior, e


def within_state_auc(Dr, score):
    """State-weighted AUC of score inside the rung's judge states on U (states with >= 5 positive and negative labels)."""
    st, sl = Dr.state[lab], score[lab]
    cells = [s for s in np.unique(Dr.state) if min((uu & (st == s) & y1).sum(), (uu & (st == s) & ~y1).sum()) >= 5]

    def stat(i):
        num = den = 0.0
        for s in cells:
            m = uu[i] & (st[i] == s)
            if y1[i][m].all() or not y1[i][m].any():
                return np.nan
            wt = w[uu & (st == s)].sum()
            num, den = num + wt * w_auc(sl[i][m], y1[i][m], w[i][m]), den + wt
        return num / den

    return boot(stat), len(cells)


def design_parts(Dr, F, cols):
    """Replica of WD.design_estimate (single predictor F for both batches) that also returns per-label variance terms.
    cols(f) -> (Zc, b0).  Returns est, var: lists over blocks of [R]; con: list over blocks of [R, n_lab] in `lab` order
    with con[b].sum(1) == var[b]."""
    pi_lab = np.concatenate([bt.pi[bt.items] for bt in batches])
    est_out, var_out, con_out = [], [], []
    for b in range(3):
        u_all = np.zeros(N)
        u_all[lab] = np.concatenate([WD.gains(bt.y)[b] for bt in batches])
        Zc, b0 = cols(F[b])
        H = W.greg_fit(Dr.G, Zc, u_all, lab, pi_lab, b0, cfg["ridge"]) @ Zc.T  # [R, N]
        num = den = vnum = 0.0
        known = np.zeros(N, bool)
        terms = []
        for t, bt in enumerate(batches):
            xi = np.zeros(N, bool)
            xi[bt.items] = True
            e_t, v_t = W.stage_estimate(Dr.G, known * u_all, np.zeros(N, bool), H, u_all, xi, bt.pi, Dr.n_q)
            m2 = WD.working_var(Dr, b, batches[:t])[None, :] + 0.0 * H
            om = 1.0 / np.sqrt(np.maximum(W.predicted_var(Dr.G, bt.pi, m2, Dr.n_q), 1e-16))
            pk = bt.pi[bt.items]
            ct = Dr.G[:, bt.items] ** 2 * ((1 - pk) / pk ** 2 * (u_all[bt.items] - H[:, bt.items]) ** 2) / Dr.n_q ** 2
            assert np.allclose(ct.sum(1), v_t, rtol=1e-9, atol=0.0)
            terms.append((om, ct))
            num, den, vnum = num + om * e_t, den + om, vnum + om ** 2 * v_t
            known |= xi
        est_out.append(num / den)
        var_out.append(vnum / den ** 2)
        con_out.append(np.concatenate([(om_ ** 2 / den ** 2)[:, None] * ct_ for om_, ct_ in terms], axis=1))
        assert np.allclose(con_out[-1].sum(1), var_out[-1], rtol=1e-9, atol=0.0)
    return est_out, var_out, con_out


def certify(est, var):
    """Endpoint intersection-union + BH / BY over the 2P one-sided nulls (as scripts/d3/pu_real_qrels.py)."""
    se = [np.sqrt(np.maximum(v[:D0.P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:D0.P] / se[0]), norm.sf(est[2][:D0.P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:D0.P] / se[0]), norm.cdf(est[2][:D0.P] / se[2]))
    return {name: W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"], dependent=dep).reshape(2, D0.P)
            for name, dep in (("BH", False), ("BY", True))}


def se_ratio(alt, base):
    """Per-pair SE ratio alt / base per block: (median, min, max, bootstrap CI of the median)."""
    out = []
    for b in range(3):
        r = np.sqrt(alt["var"][b][:D0.P] / base["var"][b][:D0.P])
        rb = np.sqrt((MULT @ alt["con"][b][:D0.P].T) / (MULT @ base["con"][b][:D0.P].T))  # [B, P]
        med = np.median(rb, axis=1)
        out.append((np.median(r), r.min(), r.max(), np.quantile(med, 0.025), np.quantile(med, 0.975)))
    return out


TABLE = {}
PRED_NAMES = ("judge", "judge_cos", "pu", "pu_prior", "pu_nc", "pu_nc_prior")
for name, (J, cname, raw) in RUNGS.items():
    print(f"==== rung {name}: {len(J)} judge(s), content feature {cname}")
    Dr = rung_frame(J)
    content = STD[cname]
    # [S] strength
    grade = J.mean(0).astype(float)
    s_grade, s_f1 = auc_u(grade), auc_u(Dr.F[2])
    print(f"[S] {name}: AUC on U mean grade {fmt(s_grade)}; P(y>=1) plug-in {fmt(s_f1)}; states {len(np.unique(Dr.state))}")
    for k in range(len(J)):
        jl = J[k][lab]
        ex = boot(lambda i, jl=jl: np.average(jl[i] == y[i], weights=w[i]))
        bi = boot(lambda i, jl=jl: np.average((jl[i] >= 1) == y1[i], weights=w[i]))
        print(f"[S] {name} component {k}: weighted exact agreement {fmt(ex)}, binary {fmt(bi)} (all labels)")
    if raw is not None:
        print(f"[S] {name} continuous score before grading: AUC on U {fmt(auc_u(raw))}")
    # [P] PU fits
    pu, pu_prior, e = pu_fit(Dr, content)
    pu_nc, pu_nc_prior, e_nc = pu_fit(Dr, None)
    preds = dict(judge=Dr.F, judge_cos=Dr.F, pu=pu, pu_prior=pu_prior, pu_nc=pu_nc, pu_nc_prior=pu_nc_prior)
    for ename, ee in (("pu", e), ("pu_nc", e_nc)):
        print(f"[P] {name} {ename}: e(x,1) on U mean {ee[U, 1].mean():.3f} range [{ee[U, 1].min():.3f}, "
              f"{ee[U, 1].max():.3f}]; e(x,2) mean {ee[U, 2].mean():.3f}")
    row = dict(strength=s_grade, strength_f1=s_f1)
    for pn in ("pu", "pu_prior", "pu_nc", "pu_nc_prior", "judge"):
        p1 = preds[pn][2]
        au = auc_u(p1)
        cal = np.average((p1[lab] - y1)[uu], weights=w[uu])
        (ws, n_cells) = within_state_auc(Dr, p1) if pn != "judge" else ((np.nan, np.nan, np.nan), 0)
        row[f"auc_{pn}"] = au
        print(f"[P] {name} {pn:12s} AUC on U {fmt(au)}; mean P(y>=1) on U {p1[U].mean():.3f}; mean(p - y) on labelled U "
              f"{cal:+.3f}" + (f"; AUC within judge states ({n_cells} states) {fmt(ws)}" if pn != "judge" else ""))
    print(f"[P] {name}: corr(posterior, prior) of P(y>=1) on U: pu {np.corrcoef(pu[2][U], pu_prior[2][U])[0, 1]:.4f}, "
          f"pu_nc {np.corrcoef(pu_nc[2][U], pu_nc_prior[2][U])[0, 1]:.4f}")
    # [D] design estimator
    for extras in ([], ["fam"]):
        tag_x = "+fam" if extras else ""
        res = {}
        for pn in PRED_NAMES:
            if pn == "judge_cos":
                def cols(f, extras=extras):
                    Zc, b0 = WD.design_columns(Dr, 0, extras, f)
                    return np.column_stack([Zc, np.where(U, content - content[U].mean(), 0.0)]), np.r_[b0, 0.0]
            else:
                def cols(f, extras=extras):
                    return WD.design_columns(Dr, 0, extras, f)
            est, var, con = design_parts(Dr, preds[pn], cols)
            if pn != "judge_cos":
                est_r, var_r = WD.design_estimate(Dr, batches, extras, cfg["ridge"], preds[pn])
                assert all(np.allclose(a_, b_, rtol=1e-10, atol=1e-14) for a_, b_ in zip(est, est_r))
                assert all(np.allclose(a_, b_, rtol=1e-10, atol=1e-16) for a_, b_ in zip(var, var_r))
            c = certify(est, var)
            res[pn] = dict(est=est, var=var, con=con, cert=c)
            se = [np.sqrt(v[:D0.P]).mean() for v in var]
            print(f"[D] {name} {pn + tag_x:16s} certified BH {int(c['BH'].sum())}/{D0.P}, BY {int(c['BY'].sum())}/{D0.P}; "
                  f"mean SE rho=0 {se[0]:.4f}, 1/2 {se[1]:.4f}, 1 {se[2]:.4f}")
            row[f"cert{tag_x}_{pn}"] = int(c["BH"].sum())
            row[f"se{tag_x}_{pn}"] = se
        for pn in PRED_NAMES[1:]:
            rr = se_ratio(res[pn], res["judge"])
            flips = int((res[pn]["cert"]["BH"] != res["judge"]["cert"]["BH"]).any(0).sum())
            print(f"[D] {name} SE ratio {pn + tag_x} / judge{tag_x} per pair: " + "; ".join(
                f"rho={rl} median {m:.3f} [{lo:.3f}, {hi:.3f}] boot CI [{cl:.3f}, {ch:.3f}]"
                for rl, (m, lo, hi, cl, ch) in zip(("0", "1/2", "1"), rr)) + f"; BH decisions differing: {flips}")
            row[f"ratio{tag_x}_{pn}"] = rr
    TABLE[name] = row

# ---- [R] summary table and dose-response
print("==== [T] table (extras []; ratio = median per-pair SE ratio vs judge at rho=1/2 with bootstrap CI)")
print("rung | judge AUC(grade) | judge AUC(P>=1) | PU AUC | PU_nc AUC | ratio pu | ratio pu_prior | ratio pu_nc | "
      "ratio judge_cos | cert judge/pu/pu_nc/judge_cos | cert +fam judge/pu")
for name, r in TABLE.items():
    rt = {pn: r[f"ratio_{pn}"][1] for pn in PRED_NAMES[1:]}
    print(f"{name} | {fmt(r['strength'])} | {r['strength_f1'][0]:.3f} | {fmt(r['auc_pu'])} | {r['auc_pu_nc'][0]:.3f} | "
          + " | ".join(f"{rt[pn][0]:.3f} [{rt[pn][3]:.3f}, {rt[pn][4]:.3f}]" for pn in ("pu", "pu_prior", "pu_nc", "judge_cos"))
          + f" | {r['cert_judge']}/{r['cert_pu']}/{r['cert_pu_nc']}/{r['cert_judge_cos']} | "
            f"{r['cert+fam_judge']}/{r['cert+fam_pu']}")
names = list(TABLE)
strength = np.array([TABLE[n]["strength"][0] for n in names])


def spearman_perm(xv, yv):
    rx, ry = rankdata(xv), rankdata(yv)
    obs = np.corrcoef(rx, ry)[0, 1]
    perm = np.array([np.corrcoef(rx, ry[list(p)])[0, 1] for p in itertools.permutations(range(len(ry)))])
    return obs, (np.abs(perm) >= abs(obs) - 1e-12).mean()


for pn in ("pu", "pu_nc", "pu_prior", "judge_cos"):
    for b, rl in enumerate(("0", "1/2", "1")):
        g = np.log(np.array([TABLE[n][f"ratio_{pn}"][b][0] for n in names]))
        rs, pp = spearman_perm(strength, g)
        pr_, pp_ = pearsonr(strength, g)
        print(f"[R] {pn} rho={rl}: Spearman(judge AUC, log SE ratio) {rs:+.3f} (exact permutation p {pp:.4f}), "
              f"Pearson {pr_:+.3f} (p {pp_:.4f}); slope of log ratio per +.1 AUC "
              f"{np.polyfit(strength, g, 1)[0] * 0.1:+.4f}")
print("DONE")
