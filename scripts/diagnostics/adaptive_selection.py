"""Why does the adaptive witness refit drift?  Simulation with known truth.

Per repetition: simulated qrels o (witness.core.simulate_qrels, the mechanism of the eval config), zero-label PU fit
(witness.core.pu_fuse), then one stage-1 human sample of n1 = budget * fracs[0] items drawn two ways:
  decision  the stage-1 plan of witness_adapt in scripts/eval/evaluate.py::run (t = 0: decision-aware allocation from
            the zero-label PU predictor, plan_iters certification-weight updates, pi clipped at pi_min_frac)
  uniform   pi = n1 / |frame| over the same frame (L and U)
Each sample refits PU with variants (all warm-started from the matching zero-label fit, 30 EM iterations):
  cont      no label (zero-label EM continued for the same 30 iterations: the reference for the label effect)
  unw / w   labels with weight 1 / 1/pi (design-weighted composite likelihood, pu_fuse lab_w)
  scaled    every label with the constant weight mean(1/pi): the same total label weight as w, no design information
            (scaled = w under the uniform design)
  +fam      z gains the BM25 / dense / rerank hit shares (FAMX of scripts/eval/evaluate.py)
  +sys      z gains every system's top-k hit and discount: the decision pi is a deterministic function of
            (z_sys, x, O), so selection is exactly ignorable given z_sys
  C0.1      both M-step regressions with C = 0.1 (10x the L2 strength)
  census    every frame label, weight 1 (the refit's ceiling and the census pseudo-true fit)
Evaluation on U items of the frame not in the sample (truth u1 = 1[y>=1] known): mean(p - u1), Brier, ECE (10 bins),
mean(p - p_census), and the label effect mean(p - p_cont).  Selection diagnostic: on the sampled U items, mean(u1 - p_census) unweighted and
1/pi-weighted (Hajek) against its frame value; a gap between unweighted and frame means the sample over-represents
items whose label departs from the census fit given z (informative selection given z).
usage: python scripts/diagnostics/adaptive_selection.py <testbed.yaml> <eval.yaml> rep_lo rep_hi <budgets, e.g. 200,600>
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from witness import core as W  # noqa: E402
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
base = load_config(sys.argv[1])
cfg = load_config(sys.argv[2])
REP_LO, REP_HI = int(sys.argv[3]), int(sys.argv[4])
print("config:", cfg)
for path in (os.path.abspath(__file__), W.__file__):
    print("md5", os.path.basename(path), hashlib.md5(open(path, "rb").read()).hexdigest())
D = load_testbed(base, cfg)
G, P, n_q = D.G, D.P, D.n_q
R, n = G.shape[0], len(D.y)
U_END = [D.u0, 0.5 * D.u0 + 0.5 * D.u1, D.u1]
ans = D.J >= 0
LCE = np.log(np.clip(D.ce, 1e-4, 1 - 1e-4) / (1 - np.clip(D.ce, 1e-4, 1 - 1e-4)))
JPAT = ((D.J + 1) * 4 ** np.arange(len(D.J))[:, None]).sum(0)
JPAT_LEVELS = np.unique(JPAT)
Z_PU = np.column_stack([(JPAT[:, None] == JPAT_LEVELS[None, 1:]).astype(float), LCE, D.h, D.n_top / len(D.systems)])
X_PU = np.column_stack([D.h, D.n_top / len(D.systems)])
FAMILY = np.array(["bm25" if s.startswith("bm25") else "rerank" if s.startswith("rerank") else "dense" for s in D.systems])
FAMX = np.column_stack([(D.disc[FAMILY == k] > 0).mean(0) for k in ("bm25", "dense", "rerank")])  # [n, 3]
SYSX = np.column_stack([c for s in range(len(D.systems)) for c in ((D.disc[s] > 0).astype(float), D.disc[s])])
Z = {"": (Z_PU, 1.0), "fam": (np.column_stack([Z_PU, FAMX]), 1.0), "sys": (np.column_stack([Z_PU, SYSX]), 1.0),
     "C0.1": (Z_PU, 0.1)}
BUDGETS = [int(b) for b in sys.argv[5].split(",")]  # subset of cfg budgets (compute)
assert set(BUDGETS) <= set(cfg["budgets"])
SHIFT = None
if cfg["mnar"]["shift"] is not None:  # as scripts/eval/evaluate.py
    _fj = np.where(ans, D.J, 0).sum(0) / np.maximum(ans.sum(0), 1)
    _zs = lambda v: (v - v[D.in_union].mean()) / v[D.in_union].std()  # noqa: E731
    SHIFT = cfg["mnar"]["shift"]["judge"] * _zs(_fj) + cfg["mnar"]["shift"]["ce"] * _zs(LCE)
FR = D.in_union
print(f"items {n} frame {FR.sum()} pairs {P}; Z_PU {Z_PU.shape[1]} cols, +fam {Z['fam'][0].shape[1]}, "
      f"+sys {Z['sys'][0].shape[1]}; shift {'on' if SHIFT is not None else 'off'}")
Gsq = G ** 2
EDGES = np.linspace(0, 1, 11)


def decision_pi(F, V, budget):
    """Stage-1 inclusion probabilities of witness_adapt (scripts/eval/evaluate.py::run at t = 0, no labels yet:
    betas = b0, so H = F and m2 = max(V, v_floor))."""
    fracs = cfg["fracs"]
    n_t = budget * fracs[0]
    m2 = [np.broadcast_to(np.maximum(V[b], cfg["v_floor"])[None, :], (R, n)) for b in range(3)]
    pi_min = cfg["pi_min_frac"] * n_t / FR.sum()
    w_rows = [np.ones(P), np.ones(P)]
    for _ in range(cfg["plan_iters"]):
        cd = sum((w[:, None] * Gsq[:P] * m2_[:P]).sum(0) for w, m2_ in zip(w_rows, [m2[0], m2[2]]))
        n_rem = min(budget * sum(fracs), FR.sum() * 0.999)
        pi_rem = np.zeros(n)
        pi_rem[FR] = W.sampling_probs_clip(np.sqrt(cd[FR]), n_rem, cfg["pi_min_frac"] * n_rem / FR.sum())
        newv = []
        for b in (0, 2):
            v_rem = W.predicted_var(G, pi_rem, m2[b], n_q)[:P]
            e_now = G[:P] @ F[b] / n_q
            newv.append(W.cert_weights(e_now, v_rem, np.maximum(v_rem, 1e-30), cfg["z_plan"]))
        (w0, pc0), (w1, pc1) = newv
        w_rows = [w0 * pc1, w1 * pc0]
    wd = np.sqrt(np.maximum(cd, 0.0)) * FR
    pi = np.zeros(n)
    pi[FR] = W.sampling_probs_clip(wd[FR], n_t, pi_min)
    return pi


def calib(p, y):
    k = np.clip(np.digitize(p, EDGES[1:-1]), 0, 9)
    ece = sum((k == b).sum() * abs(p[k == b].mean() - y[k == b].mean()) for b in np.unique(k)) / len(p)
    return (p - y).mean(), ((p - y) ** 2).mean(), ece


# name: (z key, labels used, label weights: None = 1, "pi" = 1/pi, "scaled" = mean(1/pi))
VARIANTS = {"cont": ("", False, None), "unw": ("", True, None), "w": ("", True, "pi"), "scaled": ("", True, "scaled"),
            "unw+fam": ("fam", True, None), "unw+sys": ("sys", True, None), "w+sys": ("sys", True, "pi"),
            "unw C0.1": ("C0.1", True, None)}
rows = []
out = f"{cfg['out_dir']}/diag_adsel_{cfg['tag']}_r{REP_LO}-{REP_HI}_raw.csv"
for rep in range(REP_LO, REP_HI):
    t0 = time.time()
    rng = np.random.default_rng([cfg["seed"], rep])
    o, _ = W.simulate_qrels(rng, D.y, D.h, cfg["mnar"]["a"], cfg["mnar"]["b"], cfg["mnar"]["a_fp"], cfg["mnar"]["c"],
                            shift=SHIFT)
    zero = {k: W.pu_fuse(o, zm, X_PU, np.array([], int), np.array([], int), 0.5, C=C)[3] for k, (zm, C) in Z.items()}
    lab_all = np.flatnonzero(FR)
    census = W.pu_fuse(o, Z_PU, X_PU, lab_all, D.y[lab_all], 0.5, w0=zero[""], clamp_out=False)[3]
    p_cen = census[:, 1] + census[:, 2]
    F = [zero[""][:, 2], 0.5 * zero[""][:, 2] + 0.5 * (zero[""][:, 1] + zero[""][:, 2]), zero[""][:, 1] + zero[""][:, 2]]
    V = [F[0] * (1 - F[0]), 0.25 * F[0] * (1 - F[0]) + 0.25 * F[2] * (1 - F[2]) + 0.5 * F[0] * (1 - F[2]),
         F[2] * (1 - F[2])]
    for budget in BUDGETS:
        n1 = budget * cfg["fracs"][0]
        pis = {"decision": decision_pi(F, V, budget), "uniform": np.where(FR, n1 / FR.sum(), 0.0)}
        for design, pi in pis.items():
            xi = FR & (rng.random(n) < pi)
            lab = np.flatnonzero(xi)
            ev = FR & ~o & ~xi
            su = xi & ~o
            r_un = (D.u1 - p_cen)[su]
            sel = dict(sel_unw=r_un.mean(), sel_ht=np.average(r_un, weights=1.0 / pi[su]),
                       sel_frame=(D.u1 - p_cen)[FR & ~o].mean(), n_lab=len(lab), n_lab_u=int(su.sum()))
            preds = {"zero": zero[""], "census": census}
            for nm, (zk, use_lab, wt) in VARIANTS.items():
                zm, C = Z[zk]
                li = lab if use_lab else np.array([], int)
                lw = {None: None, "pi": 1.0 / pi[li], "scaled": np.full(len(li), (1.0 / pi[li]).mean())}[wt]
                preds[nm] = W.pu_fuse(o, zm, X_PU, li, D.y[li], 0.5, w0=zero[zk], clamp_out=False, lab_w=lw, C=C)[3]
            p_cont = preds["cont"][:, 1] + preds["cont"][:, 2]
            for nm, pw in preds.items():
                p1 = pw[:, 1] + pw[:, 2]
                bias, brier, ece = calib(p1[ev], D.u1[ev])
                rows.append(dict(rep=rep, budget=budget, design=design, variant=nm, bias=bias, brier=brier, ece=ece,
                                 to_census=(p1 - p_cen)[ev].mean(), to_cont=(p1 - p_cont)[ev].mean(), **sel))
    print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
g = df.groupby(["budget", "design", "variant"])
tab = g[["bias", "brier", "ece", "to_census", "to_cont"]].mean()
se = g[["bias", "brier", "to_cont"]].std() / np.sqrt(g.size().to_numpy()[:, None])
tab["bias_se"], tab["brier_se"], tab["to_cont_se"] = se["bias"], se["brier"], se["to_cont"]
print(tab.round(4).to_string())
print(df.groupby(["budget", "design"])[["sel_unw", "sel_ht", "sel_frame", "n_lab", "n_lab_u"]].mean().round(4).to_string())
print("DONE")
