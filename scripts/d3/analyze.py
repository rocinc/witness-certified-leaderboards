"""D3 certified DuRetrieval leaderboard from all human-label batches (real qrels, real judges).

Estimators of Delta_p(rho), rho in {0, 1/2, 1}, for the 21 system pairs:
  official        qrels gain 1, query CLT (the usual significance test)
  judge_plugin    qrels gain 1, holes = judges' mean gain, query CLT
  design_b1       design-based GREG on batch 1 only (single-batch reference)
  design          all batches, one GREG fit on every label ('final', PPI++-style), stage weights 1/sqrt(pred. var)
  design_fam      the same with family provenance columns
  design_prev     all batches, exactly design-unbiased: batch t's predictor is fitted on batches < t only
                  (batch 1 uses the plain judge plug-in)
Certification: endpoint intersection-union + BH and BY over the 2P one-sided nulls (alpha from the config).
Coupling test per pair: judge plug-in minus the design estimate, an HT-type estimate of the plug-in bias,
flagged when it exceeds 2 SE and moves the plug-in across 0.
Planned vs realised: the batch planner's predicted P(certified after batch t) (parsed from its log) against the
realised certification.
usage: python scripts/d3/analyze.py configs/d3_duretrieval.yaml <planner_log> <annotated_1.xlsx> [...]
"""
import os
import re
import sys

import numpy as np
from scipy.stats import norm
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
planner_log = sys.argv[2]
sheets = sys.argv[3:]
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sheets)]
P, G, n_q = D.P, D.G, D.n_q
for bt in batches:
    print(f"batch {bt.stage}: labelled rows {bt.m_rows}, unique items {len(bt.items)}/{bt.n_unique}, "
          f"repeats agreement {np.mean(bt.agree) if bt.agree else float('nan'):.3f} over {len(bt.agree)}, "
          f"label shares {np.round(np.bincount(bt.y, minlength=3) / len(bt.y), 3).tolist()}")
all_items = np.concatenate([bt.items for bt in batches])
assert len(np.unique(all_items)) == len(all_items), "an item was labelled in two batches"


def query_var(contrib):
    per_q = np.zeros((contrib.shape[0], n_q))
    np.add.at(per_q.T, D.qi, contrib.T)
    return per_q.var(1, ddof=1) / n_q


def certify(est, var):
    se = [np.sqrt(np.maximum(v[:P], 1e-30)) for v in var]
    p_plus = np.maximum(norm.sf(est[0][:P] / se[0]), norm.sf(est[2][:P] / se[2]))
    p_minus = np.maximum(norm.cdf(est[0][:P] / se[0]), norm.cdf(est[2][:P] / se[2]))
    out = {}
    for name, dep in (("BH", False), ("BY", True)):
        rej = W.bh(np.concatenate([p_plus, p_minus]), cfg["alpha"], dependent=dep)
        out[name] = (rej[:P], rej[P:])
    return out


def design_prev(batches_):
    """Exactly design-unbiased multi-batch estimator: batch t uses a GREG predictor fitted on batches < t."""
    est_out, var_out = [], []
    for b in range(3):
        Zc, b0 = WD.design_columns(D, b, [], D.F[b])
        u_all = np.zeros(D.N)
        for bt in batches_:
            u_all[bt.items] = WD.gains(bt.y)[b]
        num = den = vnum = 0.0
        known = np.zeros(D.N, bool)
        for t, bt in enumerate(batches_):
            if t == 0:
                H = np.tile(b0, (G.shape[0], 1)) @ Zc.T
            else:
                items = np.concatenate([x.items for x in batches_[:t]])
                pl = np.concatenate([x.pi[x.items] for x in batches_[:t]])
                H = W.greg_fit(G, Zc, u_all, items, pl, b0, cfg["ridge"]) @ Zc.T
            xi = np.zeros(D.N, bool)
            xi[bt.items] = True
            e_t, v_t = W.stage_estimate(G, known * u_all, np.zeros(D.N, bool), H, u_all, xi, bt.pi, n_q)
            m2 = np.broadcast_to(WD.working_var(D, b, batches_[:t]), G.shape)
            om = 1.0 / np.sqrt(np.maximum(W.predicted_var(G, bt.pi, m2, n_q), 1e-16))
            num, den, vnum = num + om * e_t, den + om, vnum + om ** 2 * v_t
            known |= xi
        est_out.append(num / den)
        var_out.append(vnum / den ** 2)
    return est_out, var_out


off = [D.o.astype(float)] * 3
results = {
    "official": ([G @ f / n_q for f in off], [query_var(G * f) for f in off]),
    "judge_plugin": ([G @ f / n_q for f in D.F], [query_var(G * f) for f in D.F]),
    "design_b1": WD.design_estimate(D, batches[:1], [], cfg["ridge"], D.F),
    "design": WD.design_estimate(D, batches, [], cfg["ridge"], D.F),
    "design_fam": WD.design_estimate(D, batches, ["fam"], cfg["ridge"], D.F),
    "design_prev": design_prev(batches),
}
cert = {}
for name, (est, var) in results.items():
    c = certify(est, var)
    cert[name] = c
    print(f"== {name}: certified (BH) {int(c['BH'][0].sum() + c['BH'][1].sum())}/{P}, "
          f"(BY) {int(c['BY'][0].sum() + c['BY'][1].sum())}/{P}")
    for p, (a, b_) in enumerate(D.pairs):
        tag = "A>B" if c["BH"][0][p] else "B>A" if c["BH"][1][p] else " ? "
        print(f"   {D.systems[a]:14s} vs {D.systems[b_]:14s} {tag}  D(.5)={est[1][p]:+.4f} se={np.sqrt(var[1][p]):.4f}  "
              f"D(0)={est[0][p]:+.4f} D(1)={est[2][p]:+.4f}")
# disagreements between the usual practice and the human-labelled design estimate
for name in ("official", "judge_plugin"):
    a_p, a_m = cert[name]["BH"]
    d_p, d_m = cert["design"]["BH"]
    unsupported = (a_p & ~d_p) | (a_m & ~d_m)
    contradicted = (a_p & d_m) | (a_m & d_p)
    print(f"== {name} vs design: certified by {name} but not by the human design estimate {unsupported.sum()}, "
          f"opposite direction certified {contradicted.sum()}")
# coupling test: plug-in minus design estimate
est_d, var_d = results["design"]
plug = results["judge_plugin"][0]
print("== judge plug-in bias per pair (rho=1/2) = plug-in - design estimate; FLAG if |bias| > 2 SE and it moves the plug-in across 0")
for p, (a, b_) in enumerate(D.pairs):
    bias, se = plug[1][p] - est_d[1][p], np.sqrt(var_d[1][p])
    flag = abs(bias) > 2 * se and np.sign(plug[1][p] - bias) != np.sign(plug[1][p])
    print(f"   {D.systems[a]:14s} vs {D.systems[b_]:14s} plug-in {plug[1][p]:+.4f} bias {bias:+.4f} (se {se:.4f}){'  FLAG' if flag else ''}")
# planned vs realised (the planner's per-pair predictions for the last batch)
pat = re.compile(r"^\s+(\S+)\s+vs (\S+)\s+D\(0\).*planned P\(certified after batch (\d+)\) = ([\d.]+)", re.M)
planned = {(m[0], m[1]): float(m[3]) for m in pat.findall(open(planner_log, encoding="utf8").read())
           if int(m[2]) == len(batches)}
assert len(planned) == P, len(planned)
real = cert["design_fam"]["BH"]
realised = real[0] | real[1]
pp = np.array([planned[(D.systems[a], D.systems[b_])] for a, b_ in D.pairs])
print(f"== planner (design_fam, z=2.5): expected certified {pp.sum():.1f}, realised {realised.sum()} (BH); "
      f"Brier score {np.mean((pp - realised) ** 2):.3f}")
import json  # noqa: E402

dump = dict(systems=D.systems, pairs=D.pairs.tolist(), n_q=n_q, n_labels=int(len(all_items)), planned=pp.tolist(),
            alpha=cfg["alpha"],
            methods={name: dict(est=[e.tolist() for e in est], var=[v.tolist() for v in var],
                                cert_plus=cert[name]["BH"][0].tolist(), cert_minus=cert[name]["BH"][1].tolist())
                     for name, (est, var) in results.items()})
json.dump(dump, open(f"{cfg['out_dir']}/d3_results.json", "w"))
print(f"wrote {cfg['out_dir']}/d3_results.json")
for p in np.argsort(-pp):
    a, b_ = D.pairs[p]
    print(f"   {D.systems[a]:14s} vs {D.systems[b_]:14s} planned {pp[p]:.2f} realised {int(realised[p])}")
print("DONE")
