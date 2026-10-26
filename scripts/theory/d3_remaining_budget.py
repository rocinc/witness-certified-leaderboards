"""D3 label budget to close the pairs that the two-batch, anchor-calibrated analysis leaves open.

For every open pair p with the two endpoint estimates (rho = 0 and rho = 1) of the same sign, the single-pair label
complexity and the joint leaderboard budget (witness.core.label_complexity, witness.core.leaderboard_budget) are
evaluated at the target level t = z*(k) + z_beta, with k the number of pairs certified by the final analysis,
z*(k) = Phi^{-1}(1 - alpha k / (2P)) and z_beta = Phi^{-1}(power).  The effect sizes Delta_p(rho_b) are the final
anchor-calibrated estimates of <out_dir>/d3_results_anchor.json; the residual scale sigma_d is the cell-calibrated
working standard deviation of witness.d3.working_var on the labels of both batches.  The labels already bought
(batch pi) are not subtracted: the budget is the total expected number of labels of a design built for the pairs.
A pair whose endpoints disagree in sign has no budget (its endpoint nulls both hold at the point estimate).
Closed-form reference: n_p = n_now (t / z_p)^2 with z_p = min_b |Delta_pb| / SE_pb, the 1/n extrapolation of the
current design.
usage: python scripts/theory/d3_remaining_budget.py configs/d3_duretrieval.yaml <annotated_1.xlsx> <annotated_2.xlsx> [power]
"""
import json
import sys

import numpy as np
from scipy.stats import norm
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
power = float(sys.argv[4]) if len(sys.argv) > 4 else 0.80
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sys.argv[2:4])]
P, G, n_q, alpha = D.P, D.G, D.n_q, cfg["alpha"]
res = json.load(open(f"{cfg['out_dir']}/d3_results_anchor.json"))["methods"]["design_fam_anchor_mi"]
est = [np.array(e)[:P] for e in res["est"]]
se = [np.sqrt(np.array(v)[:P]) for v in res["var"]]
k = int(np.sum(res["cert_plus"]) + np.sum(res["cert_minus"]))
z_star = norm.isf(alpha * k / (2 * P))
z_beta = norm.ppf(power)
t_target = z_star + z_beta
n_now = sum(len(bt.items) for bt in batches)
print(f"certified k={k} of {P}; z*={z_star:.3f}; z_beta={z_beta:.3f} (power {power}); target t={t_target:.3f}; labels so far {n_now}")

sigma = [np.sqrt(WD.working_var(D, b, batches)) for b in (0, 2)]
# The final variance includes the between-imputation part of the anchor calibration and the GREG fit, which the
# working-model variance of the realised design omits.  Scale sigma_b by the median ratio so that the working model
# reproduces the realised variance of the design that was run.
for i, b in enumerate((0, 2)):
    pred = []
    for bt in batches:
        pred.append(W.predicted_var(G, bt.pi, np.broadcast_to(sigma[i] ** 2, G.shape), n_q)[:P])
    om = [1.0 / np.sqrt(np.maximum(v, 1e-16)) for v in pred]
    pred_final = sum(o ** 2 * v for o, v in zip(om, pred)) / sum(om) ** 2
    ratio = float(np.median(np.array(res["var"][b])[:P] / pred_final))
    print(f"endpoint rho={b // 2}: realised / working-model variance, median {ratio:.2f}")
    sigma[i] = sigma[i] * np.sqrt(ratio)
same = (np.sign(est[0]) == np.sign(est[2])) & (est[0] != 0) & (est[2] != 0)
open_pairs = [p for p in range(P) if not (res["cert_plus"][p] or res["cert_minus"][p])]
rows = []
for p in open_pairs:
    a, b = D.pairs[p]
    z_p = min(abs(est[0][p]) / se[0][p], abs(est[2][p]) / se[2][p])
    extrap = n_now * (t_target / z_p) ** 2 if same[p] else float("inf")
    n_single = max(W.label_complexity(G[p][None], sigma[i], np.array([est[e][p]]), n_q, t_target)[0]
                   for i, e in enumerate((0, 2))) if same[p] else float("inf")
    rows.append((p, D.systems[a], D.systems[b], est[0][p], est[2][p], z_p, same[p], extrap, n_single))
rows.sort(key=lambda r: r[7])
print("pair | Delta(0) Delta(1) | z_p (current) | same sign | 1/n extrapolation (total labels) | single-pair complexity")
for p, a, b, d0, d1, z_p, ok, ex, ns in rows:
    print(f"{a:14s} vs {b:14s} {d0:+.3f} {d1:+.3f}  z={z_p:.2f}  {'yes' if ok else 'NO '}  {ex:10.0f}  {ns:10.0f}")

closable = [r for r in rows if r[6]]
order = [r[0] for r in closable]
print("joint budget to close the m easiest open pairs together (both endpoints, shared sample):")
for m in range(1, len(order) + 1):
    sel = np.array(order[:m])
    g_rows = np.vstack([G[sel], G[sel]])
    delta = np.concatenate([est[0][sel], est[2][sel]])
    sig_rows = np.sqrt(0.5 * (sigma[0] ** 2 + sigma[1] ** 2))
    budget, _, viol = W.leaderboard_budget(g_rows, sig_rows, delta, n_q, t_target, iters=4000)
    print(f"  m={m} ({closable[m - 1][1]} vs {closable[m - 1][2]}): {budget:9.0f} expected labels, max relative violation {viol:+.1e}")
print("DONE")
