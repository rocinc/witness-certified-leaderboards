"""D3 next human-label batch, planned from all earlier batches (M3 decision-aware allocation).

Frame for the new batch: frame items not labelled in any earlier batch, both strata (qrels entries and the rest).
Pair weights: w_p = E[t phi(t - z)] / (2 V_final) per endpoint (rho = 0, 1), with Delta and its SE from the earlier
batches' design-based estimate and V_final the variance after this batch; the endpoint weights are multiplied by the
other endpoint's predicted certification probability.  Item weights: Neyman sqrt(sum_p w_p g_pd^2 s2_d) with the
cell-calibrated working variance s2 from the earlier labels; pi = clip(kappa * weight, pi_min, 1) with sum pi = n.
Writes batch<stage>_pi.npy, batch<stage>_key.json, batch<stage>_sheet.xlsx and prints the planned certification.
usage: python scripts/d3/make_next_batch.py configs/d3_duretrieval.yaml <stage> <n_labels> <annotated_1.xlsx> [...]
"""
import json
import os
import sys

import numpy as np
from witness import core as W  # noqa: E402
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
stage, n_new = int(sys.argv[2]), int(sys.argv[3])
sheets = sys.argv[4:]
assert len(sheets) == stage - 1
B = cfg["next_batch"]
rng = np.random.default_rng([cfg["batch"]["seed"], stage])
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sheets)]
labelled = np.zeros(D.N, bool)
for bt in batches:
    labelled[bt.items] = True
frame_new = ~labelled
est, var = WD.design_estimate(D, batches, B["extras"], cfg["ridge"], D.F)
P, G, n_q = D.P, D.G, D.n_q
z = B["z_plan"]
s2 = [WD.working_var(D, b, batches) for b in (0, 2)]
pi_min = B["pi_min_frac"] * n_new / frame_new.sum()
w_rows = [np.ones(P), np.ones(P)]
for _ in range(B["plan_iters"]):
    cd = sum((w[:, None] * G[:P] ** 2 * s[None, :]).sum(0) for w, s in zip(w_rows, s2))
    pi = np.zeros(D.N)
    pi[frame_new] = W.sampling_probs_clip(np.sqrt(np.maximum(cd[frame_new], 0.0)), n_new, pi_min)
    newv = []
    for bi, b in enumerate((0, 2)):
        v_new = W.predicted_var(G, pi, np.broadcast_to(s2[bi], G.shape), n_q)[:P]
        v_fin = 1.0 / (1.0 / np.maximum(var[b][:P], 1e-30) + 1.0 / np.maximum(v_new, 1e-30))
        newv.append(W.cert_weights(est[b][:P], var[b][:P], v_fin, z))
    (w0, pc0), (w1, pc1) = newv
    w_rows = [w0 * pc1, w1 * pc0]
pc_both = pc0 * pc1
for p in np.argsort(-pc_both):
    a, b_ = D.pairs[p]
    print(f"   {D.systems[a]:14s} vs {D.systems[b_]:14s} D(0)={est[0][p]:+.4f} (se {np.sqrt(var[0][p]):.4f}) "
          f"D(1)={est[2][p]:+.4f} (se {np.sqrt(var[2][p]):.4f})  planned P(certified after batch {stage}) = {pc_both[p]:.2f}")
print(f"expected number of pairs certified after batch {stage} (both endpoints, z={z}): {pc_both.sum():.1f} of {P}")
xi = frame_new & (rng.random(D.N) < pi)
rows = np.flatnonzero(xi)
print(f"batch {stage}: designed n={n_new}, drawn {len(rows)} (L {D.o[rows].sum()}, U {(~D.o[rows]).sum()}); "
      f"pi range [{pi[frame_new].min():.2e}, {pi[frame_new].max():.3f}]")
out = cfg["out_dir"]
for path in (f"{out}/batch{stage}_pi.npy", f"{out}/batch{stage}_key.json", f"{out}/batch{stage}_sheet.xlsx"):
    assert not os.path.exists(path), path
np.save(f"{out}/batch{stage}_pi.npy", pi)
key, n_unique = WD.write_sheet(D, rows, f"{out}/batch{stage}_sheet.xlsx", cfg["batch"]["repeat_share"], rng)
json.dump(dict(stage=stage, systems=D.systems, pairs=D.pairs.tolist(), n_rows=len(key), n_unique=n_unique, rows=key),
          open(f"{out}/batch{stage}_key.json", "w"), ensure_ascii=False)
print(f"sheet rows={len(key)} unique={n_unique}")
print("DONE")
