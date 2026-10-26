"""D3 findings from the human-labelled batch (DuRetrieval, real qrels, real judges), all design-weighted (1/pi).

1. qrels quality: among qrels entries in the frame (stratum L), the share humans label 0 / 1 / 2.
2. holes: among unlabelled frame items (stratum U), the share humans label >= 1; the implied number of holes.
3. per-system pool-familiarity decomposition at rho (an exact identity for the realised qrels):
     DCG_s(true) = DCG_s(official) - E_s + B_s,
     B_s = (1/Q) sum_{d in U} disc_s u_d          (relevant mass the official metric misses: holes)
     E_s = (1/Q) sum_{d in L} disc_s (1 - u_d)    (mass the official metric over-credits: qrels gain 1 vs true u)
   HT estimates over the sampled items.
4. judges vs humans: weighted exact / binary agreement and the confusion table.
5. provenance coupling on real annotations: within joint judge state, human relevance of items retrieved only by
   lexical (BM25) systems vs only by neural systems (state-weighted gap, HT within cells).
Uncertainty: bootstrap of the sampled items with their 1/pi weights (with-replacement approximation to the Poisson
design; it ignores the finite-population correction, so it is slightly conservative).
usage: python scripts/d3/findings.py configs/d3_duretrieval.yaml <annotated.xlsx> <rho>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
rho = float(sys.argv[3])
OUT = cfg["out_dir"]
frame = pd.read_parquet(f"{OUT}/frame.parquet")
z = np.load(f"{OUT}/frame_disc.npz")
assert (z["qi"] == frame.qi.to_numpy()).all() and (z["di"] == frame.di.to_numpy()).all()
disc, systems, n_q = z["disc"], list(z["systems"]), int(z["n_q"])
key = json.load(open(f"{OUT}/batch1_key.json", encoding="utf8"))
pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(frame.qi, frame.di))}
lab_of_row = WD.sheet_labels(sys.argv[2])
first = {}
for k in sorted(lab_of_row):
    kr = key["rows"][k - 1]
    assert kr["row"] == k
    if not kr["repeat"]:
        first[pos[(kr["qi"], kr["di"])]] = lab_of_row[k]
assert len(first) == key["n_unique"], "the full sheet is expected (every unique row labelled)"
S = np.array(sorted(first))  # sampled frame items
yh = np.array([first[i] for i in S])
pi = frame.pi.to_numpy()[S]
w = 1.0 / pi
o = frame.in_qrels.to_numpy()[S]
pro, fla = frame.pro.to_numpy()[S], frame.flash.to_numpy()[S]
u = (1 - rho) * (yh == 2) + rho * (yh >= 1)
D = disc[:, S]  # [S_sys, n]
lex = np.array([s.startswith("bm25") for s in systems])
by_lex, by_neu = (D[lex] > 0).any(0), (D[~lex] > 0).any(0)
grp = np.where(by_lex & ~by_neu, 0, np.where(~by_lex & by_neu, 1, 2))
state = (pro + 1) + 4 * (fla + 1)
N_L, N_U = int(frame.in_qrels.sum()), int((~frame.in_qrels).sum())
rng = np.random.default_rng(0)
B_REPS = 2000


def boot(stat):
    """Point estimate and percentile 95% CI of stat(idx) over a with-replacement resample of the sampled items."""
    pt = stat(np.arange(len(S)))
    bs = np.array([stat(rng.integers(0, len(S), len(S))) for _ in range(B_REPS)])
    return pt, np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)


def wmean(x, m, idx):
    ww = w[idx] * m[idx]
    return (ww * x[idx]).sum() / ww.sum()


print(f"rho={rho}; sampled items={len(S)} (L {o.sum()}, U {(~o).sum()}); frame L={N_L}, U={N_U}, queries={n_q}")
for k in (0, 1, 2):
    pt, lo, hi = boot(lambda i, k=k: wmean((yh == k).astype(float), o, i))
    print(f"[1] qrels entries labelled {k} by humans: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
pt, lo, hi = boot(lambda i: wmean((yh >= 1).astype(float), ~o, i))
print(f"[2] unlabelled frame items relevant (>=1): {pt:.3f} [{lo:.3f}, {hi:.3f}]  -> about {pt * N_U:.0f} holes "
      f"vs {N_L} qrels entries in the frame")
pt, lo, hi = boot(lambda i: wmean((yh == 2).astype(float), ~o, i))
print(f"[2] unlabelled frame items fully relevant (2): {pt:.3f} [{lo:.3f}, {hi:.3f}]")


def decomp(idx):
    ww = w[idx]
    Bs = (D[:, idx] * (ww * u[idx] * ~o[idx])[None, :]).sum(1) / n_q
    Es = (D[:, idx] * (ww * (1 - u[idx]) * o[idx])[None, :]).sum(1) / n_q
    return np.concatenate([Bs, Es])


official = (disc * frame.in_qrels.to_numpy()[None, :]).sum(1) / n_q
pt, lo, hi = boot(decomp)
ns = len(systems)
print("[3] per system: official DCG@10 | B_s holes missed [CI] | E_s qrels over-credit [CI] | corrected = off - E + B")
for s in range(ns):
    print(f"   {systems[s]:14s} {official[s]:.4f} | {pt[s]:.4f} [{lo[s]:.4f}, {hi[s]:.4f}] | "
          f"{pt[ns + s]:.4f} [{lo[ns + s]:.4f}, {hi[ns + s]:.4f}] | {official[s] - pt[ns + s] + pt[s]:.4f}")
# judges vs humans (weighted)
for name, jl in (("pro", pro), ("flash", fla)):
    ex = boot(lambda i, jl=jl: (w[i] * (jl[i] == yh[i])).sum() / w[i].sum())
    bi = boot(lambda i, jl=jl: (w[i] * ((jl[i] >= 1) == (yh[i] >= 1))).sum() / w[i].sum())
    print(f"[4] judge {name}: weighted exact agreement {ex[0]:.3f} [{ex[1]:.3f}, {ex[2]:.3f}], "
          f"binary {bi[0]:.3f} [{bi[1]:.3f}, {bi[2]:.3f}]")
    tab = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            tab[a, b] = w[(jl == a) & (yh == b)].sum()
    print("   weighted confusion (rows judge 0/1/2, cols human 0/1/2, share of weight):", np.round(tab / tab.sum(), 3).tolist())
# provenance coupling on real annotations
cells = [s for s in np.unique(state) if min(((state == s) & (grp == 0)).sum(), ((state == s) & (grp == 1)).sum()) >= 5]
print("[5] provenance groups:", {n: int((grp == k).sum()) for k, n in enumerate(("lex_only", "neural_only", "both"))},
      "; judge states with >= 5 items in both groups:", [(int(s % 4) - 1, int(s // 4) - 1) for s in cells])
for s in cells:
    m0, m1 = (state == s) & (grp == 0), (state == s) & (grp == 1)
    print(f"   state (pro,flash)=({int(s % 4) - 1},{int(s // 4) - 1}): lex_only n={m0.sum()} u={np.average(u[m0], weights=w[m0]):.3f}"
          f" | neural_only n={m1.sum()} u={np.average(u[m1], weights=w[m1]):.3f}")


def gap(idx):
    num = den = 0.0
    for s in cells:
        m0 = idx[(state[idx] == s) & (grp[idx] == 0)]
        m1 = idx[(state[idx] == s) & (grp[idx] == 1)]
        if len(m0) == 0 or len(m1) == 0:
            return np.nan
        wt = min(((state == s) & (grp == 0)).sum(), ((state == s) & (grp == 1)).sum())
        num += wt * (np.average(u[m0], weights=w[m0]) - np.average(u[m1], weights=w[m1]))
        den += wt
    return num / den


if cells:
    pt, lo, hi = boot(gap)
    print(f"[5] state-weighted human relevance gap u(lex_only) - u(neural_only) = {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("DONE")
