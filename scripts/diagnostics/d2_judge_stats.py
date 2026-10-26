"""D2 summary: per-system true DCG@5 and LLM-judge agreement with the TREC DL 2019 labels.

DCG@5 per query = sum_{r=1..5} g(rel_r) / log2(r + 1), g = 1[rel=3] + 0.5*1[rel=2] (rel 0-3), ranking by the system's
score within the query, ties broken by items.parquet row order; reported as the mean over queries.
Judge agreement maps TREC rel 0,1 -> 0; 2 -> 1; 3 -> 2 (the judge's 0-2 scale); exact = label equality, binary = (>=1)
equality; refused items are excluded from the agreement denominators and counted separately.
usage: python scripts/diagnostics/d2_judge_stats.py <config.yaml> <judge_name>:<judged.jsonl>:<judge.log> [...]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from witness.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1])
items = pd.read_parquet(f"{cfg['out_dir']}/items.parquet")
assert (items.qi.diff().dropna() >= 0).all()
rel = items.rel.to_numpy()
gain = (rel == 3) + 0.5 * (rel == 2)  # [n_items]
DISC = 1 / np.log2(np.arange(2, 7))  # ranks 1..5
groups = [np.flatnonzero(items.qi.to_numpy() == q) for q in np.sort(items.qi.unique())]


def dcg5(s):
    per_q = []
    for idx in groups:
        # stable sort on -score keeps row order among ties
        top = idx[np.argsort(-s[idx], kind="stable")[:5]]
        per_q.append((gain[top] * DISC[: len(top)]).sum())
    return np.mean(per_q)


print(f"items={len(items)} queries={len(groups)} rel_counts={items.rel.value_counts().sort_index().to_dict()}")
ideal = dcg5(gain.astype(np.float64))
print(f"oracle DCG@5={ideal:.4f}")
for path in sorted(glob.glob(f"{cfg['out_dir']}/scores/*.npy")):
    s = np.load(path)
    assert s.shape == (len(items),) and np.isfinite(s).all()
    print(f"DCG@5 {os.path.basename(path)[:-4]:14s} {dcg5(s.astype(np.float64)):.4f}")

mapped = np.select([rel <= 1, rel == 2, rel == 3], [0, 1, 2])
ids = [f"dl19:{r.qi}:{r.pid}" for r in items.itertuples()]
for spec in sys.argv[2:]:
    jname, jpath, lpath = spec.split(":")
    recs = [json.loads(line) for line in open(jpath, encoding="utf8")]
    assert [r["id"] for r in recs] == ids
    assert all(r["status"] in ("ok", "refused") for r in recs)
    ok = np.array([r["status"] == "ok" for r in recs])
    lab = np.array([r["label"] if r["status"] == "ok" else -1 for r in recs])
    done = [line for line in open(lpath, encoding="utf8") if line.startswith("DONE")]
    assert len(done) == 1 and open(lpath, encoding="utf8").read().rstrip().endswith("EXIT:0")
    exact = (lab[ok] == mapped[ok]).mean()
    binary = ((lab[ok] >= 1) == (mapped[ok] >= 1)).mean()
    conf = pd.crosstab(pd.Series(rel[ok], name="trec_rel"), pd.Series(lab[ok], name="judge"))
    print(f"judge {jname}: n_ok={ok.sum()} refused={(~ok).sum()} exact={exact:.4f} binary={binary:.4f}")
    print(f"  label_counts={pd.Series(lab[ok]).value_counts().sort_index().to_dict()}")
    print("  " + conf.to_string().replace("\n", "\n  "))
    print(f"  log: {done[0].strip()}")
