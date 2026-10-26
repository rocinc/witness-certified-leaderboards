"""D1 testbed: T2Ranking dev rerank setting, every candidate carries a 0-3 human label.

Samples `n_queries` dev queries with at least `min_candidates` labelled candidates, pulls the candidate passages
from collection.tsv, and writes
  <out_dir>/items.parquet   one row per (query, candidate): qi, qid, pid, rel (0-3), query, passage
usage: python scripts/data/build_d1.py configs/testbeds/d1_t2ranking.yaml
"""
import os
import sys

import numpy as np
import pandas as pd
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
rng = np.random.default_rng(cfg["seed"])
D = cfg["data_dir"]
os.makedirs(cfg["out_dir"], exist_ok=True)

qrels = pd.read_csv(f"{D}/qrels.dev.tsv", sep="\t").rename(columns={"rel": "rel"})[["qid", "pid", "rel"]]
qrels = qrels.drop_duplicates(["qid", "pid"])
size = qrels.groupby("qid").size()
# a query needs both relevant and non-relevant candidates for a ranking to matter
has_pos = qrels[qrels.rel >= 2].groupby("qid").size()
has_neg = qrels[qrels.rel <= 1].groupby("qid").size()
ok = size[(size >= cfg["min_candidates"])].index.intersection(has_pos.index).intersection(has_neg.index)
qsel = np.sort(rng.choice(np.array(ok), size=cfg["n_queries"], replace=False))
items = qrels[qrels.qid.isin(set(qsel))].copy()
print(f"eligible queries={len(ok)} sampled={len(qsel)} items={len(items)}")

queries = pd.read_csv(f"{D}/queries.dev.tsv", sep="\t", quoting=3)
queries = queries.set_index("qid")["text"]
need = set(items.pid)
texts = {}
for chunk in pd.read_csv(f"{D}/collection.tsv", sep="\t", quoting=3, chunksize=500_000, dtype={"pid": np.int64, "text": str}):
    hit = chunk[chunk.pid.isin(need)]
    texts.update(zip(hit.pid, hit.text))
print(f"passages found={len(texts)} of {len(need)}")
assert len(texts) == len(need)

items["query"] = items.qid.map(queries)
items["passage"] = items.pid.map(texts)
assert items["query"].notna().all() and items["passage"].notna().all()
qi = {q: i for i, q in enumerate(qsel)}
items["qi"] = items.qid.map(qi)
items = items.sort_values(["qi", "pid"]).reset_index(drop=True)
items[["qi", "qid", "pid", "rel", "query", "passage"]].to_parquet(f"{cfg['out_dir']}/items.parquet")
print(items.rel.value_counts().sort_index().to_dict())
print("candidates per query", items.groupby("qi").size().describe().round(1).to_dict())
print("DONE")
